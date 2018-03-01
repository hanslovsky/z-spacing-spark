package org.janelia.thickness.similarity;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.stream.LongStream;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.thickness.KryoSerialization;
import org.janelia.thickness.ZSpacing;
import org.janelia.thickness.utility.Grids;

import net.imglib2.type.numeric.real.DoubleType;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

public class MakeMatrices
{

	@Command(
			name = "make-matrices",
			sortOptions = true )
	public static class CommandLineArguments implements Callable< Boolean >
	{
		@Parameters( index = "0", description = "Path to file containing list paths to image files" )
		private String images;

		@Parameters( index = "1", description = "N5 root for integral images and matrices." )
		private String root;

		final int[] defaultIntegralImageBlockSize = { 1000, 1000 };

		@Option(
				names = { "--integral-image-blocksize" },
				split = ",",
				required = false,
				description = "Block size for 2D integral images. Default = [1000, 1000]" )
		int[] integralImageBlockSize;

		@Option( names = { "-h", "--help" }, usageHelp = true, description = "Show this help message and exit." )
		private boolean helpRequested;

		@Option(
				names = { "-r", "--range" },
				required = true,
				split = ",",
				description = "Z-range for each level in the hierarchy. range_i >= range_{i+1} for all i!" )
		private int[] range;

		@Override
		public Boolean call() throws Exception
		{

			integralImageBlockSize = Optional.ofNullable( integralImageBlockSize ).orElse( defaultIntegralImageBlockSize );

			int minRange = range[ 0 ];
			for ( final int r : range )
			{
				if ( r > minRange ) { throw new IllegalArgumentException( "Expected ranges to follow r_0 >= r_1 >= r2 >= ... but got " + Arrays.toString( range ) ); }
				minRange = r;
			}

			return true;
		}
	}

	public static void main( final String[] args ) throws IOException
	{

		final CommandLineArguments cmdLineArgs = new CommandLineArguments();
		final boolean parsedSuccessfully = Optional.ofNullable( CommandLine.call( cmdLineArgs, System.err, args ) ).orElse( false );

		System.out.println( "Parsed successfully? " + parsedSuccessfully );

		if ( !parsedSuccessfully ) { return; }

		final List< String > filenames = Files.readAllLines( Paths.get( cmdLineArgs.images ) );

		ZSpacing.n5Writer( cmdLineArgs.root ).setAttribute( "/", "filenames", filenames );

		final SparkConf conf = new SparkConf()
				.setAppName( MethodHandles.lookup().lookupClass().getName() )
				.set( "spark.serializer", "org.apache.spark.serializer.KryoSerializer" )
				.set( "spark.kryo.registrator", KryoSerialization.Registrator.class.getName() );

		try (final JavaSparkContext sc = new JavaSparkContext( conf ))
		{
			LogManager.getRootLogger().setLevel( Level.ERROR );

			GenerateIntegralImages.run(
					sc,
					filenames,
					cmdLineArgs.integralImageBlockSize,
					cmdLineArgs.range[ 0 ],
					cmdLineArgs.root,
					"integral-sum",
					"integral-sum-squared" );

			final long[] stepSizes = new long[ 2 ];
			final long[] radii = new long[ 2 ];
			final DatasetAttributes integralSumAttrs = ZSpacing.n5( cmdLineArgs.root ).getDatasetAttributes( "integral-sum" );
			final long[] imgDim = Arrays.stream( integralSumAttrs.getDimensions() ).limit( 2 ).map( l -> l - 1 ).toArray();

			for ( int d = 0; d < stepSizes.length; ++d )
			{
				stepSizes[ d ] = imgDim[ d ];
				radii[ d ] = stepSizes[ d ] / 2;
			}

			for ( int level = 0; level < cmdLineArgs.range.length; ++level )
			{

				if ( Arrays.stream( radii ).filter( r -> r > 1 ).count() == 0 )
				{
					break;
				}

				final int range = cmdLineArgs.range[ level ];

				final List< long[] > positions = Grids
						.collectAllOffsets( imgDim, Arrays.stream( stepSizes ).mapToInt( l -> ( int ) l ).toArray(), a -> divide( a, stepSizes, a ) );
				final JavaRDD< long[] > blocksRDD = sc
						.parallelize( positions );

				final long[] maxPosition = new long[] { 0, 0 };
				positions.forEach( p -> Arrays.setAll( maxPosition, d -> Math.max( maxPosition[ d ], p[ d ] ) ) );
				final long[] currentDim = Arrays.stream( maxPosition ).map( p -> p + 1 ).toArray();
				final long[] datasetDims = LongStream.concat( Arrays.stream( currentDim ), LongStream.of( range + 1, filenames.size() ) ).toArray();

				final String matrixDataset = level + "/matrices";

				ZSpacing.n5Writer( cmdLineArgs.root ).createDataset( matrixDataset, datasetDims, new int[] { 1, 1, range + 1, filenames.size() }, DataType.FLOAT64, new GzipCompression() );

				MatricesFromN5.makeMatrices(
						blocksRDD,
						radii,
						stepSizes,
						filenames.size(),
						range,
						Arrays.stream( imgDim ).map( d -> d - 1 ).toArray(),
						cmdLineArgs.root,
						"integral-sum",
						"integral-sum-squared",
						matrixDataset,
						new GzipCompression(),
						sc.broadcast( new DoubleType() ) );

				for ( int d = 0; d < stepSizes.length; ++d )
				{
					stepSizes[ d ] = Math.max( stepSizes[ d ] / 2, 1 );
					radii[ d ] = Math.max( stepSizes[ d ] / 2, 1 );
				}

			}

		}

	}

	public static long[] divide( final long[] dividend, final int[] divisor, final long[] quotient )
	{
		Arrays.setAll( quotient, d -> dividend[ d ] / divisor[ d ] );
		return quotient;
	}

	static public long[] divide( final long[] dividend, final long[] divisor, final long[] quotient )
	{
		Arrays.setAll( quotient, d -> dividend[ d ] / divisor[ d ] );
		return quotient;
	}

}
