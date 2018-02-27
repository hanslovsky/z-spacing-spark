package org.janelia.thickness.similarity;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Callable;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.thickness.ZSpacing;
import org.janelia.thickness.utility.Grids;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;
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

		final SparkConf conf = new SparkConf().setAppName( MethodHandles.lookup().lookupClass().getName() );

		try (final JavaSparkContext sc = new JavaSparkContext( conf ))
		{

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

			for ( int d = 0; d < stepSizes.length; ++d ) {
				stepSizes[ d ] = imgDim [ d ];
				radii[ d ] = stepSizes[ d ] / 2;
			}

			for ( int i = 0; i < cmdLineArgs.range.length; ++i )
			{

				for ( int d = 0; d < stepSizes.length; ++d )
				{
					stepSizes[ d ] = Math.max( stepSizes[ d ] / 2, 1 );
					radii[ d ] = Math.max( stepSizes[ d ] / 2, 1 );
				}
				if ( Arrays.stream( radii ).filter( r -> r > 1 ).count() == 0 ) {
					break;
				}

				final JavaRDD< Interval > blocksRDD = sc
						.parallelize( Grids.collectAllOffsets( imgDim, Arrays.stream( stepSizes ).mapToInt( l -> ( int ) l ).toArray() ) )
						.map( min -> {
							final long[] max = new long[ min.length ];
							Arrays.setAll( max, d -> Math.min( min[ d ] + stepSizes[ d ], imgDim[ d ] ) - 1 );
							return new FinalInterval( min, max );
						} );

				MatricesFromN5.makeMatrices(
						blocksRDD,
						new int[] { 1, 1 },
						radii,
						stepSizes,
						filenames.size(),
						cmdLineArgs.range[ i ],
						cmdLineArgs.root,
						"integral-sum",
						"integral-sum-squared",
						"matrices",
						new GzipCompression(),
						new DoubleType() );
			}

		}
	}

}
