package org.janelia.thickness.similarity;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.Compression;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.thickness.KryoSerialization;
import org.janelia.thickness.ZSpacing;
import org.janelia.thickness.utility.Grids;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import loci.formats.FormatException;
import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class MatricesFromN5
{

	private static class Parameters
	{

		@Argument( metaVar = "ROOT_DIRECTORY" )
		private String rootDirectory;

		private boolean parsedSuccessfully;
	}

	public static void main( final String[] args ) throws FormatException, IOException
	{

		final Parameters p = new Parameters();
		final CmdLineParser parser = new CmdLineParser( p );
		try
		{
			parser.parseArgument( args );
			p.parsedSuccessfully = true;
		}
		catch ( final CmdLineException e )
		{
			System.err.println( e.getMessage() );
			parser.printUsage( System.err );
			p.parsedSuccessfully = false;
		}

		if ( p.parsedSuccessfully )

		{
			final SparkConf conf = new SparkConf()
					.setAppName( MethodHandles.lookup().lookupClass().getName() )
					.set( "spark.network.timeout", "600" )
					.set( "spark.serializer", "org.apache.spark.serializer.KryoSerializer" )
					.set( "spark.kryo.registrator", KryoSerialization.Registrator.class.getName() );

			try (JavaSparkContext sc = new JavaSparkContext( conf ))
			{
				run( sc, p.rootDirectory );
			}
		}

	}

	public static void run(
			final JavaSparkContext sc,
			final String root ) throws FormatException, IOException
	{

		final N5Reader n5 = ZSpacing.n5( root );
		if ( !n5.exists( "/" ) ) { throw new RuntimeException( "N5 root does not exist!" ); }

		final int numScaleLevels = n5.getAttribute( "/", "scaleLevels", int.class );
		final long[] dims = n5.getAttribute( "/", "dimensions", long[].class );
		final long width = dims[ 0 ];
		final long height = dims[ 1 ];
		final long size = dims[ 2 ];

		final long[] dim = new long[] { width, height };

		final String datasetSum = n5.getAttribute( "/", "sumX", String.class );
		final String datasetSumSquared = n5.getAttribute( "/", "sumXX", String.class );
		final String datasetMatrix = Optional.ofNullable( n5.getAttribute( "/", "matrices", String.class ) ).orElse( "matrices" );

		for ( int level = 0; level < numScaleLevels; ++level )
		{

			final long[] radii = n5.getAttribute( "/" + level, "radii", long[].class );
			final long[] steps = n5.getAttribute( "/" + level, "steps", long[].class );
			final int range = n5.getAttribute( "/" + level, "range", int.class );

			final long[] currentDim = new long[] {
					Math.max( 1, ( long ) Math.ceil( ( dim[ 0 ] - radii[ 0 ] ) * 1.0 / steps[ 0 ] ) ),
					Math.max( 1, ( long ) Math.ceil( ( dim[ 1 ] - radii[ 1 ] ) * 1.0 / steps[ 1 ] ) )
			};
			final long numElements = Intervals.numElements( currentDim );
			final int stepSize = ( int ) Math.ceil( Math.max( numElements / sc.defaultParallelism(), 1 ) );
			final int[] blockSize = IntStream.generate( () -> stepSize ).limit( currentDim.length ).toArray();
			final List< Interval > blocks = Grids.collectAllContainedIntervals( currentDim, blockSize );
			final JavaRDD< Interval > blocksRDD = sc.parallelize( blocks );
			makeMatrices(
					blocksRDD,
					blockSize,
					radii,
					steps,
					size,
					range,
					root,
					datasetSum,
					datasetSumSquared,
					"/" + level + "/" + datasetMatrix,
					new GzipCompression(),
					new DoubleType() );

		}

	}

	public static < T extends NativeType< T > & RealType< T >, U extends NativeType< U > & RealType< U > > void makeMatrices(
			final JavaRDD< Interval > blocksRDD,
			final int[] blockSize,
			final long[] radius,
			final long[] step,
			final long size,
			final int range,
			final String root,
			final String datasetSum,
			final String datasetSumSquared,
			final String datasetMatrix,
			final Compression compression,
			final U matrixType )
	{
		blocksRDD.foreach( block -> {
			final long[] min = Intervals.minAsLongArray( block );
			final long[] max = Intervals.maxAsLongArray( block );
			final N5Writer writer = ZSpacing.n5Writer( root );

			final long[] matrixBlockMin = LongStream.concat( Arrays.stream( min ), LongStream.of( 0, 0 ) ).toArray();
			final long[] matrixBlockMax = LongStream.concat( Arrays.stream( max ), LongStream.of( range, size ) ).toArray();
			final Interval matrixBlockInterval = new FinalInterval( matrixBlockMin, matrixBlockMax );
			final Img< U > matrixBlock = new ArrayImgFactory< U >().create( matrixBlockInterval, matrixType );
			final RandomAccessibleInterval< RandomAccessibleInterval< U > > collapsedMatrixBlock = ZSpacing.collapseToMatrices( Views.translate( matrixBlock, matrixBlockMin ) );

			final RandomAccessibleInterval< T > sum = N5Utils.< T >open( writer, datasetSum );
			final RandomAccessibleInterval< T > sumSquared = N5Utils.< T >open( writer, datasetSumSquared );

			final long[] maxLimit = Intervals.maxAsLongArray( sum );

			final long[] correlationMin = new long[ 2 ];
			final long[] correlationMax = new long[ 2 ];

			for ( final Cursor< RandomAccessibleInterval< U > > c = Views.flatIterable( collapsedMatrixBlock ).cursor(); c.hasNext(); )
			{
				final RandomAccessibleInterval< U > matrix = c.next();
				Views.flatIterable( matrix ).forEach( v -> v.setReal( Double.NaN ) );
				final net.imglib2.RandomAccess< U > matrixAccess = matrix.randomAccess();

				// TODO is this the correct min and max for integral image
				// (dimension extended by one!)
				Arrays.setAll( correlationMin, d -> c.getLongPosition( d ) * step[ d ] );
				Arrays.setAll( correlationMax, d -> Math.min( c.getLongPosition( d ) * step[ d ] + 2 * radius[ d ] + 1, maxLimit[ d ] ) );

				for ( long z1 = 0; z1 < size; ++z1 )
				{
					matrixAccess.setPosition( z1, 1 );
					final IntervalView< T > sumX = Views.hyperSlice( sum, 2, z1 );
					final IntervalView< T > sumXX = Views.hyperSlice( Views.hyperSlice( sumSquared, 3, z1 ), 2, z1 );
					for ( long z2 = z1 + 1, r = 0; z2 < size && r < range; ++z2, ++r )
					{
						final IntervalView< T > sumY = Views.hyperSlice( sum, 2, z2 );
						final IntervalView< T > sumYY = Views.hyperSlice( Views.hyperSlice( sumSquared, 3, z2 ), 2, z2 );
						final IntervalView< T > sumXY = Views.hyperSlice( Views.hyperSlice( sumSquared, 3, z2 ), 2, z1 );
						matrixAccess.setPosition( r, 0 );
						final double correlation = correlation2DSquared( sumX, sumY, sumXX, sumYY, sumXY, new FinalInterval( correlationMin, correlationMax ) );
						// TODO calculate correlation from sumX, sumY, sumXX,
						// sumYY, sumXY for interval defined by correlationMin,
						// correlationMax
						matrixAccess.get().setReal( correlation );
					}
				}

			}

			final long[] matrixBlockPosition = LongStream.concat(
					IntStream.range( 0, blockSize.length ).mapToLong( d -> matrixBlockMin[ d ] / blockSize[ d ] ),
					LongStream.of( 0, 0 ) ).toArray();

			N5Utils.saveBlock( matrixBlock, writer, datasetMatrix, matrixBlockPosition );


		} );
	}

	private static < T extends RealType< T > > double correlation2DSquared(
			final RandomAccessibleInterval< T > sumXAccessible,
			final RandomAccessibleInterval< T > sumYAccessible,
			final RandomAccessibleInterval< T > sumXXAccessible,
			final RandomAccessibleInterval< T > sumYYAccessible,
			final RandomAccessibleInterval< T > sumXYAccessible,
			final Interval interval )
	{

		// Easiest to prove with expectations:
		// rho ^ 2 =
		// ( S_xy - S_x * S_y / n ) ^ 2
		// ---------------------------------------------------
		// ( S_xx - S_x * S_x / n ) * ( S_yy - S_y * S_y / n )

		final double S_x = fromIntegralImage( sumXAccessible, interval );
		final double S_y = fromIntegralImage( sumYAccessible, interval );
		final double S_xx = fromIntegralImage( sumXXAccessible, interval );
		final double S_yy = fromIntegralImage( sumYYAccessible, interval );
		final double S_xy = fromIntegralImage( sumXYAccessible, interval );

		final long n = Intervals.numElements( interval );

		final double var_xy = S_xy - S_x * S_y / n;
		final double var_xx = S_xx - S_x * S_x / n;
		final double var_yy = S_yy - S_y * S_y / n;


		return var_xy * var_xy / ( var_xx * var_yy );

	}

	private static < T extends RealType< T > > double fromIntegralImage(
			final RandomAccessibleInterval< T > integral,
			final Interval interval
			) {
		double correlation = 0.0;
		final RandomAccess< T > access = integral.randomAccess();
		interval.min( access );
		correlation += access.get().getRealDouble();
		access.setPosition( interval.max( 0 ), 0 );
		correlation -= access.get().getRealDouble();
		interval.max( access );
		correlation += access.get().getRealDouble();
		access.setPosition( interval.min( 0 ), 0 );
		correlation -= access.get().getRealDouble();
		return correlation;
	}

}
