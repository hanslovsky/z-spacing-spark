package org.janelia.thickness.similarity;

import java.io.IOException;
import java.util.Arrays;
import java.util.stream.LongStream;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.janelia.saalfeldlab.n5.Compression;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.thickness.ZSpacing;
import org.janelia.thickness.utility.DataTypeMatcher;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class MatricesFromN5
{

	public static < T extends NativeType< T > & RealType< T >, U extends NativeType< U > & RealType< U > > void makeMatrices(
			final JavaRDD< long[] > blocksRDD,
			final long[] radius,
			final long[] step,
			final long size,
			final int range,
			final long[] dim,
			final long[] max,
			final String root,
			final String datasetSum,
			final String datasetSumSquared,
			final String datasetMatrix,
			final Compression compression,
			final Broadcast< U > matrixType ) throws IllegalArgumentException, IOException
	{
		final long[] datasetDims = LongStream.concat( Arrays.stream( dim ), LongStream.of( 2 * range + 1, size ) ).toArray();
		ZSpacing.n5Writer( root ).createDataset(
				datasetMatrix,
				datasetDims,
				new int[] { 1, 1, 2 * range + 1, ( int ) size },
				DataTypeMatcher.toDataType( matrixType.getValue() ), compression );
		blocksRDD.foreach( new MatricesFromIntegralImagesAndWrite<>( root, datasetSum, datasetSumSquared, datasetMatrix, range, size, radius, step, max, matrixType ) );
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
			final Interval interval )
	{
		double correlation = 0.0;
		final RandomAccess< T > access = integral.randomAccess();
//		System.out.println( "ACCESSING: " +
//				Point.wrap( Intervals.minAsLongArray( interval ) ) + " " + Point.wrap( Intervals.maxAsLongArray( interval ) ) + " " +
//				Point.wrap( Intervals.minAsLongArray( integral ) ) + " " + Point.wrap( Intervals.maxAsLongArray( integral ) ) );
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

	public static class MatricesFromIntegralImagesAndWrite< U extends RealType< U > & NativeType< U > > implements VoidFunction< long[] >
	{

		private final String root;

		private final String datasetSum;

		private final String datasetSumSquared;

		private final String datasetMatrix;

		private final int range;

		private final long size;

		private final long[] radius;

		private final long[] step;

		private final long[] max;

		private final Broadcast< U > matrixType;

		public MatricesFromIntegralImagesAndWrite(
				final String root,
				final String datasetSum,
				final String datasetSumSquared,
				final String datasetMatrix,
				final int range,
				final long size,
				final long[] radius,
				final long[] step,
				final long[] max,
				final Broadcast< U > matrixType )
		{
			super();
			this.root = root;
			this.datasetSum = datasetSum;
			this.datasetSumSquared = datasetSumSquared;
			this.datasetMatrix = datasetMatrix;
			this.range = range;
			this.size = size;
			this.radius = radius;
			this.step = step;
			this.max = max;
			this.matrixType = matrixType;
		}

		@Override
		public void call( final long[] block ) throws Exception
		{
			final N5Writer writer = ZSpacing.n5Writer( root );
			final Img< U > matrix = new ArrayImgFactory< U >().create( new long[] { 1, 1, 2 * range + 1, size }, matrixType.getValue() );
			final long[] min = new long[ block.length ];
			final long[] max = new long[ block.length ];
			Arrays.setAll( min, d -> block[ d ] * step[ d ] );
			Arrays.setAll( max, d -> Math.min( min[ d ] + 2 * radius[ d ] + 1, this.max[ d ] ) );
			populateMatrix( writer, datasetSum, datasetSumSquared, radius, step, size, range, Views.hyperSlice( Views.hyperSlice( matrix, 0, 0 ), 0, 0 ), new FinalInterval( min, max ) );
			final long[] blockPosition = LongStream.concat( Arrays.stream( block ), LongStream.of( 0, 0 ) ).toArray();
//			final StringBuilder vals = new StringBuilder();
//			Views.flatIterable( matrix ).forEach( v -> vals.append( v.getRealDouble() ).append( ", " ) );
//			System.out.println( "STORING BLOCK AT " + Arrays.toString( blockPosition ) + " " + vals.toString() );
			N5Utils.saveBlock( matrix, writer, datasetMatrix, blockPosition );

		}

		public static < T extends RealType< T > & NativeType< T >, U extends RealType< U > > void populateMatrix(
				final N5Reader reader,
				final String datasetSum,
				final String datasetSumSquared,
				final long[] radius,
				final long[] step,
				final long size,
				final int range,
				final RandomAccessibleInterval< U > matrix,
				final Interval interval ) throws IOException
		{

			final RandomAccessibleInterval< T > sum = N5Utils.< T >open( reader, datasetSum );
			final RandomAccessibleInterval< T > sumSquared = N5Utils.< T >open( reader, datasetSumSquared );

			initialize( matrix, Double.NaN );
			Views.hyperSlice( matrix, 0, ( long ) range ).forEach( U::setOne );
			final RandomAccess< U > matrixAccess1 = matrix.randomAccess();
			final RandomAccess< U > matrixAccess2 = matrix.randomAccess();

			for ( long z1 = 0; z1 < size; ++z1 )
			{
				final IntervalView< T > sumX = Views.hyperSlice( sum, 2, z1 );
				final IntervalView< T > sumXX = Views.hyperSlice( Views.hyperSlice( sumSquared, 3, z1 ), 2, z1 );
				for ( long z2 = z1 + 1, r = 1; z2 < size && r <= range; ++z2, ++r )
				{
					final IntervalView< T > sumY = Views.hyperSlice( sum, 2, z2 );
					final IntervalView< T > sumYY = Views.hyperSlice( Views.hyperSlice( sumSquared, 3, z2 ), 2, z2 );
					final IntervalView< T > sumXY = Views.hyperSlice( Views.hyperSlice( sumSquared, 3, z2 ), 2, z1 );
					final double correlation = correlation2DSquared( sumX, sumY, sumXX, sumYY, sumXY, interval );
					// TODO calculate correlation from sumX, sumY, sumXX,
					// sumYY, sumXY for interval defined by correlationMin,
					// correlationMax
//					System.out.println( "CORRELATION IS " + correlation );
					matrixAccess1.setPosition( z1, 1 );
					matrixAccess1.setPosition( range + r, 0 );
					matrixAccess2.setPosition( z1 + r, 1 );
					matrixAccess2.setPosition( range - r, 0 );
					matrixAccess1.get().setReal( correlation );
					matrixAccess2.get().setReal( correlation );
				}
			}

		}

	}

	public static < T extends RealType< T > > void initialize( final RandomAccessibleInterval< T > matrix, final double value )
	{
		for ( final T m : Views.flatIterable( matrix ) )
		{
			m.setReal( value );
		}
	}

}
