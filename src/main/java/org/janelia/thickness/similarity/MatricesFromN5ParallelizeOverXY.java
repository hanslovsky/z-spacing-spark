package org.janelia.thickness.similarity;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.LongStream;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.janelia.saalfeldlab.n5.Compression;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.thickness.utility.N5Helpers;

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

public class MatricesFromN5ParallelizeOverXY
{
	
	public static interface DataSupplier< T > {
		
		public T get() throws Exception;
		
	}
	
	public static < T extends NativeType< T > & RealType< T >, U extends NativeType< U > & RealType< U > > void makeMatrices(
			JavaSparkContext sc,
			final JavaRDD< List< long[] > > locations,
			final long[] matrixDims,
			final long[] radius,
			final int range,
			final DataSupplier< RandomAccessibleInterval< T > > dataSupplier,
			final String root,
			final String datasetMatrix ) throws Exception
	{
		makeMatrices( sc, locations, matrixDims, radius, range, dataSupplier, root, datasetMatrix, new GzipCompression(), sc.broadcast( new DoubleType() ) );
	}

	public static < T extends NativeType< T > & RealType< T >, U extends NativeType< U > & RealType< U > > void makeMatrices(
			JavaSparkContext sc,
			final JavaRDD< List< long[] > > locations,
			final long[] matrixDims,
			final long[] radius,
			final int range,
			final DataSupplier< RandomAccessibleInterval< T > > dataSupplier,
			final String root,
			final String datasetMatrix,
			final Compression compression,
			final Broadcast< U > matrixType ) throws Exception
	{
		locations.foreach( new GenerateMatrices<>(
				sc.broadcast( dataSupplier ),
				root,
				datasetMatrix,
				range,
				radius,
				matrixType ) );
	}

	public static class GenerateMatrices< T extends RealType< T >, U extends RealType< U > & NativeType< U > > implements VoidFunction< List< long[] > >
	{

		private final Broadcast< DataSupplier< RandomAccessibleInterval< T > > > dataSupplier;

		private final String root;

		private final String datasetMatrix;

		private final int range;

		private final long[] radius;

		private final Broadcast< U > matrixType;

		public GenerateMatrices(
				final Broadcast< DataSupplier< RandomAccessibleInterval< T > > > dataSupplier,
				final String root,
				final String datasetMatrix,
				final int range,
				final long[] radius,
				final Broadcast< U > matrixType )
		{
			super();
			this.dataSupplier = dataSupplier;
			this.root = root;
			this.datasetMatrix = datasetMatrix;
			this.range = range;
			this.radius = radius;
			this.matrixType = matrixType;
		}

		@Override
		public void call( final List< long[] > blocks ) throws Exception
		{
			RandomAccessibleInterval< T > data = dataSupplier.getValue().get();
			long[] dims = Intervals.dimensionsAsLongArray( data );

			assert dims.length == 2;

			for ( long[] block : blocks )
			{
				final N5Writer writer = N5Helpers.n5Writer( root );
				final Img< U > matrix = new ArrayImgFactory< U >().create( new long[] { 1, 1, 2 * range + 1, dims[ 2 ] }, matrixType.getValue() );
				populateMatrix( data, block, radius, range, ( RandomAccessibleInterval< U > ) Views.hyperSlice( Views.hyperSlice( matrix, 0, 0 ), 0, 0 ) );
				// TODO use block sizes other than [1, 1, 2 * range + 1, size]
				final long[] blockPosition = LongStream.concat( Arrays.stream( block ), LongStream.of( 0, 0 ) ).toArray();
				N5Utils.saveBlock( matrix, writer, datasetMatrix, blockPosition );
			}

		}

		public static < T extends RealType< T >, U extends RealType< U > > void populateMatrix(
				RandomAccessibleInterval< T > data,
				final long[] position,
				final long[] radius,
				final int range,
				final RandomAccessibleInterval< U > matrix ) throws IOException
		{

			if ( !Views.isZeroMin( data ) )
			{
				populateMatrix( Views.zeroMin( data ), position, radius, range, matrix );
				return;
			}

			assert data.numDimensions() == 3;
			long[] min = Intervals.minAsLongArray( data );
			long[] max = Intervals.maxAsLongArray( data );
			long[] dim = Intervals.dimensionsAsLongArray( data );
			long[] blockMin = new long[ 2 ];
			long[] blockMax = new long[ 2 ];
			Arrays.setAll( blockMin, d -> Math.max( position[ d ] - radius[ d ] + 0, min[ d ] ) );
			Arrays.setAll( blockMax, d -> Math.min( position[ d ] + radius[ d ] + 1, max[ d ] ) );

			initialize( matrix, Double.NaN );
			Views.hyperSlice( matrix, 0, ( long ) range ).forEach( U::setOne );
			final RandomAccess< U > matrixAccess1 = matrix.randomAccess();
			final RandomAccess< U > matrixAccess2 = matrix.randomAccess();

			for ( long z1 = 0; z1 < dim[ 2 ]; ++z1 )
			{
				IntervalView< T > s1 = Views.interval( Views.hyperSlice( data, 2, z1 ), blockMin, blockMax );
				for ( long z2 = z1 + 1, r = 1; z2 < dim[ 2 ] && r <= range; ++z2, ++r )
				{
					IntervalView< T > s2 = Views.interval( Views.hyperSlice( data, 2, z2 ), blockMin, blockMax );
					final double correlation = Correlations.pearsonCorrelationCoefficientSquared( s1, s2 );
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
