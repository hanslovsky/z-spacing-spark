package org.janelia.thickness.similarity;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.thickness.similarity.MatricesFromN5ParallelizeOverXY.DataSupplier;
import org.janelia.thickness.utility.N5Helpers;

import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import scala.Tuple2;

public class MatricesFromN5ParallelizeOverZ
{

	public static < T extends NativeType< T > & RealType< T >, U extends RealType< U > & NativeType< U > > void makeMatrices(
			JavaSparkContext sc,
			List< CorrelationBlockSpec > blockSpecs,
			final int range,
			final DataSupplier< RandomAccessibleInterval< T > > dataSupplier,
			final String root,
			final String datasetMatrix,
			final U matrixType ) throws Exception
	{
		long[] dims = Intervals.dimensionsAsLongArray( dataSupplier.get() );

		assert dims.length == 3;

		JavaRDD< Long > sliceIndices = sc.parallelize( LongStream.range( 0, dims[ 2 ] ).mapToObj( Long::new ).collect( Collectors.toList() ) );

		Broadcast< U > matrixTypeBC = sc.broadcast( matrixType );

		Broadcast< DataSupplier< RandomAccessibleInterval< T > > > dataSupplierBroadcast = sc.broadcast( dataSupplier );
		Broadcast< List< CorrelationBlockSpec > > blockSpecsBroadcast = sc.broadcast( blockSpecs );

		JavaPairRDD< Tuple2< Long, Long >, Tuple2< Long, double[] > > xyToSectionAndCorrelationsMapping = sliceIndices
				.map( new PairwiseSimilarities<>( dataSupplierBroadcast, range, blockSpecsBroadcast ) )
				.flatMapToPair( new FlattenList<>() );

		JavaPairRDD< Tuple2< Long, Long >, RandomAccessibleInterval< U > > matrices = xyToSectionAndCorrelationsMapping
				.mapValues( new AsList<>() )
				.reduceByKey( new CombineLists<>() )
				.mapValues( new CollectAsMatrix<>( range, dims, matrixTypeBC ) );

		matrices.foreach( new SaveMatrix<>( root, datasetMatrix ) );

	}

	public static class PairwiseSimilarities< T extends RealType< T > > implements Function< Long, List< Tuple2< Tuple2< Long, Long >, Tuple2< Long, double[] > > > >
	{

		/**
		 * 
		 */
		private static final long serialVersionUID = -192278321737762028L;

		private final Broadcast< DataSupplier< RandomAccessibleInterval< T > > > dataSupplierBroadcast;

		private final int range;

		private final Broadcast< List< CorrelationBlockSpec > > blockSpecsBroadcast;

		public PairwiseSimilarities(
				Broadcast< DataSupplier< RandomAccessibleInterval< T > > > dataSupplierBroadcast,
				int range,
				Broadcast< List< CorrelationBlockSpec > > blockSpecsBroadcast )
		{
			super();
			this.dataSupplierBroadcast = dataSupplierBroadcast;
			this.range = range;
			this.blockSpecsBroadcast = blockSpecsBroadcast;
		}

		@Override
		public List< Tuple2< Tuple2< Long, Long >, Tuple2< Long, double[] > > > call( Long sectionIndex ) throws Exception
		{
			RandomAccessibleInterval< T > data = dataSupplierBroadcast.getValue().get();
			long[] dataDims = Intervals.dimensionsAsLongArray( data );
			long maxOtherSectionIndex = Math.min( sectionIndex + range + 1, dataDims[ 2 ] - 1 );
			final List< Tuple2< Tuple2< Long, Long >, Tuple2< Long, double[] > > > allCorrelations = new ArrayList<>();
			for ( CorrelationBlockSpec blockSpec : blockSpecsBroadcast.getValue() )
			{
				double[] correlations = new double[ ( int ) Math.min( range, maxOtherSectionIndex - sectionIndex ) ];
				allCorrelations.add( new Tuple2<>( fromArray( blockSpec.blockPosition ), new Tuple2<>( sectionIndex, correlations ) ) );
				RandomAccessibleInterval< T > s1 = Views.interval( Views.hyperSlice( data, 2, sectionIndex ), blockSpec.min, blockSpec.max );
				for ( long z = sectionIndex + 1, r = 1; z <= maxOtherSectionIndex && r <= range; ++z, ++r )
				{
					RandomAccessibleInterval< T > s2 = Views.interval( Views.hyperSlice( data, 2, z ), blockSpec.min, blockSpec.max );
					correlations[ ( int ) ( r - 1 ) ] = Correlations.pearsonCorrelationCoefficientSquared( s1, s2 );
				}
			}

			return allCorrelations;
		}

	}

	public static class FlattenList< K, V, C extends Collection< Tuple2< K, V > > > implements PairFlatMapFunction< C, K, V >
	{

		/**
		 * 
		 */
		private static final long serialVersionUID = -1265450953827609702L;

		@Override
		public Iterator< Tuple2< K, V > > call( C collection ) throws Exception
		{
			return collection.iterator();
		}
	}

	public static class AsList< T > implements Function< T, List< T > >
	{

		/**
		 * 
		 */
		private static final long serialVersionUID = -7563249767336749747L;

		@Override
		public List< T > call( T t ) throws Exception
		{
			return Arrays.asList( t );
		}
	}

	public static class CombineLists< T > implements Function2< List< T >, List< T >, List< T > >
	{

		/**
		 * 
		 */
		private static final long serialVersionUID = -8042760565432939030L;

		@Override
		public List< T > call( List< T > v1, List< T > v2 ) throws Exception
		{
			ArrayList< T > list = new ArrayList<>();
			list.addAll( v1 );
			list.addAll( v2 );
			return list;
		}

	}

	public static class CollectAsMatrix< U extends RealType< U > & NativeType< U > > implements Function< List< Tuple2< Long, double[] > >, RandomAccessibleInterval< U > >
	{

		/**
		 * 
		 */
		private static final long serialVersionUID = 749990630192281895L;

		private final int range;

		private final long[] dims;

		private final Broadcast< U > matrixTypeBC;

		public CollectAsMatrix( int range, long[] dims, Broadcast< U > matrixTypeBC )
		{
			super();
			this.range = range;
			this.dims = dims;
			this.matrixTypeBC = matrixTypeBC;
		}

		@Override
		public RandomAccessibleInterval< U > call( List< Tuple2< Long, double[] > > indicesWithCorrelations ) throws Exception
		{
			RandomAccessibleInterval< U > matrix = new ArrayImgFactory< U >().create(
					new long[] { 1, 1, 2 * range + 1, dims[ 2 ] },
					matrixTypeBC.getValue().createVariable() );
			initialize( matrix, Double.NaN );
			Views.hyperSlice( matrix, 2, range ).forEach( RealType::setOne );
			RandomAccess< U > matrixAccess1 = Views.hyperSlice( Views.hyperSlice( matrix, 0, 0l ), 0, 0l ).randomAccess();
			RandomAccess< U > matrixAccess2 = Views.hyperSlice( Views.hyperSlice( matrix, 0, 0l ), 0, 0l ).randomAccess();

			for ( Tuple2< Long, double[] > indexWithCorrelations : indicesWithCorrelations )
			{
				long z1 = indexWithCorrelations._1();
				double[] correlations = indexWithCorrelations._2();

				for ( int i = 0, r = 1; i < correlations.length; ++i, ++r )
				{
					double correlation = correlations[ i ];
					long z2 = z1 + r;

					matrixAccess1.setPosition( range + r, 0 );
					matrixAccess2.setPosition( range - r, 0 );
					matrixAccess1.setPosition( z1, 1 );
					matrixAccess2.setPosition( z2, 1 );
					matrixAccess1.get().setReal( correlation );
					matrixAccess2.get().setReal( correlation );
				}

			}

			return matrix;
		}

	}

	public static class SaveMatrix< U extends NativeType< U > > implements VoidFunction< Tuple2< Tuple2< Long, Long >, RandomAccessibleInterval< U > > >
	{
		
		/**
		 * 
		 */
		private static final long serialVersionUID = -7524659709110262253L;

		private final String root;
		
		private final String dataset;

		public SaveMatrix( String root, String dataset )
		{
			super();
			this.root = root;
			this.dataset = dataset;
		}

		@Override
		public void call( Tuple2< Tuple2< Long, Long >, RandomAccessibleInterval< U > > positionAndData ) throws Exception
		{
			Tuple2< Long, Long > position = positionAndData._1();
			RandomAccessibleInterval< U > matrix = positionAndData._2();
			long[] blockPosition = new long[] { position._1(), position._2(), 0, 0 };
			N5Writer n5 = N5Helpers.n5Writer( root );
			N5Utils.saveBlock( matrix, n5, dataset, blockPosition );
		}

	}

	public static < T extends RealType< T > > void initialize( final RandomAccessibleInterval< T > matrix, final double value )
	{
		for ( final T m : Views.flatIterable( matrix ) )
		{
			m.setReal( value );
		}
	}

	public static Tuple2< Long, Long > fromArray( long[] arr )
	{

		assert arr.length >= 2;

		return new Tuple2<>( arr[ 0 ], arr[ 1 ] );
	}

}
