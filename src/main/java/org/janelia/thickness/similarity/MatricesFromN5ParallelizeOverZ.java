package org.janelia.thickness.similarity;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import java.util.stream.Stream;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.thickness.similarity.MatricesFromN5ParallelizeOverXY.DataSupplier;
import org.janelia.thickness.utility.N5Helpers;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

	private static final Logger LOG = LoggerFactory.getLogger( MethodHandles.lookup().lookupClass() );

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
				.map( sectionIndex -> {
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

				} )
				.flatMapToPair( List::iterator );

		JavaPairRDD< Tuple2< Long, Long >, RandomAccessibleInterval< U > > matrices = xyToSectionAndCorrelationsMapping
				.mapValues( Arrays::asList )
				.reduceByKey( ( l1, l2 ) -> Stream.concat( l1.stream(), l2.stream() ).collect( Collectors.toList() ) )
				.mapValues( indicesWithCorrelations -> {

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
				} );

		matrices.foreach( positionAndMatrix -> {
			Tuple2< Long, Long > position = positionAndMatrix._1();
			RandomAccessibleInterval< U > matrix = positionAndMatrix._2();
			long[] blockPosition = new long[] { position._1(), position._2(), 0, 0 };
			N5Writer n5 = N5Helpers.n5Writer( root );
			N5Utils.saveBlock( matrix, n5, datasetMatrix, blockPosition );
		} );

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
