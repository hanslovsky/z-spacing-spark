package org.janelia.thickness.similarity;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.saalfeldlab.n5.spark.N5ConvertSpark;
import org.janelia.thickness.utility.N5Helpers;

import ij.ImagePlus;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.integral.IntegralImgLong;
import net.imglib2.converter.Converters;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.util.Pair;
import net.imglib2.util.Util;
import net.imglib2.view.Views;
import scala.Tuple2;

public class GenerateIntegralImages
{

	public static < T extends IntegerType< T> & NativeType< T > > void run(
			final JavaSparkContext sc,
			final List< String > filenames,
			final int[] blockSize,
			final int range,
			final String root,
			final String datasetSumX,
			final String datasetSumXY ) throws IOException
	{

		final String datasetSumXTmp = datasetSumX + "-tmp";
		final String datasetSumXYTmp = datasetSumXY + "-tmp";

		{
			final ImagePlus img = new ImagePlus( filenames.get( 0 ) );
			final int[] integralPlaneDimensions = { img.getWidth() + 1, img.getHeight() + 1 };
			final N5Writer writer = N5Helpers.n5Writer( root );
			final long[] dimensionsX = LongStream.concat( Arrays.stream( integralPlaneDimensions ).asLongStream(), LongStream.of( filenames.size() ) ).toArray();
			final long[] dimensionsXY = LongStream.concat( Arrays.stream( dimensionsX ), LongStream.of( range + 1 ) ).toArray();
			final int[] blockSizeX = IntStream.concat( Arrays.stream( integralPlaneDimensions ), IntStream.of( 1 ) ).toArray();
			final int[] blockSizeXY = IntStream.concat( Arrays.stream( blockSizeX ), IntStream.of( 1 ) ).toArray();
			writer.createDataset( datasetSumXTmp, dimensionsX, blockSizeX, DataType.INT64, new GzipCompression() );
			writer.createDataset( datasetSumXYTmp, dimensionsXY, blockSizeXY, DataType.INT64, new GzipCompression() );
		}

		sc
		.parallelize( IntStream.range( 0, filenames.size() ).mapToObj( i -> new Tuple2<>( i, filenames.get( i ) ) ).collect( Collectors.toList() ) )
		.foreach( filename -> {
			// load data into img
			final ImagePlus imp = new ImagePlus( filename._2() );
			final RandomAccessibleInterval< T > img = ImageJFunctions.wrap( imp );
			final IntegralImgLong< T > iimg = new IntegralImgLong<>( img, new LongType(), ( s, t ) -> t.set( s.getIntegerLong() ) );
			iimg.process();
			final Img< LongType > store = iimg.getResult();
			final N5Writer writer = N5Helpers.n5Writer( root );

			final long[] offset = new long[] { 0, 0, filename._1 };
			N5Utils.saveBlock( Views.addDimension( store, 0, 0 ), writer, datasetSumXTmp, offset );
		} );

		final List< Tuple2< Tuple2< Integer, String >, Tuple2< Integer, String > > > pairs = new ArrayList<>();
		for ( int z1 = 0; z1 < filenames.size(); ++z1 )
		{
			final Tuple2< Integer, String > p1 = new Tuple2<>( z1, filenames.get( z1 ) );
			for ( int z2 = z1, r = 0; z2 < filenames.size() && r <= range; ++z2, ++r )
			{
				final Tuple2< Integer, String > p2 = new Tuple2<>( z2, filenames.get( z2 ) );
				pairs.add( new Tuple2<>( p1, p2 ) );
			}
		}

		sc
		.parallelize( pairs )
		.foreach( pair -> {
			final int z1 = pair._1()._1();
			final int z2 = pair._2()._1();
			final String fn1 = pair._1()._2();
			final String fn2 = pair._2()._2();
			final RandomAccessibleInterval< T > img1 = ImageJFunctions.wrap( new ImagePlus( fn1 ) );
			final RandomAccessibleInterval< T > img2 = ImageJFunctions.wrap( new ImagePlus( fn2 ) );
			final IntegralImgLong< T > iimgSquared = new IntegralImgLong<>( multiply( img1, img2 ), new LongType(), ( s, t ) -> t.set( s.getIntegerLong() ) );
			iimgSquared.process();
			final Img< LongType > storeSquared = iimgSquared.getResult();
			final N5Writer writer = N5Helpers.n5Writer( root );

			final long[] offset = new long[] { 0, 0, z1, z2 - z1 };
			N5Utils.saveBlock( Views.addDimension( Views.addDimension( storeSquared, 0, 0 ), 0, 0 ), writer, datasetSumXYTmp, offset );
		} );
		
		N5ConvertSpark.convert( 
				sc, 
				() -> N5Helpers.n5( root ), 
				datasetSumXTmp, 
				() -> N5Helpers.n5Writer( root ), 
				datasetSumX, 
				Optional.of( blockSize ), 
				Optional.empty(), 
				Optional.empty(), 
				Optional.empty(),
				true );
		
		N5ConvertSpark.convert( 
				sc, 
				() -> N5Helpers.n5( root ), 
				datasetSumXYTmp,
				() -> N5Helpers.n5Writer( root ), 
				datasetSumXY, 
				Optional.of( IntStream.concat( Arrays.stream( blockSize ), IntStream.of( range + 1 ) ).toArray() ), 
				Optional.empty(), 
				Optional.empty(), 
				Optional.empty(),
				true );


	}

	public static < T extends NumericType< T > > RandomAccessibleInterval< T > multiply(
			final RandomAccessibleInterval< T > rai1,
			final RandomAccessibleInterval< T > rai2 )
	{
		final T type = Util.getTypeFromInterval( rai1 ).createVariable();
		final RandomAccessibleInterval< Pair< T, T > > paired = Views.interval( Views.pair( rai1, rai2 ), rai1 );
		return Converters.convert( paired, ( s, t ) -> {
			t.set( s.getA() );
			t.mul( s.getB() );
		}, type );
	}

}
