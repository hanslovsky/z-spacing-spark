package org.janelia.thickness;

import java.io.Serializable;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.janelia.thickness.inference.InferFromMatrix;
import org.janelia.thickness.inference.Options;
import org.janelia.thickness.inference.fits.AbstractCorrelationFit;
import org.janelia.thickness.inference.fits.GlobalCorrelationFitAverage;
import org.janelia.thickness.inference.fits.LocalCorrelationFitAverage;
import org.janelia.thickness.inference.visitor.LazyVisitor;
import org.janelia.thickness.inference.visitor.Visitor;
import org.janelia.thickness.utility.Utility;
import org.janelia.thickness.weight.Weights;
import org.janelia.utility.MatrixStripConversion;

import ij.process.FloatProcessor;
import mpicbg.models.NotEnoughDataPointsException;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Pair;
import net.imglib2.view.ExtendedRandomAccessibleInterval;
import net.imglib2.view.Views;
import scala.Tuple2;

public class SparkInference
{

	public static class Variables implements Serializable
	{

		public final double[] coordinates;

		public final double[] scalingFactors;

		public final double[] estimate;

		public Variables( final double[] coordinates, final double[] scalingFactors, final double[] estimate )
		{
			super();
			this.coordinates = coordinates;
			this.scalingFactors = scalingFactors;
			this.estimate = estimate;
		}

	}

	public static class Input implements Serializable
	{

		public final ArrayImg< FloatType, ? > matrix;

		public final ArrayImg< FloatType, ? > estimateWeights;

		public final Variables variables;

		public final Weights weights;

		public Input( final ArrayImg< FloatType, ? > matrix, final ArrayImg< FloatType, ? > estimateWeights, final Variables variables, final Weights weights )
		{
			super();
			this.matrix = matrix;
			this.estimateWeights = estimateWeights;
			this.variables = variables;
			this.weights = weights;
		}


	}

	public static JavaPairRDD< Tuple2< Integer, Integer >, Variables > inferCoordinates(
			final JavaSparkContext sc,
			final JavaPairRDD< Tuple2< Integer, Integer >, Input > matricesWithStartingCoordinates,
			final Options options,
			final String pattern )
	{
		//		final JavaPairRDD< Tuple2< Integer, Integer >, Input > matricesWithStartingCoordinates = matricesAndEstimateWeights
		//				.join( startingVariablesAndWeights )
		//				.mapValues( t -> new Input( t._1()._1(), t._1()._2(), t._2()._1(), t._2()._2() ) );
		final JavaPairRDD< Tuple2< Integer, Integer >, Variables > result =
				matricesWithStartingCoordinates.mapToPair( new Inference< Tuple2< Integer, Integer > >( options, pattern ) );

		return result;
	}

	public static class Inference< K > implements PairFunction< Tuple2< K, Input >, K, Variables >
	{
		private static final long serialVersionUID = 8094812748656050753L;

		private final Options options;

		private final String pattern;

		public Inference( final Options options, final String pattern )
		{
			super();
			this.options = options;
			this.pattern = pattern;
		}

		@Override
		public Tuple2< K, Variables > call( final Tuple2< K, Input > t ) throws Exception
		{

			final Input input = t._2();
			final RandomAccessibleInterval< FloatType > matrix = MatrixStripConversion.stripToMatrix( halfStripToStrip( input.matrix, new ArrayImgFactory<>(), new FloatType( Float.NaN ) ), new FloatType( Float.NaN ) );
			final RandomAccessibleInterval< FloatType > estimateWeights = MatrixStripConversion.stripToMatrix( halfStripToStrip( input.estimateWeights, new ArrayImgFactory<>(), new FloatType( Float.NaN ) ), new FloatType( Float.NaN ) );

			for ( final Cursor< FloatType > c = Views.iterable( matrix ).cursor(); c.hasNext(); )
			{
				final float val = c.next().get();
				final long x = c.getLongPosition( 0 );
				final long y = c.getLongPosition( 1 );
				if ( Math.abs( x - y ) <= options.comparisonRange && ( Float.isNaN( val ) || val == 0.0f ) )
					return Utility.tuple2( t._1(), input.variables );
			}

			final AbstractCorrelationFit corrFit = options.estimateWindowRadius < 0 ? new GlobalCorrelationFitAverage() : new LocalCorrelationFitAverage( ( int ) matrix.dimension( 1 ), options );;
			final InferFromMatrix inference = new InferFromMatrix( corrFit );
			final Visitor visitor = new LazyVisitor();
			//			final ArrayImg< DoubleType, DoubleArray > img = ArrayImgs.doubles( input.variables.coordinates.length, options.nIterations + 1 );
			//			visitor = new WriteTransformationVisitor( img );
			try
			{
				//				final double[] coordinates = inference.estimateZCoordinates( matrix, input.variables.coordinates, options );
				final double[] coordinates = inference.estimateZCoordinates(
						matrix,
						input.variables.coordinates,
						input.variables.estimate,
						input.variables.scalingFactors,
						estimateWeights, // ConstantUtils.constantRandomAccessibleInterval(
											// new FloatType( 1.0f ),
											// estimateWeights.numDimensions(),
											// estimateWeights ),
						input.weights.shiftWeights,
						visitor,
						options );
				for ( final double c : coordinates )
					if ( Double.isNaN( c ) )
					{
						System.err.println( "Inferred NaN value for coordinate " + t._1() );
						return Utility.tuple2( t._1(), null );
					}
				//				final String path = String.format( pattern, t._1().toString() );
				//				Files.createDirectories( new File( path ).getParentFile().toPath() );
				//				new FileSaver( ImageJFunctions.wrapFloat( img, "" ) ).saveAsTiff( path );
				return Utility.tuple2( t._1(), new Variables( coordinates, input.variables.scalingFactors, input.variables.estimate ) );
			}
			catch ( final NotEnoughDataPointsException e )
			{
				//                String msg = e.getMessage();
				//                new ImagePlus(t._1().toString(),t._2()._1().rebuild()).show();
				System.err.println( "Fail at inference for coordinate " + t._1() );
				e.printStackTrace( System.err );
				return Utility.tuple2( t._1(), null );
				//                throw e;
				//                throw new NotEnoughDataPointsException( t._1() + " " + msg );
			}
		}
	}

	public static class WriteTransformationVisitor implements Visitor
	{

		private final RandomAccessibleInterval< DoubleType > img;

		public WriteTransformationVisitor( final RandomAccessibleInterval< DoubleType > img )
		{
			this.img = img;
		}

		@Override
		public < T extends RealType< T > > void act( final int iteration, final RandomAccessibleInterval< T > matrix, final RandomAccessibleInterval< T > scaledMatrix, final double[] lut, final int[] permutation, final int[] inversePermutation, final double[] multipliers, final RandomAccessibleInterval< double[] > estimatedFit )
		{
			final Cursor< DoubleType > current = Views.flatIterable( Views.hyperSlice( img, 1, iteration ) ).cursor();
			for ( int z = 0; current.hasNext(); ++z )
				current.next().set( lut[ z ] );
		}
	}

	public static < T extends NumericType< T > > Img< T > halfStripToStrip( final RandomAccessibleInterval< T > halfStrip, final ImgFactory< T > fac, final T t )
	{
		final T t1 = t.createVariable();
		final T t0 = t.createVariable();
		t1.setOne();
		t0.setZero();

		final long r = halfStrip.dimension( 0 );
		final Img< T > img = fac.create( new long[] { r * 2 + 1, halfStrip.dimension( 1 ) }, t );
		for ( final T v : Views.hyperSlice( img, 0, r ) )
			v.setOne();

		for ( final Pair< T, T > p : Views.interval( Views.pair( Views.offset( img, r + 1, 0 ), halfStrip ), halfStrip ) )
			p.getA().set( p.getB() );

		final ExtendedRandomAccessibleInterval< T, RandomAccessibleInterval< T > > ext = Views.extendValue( halfStrip, t );

		for ( long i = 1; i <= r; ++i )
		{
			final Cursor< T > target = Views.hyperSlice( img, 0, r - i ).cursor();
			final Cursor< T > source = Views.offsetInterval( Views.hyperSlice( ext, 0, i - 1 ), new long[] { -i }, new long[] { halfStrip.dimension( 1 ) } ).cursor();
			while ( source.hasNext() )
				target.next().set( source.next() );
		}

		return img;
	}

	public static RandomAccessibleInterval< FloatType > wrapMatrix( final FloatProcessor fp )
	{
		final int w = fp.getWidth();
		final int h = fp.getHeight();
		final ArrayImg< FloatType, FloatArray > wrapped = ArrayImgs.floats( ( float[] ) fp.getPixels(), w, h );
		return w == h ? wrapped : MatrixStripConversion.stripToMatrix( wrapped, new FloatType() );
	}
}
