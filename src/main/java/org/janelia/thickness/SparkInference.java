package org.janelia.thickness;

import java.util.Arrays;
import java.util.stream.LongStream;

import org.apache.spark.api.java.function.Function;
import org.janelia.thickness.inference.InferFromMatrix;
import org.janelia.thickness.inference.Options;
import org.janelia.thickness.inference.fits.AbstractCorrelationFit;
import org.janelia.thickness.inference.fits.GlobalCorrelationFitAverage;
import org.janelia.thickness.inference.fits.LocalCorrelationFitAverage;
import org.janelia.thickness.inference.visitor.LazyVisitor;
import org.janelia.thickness.inference.visitor.Visitor;

import mpicbg.models.NotEnoughDataPointsException;
import net.imglib2.Cursor;
import net.imglib2.Point;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.Views;
import net.imglib2.view.composite.Composite;
import net.imglib2.view.composite.RealComposite;

public class SparkInference
{

	public static final class InputData< T, U extends RealType< U > >
	{
		public final RandomAccessibleInterval< RandomAccessibleInterval< T > > matrix;

		public final RandomAccessibleInterval< ? extends Composite< U > > coordinates;

		public InputData( final RandomAccessibleInterval< RandomAccessibleInterval< T > > matrix, final RandomAccessibleInterval< ? extends Composite< U > > coordinates )
		{
			super();
			this.matrix = matrix;
			this.coordinates = coordinates;
		}
	}

	public static class Inference< T extends RealType< T > & NativeType< T >, U extends RealType< U > & NativeType< U > >
	implements Function< InputData< T, U >, RandomAccessibleInterval< U > >
	{
		private static final long serialVersionUID = 8094812748656050753L;

		private final Options options;

		private final long sectionMin;

		private final long sectionMax;

		public Inference( final Options options, final long numSections )
		{
			this( options, 0, numSections - 1 );
		}

		public Inference( final Options options, final long sectionMin, final long sectionMax )
		{
			super();
			this.options = options;
			this.sectionMin = sectionMin;
			this.sectionMax = sectionMax;
		}

		@Override
		public RandomAccessibleInterval< U > call( final InputData< T, U > t ) throws Exception
		{

			final RandomAccessibleInterval< RandomAccessibleInterval< T > > matrices = t.matrix;
			final RandomAccessibleInterval< ? extends Composite< U > > startingCoordinates = t.coordinates;
			final long[] min = Intervals.minAsLongArray( matrices );
			final long[] max = Intervals.maxAsLongArray( matrices );
			assert Arrays.equals( Intervals.minAsLongArray( startingCoordinates ), min ) && Arrays.equals( Intervals.maxAsLongArray( startingCoordinates ), max );

			final long[] coordinatesDims = LongStream.concat(
					Arrays.stream( Intervals.dimensionsAsLongArray( startingCoordinates ) ),
					LongStream.of( sectionMax - sectionMin + 1 ) ).toArray();

			final Img< U > targetCoordinates = new ArrayImgFactory< U >().create( coordinatesDims, Util.getTypeFromInterval( t.coordinates ).get( sectionMin ).copy() );



//			for ( final Cursor< FloatType > c = Views.iterable( matrix ).cursor(); c.hasNext(); )
//			{
//				final float val = c.next().get();
//				final long x = c.getLongPosition( 0 );
//				final long y = c.getLongPosition( 1 );
//				if ( Math.abs( x - y ) <= options.comparisonRange && ( Float.isNaN( val ) || val == 0.0f ) ) {
//					return Utility.tuple2( t._1(), t._2()._2() );
//				}
//			}
			final Cursor< RandomAccessibleInterval< T > > matrixCursor = Views.flatIterable( matrices ).cursor();
			final Cursor< ? extends Composite< U > > initialCoordinatesCursor = Views.flatIterable( startingCoordinates ).cursor();
			final Cursor< RealComposite< U > > targetCoordinatesCursor = Views.flatIterable( Views.collapseReal( targetCoordinates ) ).cursor();
			while ( matrixCursor.hasNext() )
			{
				final RandomAccessibleInterval< T > matrix = matrixCursor.next();
				final Composite< U > initCoord = initialCoordinatesCursor.next();
				final RealComposite< U > targetCoord = targetCoordinatesCursor.next();
				final AbstractCorrelationFit corrFit = options.estimateWindowRadius < 0 ? new GlobalCorrelationFitAverage() : new LocalCorrelationFitAverage( ( int ) matrix.dimension( 1 ), options );;
				final InferFromMatrix inference = new InferFromMatrix( corrFit );
				final Visitor visitor = new LazyVisitor();
				try
				{
					final double[] initialCoordinatesArray = new double[ ( int ) ( sectionMax - sectionMin ) ];
					for ( int i = 0; i < initialCoordinatesArray.length; ++i )
					{
						initialCoordinatesArray[ i ] = initCoord.get( i + sectionMin ).getRealDouble();
					}
					final double[] coordinates = inference.estimateZCoordinates( matrix, initialCoordinatesArray, visitor, options );
					boolean coordinatesAreFinite = true;
					for ( final double c : coordinates )
					{
						if ( Double.isNaN( c ) )
						{
							System.err.println( "Inferred NaN value for coordinate " + new Point( matrixCursor ) );
							coordinatesAreFinite = false;
							break;
						}
					}
					for ( long m = sectionMin, i = 0; m < sectionMax; ++m, ++i )
					{
						targetCoord.get( m ).setReal( coordinatesAreFinite ? coordinates[ ( int ) i ] : initCoord.get( m ).getRealDouble() );
					}
				}
				catch ( final NotEnoughDataPointsException e )
				{
					System.err.println( "Fail at inference for coordinate " + new Point( matrixCursor ) );
					e.printStackTrace( System.err );
				}
			}
			return Views.translate( targetCoordinates, Intervals.minAsLongArray( t.coordinates ) );
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
			{
				current.next().set( lut[ z ] );
			}
		}

	}
}
