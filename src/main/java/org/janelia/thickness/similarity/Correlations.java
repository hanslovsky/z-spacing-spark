package org.janelia.thickness.similarity;

import ij.process.FloatProcessor;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.util.RealSum;
import net.imglib2.view.Views;

/**
 * @author Philipp Hanslovsky &lt;hanslovskyp@janelia.hhmi.org&gt;
 */
public class Correlations
{

	public static double calculate( final FloatProcessor img1, final FloatProcessor img2 )
	{
		final RealSum sumA = new RealSum();
		final RealSum sumAA = new RealSum();
		final RealSum sumB = new RealSum();
		final RealSum sumBB = new RealSum();
		final RealSum sumAB = new RealSum();
		int n = 0;
		float[] d1 = ( float[] ) img1.getPixels();
		float[] d2 = ( float[] ) img2.getPixels();
		for ( int i = 0; i < d1.length; ++i )
		{
			final float va = d1[ i ];
			final float vb = d2[ i ];

			if ( Float.isNaN( va ) || Float.isNaN( vb ) )
				continue;
			++n;
			sumA.add( va );
			sumAA.add( va * va );
			sumB.add( vb );
			sumBB.add( vb * vb );
			sumAB.add( va * vb );
		}
		final double suma = sumA.getSum();
		final double sumaa = sumAA.getSum();
		final double sumb = sumB.getSum();
		final double sumbb = sumBB.getSum();
		final double sumab = sumAB.getSum();

		return ( n * sumab - suma * sumb ) / Math.sqrt( n * sumaa - suma * suma ) / Math.sqrt( n * sumbb - sumb * sumb );
	}

	public static < T extends RealType< T > > double pearsonCorrelationCoefficientSquared(
			final RandomAccessibleInterval< T > sampleA,
			final RandomAccessibleInterval< T > sampleB )
	{
	
		// Easiest to prove with expectations:
		// rho ^ 2 =
		// ( S_ab - S_a * S_b / n ) ^ 2
		// ---------------------------------------------------
		// ( S_aa - S_a * S_a / n ) * ( S_b - S_b * S_b / n )
	
		RealSum sumA = new RealSum();
		RealSum sumB = new RealSum();
		RealSum sumAA = new RealSum();
		RealSum sumAB = new RealSum();
		RealSum sumBB = new RealSum();
	
		for ( Cursor< T > cursorA = Views.flatIterable( sampleA ).cursor(), cursorB = Views.flatIterable( sampleB ).cursor(); cursorA.hasNext(); )
		{
			double a = cursorA.next().getRealDouble();
			double b = cursorB.next().getRealDouble();
			sumA.add( a );
			sumB.add( b );
			sumAA.add( a * a );
			sumAB.add( a * b );
			sumBB.add( b * b );
		}
	
		long count = Intervals.numElements( sampleA );
		double sumAResult = sumA.getSum();
		double sumBResult = sumB.getSum();
		double sumAAResult = sumAA.getSum();
		double sumABResult = sumAB.getSum();
		double sumBBResult = sumBB.getSum();
	
		double varAB = sumABResult - sumAResult * sumBResult / count;
		double varA = sumAAResult - sumAResult * sumAResult / count;
		double varB = sumBBResult - sumBResult * sumBResult / count;
	
		return varAB * varAB / ( varA * varB );
	
	}
}
