package org.janelia.thickness.similarity;

import java.io.IOException;

import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.thickness.ZSpacing;

import bdv.util.BdvFunctions;
import bdv.util.BdvOptions;
import bdv.util.BdvStackSource;
import bdv.util.volatiles.VolatileViews;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.volatiles.VolatileDoubleType;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

public class ShowMatrices
{

	public static void main( final String[] args ) throws IOException
	{
		System.out.println( 1 );
		final String root = "/home/phil/workspace/z-spacing-n5/z-spacing-spark/test.n5";
		System.out.println( 2 );
		final String dataset = "0/matrices";
		System.out.println( 3 );
		final N5Reader n5 = ZSpacing.n5( root );
		System.out.println( 4 );
		final RandomAccessibleInterval< DoubleType > matrices = N5Utils.openVolatile( n5, dataset );
		System.out.println( 5 );
		final RandomAccessibleInterval< VolatileDoubleType > volatileMatrices = VolatileViews.wrapAsVolatile( matrices );
		System.out.println( 6 );

		final long x = 0;
		final long y = 0;
		System.out.println( 7 );
//		final BdvStackSource< VolatileDoubleType > bdv = BdvFunctions.show( Views.hyperSlice( Views.hyperSlice( volatileMatrices, 1, y ), 0, x ), "matrix", BdvOptions.options().is2D() );
		final BdvStackSource< DoubleType > bdv = BdvFunctions.show( Views.hyperSlice( Views.hyperSlice( multiply( matrices, 255 ), 1, y ), 0, x ), "matrix", BdvOptions.options().is2D() );
		bdv.setDisplayRange( 0, 255 );
		Views.flatIterable( matrices ).forEach( System.out::println );
		System.out.println( 8 );

	}

	public static < T extends RealType< T > > RandomAccessibleInterval< T > multiply( final RandomAccessibleInterval< T > input, final double mul )
	{
		return Converters.convert( input, ( s, t ) -> t.setReal( s.getRealDouble() * mul ), Util.getTypeFromInterval( input ).createVariable() );
	}

}
