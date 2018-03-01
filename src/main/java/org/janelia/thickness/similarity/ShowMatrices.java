package org.janelia.thickness.similarity;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.thickness.CollapsedRandomAccessibleInterval;
import org.janelia.thickness.ZSpacing;
import org.janelia.utility.MatrixStripConversion;

import bdv.util.AxisOrder;
import bdv.util.BdvFunctions;
import bdv.util.BdvOptions;
import bdv.util.BdvStackSource;
import ij.ImageJ;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

public class ShowMatrices
{

	public static void main( final String[] args ) throws IOException
	{
		final String root = "/home/hanslovskyp/workspace/z-spacing-n5/test.n5";
		final String dataset = "3/matrices";
		final N5Reader n5 = ZSpacing.n5( root );
		final RandomAccessibleInterval< DoubleType > matrices = N5Utils.openVolatile( n5, dataset );

		RandomAccessibleInterval< RandomAccessibleInterval< DoubleType > > matricesCollapsed = new CollapsedRandomAccessibleInterval<>( matrices, 2 );
		List< RandomAccessibleInterval< DoubleType > > matricesList = new ArrayList<>();
		Views.flatIterable( matricesCollapsed ).forEach( matricesList::add );
		RandomAccessibleInterval< DoubleType > matricesStacked = Views.stack( matricesList
				.stream()
				.map( m -> MatrixStripConversion.stripToMatrix( m, new DoubleType( Double.NaN ) ) )
				.collect( Collectors.toList() ) );

		final BdvStackSource< DoubleType > bdv = BdvFunctions.show(
				matricesStacked,
				"matrix",
				BdvOptions.options().is2D().axisOrder( AxisOrder.XYT ) );
		bdv.setDisplayRange( 0, 1 );
		System.out.println( "DIMS: " + Arrays.toString( Intervals.dimensionsAsLongArray( matrices ) ) + " " + Arrays.toString( Intervals.dimensionsAsLongArray( matricesStacked ) ) );

		new ImageJ();
		ImageJFunctions.show( matricesStacked, "ok" );

	}

	public static < T extends RealType< T > > RandomAccessibleInterval< T > multiply( final RandomAccessibleInterval< T > input, final double mul )
	{
		return Converters.convert( input, ( s, t ) -> t.setReal( s.getRealDouble() * mul ), Util.getTypeFromInterval( input ).createVariable() );
	}

}
