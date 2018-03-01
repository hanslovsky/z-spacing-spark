package org.janelia.thickness.similarity;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.thickness.CollapsedRandomAccessibleInterval;
import org.janelia.thickness.utility.N5Helpers;
import org.janelia.utility.MatrixStripConversion;

import bdv.util.AxisOrder;
import bdv.util.BdvFunctions;
import bdv.util.BdvOptions;
import bdv.util.BdvStackSource;
import ij.ImageJ;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.util.Pair;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

public class ShowMatricesAndCoordinates
{

	public static void main( final String[] args ) throws IOException
	{

		final int level = 2;

		final String root = "/home/hanslovskyp/workspace/z-spacing-n5/project.n5";
		final String dataset = level + "/matrices";
		final N5Reader n5 = N5Helpers.n5( root );
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

		new ImageJ();
		ImageJFunctions.show( matricesStacked, "ok" );

		String coordinatesDataset = level + "/forward";
		RandomAccessibleInterval< DoubleType > coordinates = N5Utils.openVolatile( n5, coordinatesDataset );
		RandomAccessible< DoubleType > fwd = Views.translate( Views.extendBorder( coordinates ), 0, 0, -1 );
		RandomAccessible< DoubleType > bck = Views.translate( Views.extendBorder( coordinates ), 0, 0, 1 );
		RandomAccessibleInterval< DoubleType > diff = Views.interval( subtract( fwd, bck, new DoubleType() ), coordinates );
		BdvStackSource< DoubleType > bdvDiff = BdvFunctions.show( diff, "diff" );
		bdvDiff.setDisplayRange( -2, 2 );

		ImageJFunctions.show( coordinates, "coord" );
		ImageJFunctions.show( diff, "diff" );

	}

	public static < T extends RealType< T > > RandomAccessibleInterval< T > multiply( final RandomAccessibleInterval< T > input, final double mul )
	{
		return Converters.convert( input, ( s, t ) -> t.setReal( s.getRealDouble() * mul ), Util.getTypeFromInterval( input ).createVariable() );
	}

	public static < T extends RealType< T > > RandomAccessibleInterval< T > subtract(
			RandomAccessibleInterval< T > minuend,
			RandomAccessibleInterval< T > subtrahend )
	{
		return Views.interval( subtract( minuend, subtrahend, Util.getTypeFromInterval( minuend ).createVariable() ), minuend );
	}

	public static < T extends RealType< T > > RandomAccessible< T > subtract(
			RandomAccessible< T > minuend,
			RandomAccessible< T > subtrahend,
			T type )
	{
		RandomAccessible< Pair< T, T > > paired = Views.pair( minuend, subtrahend );
		return Converters.convert( paired, ( s, t ) -> {
//			System.out.println( s.getA() +  " " + s.getB() );
			t.set( s.getA() );
			t.sub( s.getB() );
		}, type );
	}

}
