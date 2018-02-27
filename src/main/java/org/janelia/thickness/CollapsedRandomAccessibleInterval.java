package org.janelia.thickness;

import java.util.Arrays;

import net.imglib2.AbstractWrappedInterval;
import net.imglib2.Dimensions;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.util.Intervals;

public class CollapsedRandomAccessibleInterval< T >
extends AbstractWrappedInterval< Interval >
implements RandomAccessibleInterval< RandomAccessibleInterval< T > >
{

	private final RandomAccessibleInterval< T > source;

	private final int numCollapsedDimensions;

	private final int numDimensions;

	public CollapsedRandomAccessibleInterval( final RandomAccessibleInterval< T > source, final int numCollapsedDimensions )
	{
		super( subInterval( source, numDimensions( source, numCollapsedDimensions ) ) );

		assert numCollapsedDimensions > 0 && numCollapsedDimensions < source.numDimensions();

		this.source = source;
		this.numCollapsedDimensions = numCollapsedDimensions;
		this.numDimensions = numDimensions( source, numCollapsedDimensions );
	}

	@Override
	public RandomAccess< RandomAccessibleInterval< T > > randomAccess()
	{
		return new CollapsedRandomAccess<>( source, numDimensions );
	}

	@Override
	public RandomAccess< RandomAccessibleInterval< T > > randomAccess( final Interval interval )
	{
		return randomAccess();
	}

	public static Interval subInterval(
			final Interval interval,
			final int numDimensions )
	{
		assert numDimensions > 0 && numDimensions < interval.numDimensions();

		final long[] min = Arrays.stream( Intervals.minAsLongArray( interval ) ).limit( numDimensions ).toArray();
		final long[] max = Arrays.stream( Intervals.maxAsLongArray( interval ) ).limit( numDimensions ).toArray();

		return new FinalInterval( min, max );

	}

	public int numCollapsedDimensions()
	{
		return this.numCollapsedDimensions;
	}

	public static int numDimensions( final Dimensions dim, final int numCollapsedDimensions )
	{
		return dim.numDimensions() - numCollapsedDimensions;
	}

}
