package org.janelia.thickness;

import net.imglib2.Point;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.Sampler;
import net.imglib2.view.Views;

public class CollapsedRandomAccess< T > extends Point implements RandomAccess< RandomAccessibleInterval< T > >
{

	private final RandomAccessibleInterval< T > source;

	public CollapsedRandomAccess( final RandomAccessibleInterval< T > source, final int numDimensions )
	{
		super( numDimensions );

		assert numDimensions > 0 && numDimensions < source.numDimensions();

		this.source = source;
	}

	@Override
	public RandomAccessibleInterval< T > get()
	{
		RandomAccessibleInterval< T > result = source;
		for ( int dim = 0; dim < position.length; ++dim )
		{
			result = Views.hyperSlice( source, dim, position[ dim ] );
		}
		return result;
	}

	@Override
	public Sampler< RandomAccessibleInterval< T > > copy()
	{
		return copyRandomAccess();
	}

	@Override
	public RandomAccess< RandomAccessibleInterval< T > > copyRandomAccess()
	{
		return new CollapsedRandomAccess<>( source, numDimensions() );
	}


}
