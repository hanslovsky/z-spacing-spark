package org.janelia.thickness.similarity;

import java.io.Serializable;
import java.util.Arrays;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;

public class CorrelationBlockSpec implements Serializable
{

	/**
	 *
	 */
	private static final long serialVersionUID = 9091446265913635478L;

	public final long[] blockPosition;

	public final long[] min;

	public final long[] max;

	public static interface GridPositionToRealWorld
	{

		public double map( long position, int dimension );

	}

	public CorrelationBlockSpec( long[] blockPosition, long[] min, long[] max )
	{
		super();
		this.blockPosition = blockPosition;
		this.min = min;
		this.max = max;
	}

	public static CorrelationBlockSpec asSpec(
			final long[] gridPos,
			final long[] radius,
			final int halo,
			final long[] min,
			final long[] max,
			final GridPositionToRealWorld gridPosInRealWorld )
	{
		return asSpec( gridPos, radius, halo, new FinalInterval( min, max ), gridPosInRealWorld );
	}

	public static CorrelationBlockSpec asSpec(
			final long[] gridPos,
			final long[] radius,
			final int halo,
			Interval bounds,
			final GridPositionToRealWorld gridPosInRealWorld )
	{
		final long[] blockMin = new long[ gridPos.length ];
		final long[] blockMax = new long[ gridPos.length ];

		for ( int d = 0; d < gridPos.length; ++d )
		{
			double c = gridPosInRealWorld.map( gridPos[ d ], d );
			long r = radius[ d ] + halo;
			blockMin[ d ] = Math.max( ( long ) Math.ceil( c - r ), bounds.min( d ) );
			blockMax[ d ] = Math.min( ( long ) Math.floor( c + r ), bounds.max( d ) );
		}

		return new CorrelationBlockSpec( gridPos, blockMin, blockMax );

	}

	@Override
	public String toString()
	{
		return String.format(
				"(%s: pos=%s, min=%s, max=%s)",
				getClass().getSimpleName(),
				Arrays.toString( this.blockPosition ),
				Arrays.toString( this.min ),
				Arrays.toString( this.max ) );

	}

}
