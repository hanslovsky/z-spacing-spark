package org.janelia.thickness.similarity;

import java.io.Serializable;
import java.util.Arrays;

public class CorrelationBlockSpec implements Serializable
{

	/**
	 *
	 */
	private static final long serialVersionUID = 9091446265913635478L;

	public final long[] blockPosition;

	public final long[] min;

	public final long[] max;

	public CorrelationBlockSpec( long[] blockPosition, long[] min, long[] max )
	{
		super();
		this.blockPosition = blockPosition;
		this.min = min;
		this.max = max;
	}

	public static CorrelationBlockSpec asSpec( long[] min, int[] blockSize, long[] extent, long[] maxCap )
	{
		return asSpec( min, blockSize, extent, maxCap, false );
	}

	public static CorrelationBlockSpec asSpec( long[] min, int[] blockSize, long[] extent, long[] maxCap, boolean ensureFullSize )
	{
		return asSpec( min, blockSize, extent, maxCap, new long[ min.length ], ensureFullSize );
	}

	public static CorrelationBlockSpec asSpec( long[] min, int[] blockSize, long[] extent, long[] maxCap, long[] minCap, boolean ensureFullSize )
	{
		long[] blockPosition = new long[ min.length ];
		long[] max = new long[ min.length ];
		long[] actualMin = min.clone();
		Arrays.setAll( blockPosition, d -> min[ d ] / blockSize[ d ] );
		Arrays.setAll( max, d -> Math.min( min[ d ] + extent[ d ] - 1, maxCap[ d ] ) );
		Arrays.setAll( actualMin, d -> ( ensureFullSize && max[ d ] == maxCap[ d ] ) ? Math.max( max[ d ] - ( extent[ d ] - 1 ), minCap[ d ] ) : min[ d ] );
		return new CorrelationBlockSpec( blockPosition, actualMin, max );
	}

}
