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
		long[] blockPosition = new long[ min.length ];
		long[] max = new long[ min.length ];
		Arrays.setAll( blockPosition, d -> min[ d ] / blockSize[ d ] );
		Arrays.setAll( max, d -> Math.min( min[ d ] + extent[ d ], maxCap[ d ] ) );
		return new CorrelationBlockSpec( blockPosition, min, max );
	}

}
