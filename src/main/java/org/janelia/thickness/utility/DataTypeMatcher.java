package org.janelia.thickness.utility;

import org.janelia.saalfeldlab.n5.DataType;

import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

public class DataTypeMatcher
{
	@SuppressWarnings( "unchecked" )
	public static < T extends NativeType< T > > T toImgLib( DataType dataType ) throws IllegalArgumentException {
		switch ( dataType )
		{
		case INT8:
			return ( T )new ByteType();
		case UINT8:
			return ( T )new UnsignedByteType();
		case INT16:
			return ( T )new ShortType();
		case UINT16:
			return ( T )new UnsignedShortType();
		case INT32:
			return ( T )new IntType();
		case UINT32:
			return ( T )new UnsignedIntType();
		case INT64:
			return ( T )new LongType();
		case UINT64:
			return ( T )new UnsignedLongType();
		case FLOAT32:
			return ( T )new FloatType();
		case FLOAT64:
			return ( T )new DoubleType();
		default:
			throw new IllegalArgumentException( "Cannot match " + dataType + " to a " + NativeType.class.getName() );
		}
	}
	
	public static < T extends NativeType< T > > DataType toDataType( T t ) throws IllegalArgumentException {
		if ( t instanceof ByteType )
			return DataType.INT8;
		if ( t instanceof UnsignedByteType )
			return DataType.UINT8;
		if ( t instanceof ShortType )
			return DataType.INT16;
		if ( t instanceof UnsignedShortType )
			return DataType.UINT16;
		if ( t instanceof IntType )
			return DataType.INT32;
		if ( t instanceof UnsignedIntType )
			return DataType.UINT32;
		if ( t instanceof LongType )
			return DataType.INT64;
		if ( t instanceof UnsignedLongType )
			return DataType.UINT64;
		if ( t instanceof FloatType )
			return DataType.FLOAT32;
		if ( t instanceof DoubleType )
			return DataType.FLOAT64;
		throw new IllegalArgumentException( "Cannot match " + t.getClass().getName() + " to a " + DataType.class.getName() );
	}
}
