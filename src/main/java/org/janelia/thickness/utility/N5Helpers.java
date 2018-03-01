package org.janelia.thickness.utility;

import java.io.IOException;

import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;

public class N5Helpers
{

	public static N5Writer n5Writer( final String root ) throws IOException
	{
		return new N5FSWriter( root );
	}

	public static N5Reader n5( final String root ) throws IOException
	{
		return new N5FSReader( root );
	}

}
