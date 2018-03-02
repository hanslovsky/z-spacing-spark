package org.janelia.thickness;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Callable;

import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.thickness.utility.N5Helpers;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

public class CreateProject
{
	@Command(
			name = "create-distortion-correction-project"
			)
	public static class Arguments implements Callable< Boolean > {
		@Parameters( index = "0", description = "Path to file containing list paths to image files" )
		private Path images;

		@Parameters( index = "1", description = "N5 root for integral images and matrices." )
		private String root;
		
		@Option(
				names = { "-r", "--range" },
				required = true,
				split = ",",
				description = "Z-range for each level in the hierarchy. range_i >= range_{i+1} for all i!" )
		private int[] range;
		
		@Option(
				names = { "-i", "--inference-iterations" },
				required = true,
				split = ",",
				description = "Number of iterations for optimization." )
		private int[] inferenceIterations;
		
		@Option(
				names = { "-R", "--regularization" },
				required = true,
				split = ",",
				description = "Regularization to previous result." )
		private double[] regularization;

		private final int[] defaultIntegralImageBlockSize = { 10, 10, 1000 };

		@Option(
				names = { "--integral-image-blocksize" },
				split = ",",
				required = false,
				description = "Block size for 2D integral images. Default = [10, 10, 1000]" )
		int[] integralImageBlockSize;

		@Option( names = { "-h", "--help" }, usageHelp = true, description = "Show this help message and exit." )
		private boolean helpRequested;

		@Override
		public Boolean call() throws Exception
		{
			
			int minRange = range[ 0 ];
			for ( final int r : range )
			{
				if ( r > minRange )
				{
					this.helpRequested = true;
					throw new IllegalArgumentException( "Expected ranges to follow r_0 >= r_1 >= r2 >= ... but got " + Arrays.toString( range ) );
				}
				minRange = r;
			}
			
			this.integralImageBlockSize = Optional.ofNullable( this.integralImageBlockSize ).orElse( this.defaultIntegralImageBlockSize );
			
			return true;
		}
	}
	
	public static void main( String[] args ) throws IOException {
		Arguments arguments = new Arguments();
		Boolean parsedSuccessfully = Optional.ofNullable( CommandLine.call( arguments, System.err, args ) ).orElse( false );
		
		if ( !parsedSuccessfully || arguments.helpRequested )
			return;
		
		List< String > filenames = Files.readAllLines( arguments.images );
		N5Writer n5 = N5Helpers.n5Writer( arguments.root );
		final String rootGroup = NonPlanarAxialDistortionCorrection.SEPARATOR;
		n5.setAttribute( rootGroup, NonPlanarAxialDistortionCorrection.FILENAMES_ATTRIBUTE, filenames );
		n5.setAttribute( rootGroup, NonPlanarAxialDistortionCorrection.INTEGRAL_IMAGE_BLOCK_SIZE_ATTRIBUTE, arguments.integralImageBlockSize );
		n5.setAttribute( rootGroup, NonPlanarAxialDistortionCorrection.RANGE_ATTRIBUTE, arguments.range );
		n5.setAttribute( rootGroup, NonPlanarAxialDistortionCorrection.INFERENCE_ITERATIONS_ATTRIBUTE, arguments.inferenceIterations );
		n5.setAttribute( rootGroup, NonPlanarAxialDistortionCorrection.REGULARIZATION_ATTRIBUTE, arguments.regularization );
	}

}
