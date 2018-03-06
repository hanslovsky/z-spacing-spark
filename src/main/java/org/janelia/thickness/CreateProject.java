package org.janelia.thickness;

import java.io.IOException;
import java.util.Arrays;
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
			name = "create-distortion-correction-project" )
	public static class Arguments implements Callable< Boolean >
	{
		@Parameters( index = "0", description = "N5 root for source data" )
		private String sourceRoot;

		@Parameters( index = "1", description = "N5 dataset for source data" )
		private String sourceDataset;

		@Parameters( index = "2", description = "N5 root for project." )
		private String root;

		@Option(
				names = { "-r", "--range" },
				required = true,
				split = ",",
				description = "Z-range for each level in the hierarchy. range_i >= range_{i+1} for all i!" )
		private int[] range;

		@Option(
				names = { "-n", "--num-inference-iterations" },
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

		@Option( names = { "-h", "--help" }, usageHelp = true, description = "Show this help message and exit." )
		private boolean helpRequested;

		@Option( names = { "-g", "--group" }, description = "N5 group for project. Defaults to `z-spacing-correction'" )
		private String group = "z-spacing-correction";

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

			return true;
		}
	}

	public static void main( String[] args ) throws IOException
	{
		Arguments arguments = new Arguments();
		Boolean parsedSuccessfully = Optional.ofNullable( CommandLine.call( arguments, System.err, args ) ).orElse( false );

		if ( !parsedSuccessfully || arguments.helpRequested )
			return;

		N5Writer n5 = N5Helpers.n5Writer( arguments.root );
		n5.createGroup( arguments.group );
		n5.setAttribute( arguments.group, NonPlanarAxialDistortionCorrection.SOURCE_ROOT_ATTRIBUTE, arguments.sourceRoot );
		n5.setAttribute( arguments.group, NonPlanarAxialDistortionCorrection.SOURCE_DATASET_ATTRIBUTE, arguments.sourceDataset );
		n5.setAttribute( arguments.group, NonPlanarAxialDistortionCorrection.RANGE_ATTRIBUTE, arguments.range );
		n5.setAttribute( arguments.group, NonPlanarAxialDistortionCorrection.INFERENCE_ITERATIONS_ATTRIBUTE, arguments.inferenceIterations );
		n5.setAttribute( arguments.group, NonPlanarAxialDistortionCorrection.REGULARIZATION_ATTRIBUTE, arguments.regularization );
	}

}
