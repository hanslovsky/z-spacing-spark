package org.janelia.thickness;

import java.io.File;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.janelia.thickness.inference.Options;
import org.janelia.thickness.similarity.ComputeMatricesChunked;
import org.janelia.thickness.similarity.DefaultMatrixGenerator;
import org.janelia.thickness.utility.DPTuple;
import org.janelia.thickness.utility.Utility;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import ij.ImagePlus;
import ij.io.FileSaver;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import loci.formats.FormatException;
import scala.Tuple2;

/**
 *
 * @author Philipp Hanslovsky
 *
 */
public class ZSpacing
{

	public static Logger LOG = LogManager.getLogger( MethodHandles.lookup().lookupClass() );
	static
	{
		LOG.setLevel( Level.INFO );
	}

	private static class Parameters
	{

		@Argument( metaVar = "CONFIG_PATH" )
		private String configPath;

		private boolean parsedSuccessfully;
	}

	public static void main( final String[] args ) throws FormatException, IOException
	{

		final Parameters p = new Parameters();
		final CmdLineParser parser = new CmdLineParser( p );
		try
		{
			parser.parseArgument( args );
			p.parsedSuccessfully = true;
		}
		catch ( final CmdLineException e )
		{
			System.err.println( e.getMessage() );
			parser.printUsage( System.err );
			p.parsedSuccessfully = false;
		}

		if ( p.parsedSuccessfully )

		{
			final SparkConf conf = new SparkConf()
					.setAppName( "ZSpacing" )
					.set( "spark.network.timeout", "600" )
					.set( "spark.serializer", "org.apache.spark.serializer.KryoSerializer" )
					.set( "spark.kryo.registrator", KryoSerialization.Registrator.class.getName() );

			final JavaSparkContext sc = new JavaSparkContext( conf );
			final ScaleOptions scaleOptions = ScaleOptions.createFromFile( p.configPath );

			run( sc, scaleOptions );

			sc.close();
		}


		final SparkConf conf = new SparkConf().setAppName( "ZSpacing" );

		final JavaSparkContext sc = new JavaSparkContext( conf );

		final String scaleOptionsPath = args[ 0 ];
		final ScaleOptions scaleOptions = ScaleOptions.createFromFile( scaleOptionsPath );

		run( sc, scaleOptions );

	}

	public static void run( final JavaSparkContext sc, final ScaleOptions scaleOptions ) throws FormatException, IOException
	{

		final Logger log = LOG;// LogManager.getRootLogger();

		final String sourcePattern = scaleOptions.source;
		final String root = scaleOptions.target;
		final String outputFolder = root + "/%02d";
		final int imageScaleLevel = scaleOptions.scale;
		final int start = scaleOptions.start;
		final int stop = scaleOptions.stop;
		final int size = stop - start;

		final ArrayList< Integer > indices = Utility.arange( size );
		final JavaPairRDD< Integer, FloatProcessor > sections = sc.parallelize( indices ).mapToPair( arg0 -> Utility.tuple2( arg0, arg0 ) ).sortByKey().map( arg0 -> arg0._1() ).mapToPair( new Utility.LoadFileFromPattern( sourcePattern ) ).mapToPair( new Utility.DownSample< Integer >( imageScaleLevel ) );
		sections.cache();
		sections.count();

		final FloatProcessor firstImg = sections.take( 1 ).get( 0 )._2();
		final int width = firstImg.getWidth();
		final int height = firstImg.getHeight();

		final double[] startingCoordinates = new double[ size ];
		for ( int i = 0; i < size; ++i )
			startingCoordinates[ i ] = i;

		final ArrayList< Tuple2< Tuple2< Integer, Integer >, double[] > > coordinatesList = new ArrayList<>();
		coordinatesList.add( Utility.tuple2( Utility.tuple2( 0, 0 ), startingCoordinates ) );
		JavaPairRDD< Tuple2< Integer, Integer >, double[] > coordinates = sc.parallelizePairs( coordinatesList, 1 );

		final int[][] radiiArray = scaleOptions.radii;
		final int[][] stepsArray = scaleOptions.steps;
		final Options[] options = scaleOptions.inference;

		int maxRange = 0;
		for ( int i = 0; i < options.length; ++i )
			maxRange = Math.max( maxRange, options[ i ].comparisonRange );

		final int[] dim = new int[] { width, height };

		final ArrayList< Tuple2< Long, Long > > times = new ArrayList<>();

		final HashMap< Integer, ArrayList< Integer > > indexPairs = new HashMap<>();
		for ( int i = 0; i < size; ++i )
			indexPairs.put( i, Utility.arange( i + 1, Math.min( i + maxRange + 1, size ) ) );

		final ComputeMatricesChunked computer = new ComputeMatricesChunked( sc, sections, scaleOptions.joinStepSize, maxRange, dim, size, StorageLevel.MEMORY_ONLY() );
		sections.unpersist();

		for ( int i = 0; i < radiiArray.length; ++i )
		{
			log.info( MethodHandles.lookup().lookupClass().getSimpleName() + ": i=" + i );
			log.info( MethodHandles.lookup().lookupClass().getSimpleName() + ": Options [" + i + "] = \n" + options[ i ].toString() );

			final long tStart = System.currentTimeMillis();

			final int[] currentOffset = radiiArray[ i ];
			final int[] currentStep = stepsArray[ i ];
			final int[] currentDim = new int[] { Math.max( 1, ( int ) Math.ceil( ( dim[ 0 ] - currentOffset[ 0 ] ) * 1.0 / currentStep[ 0 ] ) ), Math.max( 1, ( int ) Math.ceil( ( dim[ 1 ] - currentOffset[ 1 ] ) * 1.0 / currentStep[ 1 ] ) )
			};
			// IJ.log( "i=" + i + ": " + Arrays.toString( currentDim) +
			// Arrays.toString( currentOffset ) + Arrays.toString( currentStep )
			// + " " + size );

			final int[] previousOffset = i > 0 ? radiiArray[ i - 1 ] : new int[] { 0, 0 };
			final int[] previousStep = i > 0 ? stepsArray[ i - 1 ] : new int[] { width, height };
			final int[] previousDim = new int[] { Math.max( 1, ( int ) Math.ceil( ( dim[ 0 ] - previousOffset[ 0 ] ) * 1.0 / previousStep[ 0 ] ) ), Math.max( 1, ( int ) Math.ceil( ( dim[ 1 ] - previousOffset[ 1 ] ) * 1.0 / previousStep[ 1 ] ) )
			};

			final CorrelationBlocks cbs = new CorrelationBlocks( currentOffset, currentStep );
			final ArrayList< CorrelationBlocks.Coordinate > xyCoordinatesLocalAndWorld = cbs.generateFromBoundingBox( dim );
			final ArrayList< Tuple2< Integer, Integer > > xyCoordinates = new ArrayList<>();
			for ( final CorrelationBlocks.Coordinate xy : xyCoordinatesLocalAndWorld )
				xyCoordinates.add( xy.getLocalCoordinates() );

			if ( xyCoordinates.size() == 0 )
				xyCoordinates.add( Utility.tuple2( currentOffset[ 0 ], currentOffset[ 1 ] ) );

			JavaPairRDD< Tuple2< Integer, Integer >, double[] > currentCoordinates;
			if ( i == 0 )
				currentCoordinates = coordinates;
			else
			{
				final CorrelationBlocks cbs1 = new CorrelationBlocks( previousOffset, previousStep );
				final CorrelationBlocks cbs2 = new CorrelationBlocks( currentOffset, currentStep );
				final ArrayList< CorrelationBlocks.Coordinate > newCoords = cbs2.generateFromBoundingBox( dim );
				final ArrayList< Tuple2< Tuple2< Integer, Integer >, Tuple2< Double, Double > > > mapping = new ArrayList<>();
				for ( final CorrelationBlocks.Coordinate n : newCoords )
					mapping.add( Utility.tuple2( n.getLocalCoordinates(), cbs1.translateCoordinateIntoThisBlockCoordinates( n ) ) );
				currentCoordinates = SparkInterpolation.interpolate( sc, coordinates, sc.broadcast( mapping ), previousDim, new SparkInterpolation.MatchCoordinates.NearestNeighborMatcher() );
			}
			coordinates.unpersist();

			//			final JavaPairRDD< Tuple2< Integer, Integer >, FloatProcessor > matrices = matrixGenerator.generateMatrices( currentStep, currentOffset, options[ i ].comparisonRange );
			final JavaPairRDD< Tuple2< Integer, Integer >, FloatProcessor > matrices = computer.run( new DefaultMatrixGenerator.Factory( dim ), options[ i ].comparisonRange, currentStep, currentOffset );
			matrices.cache();

			log.info( MethodHandles.lookup().lookupClass().getSimpleName() + ": Calculated " + matrices.count() + " matrices" );

			if ( scaleOptions.logMatrices[ i ] )
			{
				final String outputFormatMatrices = String.format( outputFolder, i ) + "/matrices/%s.tif";
				// Write matrices
				matrices.mapToPair( new Utility.WriteToFormatString< Tuple2< Integer, Integer > >( outputFormatMatrices ) ).collect();
			}

			final String lutPattern = String.format( outputFolder, i ) + "/luts/%s.tif";

			final JavaPairRDD< Tuple2< Integer, Integer >, double[] > result = SparkInference.inferCoordinates( sc, matrices, currentCoordinates, options[ i ], lutPattern );
			result.cache();
			final long t0Inference = System.nanoTime();
			result.count();
			final long t1Inference = System.nanoTime();
			final long dtInference = t1Inference - t0Inference;
			log.info( MethodHandles.lookup().lookupClass().getSimpleName() + ": Inference done! (" + dtInference + "ns) " );

			// last occurence of matrices, unpersist!
			matrices.unpersist();

			// log success and failure
			final String successAndFailurePath = String.format( outputFolder, i ) + "/successAndFailure.tif";
			final ByteProcessor ip = LogSuccessAndFailure.log( sc, result, currentDim );
			Files.createDirectories( new File( successAndFailurePath ).getParentFile().toPath() );
			log.info( MethodHandles.lookup().lookupClass().getSimpleName() + ": Wrote status image? " + new FileSaver( new ImagePlus( "", ip ) ).saveAsTiff( successAndFailurePath ) );

			final ColumnsAndSections columnsAndSections = new ColumnsAndSections( currentDim, size );
			final JavaPairRDD< Integer, DPTuple > coordinateSections = columnsAndSections.columnsToSections( sc, result );

			// last occurence of result, unpersist!
			result.unpersist();

			final JavaPairRDD< Integer, DPTuple > diffused = coordinateSections
					// TODO write something like diffusion: done?
					.mapToPair( t -> Utility.tuple2( t._1(), FillHoles.fill( t._2().clone() ) ) );

			coordinates = columnsAndSections.sectionsToColumns( sc, diffused );
			coordinates.cache();
			final JavaPairRDD< Tuple2< Integer, Integer >, double[] > backward = Render.invert( sc, coordinates ).cache();
			final JavaPairRDD< Integer, DPTuple > backwardImages = columnsAndSections.columnsToSections( sc, backward );

			final String outputFormat = String.format( outputFolder, i ) + "/forward/%04d.tif";
			final List< Tuple2< Integer, Boolean > > success = diffused.mapToPair( new Utility.WriteToFormatStringDouble< Integer >( outputFormat ) ).collect();

			final String outputFormatBackward = String.format( outputFolder, i ) + "/backward/%04d.tif";
			final List< Tuple2< Integer, Boolean > > successBackward = backwardImages.mapToPair( new Utility.WriteToFormatStringDouble< Integer >( outputFormatBackward ) ).collect();

			if ( scaleOptions.logMatrices[ i ] )
			{
				// write transformed matrices
				final String outputFormatTransformedMatrices = String.format( outputFolder, i ) + "/transformed-matrices/%s.tif";
				matrices.join( backward ).mapToPair( new Utility.Transform< Tuple2< Integer, Integer > >() ).mapToPair( new Utility.WriteToFormatString< Tuple2< Integer, Integer > >( outputFormatTransformedMatrices ) ).collect();
			}

			// last occurence of backward, unpersist!
			backward.unpersist();

			int count = 0;
			for ( final Tuple2< Integer, Boolean > s : success )
			{
				if ( s._2().booleanValue() )
					continue;
				++count;
				log.warn( MethodHandles.lookup().lookupClass().getSimpleName() + ": Failed to write forward image " + s._1().intValue() );
			}

			int countBackward = 0;
			for ( final Tuple2< Integer, Boolean > s : successBackward )
			{
				if ( s._2().booleanValue() )
					continue;
				++countBackward;
				log.warn( MethodHandles.lookup().lookupClass().getSimpleName() + ": Failed to write backward image " + s._1().intValue() );
			}

			final long tEnd = System.currentTimeMillis();

			log.info( MethodHandles.lookup().lookupClass().getSimpleName() + ": Successfully wrote " + ( success.size() - count ) + "/" + success.size() + " (forward) and " + ( successBackward.size() - countBackward ) + "/" + successBackward.size() + " (backward) " + "images at iteration " + i + String.format( " in %25dms", tEnd - tStart ) );

			times.add( Utility.tuple2( tStart, tEnd ) );
		}

		for ( final Tuple2< Long, Long > t : times )
		{
			final long diff = t._2().longValue() - t._1().longValue();
			log.info( String.format( "%s: Run time for complete iteration: %25dms", MethodHandles.lookup().lookupClass().getSimpleName(), diff ) );
		}

		sc.close();
	}

}
