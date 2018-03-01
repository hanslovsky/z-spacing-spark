package org.janelia.thickness;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.thickness.SparkInference.InputData;
import org.janelia.thickness.inference.Options;
import org.janelia.thickness.utility.Grids;
import org.janelia.thickness.utility.N5Helpers;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import loci.formats.FormatException;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccessible;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.interpolation.InterpolatorFactory;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.realtransform.InverseRealTransform;
import net.imglib2.realtransform.RealTransformRealRandomAccessible;
import net.imglib2.realtransform.RealViews;
import net.imglib2.realtransform.ScaleAndTranslation;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import net.imglib2.view.composite.Composite;
import net.imglib2.view.composite.RealComposite;
import scala.Tuple2;

/**
 *
 * @author Philipp Hanslovsky
 *
 */
@Deprecated
public class ZSpacing
{

	public static Logger LOG = LogManager.getLogger( MethodHandles.lookup().lookupClass() );
	static
	{
		LOG.setLevel( Level.INFO );
	}

	private static class Parameters
	{

		@Argument( metaVar = "ROOT_DIRECTORY" )
		private String rootDirectory;

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
			final String root = p.rootDirectory;

			run( sc, root );

			sc.close();
		}

	}

	public static void run( final JavaSparkContext sc, final String root ) throws FormatException, IOException
	{

		final Logger log = LOG;// LogManager.getRootLogger();

		final N5Reader n5 = N5Helpers.n5( root );
		if ( !n5.exists( "/" ) ) { throw new RuntimeException( "N5 root does not exist!" ); }

		final int numScaleLevels = n5.getAttribute( "/", "scaleLevels", int.class );
		final long[] dims = n5.getAttribute( "/", "dimensions", long[].class );
		final long width = dims[ 0 ];
		final long height = dims[ 1 ];
		final long size = dims[ 2 ];

		final double[] startingCoordinates = LongStream.range( 0, size ).asDoubleStream().toArray();


		final long[] dim = new long[] { width, height };

		final ArrayList< Tuple2< Long, Long > > times = new ArrayList<>();

		for ( int level = 0; level < numScaleLevels; ++level )
		{

			final Options options = n5.getAttribute( "/" + level, "inferenceOptions", Options.class );
			final long[] previousRadii = level > 0 ? n5.getAttribute( "/" + level, "radii", long[].class ) : new long[] { 0, 0 };
			final long[] previousSteps = level > 0 ? n5.getAttribute( "/" + level, "steps", long[].class ) : new long[] { width, height };
			final long[] radii = n5.getAttribute( "/" + level, "radii", long[].class );
			final long[] steps = n5.getAttribute( "/" + level, "steps", long[].class );

			final long[] currentDim = new long[] {
					Math.max( 1, ( long ) Math.ceil( ( dim[ 0 ] - radii[ 0 ] ) * 1.0 / steps[ 0 ] ) ),
					Math.max( 1, ( long ) Math.ceil( ( dim[ 1 ] - radii[ 1 ] ) * 1.0 / steps[ 1 ] ) )
			};
			final long numElements = Intervals.numElements( currentDim );
			final int stepSize = ( int ) Math.ceil( Math.max( numElements / sc.defaultParallelism(), 1 ) );
			final int[] blockSize = IntStream.generate( () -> stepSize ).limit( currentDim.length ).toArray();
			final List< Interval > blocks = Grids.collectAllContainedIntervals( currentDim, blockSize );

			final ScaleAndTranslation previousToWorld = new ScaleAndTranslation(
					Arrays.stream( previousSteps ).asDoubleStream().toArray(),
					Arrays.stream( previousRadii ).asDoubleStream().toArray() );
			final ScaleAndTranslation currentToWorld = new ScaleAndTranslation(
					Arrays.stream( steps ).asDoubleStream().toArray(),
					Arrays.stream( radii ).asDoubleStream().toArray() );

			final ScaleAndTranslation previousToCurrent = currentToWorld.inverse().concatenate( previousToWorld );

			final JavaRDD< Interval > blocksRDD = sc.parallelize( blocks );

			final String coordinateDataset = "/" + ( level - 1 ) + "/forward";
			final String matrixDataset = "/" + level + "/matrices";
			final Supplier< RandomAccessibleInterval< RealComposite< DoubleType > > > coordinateSupplier = level == 0 ? () -> asComposite( startingCoordinates ) : collapsedDataSupplier( root, coordinateDataset );
			final Supplier< RandomAccessibleInterval< RandomAccessibleInterval< FloatType > > > matrixSupplier = matrixDataSupplier( root, matrixDataset );
			final JavaRDD< InputData< FloatType, DoubleType > > matricesAndCoordinates = getMatricesAndCoordinates(
					blocksRDD,
					matrixSupplier,
					coordinateSupplier,
					previousToCurrent,
					new NLinearInterpolatorFactory<>() );

			// TODO do we need pattern?
			final JavaRDD< RandomAccessibleInterval< DoubleType > > newCoordinates =
					matricesAndCoordinates.map( new SparkInference.Inference<>( options, startingCoordinates.length ) );

			final String outputDataset = "/" + level + "/forward";
			N5Helpers.n5Writer( root ).createDataset( outputDataset, currentDim, blockSize, DataType.FLOAT64, new GzipCompression() );
			newCoordinates.foreach( coordinates -> {
				final long[] blockPosition = new long[ coordinates.numDimensions() ];
				Arrays.setAll( blockPosition, d -> coordinates.min( d ) / blockSize[ d ] );
				N5Utils.saveBlock( coordinates, N5Helpers.n5Writer( root ), outputDataset, blockPosition );
			} );


		}

		for ( final Tuple2< Long, Long > t : times )
		{
			final long diff = t._2().longValue() - t._1().longValue();
			log.info( String.format( "%s: Run time for complete iteration: %25dms", MethodHandles.lookup().lookupClass().getSimpleName(), diff ) );
		}
		sc.close();
	}

	public static < T, U extends RealType< U >, C extends Composite< U > >
	JavaRDD< SparkInference.InputData< T, U > > getMatricesAndCoordinates(
			final JavaRDD< Interval > blocks,
			final Supplier< RandomAccessibleInterval< RandomAccessibleInterval< T > > > matrixSupplier,
			final Supplier< RandomAccessibleInterval< C > > previousCoordinateSupplier,
			final ScaleAndTranslation fromPreviousToCurrent,
			final InterpolatorFactory< C, RandomAccessible< C > > interpolation
			)
	{
		final JavaRDD< SparkInference.InputData< T, U > > inputData = blocks.map( block -> {
			final RandomAccessible< C > extended = Views.extendBorder( previousCoordinateSupplier.get() );
			final RealRandomAccessible< C > interpolated = Views.interpolate( extended, interpolation );
			final RealTransformRealRandomAccessible< C, InverseRealTransform > transformed = RealViews.transformReal( interpolated, fromPreviousToCurrent );
			return new SparkInference.InputData<>( Views.interval( matrixSupplier.get(), block ), Views.interval( Views.raster( transformed ), block ) );
		} );
		return inputData;

	}

	public static < T extends NativeType< T > > Supplier< RandomAccessibleInterval< RandomAccessibleInterval< T > > > matrixDataSupplier(
			final String root,
			final String dataset ) throws IOException
	{
		final Supplier< RandomAccessibleInterval< T > > dataSupplier = dataSupplier( root, dataset );
		return () -> collapseToMatrices( dataSupplier.get() );
	}

	public static < T extends NativeType< T > & RealType< T > > Supplier< RandomAccessibleInterval< RealComposite< T > > > collapsedDataSupplier(
			final String root,
			final String dataset
			) throws IOException
	{
		final Supplier< RandomAccessibleInterval< T > > dataSupplier = dataSupplier( root, dataset );
		return () -> Views.collapseReal( dataSupplier.get() );
	}

	public static < T extends NativeType< T > > Supplier< RandomAccessibleInterval< T > > dataSupplier(
			final String root,
			final String dataset ) throws IOException
	{
		return dataSupplier( N5Helpers.n5( root ), dataset );
	}

	public static < T extends NativeType< T > > Supplier< RandomAccessibleInterval< T > > dataSupplier(
			final N5Reader reader,
			final String dataset )
	{
		return () -> {
			try
			{
				return N5Utils.open( reader, dataset );
			}
			catch ( final IOException e )
			{
				throw new RuntimeException( e );
			}
		};
	}

	public static RandomAccessibleInterval< RealComposite< DoubleType > > asComposite( final double[] data )
	{
		return Views.collapseReal( ArrayImgs.doubles( data, data.length, 1 ) );
	}

	public static < T > RandomAccessibleInterval< RandomAccessibleInterval< T > > collapseToMatrices(
			final RandomAccessibleInterval< T > matrixData )
	{
		assert matrixData.numDimensions() > 2;
		return new CollapsedRandomAccessibleInterval<>( matrixData, 2 );
	}


}
