package org.janelia.thickness;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.thickness.SparkInference.InputData;
import org.janelia.thickness.inference.InferFromMatrix.RegularizationType;
import org.janelia.thickness.inference.Options;
import org.janelia.thickness.similarity.GenerateIntegralImages;
import org.janelia.thickness.similarity.MatricesFromN5;
import org.janelia.thickness.utility.Grids;
import org.janelia.thickness.utility.N5Helpers;

import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccessible;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.interpolation.InterpolatorFactory;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.realtransform.InverseRealTransform;
import net.imglib2.realtransform.InvertibleRealTransform;
import net.imglib2.realtransform.RealTransformRealRandomAccessible;
import net.imglib2.realtransform.RealViews;
import net.imglib2.realtransform.ScaleAndTranslation;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import net.imglib2.view.composite.Composite;
import net.imglib2.view.composite.RealComposite;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

public class NonPlanarAxialDistortionCorrection
{

	public static final String INTEGRAL_SUM_DATASET = "integral-sum";

	public static final String INTEGRAL_SUM_SQUARED_DATASET = "integral-sum-squared";

	public static final String MATRICES_DATASET = "matrices";

	public static String FILENAMES_ATTRIBUTE = "filenames";

	public static String SEPARATOR = "/";

	public static String FORWARD_DATASET = "forward";

	public static String BACKWARD_DATASET = "backward";

	public static String INTEGRAL_IMAGE_BLOCK_SIZE_ATTRIBUTE = "integralImageBlockSize";

	public static String RANGE_ATTRIBUTE = "range";

	public static String INFERENCE_ITERATIONS_ATTRIBUTE = "inferenceIterations";

	public static String REGULARIZATION_ATTRIBUTE = "regularization";

	@Command(
			name = "distortion-correction",
			sortOptions = true )
	public static class CommandLineArguments implements Callable< Boolean >
	{
		@Parameters( index = "0", description = "N5 root for integral images and matrices." )
		private String root;

		@Option( 
				names = "--generate-integral-images", 
				required = false,
				description ="If specified, generate integral images even when dataset exists. This will switch on --generate-matrices as well." )
		private boolean generateIntegralImages;

		@Option( 
				names = "--generate-matrices", 
				required = false, 
				description = "If specified, generate matrices even when dataset exists." )
		private boolean generateMatrices;

		@Override
		public Boolean call() throws Exception
		{
			this.generateIntegralImages |= this.generateIntegralImages;
			return true;
		}
	}

	public static void main( final String[] args ) throws IOException
	{

		final CommandLineArguments cmdLineArgs = new CommandLineArguments();
		final boolean parsedSuccessfully = Optional.ofNullable( CommandLine.call( cmdLineArgs, System.err, args ) ).orElse( false );

		System.out.println( "Parsed successfully? " + parsedSuccessfully );

		if ( !parsedSuccessfully ) { return; }

		final N5Writer n5 = N5Helpers.n5Writer( cmdLineArgs.root );
		String rootGroup = SEPARATOR;
		List< String > filenames = Arrays.asList( n5.getAttribute( rootGroup, FILENAMES_ATTRIBUTE, String[].class ) );
		final int[] integralImageBlockSize = n5.getAttribute( rootGroup, INTEGRAL_IMAGE_BLOCK_SIZE_ATTRIBUTE, int[].class );
		final int[] ranges = n5.getAttribute( rootGroup, RANGE_ATTRIBUTE, int[].class );
		final int[] iterations = n5.getAttribute( rootGroup, INFERENCE_ITERATIONS_ATTRIBUTE, int[].class );
		final double[] regularization = n5.getAttribute( rootGroup, REGULARIZATION_ATTRIBUTE, double[].class );
		
		int lastIteration = iterations[ 0 ];
		double lastRegularization = regularization[ 0 ];

		final SparkConf conf = new SparkConf()
				.setAppName( MethodHandles.lookup().lookupClass().getName() )
				.set( "spark.serializer", "org.apache.spark.serializer.KryoSerializer" )
				.set( "spark.kryo.registrator", KryoSerialization.Registrator.class.getName() );

		double[] startingCoordinates = LongStream.range( 0, filenames.size() ).asDoubleStream().toArray();

		try (final JavaSparkContext sc = new JavaSparkContext( conf ))
		{
			LogManager.getRootLogger().setLevel( Level.ERROR );

			if ( cmdLineArgs.generateIntegralImages || !n5.datasetExists( INTEGRAL_SUM_DATASET ) || !n5.datasetExists( INTEGRAL_SUM_SQUARED_DATASET ) )
			{
				System.out.println( "Creating integral images." );
				GenerateIntegralImages.run(
						sc,
						filenames,
						integralImageBlockSize,
						ranges[ 0 ],
						cmdLineArgs.root,
						INTEGRAL_SUM_DATASET,
						INTEGRAL_SUM_SQUARED_DATASET );
			}
			else
			{
				System.out.println( "Integral images already exist -- skipping" );
			}

			final long[] steps = new long[ 2 ];
			final long[] radii = new long[ 2 ];
			final DatasetAttributes integralSumAttrs = N5Helpers.n5( cmdLineArgs.root ).getDatasetAttributes( INTEGRAL_SUM_DATASET );
			final long[] imgDim = Arrays.stream( integralSumAttrs.getDimensions() ).limit( 2 ).map( l -> l - 1 ).toArray();

			for ( int d = 0; d < steps.length; ++d )
			{
				steps[ d ] = imgDim[ d ];
				radii[ d ] = steps[ d ] / 2;
			}

			long[] previousSteps = imgDim.clone();
			long[] previousRadii = { 0, 0 };

			final String root = cmdLineArgs.root;

			for ( int level = 0; level < ranges.length; ++level )
			{

				if ( Arrays.stream( radii ).filter( r -> r > 1 ).count() == 0 )
				{
					break;
				}

				final int range = ranges[ level ];

				final List< long[] > positions = Grids
						.collectAllOffsets( imgDim, Arrays.stream( steps ).mapToInt( l -> ( int ) l ).toArray(), a -> divide( a, steps, a ) );
				final JavaRDD< long[] > blocksRDD = sc
						.parallelize( positions );

				final long[] maxPosition = new long[] { 0, 0 };
				positions.forEach( p -> Arrays.setAll( maxPosition, d -> Math.max( maxPosition[ d ], p[ d ] ) ) );
				final long[] currentDim = Arrays.stream( maxPosition ).map( p -> p + 1 ).toArray();

				final String matrixDataset = level + SEPARATOR + MATRICES_DATASET;
				final String coordinateDataset = level + SEPARATOR + FORWARD_DATASET;
				final String previousCoordinateDataset = ( level - 1 ) + SEPARATOR + FORWARD_DATASET;

				if ( cmdLineArgs.generateMatrices || !n5.datasetExists( matrixDataset ) )
				{
					MatricesFromN5.makeMatrices(
							blocksRDD,
							radii,
							steps,
							filenames.size(),
							range,
							currentDim,
							Arrays.stream( imgDim ).map( dim -> dim - 1 ).toArray(),
							cmdLineArgs.root,
							INTEGRAL_SUM_DATASET,
							INTEGRAL_SUM_SQUARED_DATASET,
							matrixDataset,
							new GzipCompression(),
							sc.broadcast( new DoubleType() ) );
				}

				final ScaleAndTranslation previousToWorld = new ScaleAndTranslation(
						Arrays.stream( previousSteps ).asDoubleStream().toArray(),
						Arrays.stream( previousRadii ).asDoubleStream().toArray() );
				final ScaleAndTranslation currentToWorld = new ScaleAndTranslation(
						Arrays.stream( steps ).asDoubleStream().toArray(),
						Arrays.stream( radii ).asDoubleStream().toArray() );

				final ScaleAndTranslation previousToCurrent = currentToWorld.inverse().concatenate( previousToWorld );

				final Supplier< RandomAccessibleInterval< RealComposite< DoubleType > > > coordinateSupplier =
						level == 0 ? () -> asComposite( startingCoordinates ) : collapsedDataSupplier( cmdLineArgs.root, previousCoordinateDataset );
				final Supplier< RandomAccessibleInterval< RandomAccessibleInterval< DoubleType > > > matrixSupplier = matrixDataSupplier( cmdLineArgs.root, matrixDataset );

				final long numElements = Intervals.numElements( currentDim );
				final int stepSize = ( int ) Math.ceil( Math.sqrt( Math.max( numElements * 1.0 / sc.defaultParallelism(), 1 ) ) );
				final int[] blockSize = IntStream.generate( () -> stepSize ).limit( currentDim.length ).toArray();
				List< long[] > inferenceBlocks = Grids.collectAllOffsets( currentDim, blockSize );

				JavaRDD< Interval > inferenceBlocksRdd = sc
						.parallelize( inferenceBlocks )
						.map( min -> {
							long[] max = new long[ min.length ];
							Arrays.setAll( max, d -> Math.min( min[ d ] + blockSize[ d ], currentDim[ d ] ) - 1 );
							return new FinalInterval( min, max );
						} );

				final JavaRDD< SparkInference.InputData< DoubleType, DoubleType > > matricesAndCoordinates = getMatricesAndCoordinates(
						sc,
						inferenceBlocksRdd,
						matrixSupplier,
						coordinateSupplier,
						previousToCurrent,
						new NLinearInterpolatorFactory<>() );

				// update steps and radii
				Arrays.setAll( previousSteps, d -> steps[ d ] );
				Arrays.setAll( previousRadii, d -> radii[ d ] );

				for ( int d = 0; d < steps.length; ++d )
				{
					steps[ d ] = Math.max( steps[ d ] / 2, 1 );
					radii[ d ] = Math.max( steps[ d ] / 2, 1 );
				}

				Options options = Options.generateDefaultOptions();
				options.comparisonRange = range;
				options.coordinateUpdateRegularizerWeight = level < regularization.length ? regularization[ level ] : lastRegularization / 2.0;
				options.nIterations = level < iterations.length ? iterations[ level ] : Math.max( lastIteration / 2, 1 );
				options.regularizationType = level == 0 ? RegularizationType.BORDER : RegularizationType.NONE;
				lastRegularization = options.coordinateUpdateRegularizerWeight;
				lastIteration = options.nIterations;

				final JavaRDD< RandomAccessibleInterval< DoubleType > > newCoordinates =
						matricesAndCoordinates.map( new SparkInference.Inference<>(
								options,
								startingCoordinates.length ) );

				long[] coordinatesDims = append( currentDim, filenames.size() );
				int[] coordinatesBlockSize = append( blockSize, filenames.size() );
				n5.createDataset( coordinateDataset, coordinatesDims, coordinatesBlockSize, DataType.FLOAT64, new GzipCompression() );
				newCoordinates.foreach( coordinates -> {
					final long[] blockPosition = new long[ coordinates.numDimensions() ];
					Arrays.setAll( blockPosition, d -> coordinates.min( d ) / coordinatesBlockSize[ d ] );
					N5Utils.saveBlock( coordinates, N5Helpers.n5Writer( root ), coordinateDataset, blockPosition );
				} );

			}

		}

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
			final String dataset ) throws IOException
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
		return Views.collapseReal( ArrayImgs.doubles( data, 1, 1, data.length ) );
	}

	public static < T > RandomAccessibleInterval< RandomAccessibleInterval< T > > collapseToMatrices(
			final RandomAccessibleInterval< T > matrixData )
	{
		assert matrixData.numDimensions() > 2;
		return new CollapsedRandomAccessibleInterval<>( matrixData, 2 );
	}

	public static long[] divide( final long[] dividend, final int[] divisor, final long[] quotient )
	{
		Arrays.setAll( quotient, d -> dividend[ d ] / divisor[ d ] );
		return quotient;
	}

	static public long[] divide( final long[] dividend, final long[] divisor, final long[] quotient )
	{
		Arrays.setAll( quotient, d -> dividend[ d ] / divisor[ d ] );
		return quotient;
	}

	public static < T, U extends RealType< U >, C extends Composite< U > >
			JavaRDD< SparkInference.InputData< T, U > > getMatricesAndCoordinates(
					final JavaSparkContext sc,
					final JavaRDD< Interval > blocks,
					final Supplier< RandomAccessibleInterval< RandomAccessibleInterval< T > > > matrixSupplier,
					final Supplier< RandomAccessibleInterval< C > > previousCoordinateSupplier,
					final ScaleAndTranslation fromPreviousToCurrent,
					final InterpolatorFactory< C, RandomAccessible< C > > interpolation )
	{
		return blocks.map( new GetMatrixAndCoordinates<>(
				sc.broadcast( previousCoordinateSupplier ),
				sc.broadcast( interpolation ),
				sc.broadcast( fromPreviousToCurrent ),
				sc.broadcast( matrixSupplier ) ) );
	}

	public static class GetMatrixAndCoordinates< T, U extends RealType< U >, C extends Composite< U > > implements Function< Interval, SparkInference.InputData< T, U > >
	{

		private final Broadcast< Supplier< RandomAccessibleInterval< C > > > previousCoordinateSupplier;

		private final Broadcast< InterpolatorFactory< C, RandomAccessible< C > > > interpolation;

		private final Broadcast< InvertibleRealTransform > fromPreviousToCurrent;

		private final Broadcast< Supplier< RandomAccessibleInterval< RandomAccessibleInterval< T > > > > matrixSupplier;

		public GetMatrixAndCoordinates(
				Broadcast< Supplier< RandomAccessibleInterval< C > > > previousCoordinateSupplier,
				Broadcast< InterpolatorFactory< C, RandomAccessible< C > > > interpolation,
				Broadcast< InvertibleRealTransform > fromPreviousToCurrent,
				Broadcast< Supplier< RandomAccessibleInterval< RandomAccessibleInterval< T > > > > matrixSupplier )
		{
			super();
			this.previousCoordinateSupplier = previousCoordinateSupplier;
			this.interpolation = interpolation;
			this.fromPreviousToCurrent = fromPreviousToCurrent;
			this.matrixSupplier = matrixSupplier;
		}

		@Override
		public InputData< T, U > call( Interval block ) throws Exception
		{
			final RandomAccessible< C > extended = Views.extendBorder( previousCoordinateSupplier.getValue().get() );
			final RealRandomAccessible< C > interpolated = Views.interpolate( extended, interpolation.getValue() );
			final RealTransformRealRandomAccessible< C, InverseRealTransform > transformed = RealViews.transformReal( interpolated, fromPreviousToCurrent.getValue() );
			return new SparkInference.InputData<>( Views.interval( matrixSupplier.getValue().get(), block ), Views.interval( Views.raster( transformed ), block ) );
		}

	}

	public static long[] append( long[] array, long... appendix )
	{
		return LongStream
				.concat( Arrays.stream( array ), LongStream.of( appendix ) )
				.toArray();
	}

	public static int[] append( int[] array, int... appendix )
	{
		return IntStream
				.concat( Arrays.stream( array ), IntStream.of( appendix ) )
				.toArray();
	}

}
