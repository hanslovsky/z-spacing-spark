package org.janelia.thickness;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.function.Supplier;
import java.util.stream.Collectors;
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
import org.janelia.thickness.similarity.CorrelationBlockSpec;
import org.janelia.thickness.similarity.MatricesFromN5ParallelizeOverXY;
import org.janelia.thickness.similarity.MatricesFromN5ParallelizeOverXY.DataSupplier;
import org.janelia.thickness.similarity.MatricesFromN5ParallelizeOverZ;
import org.janelia.thickness.utility.DataTypeMatcher;
import org.janelia.thickness.utility.Grids;
import org.janelia.thickness.utility.N5Helpers;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

	private static final Logger LOG = LoggerFactory.getLogger( MethodHandles.lookup().lookupClass() );

	public static final String MATRICES_DATASET = "matrices";

	public static String SEPARATOR = "/";

	public static String FORWARD_DATASET = "forward";

	public static String BACKWARD_DATASET = "backward";

	public static String RANGE_ATTRIBUTE = "range";

	public static String INFERENCE_ITERATIONS_ATTRIBUTE = "inferenceIterations";

	public static String REGULARIZATION_ATTRIBUTE = "regularization";

	public static String STEP_SIZE_ATTRIBUTE = "stepSize";

	public static String SOURCE_ROOT_ATTRIBUTE = "sourceRoot";

	public static String SOURCE_DATASET_ATTRIBUTE = "sourceDataset";

	public static String HALO_ATTRIBUTE = "halo";

	private static final int[] ONES_2D = { 1, 1 };

	@Command(
			name = "distortion-correction",
			sortOptions = true )
	public static class CommandLineArguments implements Callable< Boolean >
	{
		@Parameters( index = "0", description = "N5 root for integral images and matrices." )
		private String root;

		@Option( names = { "-g", "--group" }, description = "N5 group for project. Defaults to `z-spacing-correction'" )
		private String group = "z-spacing-correction";

		@Option(
				names = "--overwrite",
				required = false,
				description = "If specified, overwrite existing data. Otherwise, skip matrix generation/inference if data is not present." )
		private boolean overwrite;

		@Option(
				names = { "--matrices-only", "-m" },
				required = false,
				description = "Generate matrices only, do not run inference." )
		private boolean matricesOnly;

		@Override
		public Boolean call() throws Exception
		{
			return true;
		}
	}

	public static void main( final String[] args ) throws Exception
	{

		final CommandLineArguments cmdLineArgs = new CommandLineArguments();
		final boolean parsedSuccessfully = Optional.ofNullable( CommandLine.call( cmdLineArgs, System.err, args ) ).orElse( false );

		System.out.println( "Parsed successfully? " + parsedSuccessfully );

		if ( !parsedSuccessfully ) { return; }

		final N5Writer n5 = N5Helpers.n5Writer( cmdLineArgs.root );
		String rootGroup = cmdLineArgs.group;
		final int[] ranges = n5.getAttribute( rootGroup, RANGE_ATTRIBUTE, int[].class );
		final int[] iterations = n5.getAttribute( rootGroup, INFERENCE_ITERATIONS_ATTRIBUTE, int[].class );
		final double[] regularization = n5.getAttribute( rootGroup, REGULARIZATION_ATTRIBUTE, double[].class );
		final double[] inferenceStepSize = n5.getAttribute( rootGroup, STEP_SIZE_ATTRIBUTE, double[].class );
		final int[] halos = Optional.ofNullable( n5.getAttribute( rootGroup, HALO_ATTRIBUTE, int[].class ) ).orElse( new int[ ranges.length ] );

		final String root = cmdLineArgs.root;
		N5Reader rootN5 = N5Helpers.n5( root );
		String sourceRoot = rootN5.getAttribute( rootGroup, SOURCE_ROOT_ATTRIBUTE, String.class );
		String sourceDataset = rootN5.getAttribute( rootGroup, SOURCE_DATASET_ATTRIBUTE, String.class );
		N5Reader sourceN5 = N5Helpers.n5( sourceRoot );
		DatasetAttributes sourceAttrs = sourceN5.getDatasetAttributes( sourceDataset );
		long[] sourceDim = sourceAttrs.getDimensions();

		double[] startingCoordinates = LongStream.range( 0, sourceDim[ 2 ] ).asDoubleStream().toArray();

		int lastIteration = iterations[ 0 ];
		double lastRegularization = regularization[ 0 ];

		final SparkConf conf = new SparkConf()
				.setAppName( MethodHandles.lookup().lookupClass().getName() )
				.set( "spark.serializer", "org.apache.spark.serializer.KryoSerializer" )
				.set( "spark.kryo.registrator", KryoSerialization.Registrator.class.getName() );

		try (final JavaSparkContext sc = new JavaSparkContext( conf ))
		{
			LogManager.getRootLogger().setLevel( Level.ERROR );
			sc.setLogLevel( "ERROR" );

			final long[] imgDim = { sourceDim[ 0 ], sourceDim[ 1 ] };
			final Interval bounds = new FinalInterval( imgDim );

			final long smallestDim = Math.min( imgDim[ 0 ], imgDim[ 1 ] );
			final long[] steps = { smallestDim, smallestDim };
			final long[] radii = { steps[ 0 ] / 2, steps[ 1 ] / 2 };
			final double[] offset = Arrays.stream( steps ).mapToDouble( s -> 0.5 * s ).toArray();


			// real world coordinates = step * c + offset - 1/2

			long[] previousSteps = steps.clone();
			long[] previousRadii = { 0, 0 };
			double[] previousOffset = { 0, 0 };

			for ( int level = 0; level < ranges.length; ++level )
			{

				if ( Arrays.stream( radii ).filter( r -> r > 1 ).count() == 0 )
				{
					break;
				}

				final int range = ranges[ level ];

				final boolean isShiftedByHalfStep = level != 0;
				final long[] currentDim = gridDims( steps, imgDim, isShiftedByHalfStep );

				final List< long[] > positions = Grids.collectAllOffsets( currentDim, ONES_2D );

				int halo = level < halos.length ? halos[ level ] : 0;
//				CorrelationBlockSpec.asSpec( positions.get( 0 ), radii, halo, bounds, (pos, d) -> pos * steps[ d ] + offset[ d ] - 0.5 );
				CorrelationBlockSpec.GridPositionToRealWorld mapper = (pos, d) -> pos * steps[ d ] + offset[ d ] - 0.5;
				final List< CorrelationBlockSpec > specs = positions
						.stream()
						.map( p -> CorrelationBlockSpec.asSpec( p, radii, halo, bounds, mapper ) )
						.collect( Collectors.toList() );

				final long[] matrixDim = { currentDim[ 0 ], currentDim[ 1 ], 2 * range + 1, sourceDim[ 2 ] };

				final String matrixDataset = rootGroup + SEPARATOR + level + SEPARATOR + MATRICES_DATASET;
				final String coordinateDataset = rootGroup + SEPARATOR + level + SEPARATOR + FORWARD_DATASET;
				final String previousCoordinateDataset = rootGroup + SEPARATOR + ( level - 1 ) + SEPARATOR + FORWARD_DATASET;

				if ( cmdLineArgs.overwrite || !n5.datasetExists( matrixDataset ) )
				{
					makeMatrices(
							sc,
							specs,
							sourceDim,
							matrixDim,
							range,
							sourceRoot,
							sourceDataset,
							root,
							matrixDataset );
				}

				if ( !cmdLineArgs.matricesOnly && ( cmdLineArgs.overwrite || !n5.datasetExists( coordinateDataset ) ) )
				{

					final ScaleAndTranslation previousToWorld = new ScaleAndTranslation(
							Arrays.stream( previousSteps ).asDoubleStream().toArray(),
							Arrays.stream( previousOffset ).map( o -> o - 0.5 ).toArray() );
					final ScaleAndTranslation currentToWorld = new ScaleAndTranslation(
							Arrays.stream( steps ).asDoubleStream().toArray(),
							Arrays.stream( offset ).map( o -> o - 0.5 ).toArray() );

					final ScaleAndTranslation previousToCurrent = currentToWorld.inverse().concatenate( previousToWorld );

					final Supplier< RandomAccessibleInterval< RealComposite< DoubleType > > > coordinateSupplier =
							level == 0 ? new ConstantCompositeDataSupplier( startingCoordinates ) : collapsedCoordinateSupplier( cmdLineArgs.root, previousCoordinateDataset );
					final Supplier< RandomAccessibleInterval< RandomAccessibleInterval< DoubleType > > > matrixSupplier = collapsedMatrixSupplier( cmdLineArgs.root, matrixDataset );

					final long numElements = Intervals.numElements( currentDim );
					final int stepSize = ( int ) Math.ceil( Math.sqrt( Math.max( numElements * 1.0 / sc.defaultParallelism(), 1 ) ) );
					List< long[] > inferenceBlocks = Grids.collectAllOffsets( currentDim, new int[] { 1, 1 } );

					JavaRDD< Interval > inferenceBlocksRdd = sc
							.parallelize( inferenceBlocks, Math.min( inferenceBlocks.size(), sc.defaultParallelism() ) )
							.map( min -> {
								long[] max = min.clone();
								// long[] max = new long[ min.length ];
								// Arrays.setAll( max, d -> Math.min( min[ d ] +
								// blockSize[ d ], currentDim[ d ] ) - 1 );
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
					Arrays.setAll( previousOffset, d -> offset[ d ] );

					Options options = Options.generateDefaultOptions();
					options.comparisonRange = range;
					options.shiftProportion = level < inferenceStepSize.length ? inferenceStepSize[ level ] : 0.5; // 1
																													// -
																													// (
																													// level
																													// <
																													// regularization.length
																													// ?
																													// regularization[
																													// level
																													// ]
																													// :
																													// lastRegularization
																													// /
																													// 2.0
																													// );
					options.nIterations = level < iterations.length ? iterations[ level ] : Math.max( lastIteration / 2, 1 );
					options.regularizationType = level == 0 ? RegularizationType.BORDER : RegularizationType.NONE;
					options.coordinateUpdateRegularizerWeight = level < regularization.length ? regularization[ level ] : 1 - 0.5 * ( 1 - lastRegularization );
					options.withReorder = false;
					lastRegularization = options.coordinateUpdateRegularizerWeight;
					lastIteration = options.nIterations;

					final JavaRDD< RandomAccessibleInterval< DoubleType > > newCoordinates =
							matricesAndCoordinates.map( new SparkInference.Inference<>(
									options,
									startingCoordinates.length ) );

					long[] coordinatesDims = append( currentDim, startingCoordinates.length );
					int[] coordinatesBlockSize = { 1, 1, startingCoordinates.length };
					// append( blockSize, startingCoordinates.length );
					n5.createDataset( coordinateDataset, coordinatesDims, coordinatesBlockSize, DataType.FLOAT64, new GzipCompression() );
					newCoordinates.foreach( coordinates -> {
						final long[] blockPosition = new long[ coordinates.numDimensions() ];
						Arrays.setAll( blockPosition, d -> coordinates.min( d ) / coordinatesBlockSize[ d ] );
						N5Utils.saveBlock( coordinates, N5Helpers.n5Writer( root ), coordinateDataset, blockPosition );
					} );
				}

				for ( int d = 0; d < steps.length; ++d )
				{
					steps[ d ] = Math.max( steps[ d ] / 2, 1 );
					radii[ d ] = Math.max( steps[ d ] / 2, 1 );
					offset[ d ] = 0;
				}

			}

		}

	}

	public static class ConstantCompositeDataSupplier implements Supplier< RandomAccessibleInterval< RealComposite< DoubleType > > >
	{

		private final double[] data;

		public ConstantCompositeDataSupplier( double[] data )
		{
			super();
			this.data = data;
		}

		@Override
		public RandomAccessibleInterval< RealComposite< DoubleType > > get()
		{
			return asComposite( data );
		}

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

	public static < T extends NativeType< T > & RealType< T > > void makeMatrices(
			final JavaSparkContext sc,
			final List< CorrelationBlockSpec > specs,
			final long[] sourceDim,
			final long[] matrixDim,
			final int range,
			final String sourceRoot,
			final String sourceDataset,
			final String root,
			final String matrixDataset ) throws Exception
	{

		N5Helpers.n5Writer( root ).createDataset(
				matrixDataset,
				matrixDim,
				new int[] { 1, 1, ( int ) matrixDim[ 2 ], ( int ) matrixDim[ 3 ] },
				DataTypeMatcher.toDataType( new DoubleType() ),
				new GzipCompression() );

		if ( specs.size() > sourceDim[ 2 ] )
		{
			MatricesFromN5ParallelizeOverXY.makeMatrices(
					sc,
					sc.parallelize( specs ).map( Arrays::asList ),
					range,
					new DataFromN5< DoubleType >( sourceRoot, sourceDataset ),
					root,
					matrixDataset,
					new DoubleType() );
		}
		else
		{
			MatricesFromN5ParallelizeOverZ.makeMatrices(
					sc,
					specs,
					range,
					new DataFromN5< DoubleType >( sourceRoot, sourceDataset ),
					root,
					matrixDataset,
					new DoubleType() );
		}
	}

	public static class DataFromN5< T extends NativeType< T > > implements DataSupplier< RandomAccessibleInterval< T > >
	{

		private final String root;

		private final String dataset;

		public DataFromN5( String root, String dataset )
		{
			super();
			this.root = root;
			this.dataset = dataset;
		}

		@Override
		public RandomAccessibleInterval< T > get() throws Exception
		{
			@SuppressWarnings( "unchecked" )
			RandomAccessibleInterval< T > data = ( RandomAccessibleInterval< T > ) N5Utils.open( N5Helpers.n5( root ), dataset );
			return data;
		}
	}

	public static class CollapsedSupplier< T extends RealType< T > > implements Supplier< RandomAccessibleInterval< RealComposite< T > > >
	{

		private final Supplier< RandomAccessibleInterval< T > > dataSupplier;

		public CollapsedSupplier( Supplier< RandomAccessibleInterval< T > > dataSupplier )
		{
			super();
			this.dataSupplier = dataSupplier;
		}

		@Override
		public RandomAccessibleInterval< RealComposite< T > > get()
		{
			return Views.collapseReal( dataSupplier.get() );
		}

	}

	public static class DoubleDataSupplier implements Supplier< RandomAccessibleInterval< DoubleType > >
	{

		private final String n5;

		private final String dataset;

		public DoubleDataSupplier( String n5, String dataset )
		{
			super();
			this.n5 = n5;
			this.dataset = dataset;
		}

		@Override
		public RandomAccessibleInterval< DoubleType > get()
		{
			try
			{
				return N5Utils.open( N5Helpers.n5( n5 ), dataset );
			}
			catch ( IOException e )
			{
				throw new RuntimeException( e );
			}
		}

	}

	public static Supplier< RandomAccessibleInterval< RealComposite< DoubleType > > > collapsedCoordinateSupplier( String n5, String dataset )
	{
		return new CollapsedSupplier<>( new DoubleDataSupplier( n5, dataset ) );
	}

	public static class CollapsedRandomAccessibleIntervalSupplier< T > implements Supplier< RandomAccessibleInterval< RandomAccessibleInterval< T > > >
	{

		private final Supplier< RandomAccessibleInterval< T > > dataSupplier;

		private final int numCollapsedDimensions;

		public CollapsedRandomAccessibleIntervalSupplier( Supplier< RandomAccessibleInterval< T > > dataSupplier, int numCollapsedDimensions )
		{
			super();
			this.dataSupplier = dataSupplier;
			this.numCollapsedDimensions = numCollapsedDimensions;
		}

		@Override
		public RandomAccessibleInterval< RandomAccessibleInterval< T > > get()
		{
			return new CollapsedRandomAccessibleInterval<>( dataSupplier.get(), numCollapsedDimensions );
		}
	}

	public static Supplier< RandomAccessibleInterval< RandomAccessibleInterval< DoubleType > > > collapsedMatrixSupplier(
			final String n5,
			final String dataset )
	{
		return new CollapsedRandomAccessibleIntervalSupplier<>( new DoubleDataSupplier( n5, dataset ), 2 );
	}

	public static long[] gridDims( final long[] step, final long[] dim, final boolean isShiftedByHalfStep )
	{
		long[] gridDims = new long[ dim.length ];
		for ( int d = 0; d < dim.length; ++d )
		{
			long s = step[ d ];
			gridDims[ d ] = ( long ) Math.ceil( ( dim[ d ] + ( isShiftedByHalfStep ? 0.5 * s : 0 ) ) * 1.0 / s );
		}
		return gridDims;
	}

}
