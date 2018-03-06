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
import org.janelia.thickness.similarity.MatricesFromN5ParallelizeOverZ;
import org.janelia.thickness.utility.DataTypeMatcher;
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

	public static final String MATRICES_DATASET = "matrices";

	public static String SEPARATOR = "/";

	public static String FORWARD_DATASET = "forward";

	public static String BACKWARD_DATASET = "backward";

	public static String RANGE_ATTRIBUTE = "range";

	public static String INFERENCE_ITERATIONS_ATTRIBUTE = "inferenceIterations";

	public static String REGULARIZATION_ATTRIBUTE = "regularization";

	public static String SOURCE_ROOT_ATTRIBUTE = "sourceRoot";

	public static String SOURCE_DATASET_ATTRIBUTE = "sourceDataset";

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
				names = "--generate-matrices",
				required = false,
				description = "If specified, generate matrices even when dataset exists." )
		private boolean generateMatrices;

		@Option(
				names = { "--matrices-only", "-m" },
				required = false,
				description = "Generamte matrices only, do not run inference." )
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

			final long[] steps = new long[ 2 ];
			final long[] radii = new long[ 2 ];
			final long[] imgDim = LongStream.of( sourceDim ).limit( 2 ).toArray();
			final long[] imgMax = LongStream.of( imgDim ).map( d -> d - 1 ).toArray();

			for ( int d = 0; d < steps.length; ++d )
			{
				steps[ d ] = imgDim[ d ];
				radii[ d ] = steps[ d ] / 2;
			}

			long[] previousSteps = imgDim.clone();
			long[] previousRadii = { 0, 0 };

			for ( int level = 0; level < ranges.length; ++level )
			{

				if ( Arrays.stream( radii ).filter( r -> r > 1 ).count() == 0 )
				{
					break;
				}

				final int range = ranges[ level ];

				final int[] blockSize = Arrays.stream( steps ).mapToInt( l -> ( int ) l ).toArray();

				final List< long[] > positions = Grids.collectAllOffsets( imgDim, blockSize );
				final long[] extent = Arrays.stream( radii ).map( r -> 2 * r + 1 ).toArray();

				final List< CorrelationBlockSpec > specs = positions
						.stream()
						.map( p -> CorrelationBlockSpec.asSpec( p, blockSize, extent, imgMax, true ) )
						.collect( Collectors.toList() );

				final long[] maxPosition = new long[] { 0, 0 };
				positions
						.stream()
						.map( long[]::clone )
						.map( a -> divide( a, steps, a ) )
						.forEach( p -> Arrays.setAll( maxPosition, d -> Math.max( maxPosition[ d ], p[ d ] ) ) );

				final long[] currentDim = Arrays.stream( maxPosition ).map( p -> p + 1 ).toArray();
				final long[] matrixDim = { currentDim[ 0 ], currentDim[ 1 ], 2 * range + 1, sourceDim[ 2 ] };

				final String matrixDataset = rootGroup + SEPARATOR + level + SEPARATOR + MATRICES_DATASET;
				final String coordinateDataset = rootGroup + SEPARATOR + level + SEPARATOR + FORWARD_DATASET;
				final String previousCoordinateDataset = rootGroup + SEPARATOR + ( level - 1 ) + SEPARATOR + FORWARD_DATASET;

				if ( cmdLineArgs.generateMatrices || !n5.datasetExists( matrixDataset ) )
				{
					makeMatrices( sc, specs, sourceDim, matrixDim, range, sourceRoot, sourceDataset, root, matrixDataset );
				}

				if ( !cmdLineArgs.matricesOnly )
				{

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
					List< long[] > inferenceBlocks = Grids.collectAllOffsets( currentDim, new int[] { 1, 1 } );

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

					Options options = Options.generateDefaultOptions();
					options.comparisonRange = range;
					options.shiftProportion = 1 - ( level < regularization.length ? regularization[ level ] : lastRegularization / 2.0 );
					options.nIterations = level < iterations.length ? iterations[ level ] : Math.max( lastIteration / 2, 1 );
					options.regularizationType = level == 0 ? RegularizationType.NONE : RegularizationType.NONE;
					options.withReorder = false;
					lastRegularization = regularization[ level ];
					lastIteration = options.nIterations;

					final JavaRDD< RandomAccessibleInterval< DoubleType > > newCoordinates =
							matricesAndCoordinates.map( new SparkInference.Inference<>(
									options,
									startingCoordinates.length ) );

					long[] coordinatesDims = append( currentDim, startingCoordinates.length );
					int[] coordinatesBlockSize = { 1, 1, startingCoordinates.length };// append( blockSize, startingCoordinates.length );
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
				}

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
					() -> ( RandomAccessibleInterval< T > ) N5Utils.open( N5Helpers.n5( sourceRoot ), sourceDataset ),
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
					() -> ( RandomAccessibleInterval< T > ) N5Utils.open( N5Helpers.n5( sourceRoot ), sourceDataset ),
					root,
					matrixDataset,
					new DoubleType() );
		}
	}

}
