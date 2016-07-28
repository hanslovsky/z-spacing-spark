package org.janelia.thickness.similarity;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.process.FloatProcessor;
import mpicbg.ij.integral.BlockPMCC;
import org.apache.commons.lang3.StringUtils;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.janelia.similarities.NCC;
import org.janelia.thickness.utility.Utility;
import scala.Tuple2;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

/**
 * @author Philipp Hanslovsky &lt;hanslovskyp@janelia.hhmi.org&gt;
 */
public class TolerantNCC {

    private final JavaPairRDD<Tuple2< Integer, Integer >, Tuple2<FloatProcessor, FloatProcessor>> overcompleteSections;

    public TolerantNCC(JavaPairRDD<Tuple2< Integer, Integer >, Tuple2<FloatProcessor, FloatProcessor>> overcompleteSections) {
        this.overcompleteSections = overcompleteSections;
    }

    public void ensurePersistence()
    {
        overcompleteSections.cache();
        overcompleteSections.count();
    }

    public JavaPairRDD<Tuple2<Integer, Integer>, FloatProcessor> calculate(
            JavaSparkContext sc,
            final int[] blockRadius,
            final int[] stepSize,
            final int[] correlationBlockRadius,
            final int[] maxOffset,
            final int size,
            final int range
    )
    {

        JavaPairRDD<Tuple2<Integer, Integer>, Tuple2<FloatProcessor, FloatProcessor>> sections = overcompleteSections
                .filter(new MatrixGenerationFromImagePairs.SelectInRange<>(range))
                ;

        System.out.println( "sections: " + sections.count() );

        JavaPairRDD<Tuple2<Integer, Integer>, Tuple2<FloatProcessor, FloatProcessor>> maxProjections = sections
                .mapToPair(new FPToSimilarities<>(
                        maxOffset,
                        correlationBlockRadius
                ))
                .cache();

        System.out.println( "maxOffset=" + Arrays.toString( maxOffset ) );
        System.out.println( "correlationBlockRadius=" + Arrays.toString( correlationBlockRadius ) );

        System.out.println( "maxProjections: " + maxProjections.count() );

        JavaPairRDD<Tuple2<Integer, Integer>, HashMap<Tuple2<Integer, Integer>, Double>> averages = maxProjections
                .mapToPair(new AverageBlocks<Tuple2<Integer, Integer>>(blockRadius, stepSize))
                .cache()
                ;

        System.out.println( "averages: " + averages.count() );

        JavaPairRDD<Tuple2<Integer, Integer>, Tuple2<Tuple2<Integer, Integer>, Double>> flatAverages = averages
                .flatMapToPair(
                        new Utility.FlatmapMap<Tuple2<Integer, Integer>, Tuple2<Integer, Integer>, Double, HashMap<Tuple2<Integer, Integer>, Double>>()
                )
                .cache();

        System.out.println( "flatAverages: " + flatAverages.count() );


        JavaPairRDD<Tuple2<Integer, Integer>, HashMap<Tuple2<Integer, Integer>, Double>> averagesIndexedByXYTuples = flatAverages
                .mapToPair(new Utility.Swap<Tuple2<Integer, Integer>, Tuple2<Integer, Integer>, Double>())
                .mapToPair( new Utility.ValueAsMap<Tuple2<Integer,Integer>,Tuple2<Integer,Integer>,Double>())
                .cache()
                ;

        averagesIndexedByXYTuples.count();

        JavaPairRDD<Tuple2<Integer, Integer>, FloatProcessor> matrices = averagesIndexedByXYTuples
                .reduceByKey(new Utility.ReduceMapsByUnion<Tuple2<Integer, Integer>, Double, HashMap<Tuple2<Integer, Integer>, Double>>())
                .mapToPair(new MatrixGenerationFromImagePairs.MapToFloatProcessor(size, 0))
                .cache()
                ;

        matrices.count();

        return matrices;


    }

    public static class FPToSimilarities<K>
    implements PairFunction<Tuple2<K,Tuple2<FloatProcessor,FloatProcessor>>, K, Tuple2< FloatProcessor, FloatProcessor > > {

        /**
		 * 
		 */
		private static final long serialVersionUID = 572711725174439812L;
		private final int[] maxOffsets;
        private final int[] blockRadius;

        public FPToSimilarities(int[] maxOffsets, int[] blockRadius) {
            this.maxOffsets = maxOffsets;
            this.blockRadius = blockRadius;
        }

        @SuppressWarnings("rawtypes")
		@Override
        public Tuple2<K, Tuple2< FloatProcessor, FloatProcessor > > 
        call(Tuple2<K, Tuple2<FloatProcessor,FloatProcessor>> t) throws Exception {
            FloatProcessor fixed = t._2()._1();
            FloatProcessor moving = t._2()._2();

            K k = t._1();

            int x =0, y = 0;
            if ( k instanceof Tuple2 )
            {
                if ( ((Tuple2) k)._1() instanceof Integer )
                    x = ((Integer) ((Tuple2) k)._1()).intValue();
                if ( ((Tuple2) k)._2() instanceof Integer )
                	y = ((Integer) ((Tuple2) k)._2()).intValue();
            }

            Tuple2<FloatProcessor, FloatProcessor> ccs = tolerantNCC(
                    (FloatProcessor) fixed.duplicate(),
                    (FloatProcessor) moving.duplicate(),
                    maxOffsets,
                    blockRadius,
                    x ,
                    y );
            return Utility.tuple2( t._1(), Utility.tuple2( ccs._1(), ccs._2() ) );
        }
    }

    public static class AverageBlocks<K>
    implements PairFunction<Tuple2<K,Tuple2<FloatProcessor,FloatProcessor>>,K,HashMap<Tuple2<Integer,Integer>,Double>>
    {

        /**
		 * 
		 */
		private static final long serialVersionUID = 6067709319074903557L;
		private final int[] blockRadius;
        private final int[] stepSize;

        public AverageBlocks(int[] blockRadius, int[] stepSize) {
            this.blockRadius = blockRadius;
            this.stepSize = stepSize;
        }

        @Override
        public Tuple2<K, HashMap<Tuple2<Integer, Integer>, Double>> call(Tuple2<K, Tuple2<FloatProcessor,FloatProcessor>> t) throws Exception {
            return Utility.tuple2( t._1(), average( t._2()._1(), t._2()._2(), blockRadius, stepSize ) );
        }
    }

    public static FloatProcessor generateMask( FloatProcessor img, HashSet< Float > values )
    {
        FloatProcessor mask = new FloatProcessor(img.getWidth(), img.getHeight());
        float[] i = (float[]) img.getPixels();
        float[] m = (float[]) mask.getPixels();
        for( int k = 0; k < i.length; ++k )
            m[k] = values.contains( i[k] ) ? 0.0f : 1.0f;
        return mask;
    }

    public static FloatProcessor generateMask( FloatProcessor fp )
    {
        FloatProcessor weights = new FloatProcessor(fp.getWidth(), fp.getHeight());
        float[] weightsPixels = (float[]) weights.getPixels();
        float[] fpPixels = (float[]) fp.getPixels();
        for( int i = 0; i < fpPixels.length; ++i )
        {
            boolean isNaN = Float.isNaN(fpPixels[i]);
            // ignore NaNs (leave them 0.0f in mask)
            // still need to replace NaNs in image because 0.0 * NaN = NaN
            if ( isNaN )
                fpPixels[i] = 0.0f;
            else
                weightsPixels[i] = 1.0f;
        }
        return weights;
    }

    public static Tuple2< FloatProcessor, FloatProcessor > tolerantNCC(
            FloatProcessor fixed,
            FloatProcessor moving,
            final int[] maxOffsets,
            final int[] blockRadiusInput,
            int z1,
            int z2
    ) {
        int width = moving.getWidth();
        int height = moving.getHeight();

        int[] blockRadius = new int[]{
                Math.min(blockRadiusInput[0], width - 1 ),
                Math.min(blockRadiusInput[1], height - 1 )
        };

        FloatProcessor maxCorrelations = new FloatProcessor(width, height);
//        maxCorrelations.add( Double.NaN );

        final int xStart = -1 * maxOffsets[0];
        final int yStart = -1 * maxOffsets[1];

        final int xStop = 1 * maxOffsets[0]; // inclusive
        final int yStop = 1 * maxOffsets[1]; // inclusive
        
        BlockPMCC pmcc = new BlockPMCC( fixed, moving );
        FloatProcessor tp = pmcc.getTargetProcessor();

        FloatProcessor weights = new FloatProcessor( width, height );

        for( int yOff = yStart; yOff <= yStop; ++yOff )
        {
            for ( int xOff = xStart; xOff <= xStop; ++ xOff )
            {
                pmcc.setOffset( xOff, yOff );
                pmcc.r( blockRadius[0], blockRadius[1] );

                for ( int y = 0; y < height; ++y )
                {
                    for ( int x = 0; x < width; ++x )
                    {
                        // if full correlation block is not contained within image, ignore it!
                        if(
                                x + xOff - blockRadius[0] < 0 || y + yOff - blockRadius[1] < 0 ||
                                x + xOff + blockRadius[0] > width || y + yOff + blockRadius[1] > height )
                            continue;

                        // if full correlation block is not contained within moving image, ignore it!
                        if(
                                x - blockRadius[0] < 0 || y - blockRadius[1] < 0 ||
                                x + blockRadius[0] > width || y + blockRadius[1] > height )
                            continue;

//                        if ( maxOffsets[0] > 0 )
//                        	System.out.println( x + " -- " + y );
                        float val = tp.getf(x, y);
                        if ( !Double.isNaN( val ) && val > maxCorrelations.getf( x, y ) ) {
                            maxCorrelations.setf( x, y, val );
                        }
                    }
                }

            }
        }
        
        for ( int y = 0; y < height; ++y )
        {
            for ( int x = 0; x < width; ++x )
            {
                float weight = (
                        (x < blockRadius[0]) || (x >= (width - blockRadius[0])) ||
                        (y < blockRadius[1]) || (y >= (height - blockRadius[1]))
                ) ?
                Float.NaN : 1.0f;
//                if ( !Float.isNaN( weight ) )
//                	System.out.println( x + " ~~ " + y + " ~~ " + Arrays.toString( blockRadius ) );
                weights.setf( x, y, weight );
            }
        }

        return Utility.tuple2( maxCorrelations, weights );
    }

    public static HashMap<Tuple2< Integer, Integer >, Double> average(
            FloatProcessor maxCorrelations,
            FloatProcessor weights,
            int[] blockSize,
            int[] stepSize
    )
    {
        HashMap<Tuple2<Integer, Integer>, Double> hm = new HashMap<Tuple2<Integer, Integer>, Double>();
        int width = maxCorrelations.getWidth();
        int height = maxCorrelations.getHeight();

        int maxX = width - 1;
        int maxY = height - 1;

        for( int  y = blockSize[1], yIndex = 0; y < height; y += stepSize[1], ++yIndex )
        {
            int lowerY = y - blockSize[1];
            int upperY = Math.min(y + blockSize[1], maxY);
            for ( int x = blockSize[0], xIndex = 0; x < width; x += stepSize[0], ++xIndex )
            {
                int lowerX = x - blockSize[0];
                int upperX = Math.min(x + blockSize[0], maxX);
                double sum = 0.0;
                double weightSum = 0.0;
                for ( int yLocal = lowerY; yLocal <= upperY; ++yLocal )
                {
                    for ( int xLocal = lowerX; xLocal <= upperX; ++xLocal )
                    {
                        double weight = weights.getf(xLocal, yLocal);
                        double corr   = maxCorrelations.getf( xLocal, yLocal );
                        if ( Double.isNaN( weight ) || Double.isNaN( corr ))
                            continue;
                        sum += weight*corr;
                        weightSum += weight;
                    }
                }
//                if ( x == 200 && y == 120 )
//                	System.out.println( weightSum );
                if ( weightSum > 0.0 ) {
//                	System.out.println( x + " ~~ " + y + Arrays.toString( blockSize ) );
                	sum /= weightSum; // ( upperY - lowerY ) * ( upperX - lowerX );
                	hm.put( Utility.tuple2( xIndex, yIndex ), sum );
                }
            }
        }
        return hm;
    }

    public static void main(String[] args) {

       

    }

}
