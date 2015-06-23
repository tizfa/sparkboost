/*
 *
 * ****************
 * Copyright 2015 Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ******************
 */

package it.tizianofagni.sparkboost;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;

/**
 * A Spark implementation of MP-Boost algorithm, an improved variant of the
 * well known AdaBoost.MH boosting algorithm.<br/><br/>
 * The original article describing the algorithm can be found at
 * http://link.springer.com/chapter/10.1007/11880561_1
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class MpBoostLearner {

    static class DMPartialResult implements Serializable {
        private final int docID;
        private double[] labelsRes;

        DMPartialResult(int docID, double[] labelsRes) {
            this.docID = docID;
            this.labelsRes = labelsRes;
        }

        public int getDocID() {
            return docID;
        }

        public double[] getLabelsRes() {
            return labelsRes;
        }

        public void setLabelsRes(double[] labelsRes) {
            this.labelsRes = labelsRes;
        }
    }

    private static class WeakHypothesisResults implements Serializable {
        private final int[] pivot;
        private final double[] c0;
        private final double[] c1;
        private final double[] z_s;

        public WeakHypothesisResults(int[] pivot, double[] c0, double[] c1, double[] z_s) {
            this.pivot = pivot;
            this.c0 = c0;
            this.c1 = c1;
            this.z_s = z_s;
        }

        public int[] getPivot() {
            return pivot;
        }

        public double[] getC0() {
            return c0;
        }

        public double[] getC1() {
            return c1;
        }

        public double[] getZ_s() {
            return z_s;
        }
    }


    /**
     * The number of iterations.
     */
    private int numIterations;

    private final JavaSparkContext sc;

    private int parallelismDegree;

    public MpBoostLearner(JavaSparkContext sc) {
        if (sc == null)
            throw new NullPointerException("The SparkContext is 'null'");
        this.sc = sc;
        this.parallelismDegree = 8;
        this.numIterations = 200;
    }

    public int getParallelismDegree() {
        return parallelismDegree;
    }

    public void setParallelismDegree(int parallelismDegree) {
        this.parallelismDegree = parallelismDegree;
    }

    public MPBoostClassifier buildModel(String libSvmFile) {
        System.out.println("Load initial data and generating all necessary internal data representations...");
        JavaRDD<MultilabelPoint> docs = DataUtils.loadLibSvmFileFormatData(sc, libSvmFile).repartition(getParallelismDegree()).persist(StorageLevel.MEMORY_ONLY_SER());
        int numDocs = DataUtils.getNumDocuments(docs);
        int numLabels = DataUtils.getNumLabels(docs);
        JavaRDD<DataUtils.LabelDocuments> labelDocuments = DataUtils.getLabelDocuments(docs).repartition(getParallelismDegree()).persist(StorageLevel.MEMORY_ONLY_SER());
        JavaRDD<DataUtils.FeatureDocuments> featureDocuments = DataUtils.getFeatureDocuments(docs).repartition(getParallelismDegree()).persist(StorageLevel.MEMORY_ONLY_SER());
        System.out.println("Ok, done!");

        WeakHypothesis[] computedWH = new WeakHypothesis[numIterations];
        double[][] localDM = initDistributionMatrix(numLabels, numDocs);
        for (int i = 0; i < numIterations; i++) {

            // Generate new weak hypothesis.
            WeakHypothesis localWH = learnWeakHypothesis(localDM, labelDocuments, featureDocuments);

            // Update distribution matrix with the new hypothesis.
            updateDistributionMatrix(sc, docs, localDM, localWH);

            // Save current generated weak hypothesis.
            computedWH[i] = localWH;

            if (((i + 1) % 5) == 0)
                System.out.print("" + (i + 1));
            else
                System.out.print(".");

            if (((i + 1) % 50) == 0)
                System.out.println("");
        }

        System.out.println("Model built!");

        return new MPBoostClassifier(computedWH);
    }


    protected void updateDistributionMatrix(JavaSparkContext sc, JavaRDD<MultilabelPoint> docs, double[][] localDM, WeakHypothesis localWH) {
        Broadcast<WeakHypothesis> distWH = sc.broadcast(localWH);
        Broadcast<double[][]> distDM = sc.broadcast(localDM);
        JavaRDD<DMPartialResult> partialResults = docs.map(doc -> {
            int[] validFeatures = doc.getFeatures().indices();
            HashMap<Integer, Integer> dictFeatures = new HashMap<>();
            for (int featID : validFeatures)
                dictFeatures.put(featID, featID);
            HashMap<Integer, Integer> dictLabels = new HashMap<>();
            for (int idx = 0; idx < doc.getLabels().length; idx++)
                dictLabels.put(doc.getLabels()[idx], doc.getLabels()[idx]);

            double[][] dm = distDM.getValue();
            WeakHypothesis wh = distWH.getValue();
            double[] labelsRes = new double[dm.length];
            for (int labelID = 0; labelID < dm.length; labelID++) {
                float catValue = 1;
                if (dictLabels.containsKey(labelID)) {
                    catValue = -1;
                }

                // Compute the weak hypothesis value.
                double value = 0;
                WeakHypothesis.WeakHypothesisData v = wh.getLabelData(labelID);
                int pivot = v.getFeatureID();
                if (dictFeatures.containsKey(pivot))
                    value = v.getC1();
                else
                    value = v.getC0();


                double partialRes = dm[labelID][doc.getDocID()] * Math.exp(catValue * value);
                labelsRes[labelID] = partialRes;
            }

            return new DMPartialResult(doc.getDocID(), labelsRes);
        });

        Iterator<DMPartialResult> itResults = partialResults.toLocalIterator();
        // Update partial results.
        while (itResults.hasNext()) {
            DMPartialResult r = itResults.next();
            for (int labelID = 0; labelID < localDM.length; labelID++) {
                localDM[labelID][r.docID] = r.labelsRes[labelID];
            }
        }

        // Normalize all values per label.
        for (int labelID = 0; labelID < localDM.length; labelID++){
            double normalization = 0;
            for (int docID = 0; docID < localDM[0].length; docID++) {
                normalization += localDM[labelID][docID];
            }
            for (int docID = 0; docID < localDM[0].length; docID++) {
                localDM[labelID][docID] = localDM[labelID][docID] / normalization;
            }
        }
    }


    protected double[][] initDistributionMatrix(int numLabels, int numDocs) {
        double[][] dist = new double[numLabels][numDocs];

        // Initialize matrix with uniform distribution.
        float uniformValue = 1 / ((float) numDocs * numLabels);
        for (int label = 0; label < dist.length; label++) {
            for (int doc = 0; doc < dist[0].length; doc++) {
                dist[label][doc] = uniformValue;
            }
        }
        return dist;
    }



    protected WeakHypothesis learnWeakHypothesis(double[][] localDM, JavaRDD<DataUtils.LabelDocuments> labelDocuments, JavaRDD<DataUtils.FeatureDocuments> featureDocuments) {
        int labelsSize = localDM.length;
        int docsSize = localDM[0].length;

        // Examples positive for a given label.
        double[] local_weight_b1 = new double[labelsSize];

        // Examples negative for a given label
        double[] local_weight_bminus_1 = new double[labelsSize];


        // Initialize structures.
        for (int pos = 0; pos < labelsSize; pos++) {
            local_weight_b1[pos] = 0;
            local_weight_bminus_1[pos] = 0;
        }

        Iterator<DataUtils.LabelDocuments> itlabels = labelDocuments.toLocalIterator();
        while (itlabels.hasNext()){
            DataUtils.LabelDocuments la = itlabels.next();
            int labelID = la.getLabelID();
            assert(labelID != -1);
            for (int idx = 0; idx < la.getDocuments().length; idx++) {
                int docID = la.getDocuments()[idx];
                assert(docID != -1);
                double distValue = localDM[labelID][docID];
                local_weight_b1[labelID] += distValue;
            }
        }


        // Compute global weight for categories.
        for (int labelID = 0; labelID < labelsSize; labelID++) {
            double global = 0;

            // Iterate over all distribution matrix.
            for (int docID = 0; docID < docsSize; docID++) {
                double distValue = localDM[labelID][docID];
                global += distValue;
            }

            local_weight_bminus_1[labelID] = global - local_weight_b1[labelID];
        }


        Broadcast<double[][]> distDM = sc.broadcast(localDM);
        Broadcast<double[]> weight_b1 = sc.broadcast(local_weight_b1);
        Broadcast<double[]> weight_bminus_1 = sc.broadcast(local_weight_bminus_1);

        // Process all features.
        WeakHypothesisResults res = featureDocuments.map(feat -> {
            double[][] dm = distDM.getValue();
            double epsilon = 1.0 / (double) (dm.length * dm[0].length);
            int numLabels = dm.length;
            double[] weight_b1_x0 = new double[numLabels];
            double[] weight_b1_x1 = new double[numLabels];
            double[] weight_bminus_1_x0 = new double[numLabels];
            double[] weight_bminus_1_x1 = new double[numLabels];
            double[] computedC0 = new double[numLabels];
            double[] computedC1 = new double[numLabels];
            double[] computedZs = new double[numLabels];
            int[] pivot = new int[numLabels];
            int realFeatID = feat.getFeatureID();
            // Initialize structures.
            for (int pos = 0; pos < numLabels; pos++) {
                weight_b1_x0[pos] = 0;
                weight_b1_x1[pos] = 0;
                weight_bminus_1_x0[pos] = 0;
                weight_bminus_1_x1[pos] = 0;
                pivot[pos] = realFeatID;
            }


            for (int docIdx = 0; docIdx < feat.getDocuments().length; docIdx++) {
                int docID = feat.getDocuments()[docIdx];
                int[] labels = feat.getLabels()[docIdx];
                HashMap<Integer, Integer> catDict = new HashMap<Integer, Integer>();
                for (int labelIdx = 0; labelIdx < labels.length; labelIdx++) {
                    int currentCatID = labels[labelIdx];
                    double distValue = dm[currentCatID][docID];
                    // Feature and category compare together.
                    weight_b1_x1[currentCatID] += distValue;
                    catDict.put(currentCatID, currentCatID);
                }
                for (int currentCatID = 0; currentCatID < numLabels; currentCatID++) {
                    if (catDict.containsKey(currentCatID))
                        continue;
                    double distValue = dm[currentCatID][docID];
                    // Feature compare on document and category not.
                    weight_bminus_1_x1[currentCatID] += distValue;
                }
            }

            // Compute the remaining values.
            for (int catID = 0; catID < numLabels; catID++) {
                double v = weight_b1.getValue()[catID] - weight_b1_x1[catID];
                if (v < 0)
                    v = 0;

                weight_b1_x0[catID] = v;

                v = weight_bminus_1.getValue()[catID] - weight_bminus_1_x1[catID];
                // Adjust round errors.
                if (v < 0)
                    v = 0;
                weight_bminus_1_x0[catID] = v;
            }

            // Compute current Z_s.
            for (int catID = 0; catID < numLabels; catID++) {
                assert (weight_b1_x0[catID] >= 0);
                assert (weight_bminus_1_x0[catID] >= 0);
                assert (weight_b1_x1[catID] >= 0);
                assert (weight_bminus_1_x1[catID] >= 0);

                double Z_s = 0;
                double first = Math.sqrt(weight_b1_x0[catID]
                        * weight_bminus_1_x0[catID]);
                double second = Math.sqrt(weight_b1_x1[catID]
                        * weight_bminus_1_x1[catID]);
                Z_s = (first + second);
                Z_s = 2 * Z_s;
                double c0 = Math.log((weight_b1_x0[catID] + epsilon)
                        / (weight_bminus_1_x0[catID] + epsilon)) / 2.0;
                double c1 = Math.log((weight_b1_x1[catID] + epsilon)
                        / (weight_bminus_1_x1[catID] + epsilon)) / 2.0;
                computedC0[catID] = c0;
                computedC1[catID] = c1;
                computedZs[catID] = Z_s;
            }

            return new WeakHypothesisResults(pivot, computedC0, computedC1, computedZs);
        }).reduce((ph1, ph2)->{
            int[] pivot = new int[ph1.getPivot().length];
            double[] c0 = new double[ph1.getPivot().length];
            double[] c1 = new double[ph1.getPivot().length];
            double[] z_s = new double[ph1.getPivot().length];
            for (int i = 0; i < ph1.getPivot().length; i++) {
                if (ph1.getZ_s()[i] < ph2.getZ_s()[i]) {
                    pivot[i] = ph1.getPivot()[i];
                    c0[i] = ph1.getC0()[i];
                    c1[i] = ph1.getC1()[i];
                    z_s[i] = ph1.getZ_s()[i];
                } else {
                    pivot[i] = ph2.getPivot()[i];
                    c0[i] = ph2.getC0()[i];
                    c1[i] = ph2.getC1()[i];
                    z_s[i] = ph2.getZ_s()[i];
                }
            }
            return new WeakHypothesisResults(pivot,c0, c1, z_s);
        });

        WeakHypothesis wh = new WeakHypothesis(labelsSize);
        for (int i = 0; i < labelsSize; i++) {
            wh.setLabelData(i, new WeakHypothesis.WeakHypothesisData(i, res.getPivot()[i], res.getC0()[i], res.getC1()[i]));
        }
        return wh;
    }

    public int getNumIterations() {
        return numIterations;
    }

    public void setNumIterations(int numIterations) {
        this.numIterations = numIterations;
    }
}
