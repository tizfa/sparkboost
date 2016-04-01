/*
 *
 * ****************
 * This file is part of nlp4sparkml software package (https://github.com/tizfa/nlp4sparkml).
 *
 * Copyright 2016 Tiziano Fagni (tiziano.fagni@isti.cnr.it)
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

package it.tizianofagni.sparkboost;/*
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

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 * A boosting classifier built with {@link AdaBoostMHLearner} or
 * {@link MpBoostLearner} classes.
 *
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class BoostClassifier implements Serializable {

    /**
     * The set of weak hypothesis of the model.
     */
    private final WeakHypothesis[] whs;

    public BoostClassifier(WeakHypothesis[] whs) {
        if (whs == null)
            throw new NullPointerException("The set of generated WHs is 'null'");
        this.whs = whs;
    }


    /**
     * Classify the set of input documents. Please be sure that that the features IDs and labels IDs
     * used in this document set match those used during training phase.
     * <p>
     * IMPORTANT NOTE: the classification results are collected on the driver process so the results
     * are stored on RAM memory. If you classify a lot of documents, be sure to give enough memory to Spark
     * driver process or just split your documents collection in small partitions and classify one partition at
     * time.
     *
     * @param sc                The Spark context.
     * @param docs              The set of documents to classify.
     * @param parallelismDegree The number of workers used to classify documents.
     * @return The results of the classification process.
     */
    public ClassificationResults classify(JavaSparkContext sc, JavaRDD<MultilabelPoint> docs, int parallelismDegree) {
        if (sc == null)
            throw new NullPointerException("The Spark context is 'null'");
        if (docs == null)
            throw new NullPointerException("The set of documents to classify is 'null'");

        Logging.l().info("Load initial data and generating all necessary internal data representations...");
        docs = docs.repartition(parallelismDegree);
        docs = docs.cache();
        Logging.l().info("done!");
        Logging.l().info("Classifying documents...");
        Broadcast<WeakHypothesis[]> whsBr = sc.broadcast(whs);
        ClassificationResults classifications = docs.map(doc -> {
            WeakHypothesis[] whs = whsBr.getValue();
            int[] indices = doc.getFeatures().indices();
            HashMap<Integer, Integer> dict = new HashMap<Integer, Integer>();
            for (int idx = 0; idx < indices.length; idx++) {
                dict.put(indices[idx], indices[idx]);
            }
            double[] scores = new double[whs[0].getNumLabels()];
            for (int i = 0; i < whs.length; i++) {
                WeakHypothesis wh = whs[i];
                for (int labelID = 0; labelID < wh.getNumLabels(); labelID++) {
                    int featureID = wh.getLabelData(labelID).getFeatureID();
                    if (dict.containsKey(featureID)) {
                        scores[labelID] += wh.getLabelData(labelID).getC1();
                    } else {
                        scores[labelID] += wh.getLabelData(labelID).getC0();
                    }
                }
            }

            PreliminaryDocumentClassification dc = new PreliminaryDocumentClassification(doc.getPointID(), doc.getLabels(), scores);
            HashSet<Integer> goldLabels = new HashSet<>();
            for (int labelID : dc.getGoldLabels())
                goldLabels.add(labelID);
            int docID = dc.getDocumentID();
            ArrayList<Integer> labelAssigned = new ArrayList<>();
            ArrayList<Double> labelScores = new ArrayList<>();
            int tp = 0, fp = 0, fn = 0, tn = 0;
            for (int labelID = 0; labelID < dc.getScores().length; labelID++) {
                double score = dc.getScores()[labelID];
                boolean hasGoldLabel = goldLabels.contains(labelID);
                if (score > 0 && hasGoldLabel) {
                    tp++;
                    labelAssigned.add(labelID);
                    labelScores.add(score);
                } else if (score > 0 && !hasGoldLabel) {
                    fp++;
                    labelAssigned.add(labelID);
                    labelScores.add(score);
                } else if (score < 0 && hasGoldLabel) {
                    fn++;
                } else {
                    tn++;
                }
            }
            ContingencyTable ct = new ContingencyTable(tp, tn, fp, fn);
            int[][] labelsRet = new int[1][];
            labelsRet[0] = labelAssigned.stream().mapToInt(i -> i).toArray();
            double[][] scoresRet = new double[1][];
            scoresRet[0] = labelScores.stream().mapToDouble(i -> i).toArray();
            int[][] goldLabelsRet = new int[1][];
            goldLabelsRet[0] = dc.getGoldLabels();
            int[] documents = new int[]{docID};

            return new ClassificationResults(1, documents, labelsRet, scoresRet, goldLabelsRet, ct);


        }).reduce((cl1, cl2) -> {
            int numDocuments = cl1.getNumDocs() + cl2.getNumDocs();
            ContingencyTable ct = new ContingencyTable(cl1.getCt().tp() + cl2.getCt().tp(),
                    cl1.getCt().tn() + cl2.getCt().tn(), cl1.getCt().fp() + cl2.getCt().fp(),
                    cl1.getCt().fn() + cl2.getCt().fn());
            int[] documents = new int[numDocuments];

            // Copy document IDs.
            for (int i = 0; i < cl1.getNumDocs(); i++)
                documents[i] = cl1.getDocuments()[i];
            for (int i = cl1.getNumDocs(); i < numDocuments; i++)
                documents[i] = cl2.getDocuments()[i - cl1.getNumDocs()];

            // Copy label IDs.
            int[][] labels = new int[numDocuments][];
            for (int i = 0; i < cl1.getNumDocs(); i++)
                labels[i] = cl1.getLabels()[i];
            for (int i = cl1.getNumDocs(); i < numDocuments; i++)
                labels[i] = cl2.getLabels()[i - cl1.getNumDocs()];

            // Copy score IDs.
            double[][] scores = new double[numDocuments][];
            for (int i = 0; i < cl1.getNumDocs(); i++)
                scores[i] = cl1.getScores()[i];
            for (int i = cl1.getNumDocs(); i < numDocuments; i++)
                scores[i] = cl2.getScores()[i - cl1.getNumDocs()];

            // Copy gold label IDs.
            int[][] goldLabels = new int[numDocuments][];
            for (int i = 0; i < cl1.getNumDocs(); i++)
                goldLabels[i] = cl1.getGoldLabels()[i];
            for (int i = cl1.getNumDocs(); i < numDocuments; i++)
                goldLabels[i] = cl2.getGoldLabels()[i - cl1.getNumDocs()];

            return new ClassificationResults(numDocuments, documents, labels, scores, goldLabels, ct);
        });

        Logging.l().info("done.");
        return classifications;
    }


    /**
     * Classify the set of input documents contained in the specified file. The file must be in LibSvm data format. Each document
     * in the input dataset will get a document ID corresponding at the original row index of the document in the dataset file.
     *
     * @param sc
     * @param libSvmFile        The input file containing documents to classify.
     * @param parallelismDegree The number of workers used to classify documents.
     * @param labels0Based      True if the label indexes specified in the input file are 0-based (i.e. the first label ID is 0), false if they
     *                          are 1-based (i.e. the first label ID is 1).
     * @param binaryProblem     True if the input file contains data for a binary problem, false if the input file contains data for a multiclass multilabel
     *                          problem.
     * @return The results of the classification process.
     */
    public ClassificationResults classify(JavaSparkContext sc, String libSvmFile, int parallelismDegree, boolean labels0Based, boolean binaryProblem) {
        if (sc == null)
            throw new NullPointerException("The Spark context is 'null'");
        if (libSvmFile == null || libSvmFile.isEmpty())
            throw new IllegalArgumentException("The data file is 'null' or empty");
        if (parallelismDegree < 1)
            throw new IllegalArgumentException("The parallelism degree is less than 1");
        System.out.println("Load initial data and generating all necessary internal data representations...");
        JavaRDD<MultilabelPoint> docs = DataUtils.loadLibSvmFileFormatDataAsList(sc, libSvmFile, labels0Based, binaryProblem);
        return classify(sc, docs, parallelismDegree);
    }

    private class PreliminaryDocumentClassification implements Serializable {
        private final int documentID;
        private final int[] goldLabels;
        private final double[] scores;


        public PreliminaryDocumentClassification(int documentID, int[] goldLabels, double[] scores) {
            this.documentID = documentID;
            this.goldLabels = goldLabels;
            this.scores = scores;
        }

        public int getDocumentID() {
            return documentID;
        }

        public double[] getScores() {
            return scores;
        }

        public int[] getGoldLabels() {
            return goldLabels;
        }
    }

    private class FinalDocumentClassification implements Serializable {
        private final int documentID;
        private final int[] labelsAssigned;
        private final double[] scoresAssigned;
        private final int[] goldLabels;
        private final ContingencyTable ct;

        public FinalDocumentClassification(int documentID, int[] labelsAssigned, double[] scoresAssigned, int[] goldLabels, ContingencyTable ct) {
            this.documentID = documentID;
            this.labelsAssigned = labelsAssigned;
            this.scoresAssigned = scoresAssigned;
            this.goldLabels = goldLabels;
            this.ct = ct;
        }

        public int getDocumentID() {
            return documentID;
        }

        public int[] getLabelsAssigned() {
            return labelsAssigned;
        }

        public double[] getScoresAssigned() {
            return scoresAssigned;
        }

        public int[] getGoldLabels() {
            return goldLabels;
        }

        public ContingencyTable getCt() {
            return ct;
        }
    }
}
