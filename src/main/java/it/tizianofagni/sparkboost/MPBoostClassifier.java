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
import org.apache.spark.storage.StorageLevel;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class MPBoostClassifier implements Serializable {

    private class DocumentClassification {
        private final int documentID;
        private final int[] goldLabels;
        private final double[] scores;


        public DocumentClassification(int documentID, int[] goldLabels, double[] scores) {
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


    private final WeakHypothesis[] whs;

    public MPBoostClassifier(WeakHypothesis[] whs) {
        if (whs == null)
            throw new NullPointerException("The set of generated WHs is 'null'");
        this.whs = whs;
    }


    ClassificationResults classify(JavaSparkContext sc, String libSvmFile) {
        System.out.println("Load initial data and generating all necessary internal data representations...");
        JavaRDD<MultilabelPoint> docs = DataUtils.loadLibSvmFileFormatDataAsList(sc, libSvmFile).persist(StorageLevel.MEMORY_AND_DISK());
        int numDocs = DataUtils.getNumDocuments(docs);
        System.out.println("done!");
        System.out.println("Classifying documents...");
        Broadcast<WeakHypothesis[]> whsBr = sc.broadcast(whs);
        Iterator<DocumentClassification> classifications = docs.map(doc -> {
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

            return new DocumentClassification(doc.getDocID(), doc.getLabels(), scores);
        }).toLocalIterator();
        int[] documents = new int[numDocs];
        int[][] labels = new int[numDocs][];
        double[][] scores = new double[numDocs][];
        int tp = 0, tn = 0, fp = 0, fn = 0;
        while (classifications.hasNext()) {
            DocumentClassification dc = classifications.next();
            HashSet<Integer> goldLabels = new HashSet<>();
            for (int labelID : dc.getGoldLabels())
                goldLabels.add(labelID);
            int docID = dc.getDocumentID();
            ArrayList<Integer> labelAssigned = new ArrayList<>();
            ArrayList<Double> labelScores = new ArrayList<>();
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
            labels[docID] = labelAssigned.stream().mapToInt(i->i).toArray();
            scores[docID] = labelScores.stream().mapToDouble(i->i).toArray();
        }
        System.out.println("done.");
        return new ClassificationResults(numDocs, labels, scores, new ContingencyTable(tp, tn, fp, fn));
    }
}
