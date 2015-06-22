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
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class DataUtils {

    public static class LabelDocuments {
        private final int labelID;
        private final int[] documents;

        public LabelDocuments(int labelID, int[] documents) {
            this.labelID = labelID;
            this.documents = documents;
        }

        public int getLabelID() {
            return labelID;
        }

        public int[] getDocuments() {
            return documents;
        }
    }


    public static class FeatureDocuments {
        private final int featureID;
        private final int[] documents;
        private final int[][] labels;

        public FeatureDocuments(int featureID, int[] documents, int[][] labels) {
            this.featureID = featureID;
            this.documents = documents;
            this.labels = labels;
        }

        public int getFeatureID() {
            return featureID;
        }

        public int[] getDocuments() {
            return documents;
        }

        public int[][] getLabels() {
            return labels;
        }
    }


    /**
     * Load data file in LibSVm format. The documents IDs are assigned according to the row inde xin the original
     * file, i.e. useful at classification time.
     *
     * @param sc       The spark context.
     * @param dataFile The data file.
     * @return An RDD containing the read points.
     */
    public static JavaRDD<MultilabelPoint> loadLibSvmFileFormatDataAsList(JavaSparkContext sc, String dataFile) {
        if (sc == null)
            throw new NullPointerException("The Spark Context is 'null'");
        if (dataFile == null || dataFile.isEmpty())
            throw new IllegalArgumentException("The dataFile is 'null'");

        JavaRDD<String> lines = sc.textFile(dataFile).cache();
        int maxFeatureID = computeMaximumFeatureID(lines);

        ArrayList<MultilabelPoint> points = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(dataFile));

            try {
                int docID = 0;
                String line = br.readLine();
                while (line != null) {
                    if (line.isEmpty())
                        return null;
                    String[] fields = line.split("\\s+");
                    String[] t = fields[0].split(",");
                    int[] labels = new int[t.length];
                    for (int i = 0; i < t.length; i++) {
                        String label = t[i];
                        labels[i] = new Double(Double.parseDouble(label)).intValue();
                    }
                    ArrayList<Integer> indexes = new ArrayList<Integer>();
                    ArrayList<Double> values = new ArrayList<Double>();
                    for (int j = 1; j < fields.length; j++) {
                        String data = fields[j];
                        if (data.startsWith("#"))
                            // Beginning of a comment. Skip it.
                            break;
                        String[] featInfo = data.split(":");
                        int featID = Integer.parseInt(featInfo[0]);
                        double value = Double.parseDouble(featInfo[1]);
                        indexes.add(featID);
                        values.add(value);
                    }

                    SparseVector v = (SparseVector) Vectors.sparse(maxFeatureID, indexes.stream().mapToInt(i -> i).toArray(), values.stream().mapToDouble(i -> i).toArray());
                    points.add(new MultilabelPoint(docID, v, labels));

                    line = br.readLine();
                    docID++;
                }
            } finally {
                br.close();
            }
        } catch (Exception e) {
            throw new RuntimeException("Reading input LibSVM data file", e);
        }

        return sc.parallelize(points);
    }


    /**
     * Load data file in LibSVm format. The documents IDs are assigned arbitrarily by Spark.
     *
     * @param sc       The spark context.
     * @param dataFile The data file.
     * @return An RDD containing the read points.
     */
    public static JavaRDD<MultilabelPoint> loadLibSvmFileFormatData(JavaSparkContext sc, String dataFile) {
        if (sc == null)
            throw new NullPointerException("The Spark Context is 'null'");
        if (dataFile == null || dataFile.isEmpty())
            throw new IllegalArgumentException("The dataFile is 'null'");
        JavaRDD<String> lines = sc.textFile(dataFile).cache();
        int maxFeatureID = computeMaximumFeatureID(lines);
        JavaRDD<MultilabelPoint> docs = lines.zipWithIndex().map(item -> {
            String line = item._1();
            long indexLong = item._2();
            int index = (int) indexLong;
            if (line.isEmpty())
                return null;
            String[] fields = line.split("\\s+");
            String[] t = fields[0].split(",");
            int[] labels = new int[t.length];
            for (int i = 0; i < t.length; i++) {
                String label = t[i];
                labels[i] = new Double(Double.parseDouble(label)).intValue();
            }
            ArrayList<Integer> indexes = new ArrayList<Integer>();
            ArrayList<Double> values = new ArrayList<Double>();
            for (int j = 1; j < fields.length; j++) {
                String data = fields[j];
                if (data.startsWith("#"))
                    // Beginning of a comment. Skip it.
                    break;
                String[] featInfo = data.split(":");
                int featID = Integer.parseInt(featInfo[0]);
                double value = Double.parseDouble(featInfo[1]);
                indexes.add(featID);
                values.add(value);
            }

            SparseVector v = (SparseVector) Vectors.sparse(maxFeatureID, indexes.stream().mapToInt(i -> i).toArray(), values.stream().mapToDouble(i -> i).toArray());
            return new MultilabelPoint(index, v, labels);
        });

        return docs;
    }


    protected static int computeMaximumFeatureID(JavaRDD<String> lines) {
        int maxFeatureID = lines.map(line -> {
            if (line.isEmpty())
                return -1;
            String[] fields = line.split("\\s+");
            ArrayList<Double> labels = new ArrayList<Double>();
            String[] t = fields[0].split(",");
            for (String label : t) {
                labels.add(Double.parseDouble(label));
            }

            int maximumFeatID = 0;
            for (int j = 1; j < fields.length; j++) {
                String data = fields[j];
                if (data.startsWith("#"))
                    // Beginning of a comment. Skip it.
                    break;
                String[] featInfo = data.split(":");
                int featID = Integer.parseInt(featInfo[0]);
                maximumFeatID = featID;
            }
            return maximumFeatID;
        }).reduce((val1, val2) -> val1 < val2 ? val2 : val1);

        return maxFeatureID;
    }


    public static int getNumDocuments(JavaRDD<MultilabelPoint> documents) {
        if (documents == null)
            throw new NullPointerException("The documents RDD is 'null'");
        return (int) documents.count();
    }

    public static int getNumLabels(JavaRDD<MultilabelPoint> documents) {
        if (documents == null)
            throw new NullPointerException("The documents RDD is 'null'");
        return (int) documents.flatMap(doc -> {
            return Arrays.asList(doc.getLabels());
        }).distinct().count();
    }

    public static int getNumFeatures(JavaRDD<MultilabelPoint> documents) {
        if (documents == null)
            throw new NullPointerException("The documents RDD is 'null'");
        return documents.take(1).get(0).getFeatures().size();
    }


    public static MultilabelPoint getDocument(JavaRDD<MultilabelPoint> documents, int docID) {
        List<MultilabelPoint> docs = documents.filter(doc -> doc.getDocID() == docID).collect();
        if (docs.size() == 0)
            throw new IllegalArgumentException("Can not find document ID " + docID);
        return docs.get(0);
    }


    public static JavaRDD<LabelDocuments> getLabelDocuments(JavaRDD<MultilabelPoint> documents) {
        return documents.flatMapToPair(doc -> {
            int[] labels = doc.getLabels();
            ArrayList<Integer> docAr = new ArrayList<Integer>();
            docAr.add(doc.getDocID());
            ArrayList<Tuple2<Integer, ArrayList<Integer>>> ret = new ArrayList<Tuple2<Integer, ArrayList<Integer>>>();
            for (int i = 0; i < labels.length; i++) {
                ret.add(new Tuple2<>(labels[i], docAr));
            }
            return ret;
        }).reduceByKey((list1, list2) -> {
            ArrayList<Integer> ret = new ArrayList<Integer>();
            ret.addAll(list1);
            ret.addAll(list2);
            Collections.sort(ret);
            return ret;
        }).map(item -> {
            return new LabelDocuments(item._1(), item._2().stream().mapToInt(i -> i).toArray());
        });
    }


    public static JavaRDD<FeatureDocuments> getFeatureDocuments(JavaRDD<MultilabelPoint> documents) {
        return documents.flatMapToPair(doc -> {
            SparseVector feats = doc.getFeatures();
            int[] indices = feats.indices();
            ArrayList<Tuple2<Integer, FeatureDocuments>> ret = new ArrayList<>();
            for (int i = 0; i < indices.length; i++) {
                int featureID = indices[i];
                int[] docs = new int[]{doc.getDocID()};
                int[][] labels = new int[1][];
                labels[0] = doc.getLabels();
                ret.add(new Tuple2<>(featureID, new FeatureDocuments(featureID, docs, labels)));
            }
            return ret;
        }).reduceByKey((f1, f2) -> {
            int numDocs = f1.getDocuments().length + f2.getDocuments().length;
            int[] docsMerged = new int[numDocs];
            int[][] labelsMerged = new int[numDocs][];
            // Add first feature info.
            for (int idx = 0; idx < f1.getDocuments().length; idx++) {
                docsMerged[idx] = f1.getDocuments()[idx];
            }
            for (int idx = 0; idx < f1.getDocuments().length; idx++) {
                labelsMerged[idx] = f1.getLabels()[idx];
            }

            // Add second feature info.
            for (int idx = f1.getDocuments().length; idx < numDocs; idx++) {
                docsMerged[idx] = f2.getDocuments()[idx - f1.getDocuments().length];
            }
            for (int idx = f1.getDocuments().length; idx < numDocs; idx++) {
                labelsMerged[idx] = f2.getLabels()[idx - f1.getDocuments().length];
            }
            return new FeatureDocuments(f1.featureID, docsMerged, labelsMerged);
        }).map(item -> item._2());
    }


    public static void saveModel(MPBoostClassifier classifier, String outputFile) {
        if (classifier == null)
            throw new NullPointerException("The classifier is 'null'");
        if (outputFile == null)
            throw new NullPointerException("The output file is 'null'");

        File fout = new File(outputFile);
        fout.getParentFile().mkdirs();
        ObjectOutputStream oos = null;
        try {
            OutputStream fo = new BufferedOutputStream(new FileOutputStream(outputFile));
            oos = new ObjectOutputStream(fo);
            oos.writeObject(classifier);
        } catch(Exception e) {
            throw new RuntimeException("Writing classifier model", e);
        }
        finally {
            if (oos != null)
                try {
                    oos.close();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
        }
    }


    public static MPBoostClassifier loadModel(String inputFile) {
        if (inputFile == null)
            throw new NullPointerException("The output file is 'null'");
        ObjectInputStream ois = null;
        try {
            InputStream fis = new BufferedInputStream(new FileInputStream(inputFile));
            ois = new ObjectInputStream(fis);
            return (MPBoostClassifier) ois.readObject();
        } catch (Exception e) {
            throw new RuntimeException("Reading classifier model", e);
        } finally {
            if (ois != null) {
                try {
                    ois.close();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }
}
