/*
 *
 * ****************
 * This file is part of sparkboost software package (https://github.com/tizfa/sparkboost).
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

package it.tizianofagni.sparkboost;


import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
@SuppressWarnings("unchecked")
public class DataUtils {


    /**
     * Write a text file on Hadoop file system by using standard Hadoop API.
     *
     * @param outputPath The file to be written.
     * @param content    The content to put in the file.
     */
    public static void saveHadoopTextFile(String outputPath, String content) {
        try {
            Configuration configuration = new Configuration();
            Path file = new Path(outputPath);
            Path parentFile = file.getParent();
            FileSystem hdfs = FileSystem.get(file.toUri(), configuration);
            if (parentFile != null)
                hdfs.mkdirs(parentFile);
            OutputStream os = hdfs.create(file, true);
            BufferedWriter br = new BufferedWriter(new OutputStreamWriter(os, "UTF-8"));
            br.write(content);
            br.close();
            hdfs.close();
        } catch (Exception e) {
            throw new RuntimeException("Writing Hadoop text file", e);
        }
    }

    public static boolean deleteHadoopFile(String filepath, boolean recursive) {
        try {
            Configuration configuration = new Configuration();
            Path file = new Path(filepath);
            FileSystem hdfs = FileSystem.get(file.toUri(), configuration);
            return hdfs.delete(file, true);
        } catch (Exception e) {
            throw new RuntimeException("Deleting Hadoop file", e);
        }
    }


    public static <R> void saveHadoopClassificationResults(String outputPath, JavaRDD<DocClassificationResults> results) {

        JavaRDD<ClassificationPartialResults> classifications = results.mapPartitionsWithIndex((id, docs) -> {

            int tp = 0, tn = 0, fp = 0, fn = 0;
            StringBuilder sb = new StringBuilder();
            while (docs.hasNext()) {
                DocClassificationResults doc = docs.next();
                int docID = doc.getDocID();
                int[] labels = doc.getLabels();
                int[] goldLabels = doc.getGoldLabels();
                sb.append("DocID: " + docID + ", Labels assigned: " + Arrays.toString(labels) + ", Labels scores: " + Arrays.toString(doc.getScores()) + ", Gold labels: " + Arrays.toString(goldLabels) + "\n");
                tp += doc.getCt().tp();
                tn += doc.getCt().tn();
                fp += doc.getCt().fp();
                fn += doc.getCt().fn();
            }
            ContingencyTable ctRes = new ContingencyTable(tp, tn, fp, fn);
            sb.append("**** Effectiveness\n");
            sb.append(ctRes.toString() + "\n");

            ArrayList<ClassificationPartialResults> tables = new ArrayList<>();
            tables.add(new ClassificationPartialResults(id, sb.toString(), ctRes));
            return tables.iterator();

        }, true).persist(StorageLevel.MEMORY_ONLY());

        classifications.saveAsTextFile(outputPath);


        ContingencyTable ctRes = classifications.map(res -> {
            return res.ct;
        }).reduce((ct1, ct2) -> {

            ContingencyTable ct = new ContingencyTable(ct1.tp() + ct2.tp(),
                    ct1.tn() + ct2.tn(), ct1.fp() + ct2.fp(), ct1.fn() + ct2.fn());
            return ct;
        });

        DataUtils.saveHadoopTextFile(outputPath + "/global_contingency_table", ctRes.toString());
    }


    /**
     * Get the number of rows (maximum number of documents) contained in the specified file in
     * LibSvm format.
     *
     * @param sc       The Spark context.
     * @param dataFile The file to analyze.
     * @return The number of rows in the file.
     */
    public static long getNumRowsFromLibSvmFile(JavaSparkContext sc, String dataFile) {
        if (sc == null)
            throw new NullPointerException("The Spark Context is 'null'");
        if (dataFile == null || dataFile.isEmpty())
            throw new IllegalArgumentException("The dataFile is 'null'");

        JavaRDD<String> lines = sc.textFile(dataFile).cache();
        return lines.count();
    }


    /**
     * Generate a new LibSvm output file giving each document an index corresponding to the index tha documents had on
     * original input LibSvm file.
     *
     * @param sc         The spark context.
     * @param dataFile   The data file.
     * @param outputFile The output file.
     */
    public static void generateLibSvmFileWithIDs(JavaSparkContext sc, String dataFile, String outputFile) {
        if (sc == null)
            throw new NullPointerException("The Spark Context is 'null'");
        if (dataFile == null || dataFile.isEmpty())
            throw new IllegalArgumentException("The dataFile is 'null'");

        ArrayList<MultilabelPoint> points = new ArrayList<>();
        try {
            Path pt = new Path(dataFile);
            FileSystem fs = FileSystem.get(pt.toUri(), new Configuration());
            BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(pt)));

            Path ptOut = new Path(outputFile);
            BufferedWriter bw = new BufferedWriter((new OutputStreamWriter(fs.create(ptOut))));

            try {
                int docID = 0;
                String line = br.readLine();
                while (line != null) {
                    bw.write("" + docID + "\t" + line + "\n");
                    line = br.readLine();
                    docID++;
                }
            } finally {
                br.close();
                bw.close();
            }
        } catch (Exception e) {
            throw new RuntimeException("Reading input LibSVM data file", e);
        }

    }


    /**
     * Load data file in LibSvm format. The documents IDs are assigned according to the row index in the original
     * file, i.e. useful at classification time. We are assuming that the feature IDs are the same as the training
     * file used to build the classification model.
     *
     * @param sc       The spark context.
     * @param dataFile The data file.
     * @param fromID   The inclusive start document ID to read from.
     * @param toID     The noninclusive end document ID to read to.
     * @return An RDD containing the read points.
     */
    public static JavaRDD<MultilabelPoint> loadLibSvmFileFormatDataAsList(JavaSparkContext sc, String dataFile, boolean labels0Based, boolean binaryProblem, long fromID, long toID, int numFeaturesInDataset) {
        if (sc == null)
            throw new NullPointerException("The Spark Context is 'null'");
        if (dataFile == null || dataFile.isEmpty())
            throw new IllegalArgumentException("The dataFile is 'null'");

        JavaRDD<String> lines = sc.textFile(dataFile).cache();
        int numFeatures = 0;
        if (numFeaturesInDataset == -1)
            numFeatures = computeNumFeatures(lines);
        else
            numFeatures = numFeaturesInDataset;


        ArrayList<MultilabelPoint> points = new ArrayList<>();
        try {
            Path pt = new Path(dataFile);
            FileSystem fs = FileSystem.get(pt.toUri(), new Configuration());
            BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(pt)));

            //BufferedReader br = new BufferedReader(new FileReader(dataFile));

            try {
                int docID = 0;
                String line = br.readLine();
                while (line != null) {
                    if (docID >= toID)
                        break;
                    if (docID < fromID || line.isEmpty()) {
                        line = br.readLine();
                        docID++;
                        continue;
                    }

                    String[] fields = line.split("\\s+");
                    String[] t = fields[0].split(",");
                    int[] labels = new int[0];
                    if (!binaryProblem) {
                        labels = new int[t.length];
                        for (int i = 0; i < t.length; i++) {
                            String label = t[i];
                            if (labels0Based)
                                labels[i] = new Double(Double.parseDouble(label)).intValue();
                            else
                                labels[i] = new Double(Double.parseDouble(label)).intValue() - 1;
                            if (labels[i] < 0)
                                throw new IllegalArgumentException("In current configuration I obtain a negative label ID value. Please check if this is a problem binary or multiclass " +
                                        "and if the labels IDs are in form 0-based or 1-based");
                        }
                    } else {
                        if (t.length > 1)
                            throw new IllegalArgumentException("In binary problem you can only specify one label ID (+1 or -1) per document as valid label IDs");
                        int label = new Double(Double.parseDouble(t[0])).intValue();
                        if (label > 0) {
                            labels = new int[]{0};
                        }
                    }
                    ArrayList<Integer> indexes = new ArrayList<>();
                    ArrayList<Double> values = new ArrayList<>();
                    for (int j = 1; j < fields.length; j++) {
                        String data = fields[j];
                        if (data.startsWith("#"))
                            // Beginning of a comment. Skip it.
                            break;
                        String[] featInfo = data.split(":");
                        // Transform feature ID value in 0-based.
                        int featID = Integer.parseInt(featInfo[0]) - 1;
                        double value = Double.parseDouble(featInfo[1]);
                        indexes.add(featID);
                        values.add(value);
                    }

                    SparseVector v = (SparseVector) Vectors.sparse(numFeatures, indexes.stream().mapToInt(i -> i).toArray(), values.stream().mapToDouble(i -> i).toArray());
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
     * @param sc               The spark context.
     * @param dataFile         The data file.
     * @param minNumPartitions The minimum number of partitions to split data in "dataFile".
     * @return An RDD containing the read points.
     */
    public static JavaRDD<MultilabelPoint> loadLibSvmFileFormatData(JavaSparkContext sc, String dataFile, boolean labels0Based, boolean binaryProblem, int minNumPartitions) {
        if (sc == null)
            throw new NullPointerException("The Spark Context is 'null'");
        if (dataFile == null || dataFile.isEmpty())
            throw new IllegalArgumentException("The dataFile is 'null'");
        JavaRDD<String> lines = sc.textFile(dataFile, minNumPartitions).cache();
        int localNumFeatures = computeNumFeatures(lines);
        Broadcast<Integer> distNumFeatures = sc.broadcast(localNumFeatures);
        JavaRDD<MultilabelPoint> docs = lines.filter(line -> !line.isEmpty()).zipWithIndex().map(item -> {
            int numFeatures = distNumFeatures.getValue();
            String line = item._1();
            long indexLong = item._2();
            int index = (int) indexLong;
            String[] fields = line.split("\\s+");
            String[] t = fields[0].split(",");
            int[] labels = new int[0];
            if (!binaryProblem) {
                labels = new int[t.length];
                for (int i = 0; i < t.length; i++) {
                    String label = t[i];
                    // Labels should be already 0-based.
                    if (labels0Based)
                        labels[i] = new Double(Double.parseDouble(label)).intValue();
                    else
                        labels[i] = new Double(Double.parseDouble(label)).intValue() - 1;
                    if (labels[i] < 0)
                        throw new IllegalArgumentException("In current configuration I obtain a negative label ID value. Please check if this is a problem binary or multiclass " +
                                "and if the labels IDs are in form 0-based or 1-based");
                    assert (labels[i] >= 0);
                }
            } else {
                if (t.length > 1)
                    throw new IllegalArgumentException("In binary problem you can only specify one label ID (+1 or -1) per document as valid label IDs");
                int label = new Double(Double.parseDouble(t[0])).intValue();
                if (label > 0) {
                    labels = new int[]{0};
                }
            }
            ArrayList<Integer> indexes = new ArrayList<>();
            ArrayList<Double> values = new ArrayList<>();
            for (int j = 1; j < fields.length; j++) {
                String data = fields[j];
                if (data.startsWith("#"))
                    // Beginning of a comment. Skip it.
                    break;
                String[] featInfo = data.split(":");
                // Transform feature ID value in 0-based.
                int featID = Integer.parseInt(featInfo[0]) - 1;
                double value = Double.parseDouble(featInfo[1]);
                indexes.add(featID);
                values.add(value);
            }

            SparseVector v = (SparseVector) Vectors.sparse(numFeatures, indexes.stream().mapToInt(i -> i).toArray(), values.stream().mapToDouble(i -> i).toArray());
            return new MultilabelPoint(index, v, labels);
        });

        lines.unpersist();
        return docs;
    }


    /**
     * Load data file in LibSVm format. The documents IDs are specified at beginning of each
     * line containing document data.
     *
     * @param sc               The spark context.
     * @param dataFile         The data file.
     * @param minNumPartitions The minimum number of partitions to split data in "dataFile".
     * @return An RDD containing the read points.
     */
    public static JavaRDD<MultilabelPoint> loadLibSvmFileFormatDataWithIDs(JavaSparkContext sc, String dataFile, boolean labels0Based, boolean binaryProblem, int minNumPartitions) {
        if (sc == null)
            throw new NullPointerException("The Spark Context is 'null'");
        if (dataFile == null || dataFile.isEmpty())
            throw new IllegalArgumentException("The dataFile is 'null'");
        JavaRDD<String> lines = sc.textFile(dataFile, minNumPartitions).cache();
        int localNumFeatures = computeNumFeatures(lines);
        Broadcast<Integer> distNumFeatures = sc.broadcast(localNumFeatures);
        JavaRDD<MultilabelPoint> docs = lines.filter(line -> !line.isEmpty()).map(entireRow -> {
            int numFeatures = distNumFeatures.getValue();
            String[] fields = entireRow.split("\t");
            String line = fields[1];
            int docID = Integer.parseInt(fields[0]);
            fields = line.split("\\s+");
            String[] t = fields[0].split(",");
            int[] labels = new int[0];
            if (!binaryProblem) {
                labels = new int[t.length];
                for (int i = 0; i < t.length; i++) {
                    String label = t[i];
                    // Labels should be already 0-based.
                    if (labels0Based)
                        labels[i] = new Double(Double.parseDouble(label)).intValue();
                    else
                        labels[i] = new Double(Double.parseDouble(label)).intValue() - 1;
                    if (labels[i] < 0)
                        throw new IllegalArgumentException("In current configuration I obtain a negative label ID value. Please check if this is a problem binary or multiclass " +
                                "and if the labels IDs are in form 0-based or 1-based");
                    assert (labels[i] >= 0);
                }
            } else {
                if (t.length > 1)
                    throw new IllegalArgumentException("In binary problem you can only specify one label ID (+1 or -1) per document as valid label IDs");
                int label = new Double(Double.parseDouble(t[0])).intValue();
                if (label > 0) {
                    labels = new int[]{0};
                }
            }
            ArrayList<Integer> indexes = new ArrayList<>();
            ArrayList<Double> values = new ArrayList<>();
            for (int j = 1; j < fields.length; j++) {
                String data = fields[j];
                if (data.startsWith("#"))
                    // Beginning of a comment. Skip it.
                    break;
                String[] featInfo = data.split(":");
                // Transform feature ID value in 0-based.
                int featID = Integer.parseInt(featInfo[0]) - 1;
                double value = Double.parseDouble(featInfo[1]);
                indexes.add(featID);
                values.add(value);
            }

            SparseVector v = (SparseVector) Vectors.sparse(numFeatures, indexes.stream().mapToInt(i -> i).toArray(), values.stream().mapToDouble(i -> i).toArray());
            return new MultilabelPoint(docID, v, labels);
        });

        lines.unpersist();
        return docs;
    }


    protected static int computeNumFeatures(JavaRDD<String> lines) {
        int maxFeatureID = lines.map(line -> {
            if (line.isEmpty())
                return -1;
            String[] fields = line.split("\\s+");
            int maximumFeatID = 0;
            for (int j = 1; j < fields.length; j++) {
                String data = fields[j];
                if (data.startsWith("#"))
                    // Beginning of a comment. Skip it.
                    break;
                String[] featInfo = data.split(":");
                int featID = Integer.parseInt(featInfo[0]);
                maximumFeatID = Math.max(featID, maximumFeatID);
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
        int maxValidLabelID = documents.map(doc -> {
            List<Integer> values = Arrays.asList(ArrayUtils.toObject(doc.getLabels()));
            if (values.size() == 0)
                return 0;
            else
                return Collections.max(values);
        }).reduce((m1, m2) -> Math.max(m1, m2));
        return maxValidLabelID + 1;
    }

    public static int getNumFeatures(JavaRDD<MultilabelPoint> documents) {
        if (documents == null)
            throw new NullPointerException("The documents RDD is 'null'");
        return documents.take(1).get(0).getFeatures().size();
    }

    public static JavaRDD<LabelDocuments> getLabelDocuments(JavaRDD<MultilabelPoint> documents) {
        return documents.flatMapToPair(doc -> {
            int[] labels = doc.getLabels();
            ArrayList<Integer> docAr = new ArrayList<>();
            docAr.add(doc.getPointID());
            ArrayList<Tuple2<Integer, ArrayList<Integer>>> ret = new ArrayList<>();
            for (int i = 0; i < labels.length; i++) {
                ret.add(new Tuple2<>(labels[i], docAr));
            }
            return ret;
        }).reduceByKey((list1, list2) -> {
            ArrayList<Integer> ret = new ArrayList<>();
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
                int[] docs = new int[]{doc.getPointID()};
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

    /**
     * Save a boosting classifier model to the specified output model path (any valid path recognized by
     * Spark/Hadoop).
     * <br/><br/>
     * IMPORTANT NOTE: if you are executing Spark in local mode under Windows, you can get this strange error
     * as described <a href="https://issues.apache.org/jira/browse/SPARK-6961?jql=project%20%3D%20SPARK%20AND%20text%20~%20%22save%20file%20local%22">here</a>.
     * Currently the workaround is to install the winutils executable on the path corresponding to Hadoop installation (see
     * <a href="http://stackoverflow.com/questions/24832284/nullpointerexception-in-spark-sql">here</a> for more details about this workaround).
     *
     * @param sc              The Spark context.
     * @param classifier      The classifier to be save.
     * @param outputModelPath The output path where to save the model.
     */
    public static void saveModel(JavaSparkContext sc, BoostClassifier classifier, String outputModelPath) {
        if (sc == null)
            throw new NullPointerException("The Spark context is 'null'");
        if (classifier == null)
            throw new NullPointerException("The classifier is 'null'");
        if (outputModelPath == null)
            throw new NullPointerException("The output model path is 'null'");

        ArrayList<BoostClassifier> clList = new ArrayList<>();
        clList.add(classifier);
        JavaRDD<BoostClassifier> rdd = sc.parallelize(clList);
        rdd.saveAsObjectFile(outputModelPath);
    }

    /**
     * Load a boosting classifier model  from the specified input model path (any valid path recognized by
     * Spark/Hadoop).
     *
     * @param sc             The Spark context.
     * @param inputModelPath The input model path.
     * @return The corresponding boosting classifier.
     */
    public static BoostClassifier loadModel(JavaSparkContext sc, String inputModelPath) {
        if (sc == null)
            throw new NullPointerException("The Spark context is 'null'");
        if (inputModelPath == null)
            throw new NullPointerException("The input model path is 'null'");

        List classifiers = sc.objectFile(inputModelPath).collect();
        if (classifiers.size() == 0)
            throw new IllegalArgumentException("The specified input model path is not valid. No data available here!");
        BoostClassifier cl = (BoostClassifier) classifiers.get(0);
        return cl;
    }

    public static class LabelDocuments implements Serializable {
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

    public static class FeatureDocuments implements Serializable {
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
}
