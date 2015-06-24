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

import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.File;
import java.util.Arrays;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class BoostClassifierExe {
    public static void main(String[] args) {
        if (args.length != 5) {
            System.out.println("Usage: " + BoostClassifierExe.class.getName() + " <inputFile> <inputModel> <outputFile> <sparkMaster> <parallelismDegree>");
            System.exit(-1);
        }

        long startTime = System.currentTimeMillis();
        String inputFile = args[0];
        String inputModel = args[1];
        String outputFile = args[2];
        String sparkMaster = args[3];
        int parallelismDegree = Integer.parseInt(args[4]);

        // Create and configure Spark context.
        SparkConf conf = new SparkConf().setAppName("Spark MPBoost classifier");
        conf.setMaster(sparkMaster);
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Load MPBoost classifier from disk.
        BoostClassifier classifier = DataUtils.loadModel(inputModel);

        // Classify documents contained in "inputFile", a file in libsvm format.
        ClassificationResults results = classifier.classify(sc, inputFile, parallelismDegree);

        // Write classification results to disk.
        StringBuilder sb = new StringBuilder();
        sb.append("**** Effectiveness\n");
        sb.append(results.getCt().toString() + "\n");
        sb.append("********\n");
        for (int i = 0; i < results.getNumDocs(); i++) {
            int docID = results.getDocuments()[i];
            int[] labels = results.getLabels()[i];
            int[] goldLabels = results.getGoldLabels()[i];
            sb.append("DocID: " + docID + ", Labels assigned: " + Arrays.toString(labels) + ", Labels scores: " + Arrays.toString(results.getScores()[i]) + ", Gold labels: " + Arrays.toString(goldLabels) + "\n");
        }
        try {
            new File(outputFile).getParentFile().mkdirs();
            FileUtils.writeStringToFile(new File(outputFile), sb.toString());
        } catch (Exception e) {
            throw new RuntimeException("Writing clasisfication results", e);
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Execution time: " + (endTime - startTime) + " milliseconds.");
    }
}
