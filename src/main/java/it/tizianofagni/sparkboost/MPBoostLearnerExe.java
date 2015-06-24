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

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class MPBoostLearnerExe {
    public static void main(String[] args) {
        if (args.length != 5) {
            System.out.println("Usage: " + MPBoostLearnerExe.class.getName() + " <inputFile> <outputFile> <numIterations> <sparkMaster> <parallelismDegree>");
            System.exit(-1);
        }

        long startTime = System.currentTimeMillis();
        String inputFile = args[0];
        String outputFile = args[1];
        int numIterations = Integer.parseInt(args[2]);
        String sparkMaster = args[3];
        int parallelismDegree = Integer.parseInt(args[4]);

        // Create and configure Spark context.
        SparkConf conf = new SparkConf().setAppName("Spark MPBoost learner");
        conf.setMaster(sparkMaster);
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Create and configure learner.
        MpBoostLearner learner = new MpBoostLearner(sc);
        learner.setNumIterations(numIterations);
        learner.setParallelismDegree(parallelismDegree);

        // Build classifier with MPBoost learner.
        MPBoostClassifier classifier = learner.buildModel(inputFile);

        // Save classifier to disk.
        DataUtils.saveModel(classifier, outputFile);

        long endTime = System.currentTimeMillis();
        System.out.println("Execution time: " + (endTime - startTime) + " milliseconds.");
    }
}
