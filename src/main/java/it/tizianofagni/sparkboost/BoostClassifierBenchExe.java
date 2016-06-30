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

import org.apache.commons.cli.*;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class BoostClassifierBenchExe {
    public static void main(String[] args) {

        Options options = new Options();
        options.addOption("b", "binaryProblem", false, "Indicate if the input dataset contains a binary problem and not a multilabel one");
        options.addOption("z", "labels0based", false, "Indicate if the labels IDs in the dataset to classifyLibSvmWithResults are already assigned in the range [0, numLabels-1] included");
        options.addOption("l", "enableSparkLogging", false, "Enable logging messages of Spark");
        options.addOption("w", "windowsLocalModeFix", true, "Set the directory containing the winutils.exe command");
        options.addOption("p", "parallelismDegree", true, "Set the parallelism degree (default: number of available cores in the Spark runtime");

        CommandLineParser parser = new BasicParser();
        CommandLine cmd = null;
        String[] remainingArgs = null;
        try {
            cmd = parser.parse(options, args);
            remainingArgs = cmd.getArgs();
            if (remainingArgs.length != 3)
                throw new ParseException("You need to specify all mandatory parameters");
        } catch (ParseException e) {
            System.out.println("Parsing failed.  Reason: " + e.getMessage());
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp(BoostClassifierBenchExe.class.getSimpleName() + " [OPTIONS] <inputFile> <inputModel> <outputFile>", options);
            System.exit(-1);
        }

        boolean binaryProblem = false;
        if (cmd.hasOption("b"))
            binaryProblem = true;
        boolean labels0Based = false;
        if (cmd.hasOption("z"))
            labels0Based = true;
        boolean enablingSparkLogging = false;
        if (cmd.hasOption("l"))
            enablingSparkLogging = true;

        if (cmd.hasOption("w")) {
            System.setProperty("hadoop.home.dir", cmd.getOptionValue("w"));
        }

        String inputFile = remainingArgs[0];
        String inputModel = remainingArgs[1];
        String outputFile = remainingArgs[2];


        long startTime = System.currentTimeMillis();

        // Disable Spark logging.
        if (!enablingSparkLogging) {
            Logger.getLogger("org").setLevel(Level.OFF);
            Logger.getLogger("akka").setLevel(Level.OFF);
        }

        // Create and configure Spark context.
        SparkConf conf = new SparkConf().setAppName("Spark MPBoost classifier");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Load boosting classifier from disk.
        BoostClassifier classifier = DataUtils.loadModel(sc, inputModel);

        // Get the parallelism degree.
        int parallelismDegree = sc.defaultParallelism();
        if (cmd.hasOption("p")) {
            parallelismDegree = Integer.parseInt(cmd.getOptionValue("p"));
        }


        Logging.l().info("Generating a temporary file containing all documents with an ID assigned to each one...");
        String dataFile = inputFile + ".withIDs";
        DataUtils.generateLibSvmFileWithIDs(sc, inputFile, dataFile);
        Logging.l().info("done!");

        // Create an RDD with the input documents to be classified.
        Logging.l().info("Creating a RDD containing all the documents to be classified...");
        JavaRDD<MultilabelPoint> docs = DataUtils.loadLibSvmFileFormatDataWithIDs(sc, dataFile, labels0Based, binaryProblem, parallelismDegree);
        Logging.l().info("done.");

        Iterator<DocClassificationResults> results = classifier.classify(sc, docs, parallelismDegree).toLocalIterator();
        while (results.hasNext()) {
            DocClassificationResults doc = results.next();
        }

        Logging.l().info("Deleting no more necessary temporary files...");
        DataUtils.deleteHadoopFile(dataFile, true);
        Logging.l().info("done.");

        long endTime = System.currentTimeMillis();
        System.out.println("Execution time: " + (endTime - startTime) + " milliseconds.");


    }
}
