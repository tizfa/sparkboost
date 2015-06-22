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

import org.apache.commons.cli.*;
import org.apache.ivy.util.cli.OptionBuilder;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class MPBoostLearnerExe {
    public static void main(String[] args) {
        if (args.length != 4) {
            System.out.println("Usage: "+MPBoostLearnerExe.class.getName()+" <inputFile> <outputFile> <numIterations> <sparkMaster>");
        }

        String inputFile = args[0];
        String outputFile = args[1];
        int numIterations = Integer.parseInt(args[2]);
        String sparkMaster = args[3];

        SparkConf conf = new SparkConf().setAppName("Spark MPBoost learner");
        conf.setMaster(sparkMaster);
        JavaSparkContext sc = new JavaSparkContext(conf);

        MpBoostLearner learner = new MpBoostLearner(sc);
        learner.setNumIterations(numIterations);
        MPBoostClassifier classifier = learner.buildModel(inputFile);
        DataUtils.saveModel(classifier, outputFile);
    }
}
