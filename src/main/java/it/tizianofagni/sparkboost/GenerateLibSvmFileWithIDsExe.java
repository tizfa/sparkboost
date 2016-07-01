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

/**
 * @author Tiziano Fagni (tiziano.fagni@isti.cnr.it)
 */
public class GenerateLibSvmFileWithIDsExe {
    public static void main(String[] args) {
        Options options = new Options();

        CommandLineParser parser = new BasicParser();
        CommandLine cmd = null;
        String[] remainingArgs = null;
        try {
            cmd = parser.parse(options, args);
            remainingArgs = cmd.getArgs();
            if (remainingArgs.length != 2)
                throw new ParseException("You need to specify all mandatory parameters");
        } catch (ParseException e) {
            System.out.println("Parsing failed.  Reason: " + e.getMessage());
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp(BoostClassifierBenchExe.class.getSimpleName() + " <inputLibSvmFile> <outputLibSvmFile>", options);
            System.exit(-1);
        }

        String inputFile = remainingArgs[0];
        String outputFile = remainingArgs[1];

        Logging.l().info("Generating a LibSvm file containing all documents with an ID assigned to each one...");
        DataUtils.generateLibSvmFileWithIDs(inputFile, outputFile);
        Logging.l().info("done!");
    }
}
