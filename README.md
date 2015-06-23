# sparkboost
This repository contains a distributed [MP-Boost](http://link.springer.com/chapter/10.1007%2F11880561_1#page-1) implementation based on [Apache Spark](https://spark.apache.org/). MP-Boost is an improved variant of the well known [AdaBoost.MH](http://link.springer.com/article/10.1023%2FA%3A1007649029923) machine learning algorithm. MP-Boost improves original AdaBoost.MH by building classifiers which allows to obtain remarkably better effectiveness and a very similar computational cost at build/classification time.

The software allows to build MP-Boost multi-label multiclass classifiers starting from dataset files available in the [LibSvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) format. A lot of ready datasets in this format are available [here](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). 

IMPORTANT NOTE: currently the software only works with multiclass datasets containing categories IDs assigned in a 0-based way, i.e. if the dataset contains 10 labels, the set of valid IDs must be in the range [0, 9] included.

## Software compilation
To build the software you need to have [Maven](https://maven.apache.org/) installed on your machine. Download a copy of this software repository on your machine on a specific folder, go inside that forlder and at the command prompt put the following commands:
```
mvn clean
mvn -P release package
```
This set of commands will build a software bundle containing all the necessary Spark libraries. You can find the software bundle in the `target` directory of the software package.

## Software usage
### Build a classifier
To build a classifier for a specific dataset file `path/to/datasetFile` (in the format libsvm), launch this command from prompt:
```
java -cp ./target/sparkboost-0.1-SNAPSHOT-bundle.jar it.tizianofagni.sparkboost.MPBoostLearnerExe path/to/datasetfile path/to/modelOutput numIterations sparkMasterName parallelismDegree
```
where `path/to/modelOutput` is the output file where the generated classifier will be save, `numIterations` is the number of iterations used in the algorithm, `sparkMasterName` is the name of Spark master host (or local[*] for executing the process locally on your machine) and `parallelismDegree` is the number of processing units to use while executing the algorithm.

### Use a classifier
To use an already built classifier over a test dataset, use this command:
```
java -cp ./target/sparkboost-0.1-SNAPSHOT-bundle.jar it.tizianofagni.sparkboost.MPBoostClassifierExe datasetfile classifierModel outputResultsFile sparkMasterName parallelismDegree
```
where `datasetFile` is the input file containing the dataset test examples, `classifierModel` is the file containing a previous generated classifier, `outputResultsFile` is the ouput file containing classification results, `sparkMasterName` is the name of Spark master host (or local[*] for executing the process locally on your machine) and `parallelismDegree` is the number of processing units to use while executing the algorithm.

IMPORTANT NOTE: Every document in the test dataset will get a document ID corresponding at the original row index of the document in the dataset file.



