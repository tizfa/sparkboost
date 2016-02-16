# sparkboost
This repository contains a distributed implementation based on [Apache Spark](https://spark.apache.org/) of [AdaBoost.MH](http://link.springer.com/article/10.1023%2FA%3A1007649029923) and [MP-Boost](http://link.springer.com/chapter/10.1007%2F11880561_1#page-1) algorithms. MP-Boost is an improved variant of the well known AdaBoost.MH machine learning algorithm. MP-Boost improves original AdaBoost.MH by building classifiers which allows to obtain remarkably better effectiveness and a very similar computational cost at build/classification time.

The software is open source and released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

The software allows to build multi-label multiclass classifiers or binary classifiers using AdaBoost.MH or MP-Boost starting from a dataset available in the [LibSvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) format. A lot of ready datasets in this format are available [here](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).

## Software installation
To use the latest release of this software in your projects, in your project POM add the following:
```
<repositories>

    <repository>
        <id>sparkboost-mvn-repo</id>
        <url>https://raw.github.com/tizfa/sparkboost/mvn-repo/</url>
        <snapshots>
            <enabled>true</enabled>
            <updatePolicy>always</updatePolicy>
        </snapshots>
    </repository>

</repositories>
```
then in the dependencies list add
```
<dependency>
    <groupId>tizfa</groupId>
    <artifactId>sparkboost</artifactId>
    <version>0.6</version>
</dependency>
```

## Software usage
### Using provided command line tools
Currently the software allow to perform multilabel multiclass classification or binary classification over datasets available on in the LibSvm format. The user
at learning and classification time must specify if the problem is or not of binary type (usage of `-b` flag in the available commands). In multiclass problems, the user should also specify
if the labels IDs are 0-based or 1-based, i.e. if the number of valid labels is n then the set of valid IDs is in the range [0, 9] included (0-based) or [1,10] included (1-based). To specify
if the labels are 0-based, the user can use the flag `-z` in the available commands.

#### Software compilation to use command line tools
If you are interested to use the command line tools available with the software, you need to download the latest release sources available from [here](https://github.com/tizfa/sparkboost/archive/master.zip) and the compile them. To perform this task, you need [Maven](https://maven.apache.org/) and a Java 8 compiler installed on your machine. Download a copy of this software repository on your machine on a specific folder, go inside that folder and at the command prompt put the following commands:
```
mvn clean
mvn -P shading package
```
This set of commands will build a software bundle containing all the necessary Spark libraries. You can find the software bundle in the `target` directory of the software package.

#### Building a MP-Boost classifier
To build a MP-Boost classifier for a specific dataset file `path/to/datasetFile` (in the format libsvm), launch this command from prompt:
```
java -cp ./target/sparkboost-0.6-bundle.jar it.tizianofagni.sparkboost.MPBoostLearnerExe path/to/datasetfile path/to/modelOutput numIterations sparkMasterName parallelismDegree
```
where `path/to/modelOutput` is the output file where the generated classifier will be save, `numIterations` is the number of iterations used in the algorithm, `sparkMasterName` is the name of Spark master host (or local[*] for executing the process locally on your machine) and `parallelismDegree` is the number of processing units to use while executing the algorithm.

#### Building an AdaBoost.MH classifier
To build a AdaBoost.MH classifier for a specific dataset file `path/to/datasetFile` (in the format libsvm), launch this command from prompt:
```
java -cp ./target/sparkboost-0.6-bundle.jar it.tizianofagni.sparkboost.AdaBoostMHLearnerExe path/to/datasetfile path/to/modelOutput numIterations sparkMasterName parallelismDegree
```
where `path/to/modelOutput` is the output file where the generated classifier will be save, `numIterations` is the number of iterations used in the algorithm, `sparkMasterName` is the name of Spark master host (or local[*] for executing the process locally on your machine) and `parallelismDegree` is the number of processing units to use while executing the algorithm.

#### Using a classifier
To use an already built classifier over a test dataset (it does not matter if the model has been built with MP-Boost or AdaBoost.MH learner, they share the same forrmat for classification models!), use this command:
```
java -cp ./target/sparkboost-0.6-bundle.jar it.tizianofagni.sparkboost.BoostClassifierExe datasetfile classifierModel outputResultsFile sparkMasterName parallelismDegree
```
where `datasetFile` is the input file containing the dataset test examples, `classifierModel` is the file containing a previous generated classifier, `outputResultsFile` is the ouput file containing classification results, `sparkMasterName` is the name of Spark master host (or local[*] for executing the process locally on your machine) and `parallelismDegree` is the number of processing units to use while executing the algorithm.

IMPORTANT NOTE: Each document in the test dataset will get a document ID corresponding at the original row index of the document in the dataset file.

#### Use case: RCV1v2
We used MP-Boost to build a multilabel classifer to automatically classify textual documents in [RCV1v2](http://www.daviddlewis.com/resources/testcollections/rcv1/), a corpus of newswire stories made available by Reuters, Ltd. The dataset (rcv1v2 (topics; full sets)) is available also in libsvm format at page http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html

The main characteristics of rcv1v2 (topics; full sets) are:
* number of classes: 101
* number of documents: 23149 / 781265 (testing)
* number of features: 47236

We used the following files for this experimentation:
* rcv1_topics_train.svm.bz2 for training (23149 documents)
* rcv1_topics_test_0.svm.bz2 for testing (199328 documents)

We built a MP-Boost classification model using 500 iterations and using a single multicore machine (AMD Fx-8350 8-cores). The training time to build a classification model for all 101 labels and by specifying a parallelismDegree of 8 has been of 1206 seconds. The classification time has been of 61 seconds to classify all 199328 documents. Here are the main results we have obtained in this specific configuration:
Precision: 0.8331455753966425, Recall: 0.6987996337970667, F1:0.7600817829421506

### Using library API to build your own programs
An example of using the API is given by the provided command line tools. Just watch the source code of classes AdaBoostMHLearnerExe.java, MPBoostLearnerExe.java and BoostClassifierExe.java.
Briefly to build a classifier you can use a code like this:
```java
JavaSparkContext sc = ... // Spark context to use;

// Create and configure AdaBoost.MH learner. For MP-Boost, just use the class
// MPBoostLearner.
AdaBoostMHLearner learner = new AdaBoostMHLearner(sc);
learner.setNumIterations(numIterations);
learner.setParallelismDegree(parallelismDegree);

// Build a new classifier. Here we assume that the training data is available in
// the input file which is written in LibSvm format.
BoostClassifier classifier = learner.buildModel(inputFile, labels0Based, binaryProblem);

// Save classifier in outputModelPath using the any valid syntax allowed by Spark/Hadoop.
DataUtils.saveModel(sc, classifier, outputModelPath);
```
Alternatively, you can build a new classifier by specifying directly the set of training documents
to use:
```java
JavaSparkContext sc = ... // Spark context to use;

// Create and configure AdaBoost.MH learner. For MP-Boost, just use the class
// MPBoostLearner.
AdaBoostMHLearner learner = new AdaBoostMHLearner(sc);
learner.setNumIterations(numIterations);
learner.setParallelismDegree(parallelismDegree);

// You can prepare yourself the training data by generating an RDD with items
// of type MultilabelPoint.
JavaRDD<MultilabelPoint> trainingData = ...

// Build a new classifier.
BoostClassifier classifier = learner.buildModel(trainingData);

// Save classifier in outputModelPath using any valid syntax allowed by Spark/Hadoop.
DataUtils.saveModel(sc, classifier, outputModelPath);
```

To load a saved classifier and use it for classification tasks, use the following code:
```java
JavaSparkContext sc = ... // Spark context to use;

// Load boosting classifier from disk.
BoostClassifier classifier = DataUtils.loadModel(sc, inputModel);

// Classify documents contained in "inputFile", a file in libsvm format.
ClassificationResults results = classifier.classify(sc, inputFile, parallelismDegree, labels0Based, binaryProblem);

// or classify documents available in already defined RDD.
JavaRDD<MultilabelPoint> rdd = ...
results = classifier.classify(sc, rdd, parallelismDegree);

// Print results in a StringBuilder.
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
```

## Software compilation for latest snapshot
If you are interested in using the latest snapshot of the software, you need to have [Maven](https://maven.apache.org/) and a Java 8 compiler installed on your machine. Download a copy of this software repository from 'develop' branch onto your machine on a specific folder, go inside that folder and at the command prompt put the following commands:
```
mvn clean
mvn -P devel package
```
This set of commands will build a software bundle containing all the necessary Spark libraries. You can find the software bundle in the `target` directory of the software package.
