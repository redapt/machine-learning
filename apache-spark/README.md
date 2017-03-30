# Apache Spark

This article will describe what Apache Spark is and provide examples of how to use it.

__Apache Spark__ is an open-source cluster-computing framework. Originally developed at the University of California, Berkeley's AMPLab, the Spark codebase was later donated to the Apache Software Foundation, which has maintained it since. Spark provides an interface for programming entire clusters with implicit data parallelism and fault-tolerance.

## Intoduction

* What is Spark?
  * Fast and general engine for large-scale data processing and analysis.
  * Parallel distributed processing on commodity hardware.
  * Easy to use.
  * A comprehensive, unified framework for Big Data analytics.
  * Open source and a top-level Apache project.

* Spark use cases
  * Big Data use case like intrusion detection, product recommendations, estimating financial risks, detecting genomic associations to a disease, etc. require analysis on large-scale data.
  * In depth analysis requires a combination of tools like SQL, statistics, and machine learning to gain meaningful insights from data.
  * Historical choices like R and Octave operate only on single node machines and are, therefore, not suitable for large volume data sets.
  * Spark allows rich programming APIs like SQL, machine learning, graph processing, etc. to run on clusters of computers to achieve large-scale data processing and analysis.

* Spark and distributed processing
  * Challenges of distributed processing:
    * Distributed programming is much more complex than single node programming.
    * Data must be partitioned across servers, increasing the latency if data has to be shared between servers over the network.
    * Chances of failure increases with the increase in the number of servers.
  * Spark makes distributed processing easy:
    * Provides a distributed and parallel processing framework.
    * Provides scalability.
    * Provides fault-tolerance.
    * Provides a programming paradigm that makes it easy to write code in a parallel manner.

* Spark and its speed
  * Lightning fast speeds due to in-memory caching and a [DAG-based](https://en.wikipedia.org/wiki/Directed_acyclic_graph) processing engine.
  * 100 times faster than Hadoop's MapReduce for in-memory computations and 10 time faster for on-disk.
  * Well suited for iterative algorithms in machine learning
  * Fast, real-time response to user queries on large in-memory data sets.
  * Low latency data analysis applied to processing live data streams

* Spark is easy to use
  * General purpose programming model using expressive languages like Scala, Python, and Java.
  * Existing libraries and API makes it easy to write programs combining batch, streaming, interactive machine learning and complex queries in a single application.
  * An interactive shell is available for Python and Scala
  * Built for performance and reliability, written in Scala and runs on top of Java Virtual Machine (JVM).
  * Operational and debugging tools from the Java stack are available for programmers.

* Spark components
  * Spark SQL
  * Spark Streaming
  * MLib (machine learning)
  * GraphX (graph)
  * SparkR

* Spark is a comprehensive unified framework for Big Data analytics
  * Collapses the data science pipeline by allowing pre-processing of data to model evaluation in one single system.
  * Spark provides an API for data munging, [Extract, transform, load](https://en.wikipedia.org/wiki/Extract,_transform,_load) (ETL), machine learning, graph processing, streaming, interactive, and batch processing. Can replace several SQL, streaming, and complex analytics systems with one unified environment.
  * Simplifies application development, deployment, and maintenance.
  * Strong integration with a variety of tools in the Hadoop ecosystem.
  * Can read and write to different data formats and data sources, including HDFS, Cassandra, S3, and HBase.

* Spark is _not_ a data storage system
  * Spark is _not_ a data store but is versatile in reading from and writing to a variety of data sources.
  * Can access traditional business intelligence (BI) tools using a server mode that provides standard [Java Database Connectivity](https://en.wikipedia.org/wiki/Java_Database_Connectivity) (JDBC) and [Open Database Connectivity](https://en.wikipedia.org/wiki/Open_Database_Connectivity) (ODBC).
  * The DataFrame API provides a pluggable mechanism to access structured data using Spark SQL.
  * Its API provides tight optimization integration, thereby enhances the speed of the Spark jobs that process vast amounts of data.

* History of Spark
  * Originated as a research project in 2009 at [UC Berkley](https://amplab.cs.berkeley.edu/).
  * Motivated by MapReduce and the need to apply machine learning in a scalable fashion.
  * Open sourced in 2010 and transferred to Apache in 2013.
  * A top-level Apache project, as of 2017.
  * Spark is winner of Daytona GraySort contesting 2014, sorting a petabyte 3 times faster and using 10 times less hardware than Hadoop's MapReduce.
  * "Apache Spark is the Taylor Swift of big data software. The open source technology has been around and popular for a few years. But 2015 was the year Spark went from an ascendant technology to a bona fide superstar". [Reference](http://fortune.com/2015/09/25/apache-spark-survey/).

* Spark use cases
  * Fraud detection: Spark streaming an machine learning applied to prevent fraud.
  * Network intrusion detection: Machine learning applied to detect cyber hacks.
  * Customer segmentation and personalization: Spark SQL and machine learning applied to maximize customer lifetime value.
  * Social media sentiment analysis: Spark streaming, Spark SQL, and Stanford's CoreNLP wrapper helps achieve sentiment analysis.
  * Real-time ad targeting: Spark used to maximize Online ad revenues.
  * Predictive healthcare: Spark used to optimize healthcare costs.

## Spark basics

The official Apache Spark website provides detailed documentation on [how to install Apache Spark](https://spark.apache.org/docs/latest/). In this article, I will assume you already have Spark installed. However, if you wish, I have also provided a [Dockerfile](Dockerfile) that will build a Docker image of everything you will need to start a Docker container running Apache Spark, along with some other tools and packages used in this article (e.g., R, iPython, Pandas, NumPy, etc.).

To build a Docker image from the provided Dockerfile, run the following command:
```
$ docker build -t christophchamp/spark:v2.1.0 - < Dockerfile
```
(note: rename the image to whatever you like)

Then start the Docker container from the built image with:
```
$ docker run -it --name apache_spark -v $(pwd):/data christophchamp/spark:v2.1.0 /bin/bash
```
_Note: The above container will have access to files in the directory the docker command was run from and mounted at `/data`._

Apache Spark will be installed under `/spark` and all of Spark's binaries/scripts will be in the user path. As such, you can start, say, `pyspark` by simply executing `pyspark` from your current working directory.

```
$ echo $PATH
/spark/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

### A simple example using Apache Spark MLlib

Let's pretend we have a very simple dataset where we want to do some basic text document classification. Our dataset consists of a CSV file with one text document per line in the file. Our example dataset is just lines that either contain the word "spark" or do not. This will be the _target_ of our MLlib training and will be what we use to create our ML model. Our training dataset has three columns: document ID (`id`), the text of the document (`text`), and a `label` value of 0.0 or 1.0 for whether or not the line has our _target_ word "spark" or not.
```
$ cat train.csv 
id,text,label
0,"a b c d e spark",1.0
1,"b d",0.0
2,"spark f g h",1.0
3,"hadoop mapreduce",0.0
```

Our test dataset (i.e., the one we will use to test how accurate our ML model is at predicting which lines contain our "spark" word) has the same structure as our training dataset, except there are different text documents (aka "lines") and the `label` column (i.e., whether or not the text document contains the "spark" word or not) is not included, as this is what we are trying to predict.
```
$ cat test.csv 
id,text
4,"spark i j k"
5,"l m n"
6,"spark hadoop spark"
7,"apache hadoop"
```

The text document classification pipeline we will use has the following workflow:
* Training workflow: Input is a set of text documents, where each document is labelled. Stages while training the ML model are:
  * Split each text document (or lines in our `trains.csv` file) into words;
  * Convert each document's words into a numerical feature vector; and
  * Create a prediction model using the feature vectors and labels.
* Test/prediction workflow: Input is a set of text documents and the goal is to predict a label for each document. Stages while testing or making predictions with the ML model are:
  * Split each text document (or lines in our `test.csv` file) into words;
  * Convert each document's words into a numerical feature vector; and
  * Use the trained model to make predictions on the feature vector.

For our simple example, we will use the LogisticRegression algorithm. We do not need to use this algorithem, however, it is one of the simplest to use, so we will start with this (I will provide examples of more complex algorithms later on).


A tokenizer that converts the input string to lowercase and then splits it by white spaces.

Start up the pyspark REPL:
```
$ pyspark
```

_Note: The following commands will be run from within the pyspark REPL._

```
# Note: This is based off of the
# examples/src/main/python/ml/pipeline_example.py example script in the Spark
# tarball.
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession
import pandas as pd

# Initialize a Spark session
spark = SparkSession\
    .builder\
    .appName("LogisticRegressionExample")\
    .getOrCreate()

# Create DataFrame from training documents
train_df = pd.read_csv("/data/train.csv")
training = spark.createDataFrame(train_df)

# Configure an ML pipeline, which consists of three stages: tokenizer,
# hashingTF, and lr
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# Fit the pipeline to training documents
model = pipeline.fit(training)

# Create DataFrame from test documents
test_df = pd.read_csv("/data/test.csv")
test = spark.createDataFrame(test_df)

# Make predictions on test documents and print columns of interest
prediction = model.transform(test)
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    rid, text, prob, prediction = row 
    print("(%d, %s) --> prob=%s, prediction=%f" % (
        rid, text, str(prob), prediction))

spark.stop()
```

The above simple script should return the following for Spark MLlib predictions for the test documents:

```
(4, spark i j k) --> prob=[0.159640773879,0.840359226121], prediction=1.000000
(5, l m n) --> prob=[0.837832568548,0.162167431452], prediction=0.000000
(6, spark hadoop spark) --> prob=[0.0692663313298,0.93073366867], prediction=1.000000
(7, apache hadoop) --> prob=[0.982157533344,0.0178424666556], prediction=0.000000
```

So, for this _very simple_ example, Spark MLlib made perfect predictions (i.e., all documents with the word "spark" in them were correctly labeled).

### A more complex text classification example

In this example, I will use some of the data found on the [Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/) dataset (aka the "sentiment polarity dataset") by Pang and Lee from Cornell University (July 2005).

The actual dataset I will use is the "[sentence polarity dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)". This data was first used in the paper by Bo Pang and Lillian Lee (2005). "Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales". _Proceedings of the ACL_.

Each line in the dataset (there are two files in the dataset: 1) positive sentiments; and 2) negative sentiments) corresponds to a single snippet (usually containing roughly one sentence) taken from the [Rotten Tomatoes](https://www.rottentomatoes.com/) website for movie reviews. Reviews marked as "fresh" are assumed to be positive, and those for reviews marked as "rotten" are negative. All text in the snippets (i.e., each line of text) is lowercased and spaces are inserted around spaces. There are 5,331 positive and 5,331 negative processed sentences / snippets in the dataset.

* Examples of negative sentiment lines:
```
simplistic , silly and tedious .
the story is also as unoriginal as they come , already having been recycled more times than i'd care to count .
unfortunately the story and the actors are served with a hack script .
```

* Examples of positive sentiment lines:
```
take care of my cat offers a refreshingly different slice of asian cinema .
offers that rare combination of entertainment and education .
if this movie were a book , it would be a page-turner , you can't wait to see what happens next .
```

The pyspark script for performing text analysis using a [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) via Spark MLlib:
```
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from sklearn.metrics import classification_report


def load_dataset(posfile, negfile):
    """Load positive and negative sentences give dataset.
    Let 0 = negative class; 1 = positive class
    Tokenize sentences and transform them into vector space model.
    """

    # Word-to-vector space converter (limit to 10000 words)
    htf = HashingTF(10000)

    positiveData = sc.textFile(posfile)
    posdata = positiveData.map(
        lambda text: LabeledPoint(1, htf.transform(text.split(" "))))
    posdata.persist()

    negativeData = sc.textFile(negfile)
    negdata = negativeData.map(
        lambda text: LabeledPoint(0, htf.transform(text.split(" "))))
    negdata.persist()

    return posdata, negdata


def naive_bayes_classifier(posdata, negdata):
    """Split, train, and calculate prediction labels."""

    # Split positive and negative data 60/40 into training and test data sets
    ptrain, ptest = posdata.randomSplit([0.6, 0.4])
    ntrain, ntest = negdata.randomSplit([0.6, 0.4])

    # Union train data with positive and negative sentences
    trainh = ptrain.union(ntrain)
    # Union test data with positive and negative sentences
    testh = ptest.union(ntest)

    # Train a Naive Bayes model on the training data
    model = NaiveBayes.train(trainh)

    # Compare predicted labels to actual labels
    prediction_and_labels = testh.map(
        lambda point: (model.predict(point.features), point.label))

    # Filter to only correct predictions
    correct = prediction_and_labels.filter(
        lambda (predicted, actual): predicted == actual)

    # Calculate and return accuracy rate
    accuracy = correct.count() / float(testh.count())

    return prediction_and_labels, accuracy


def gen_classification_report(prediction_and_labels):
    """Classification Report using Scikit-Learn.
    SEE: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    y_true = []
    y_pred = []
    for x in prediction_and_labels.collect():
        xx = list(x)
        try:
            tt = int(xx[1])
            pp = int(xx[0])
            y_true.append(tt)
            y_pred.append(pp)
        except:
            continue

    target_names = ['neg 0', 'pos 1']

    return classification_report(y_true, y_pred, target_names=target_names)


if __name__ == "__main__":

    conf = SparkConf().setMaster("local[*]").setAppName("Naive_Bayes")
    sc = SparkContext(conf=conf)

    print "Running Spark Version %s" % (sc.version)

    posdata, negdata = load_dataset(
        "/data/rt-polarity.pos", "/data/rt-polarity.neg")
    print "No. of Positive Sentences: " + str(posdata.count())
    print "No. of Negative Sentences: " + str(negdata.count())

    prediction_and_labels, accuracy = naive_bayes_classifier(posdata, negdata)
    msg = "Classifier correctly predicted category "
    msg += str(accuracy * 100)
    msg += " percent of the time"
    print msg

    report = gen_classification_report(prediction_and_labels)
    print report
```

The results of running the above script:
```
Running Spark Version 2.1.0
No. of Positive Sentences: 5331
No. of Negative Sentences: 5331
Classifier correctly predicted category 73.3380314776 percent of the time
             precision    recall  f1-score   support

      neg 0       0.74      0.73      0.74      2143
      pos 1       0.73      0.73      0.73      2114

avg / total       0.73      0.73      0.73      4257
```

We can see that our Naive Bayes classifier correctly predicted ~73% of the sentiments (either positive or negative) from the dataset.
