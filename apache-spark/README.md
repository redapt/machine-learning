# Apache Spark

This article will describe what Apache Spark is and provide examples of how to use it.

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

Let's pretend we have a very simple dataset where we want to do some basic text document classification. Our dataset consists of a CSV file with one text document per line the file. Our example dataset is just lines that either contain the word "spark" or do not. This will be the _target_ of our MLlib training and will be what we use to create our ML model. Our training dataset has three columns: document ID (`id`), the text of the document (`text`), and a `label` value of 0.0 or 1.0 for whether or not the line has our _target_ word "spark" or not.
```
$ cat train.csv 
```
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

Start up the pyspark REPL:
```
$ pyspark
```

_Note: The following commands will be run from within the pyspark REPL._

```
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession
import pandas as pd
spark = SparkSession\
    .builder\
    .appName("PipelineExample")\
    .getOrCreate()

# Create DataFrame from training documents
train_df = pd.read_csv("/data/train.csv")
training = spark.createDataFrame(train_df)

# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr. 
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# Fit the pipeline to training documents.
model = pipeline.fit(training)

# Create DataFrame from test documents
test_df = pd.read_csv("/data/test.csv")
test = spark.createDataFrame(test_df)

# Make predictions on test documents and print columns of interest.
prediction = model.transform(test)
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    rid, text, prob, prediction = row 
    print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))

spark.stop()
```

```
(4, spark i j k) --> prob=[0.159640773879,0.840359226121], prediction=1.000000
(5, l m n) --> prob=[0.837832568548,0.162167431452], prediction=0.000000
(6, spark hadoop spark) --> prob=[0.0692663313298,0.93073366867], prediction=1.000000
(7, apache hadoop) --> prob=[0.982157533344,0.0178424666556], prediction=0.000000
```
