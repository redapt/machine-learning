# Credit Card Fraud Detection dataset
Download the dataset from [here](https://www.kaggle.com/dalpozz/creditcardfraud)

* Organize dataset:
```
$ wget https://www.kaggle.com/dalpozz/creditcardfraud/downloads/creditcard.csv.zip
$ unzip creditcard.csv.zip
$ mv creditcard.csv creditcard-train.csv
$ cut -d, -f1-30 creditcard-train.csv > creditcard-test.csv
```
