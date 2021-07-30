

#Loading the python and spark libraries
import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import isnull, when, count, col
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


if __name__ == "__main__":
    session=SparkSession.builder.appName("winequality").getOrCreate()
    sc=session.sparkContext
    df = session.read.format("csv").load("s3://aws-logs-444558833244-us-east-1/elasticmapreduce/Nishi/TrainingDataset.csv" , header = True ,sep =";")
    df.printSchema()
    
    from pyspark.sql.functions import col

    dataset1 = df.select(col('""""alcohol""""').cast('float'),
             col('""""quality"""""').cast('float'),
             col('""""volatile acidity""""').cast('float'),
             col('""""total sulfur dioxide""""').cast('float'),
             col('""""citric acid""""').cast('float'),
             col('""""sulphates""""').cast('float'))

    dataset1.show()
    
    # Feature extraction which features influence quality most
    # See if we have missing values
    from pyspark.sql.functions import isnull, when, count, col

    dataset1.select([count(when(isnull(c), c)).alias(c) for c in dataset1.columns]).show()
    # Drop missing values
    dataset1 = dataset1.replace('null', None).dropna(how='any')
    # Index categorical columns with StringIndexer
    from pyspark.ml.feature import StringIndexer

    # Check data types
    #dataset.dtypes
    # Drop unnecessary columns
    dataset1 = dataset1.drop('""""residual sugar""""')
    dataset1 = dataset1.drop('""""chlorides""""')
    dataset1 = dataset1.drop('""""free sulfur dioxide""""')
    dataset1 = dataset1.drop('""""pH""""')
    # Assemble all the features with VectorAssembler

    required_features = ['""""alcohol""""',
                    '""""volatile acidity""""',
                    '""""total sulfur dioxide""""',
                    '""""citric acid""""',
                    '""""sulphates""""' ]

    #Add a column Qualitylabel derived from 'quality' column
    for col_name in df.columns[1:-1]+['""""quality"""""']:
        df = df.withColumn(col_name, col(col_name).cast('float'))
    df = df.withColumnRenamed('""""quality"""""', "Qualitylabel")


    #Convert features and Qualitylabel into numpy array
    features =np.array(df.select(df.columns[1:-1]).collect())
    Qualitylabel = np.array(df.select('Qualitylabel').collect())

    #create the feature vector using vector assembler
    VectorAssembler = VectorAssembler(inputCols = df.columns[1:-1] , outputCol = 'features')
    df_tr = VectorAssembler.transform(df)
    df_tr = df_tr.select(['features','Qualitylabel'])

    #create the labeledpoint and parallelize it to convert it into RDD
    def to_labeled_point(sc, features, labels, categorical=False):
        labeled_points = []
        for x, y in zip(features, labels):        
            lp = LabeledPoint(y, x)
            labeled_points.append(lp)
        return sc.parallelize(labeled_points) 

    #rdd converted dataset
    dataset = to_labeled_point(sc, features, Qualitylabel)

    #Splitting the dataset into train and test
    train, test = dataset.randomSplit([0.75, 0.25],seed =15)


    #Creating a random forest training classifier
    RFmodel = RandomForest.trainClassifier(train, numClasses=10, categoricalFeaturesInfo={},
                                     numTrees=21, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=30, maxBins=32)

    #Find predictions
    predictions = RFmodel.predict(test.map(lambda x: x.features))

    # RDD of label and predictions
    labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
    labelsAndPredictions_df = labelsAndPredictions.toDF()
    
    #convert rdd to spark dataframe and pandas dataframe 
    labelpred = labelsAndPredictions.toDF(["Qualitylabel", "Prediction"])
    labelpred.show()
    labelpred_df = labelpred.toPandas()


    #Calculating the F1score
    F1score = f1_score(labelpred_df['Qualitylabel'], labelpred_df['Prediction'], average='micro')
    print(confusion_matrix(labelpred_df['Qualitylabel'],labelpred_df['Prediction']))
    print(classification_report(labelpred_df['Qualitylabel'],labelpred_df['Prediction']))
    print("Accuracy" , accuracy_score(labelpred_df['Qualitylabel'], labelpred_df['Prediction']))
    print("F1- score: ", F1score)

#calculating the test error
    testError = labelsAndPredictions.filter(
        lambda lp: lp[0] != lp[1]).count() / float(test.count())    
    print('Test Error = ' + str(testError))
    print("F1- score: of test data ", F1score)

#save the classification training model 
    RFmodel.save(sc, 's3://aws-logs-444558833244-us-east-1/elasticmapreduce/Nishi/trainingmodel.model')
    

   