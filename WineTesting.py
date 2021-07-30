import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

if __name__ == "__main__":
    session=SparkSession.builder.appName("wineTesting").getOrCreate()
    sc=session.sparkContext
    path  = sys.argv[1]
   

    #load the validation dataset
    val = session.read.format("csv").load(path, header = True , sep=";")
    val.printSchema()

    
    #change 'quality' column name to 'label'
    for col_name in val.columns[1:-1]+['""""quality"""""']:
        val = val.withColumn(col_name, col(col_name).cast('float'))
    val = val.withColumnRenamed('""""quality"""""', "Qualitylabel")

    #Converting features and label into numpy array
    features =np.array(val.select(val.columns[1:-1]).collect())
    Qualitylabel = np.array(val.select('Qualitylabel').collect())

    #creating the feature vector
    VectorAssembler = VectorAssembler(inputCols = val.columns[1:-1] , outputCol = 'features')
    df_tr = VectorAssembler.transform(val)
    df_tr = df_tr.select(['features','Qualitylabel'])

    #The following function creates the labeledpoint and parallelize it to convert it into RDD
    def to_labeled_point(sc, features, labels, categorical=False):
        labeled_points = []
        for x, y in zip(features, labels):        
            lp = LabeledPoint(y, x)
            labeled_points.append(lp)
        return sc.parallelize(labeled_points) 

    #rdd converted dataset
    dataset = to_labeled_point(sc, features, Qualitylabel)

    #load the model from s3
    RFModel = RandomForestModel.load(sc, "s3://aws-logs-444558833244-us-east-1/elasticmapreduce/Nishi/trainingmodel.model/")

    print("model loaded successfully")
    predictions = RFModel.predict(dataset.map(lambda x: x.features))

    #get RDD of label and predictions
    labelsAndPredictions = dataset.map(lambda lp: lp.label).zip(predictions)
 
    labelsAndPredictions_df = labelsAndPredictions.toDF()
    #convert rdd to spark dataframe to pandas dataframe 
    labelpred = labelsAndPredictions.toDF(["Qualitylabel", "Prediction"])
    labelpred.show()
    labelpred_df = labelpred.toPandas()


    #Calculating the F1score
    F1score = f1_score(labelpred_df['Qualitylabel'], labelpred_df['Prediction'], average='micro')
    print("F1- score: ", F1score)
    print("Accuracy" , accuracy_score(labelpred_df['Qualitylabel'], labelpred_df['Prediction']))

    #calculating the test error
    testError = labelsAndPredictions.filter(
        lambda lp: lp[0] != lp[1]).count() / float(dataset.count())    
    print('Test Error = ' + str(testError))
    print("F1- score: ", F1score)
