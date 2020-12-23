import pyspark as ps
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import udf,col,when
import numpy as np

#to show the images
from IPython.display import Image
from IPython.display import display

#creation of spark context and spark session

spark=ps.sql.SparkSession.builder.Master('local[*]').appName('ALS_Recommender').getOrCreate()

sc=spark.SparkContext()
sqlContext=SQLContext(sc)

#Creating Dataframe for ratings of each books

ratings_df=spark.read.csv('RowData/ratings.csv',header=True,inferSchema=True)
#rating_df.printSchema()
#rating_df.show(5)

#Creating Dataframe for book dataset
books_df=spark.read.csv('RawData/books.csv',header=True,inferSchema=True)
#bookss_df.printSchema()
#books_df.show(1)

#partition of rating_df in training and validation df

training_df,validation_df=ratings_df.randomSplit([.8,.2])
 



