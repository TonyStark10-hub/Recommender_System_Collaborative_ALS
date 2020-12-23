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

#using pre defined ALS parameters to try and find root mean square error with rank=4
als = ALS(maxIter=10, regParam=0.1, rank=4, userCol="user_id", itemCol="book_id", ratingCol="rating")
model = als.fit(training_df)
predictions = model.transform(validation_df)
new_predictions = predictions.filter(col('prediction') != np.nan)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(new_predictions)
print("Root-mean-square error for rank(4)= " + str(rmse))

#using pre defined ALS parameters to try and find root mean square error with rank=5
als = ALS(maxIter=10, regParam=0.1, rank=5, userCol="user_id", itemCol="book_id", ratingCol="rating")
model = als.fit(training_df)
predictions = model.transform(validation_df)
new_predictions = predictions.filter(col('prediction') != np.nan)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(new_predictions)
print("Root-mean-square error for rank(5)= " + str(rmse))


als=ALS(maxIter=10,regParam=Regularization_parameter,rank=rank,userCol='user_id',itemCol='book_id',ratingCol='rating')

#selecting most efficient parameter selection for ALS algorithm using ML CrossValidator

paramGrid=ParamGridBuilder()\
    .addGrid(als.regParam,[0.1,0.01,0.18])\
    .addGrid(als.rank,range(4,7))\
    .build()

evaluator=RegressionEvaluator(matrixName='rmse',lableCol='rating',predictionCol='prediction')
crossval=CrossValidator(estimator=als,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=5)

#creating a model with optimal parameters on training data frame

final_model=crossval.fit(training_df)
final_model_pred=final_model.transform(validation_df)
final_model_pred=final_model_pred.filter(col('prediction') != np.nan)

final_model_pred.show(5)

#join the books df with our prediction model with selecting useful rows
final_model_pred.join(books_df,'book_id').select('user_id','title','prediction').show(5)

#get the user id from user and reccomend 10 top books for that particular user

user=input('Enter your user ID :')
userRecommendations=final_model.recommendForUserSubset(user,10)
recommendedBooks=list(userRecommendations.select('recommendations,book_id'))

print('TOP 10 recommended books for you')
for bookid in recommendedBooks:
    bookdata=books_df.filter(col('book_id')==bookid)
    print(bookdata[10])
    display(Image(url=bookdata[20]))




