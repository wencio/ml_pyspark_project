import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
df_pandas = pd.concat([X, y], axis=1)

# Create Spark session
spark = SparkSession.builder \
    .appName("PySpark MLlib Example") \
    .getOrCreate()

# create a Spark dataframe
df = spark.createDataFrame(df_pandas)

# Show initial data
df.show(5)

# Index labels (convert categorical labels to numerical)
indexer = StringIndexer(inputCol="species", outputCol="label")
df = indexer.fit(df).transform(df)

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="features")
df = assembler.transform(df)

# Select only necessary columns
df = df.select("features", "label")
df.show(5)

# Split the data
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

# Initialize logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Fit the model
lr_model = lr.fit(train_data)

# Make predictions on the test set
predictions = lr_model.transform(test_data)

# Show predictions
predictions.select("features", "label", "prediction").show(5)

# Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# Evaluate F1 score
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"F1 Score: {f1_score}")
