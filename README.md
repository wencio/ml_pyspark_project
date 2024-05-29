

## Overview

This project demonstrates the implementation of a machine learning workflow using PySpark MLlib. The workflow includes fetching the Iris dataset, preprocessing the data, training a logistic regression model, and evaluating its performance.

## Prerequisites

To run this project, you need to have the following installed on your machine:
- Python (3.6 or later)
- Apache Spark (3.x)
- PySpark
- Pandas
- UCI Machine Learning Repository Python client (`ucimlrepo`)

### Installing Dependencies

You can install the necessary Python packages using pip:

```bash
pip install pyspark pandas ucimlrepo
```

## Project Structure

- `ml_pyspark_project.py`: The main Python script that runs the PySpark MLlib workflow.

## Dataset

The Iris dataset is fetched from the UCI Machine Learning Repository using the `ucimlrepo` Python client.

## Running the Project

1. **Fetch the Dataset**:
   The script uses the `fetch_ucirepo` function to fetch the Iris dataset and load it into pandas DataFrames.

2. **Create a Spark Session**:
   A Spark session is created to facilitate the data processing and machine learning tasks.

3. **Preprocess the Data**:
   - Convert the pandas DataFrames to a Spark DataFrame.
   - Index the labels (convert categorical labels to numerical).
   - Assemble features into a single vector.

4. **Train-Test Split**:
   The data is split into training and testing sets.

5. **Train the Model**:
   A logistic regression model is initialized and trained on the training set.

6. **Make Predictions**:
   The model makes predictions on the test set.

7. **Evaluate the Model**:
   The model's performance is evaluated using accuracy and F1 score.

### Example Code

Here's the complete code for the project:

```python
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from ucimlrepo import fetch_ucirepo

# Fetch dataset
iris = fetch_ucirepo(id=53)

# Data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

df_pandas = pd.concat([X, y], axis=1)

# Create Spark session
spark = SparkSession.builder \
    .appName("PySpark MLlib Example") \
    .getOrCreate()

# Create a Spark dataframe
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
```

### Steps to Execute

1. **Clone the Repository**:
   If you have the script in a GitHub repository, clone it using:
   ```bash
   git clone https://github.com/yourusername/pyspark-mllib-example.git
   cd pyspark-mllib-example
   ```

2. **Run the Script**:
   Execute the Python script:
   ```bash
   python ml_pyspark_project.py
   ```

## Conclusion

This project showcases how to use PySpark MLlib to build and evaluate a machine learning model. The workflow includes fetching the Iris dataset, preprocessing the data, training a logistic regression model, and evaluating the model's performance using accuracy and F1 score.
