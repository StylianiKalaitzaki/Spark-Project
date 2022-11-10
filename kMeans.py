import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
from math import sqrt

#0 for random, 1 for kmeans||, default: kmeans||
i = int(sys.argv[1]) if len(sys.argv) > 1 else 1
initialization=['random','k-means||']

spark = SparkSession.builder.appName("hw2").getOrCreate()
# Loads data.
dataframe_csv  = spark.read.csv("hdfs://master:9000/user/user/meddata2022.csv")

df2=dataframe_csv.select(*(col(c).cast(DoubleType()).alias(c) for c in dataframe_csv.columns))

assemble=VectorAssembler(inputCols=[df2.columns[0], df2.columns[1], df2.columns[2], df2.columns[3],df2.columns[4]], outputCol='features')

assembled_data=assemble.transform(df2)


scale=StandardScaler(inputCol='features',outputCol='standardized')
data_scale=scale.fit(assembled_data)
data_scale_output=data_scale.transform(assembled_data)

evaluator = ClusteringEvaluator(predictionCol='prediction',featuresCol='standardized', metricName='silhouette',
distanceMeasure='squaredEuclidean')

# Apply KMEANS
KMeans_algo=KMeans(featuresCol='standardized', k=6,initMode=initialization[i])
if(i==0):
   KMeans_algo.setSeed(17)
model=KMeans_algo.fit(data_scale_output)
output=model.transform(data_scale_output)
centers = model.clusterCenters()
print(KMeans_algo.getInitMode()," initialization")

#print centers
print("Cluster Centers: ")
for center in centers:
	print(center)

#μετρική περιγράμματος
score=evaluator.evaluate(output)
print("Silhouette Score:",score)

#cluster size
summary = model.summary
print("Cluster sizes:",summary.clusterSizes)
#sse
print("Sum of squared distances :",summary.trainingCost)

spark.stop()
