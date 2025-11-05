import os
from pyspark.sql import SparkSession
import subprocess

os.environ["JAVA_HOME"] = "C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.17.10-hotspot"
os.environ["SPARK_HOME"] = "C:\\dev\\spark\\spark-3.5.1-bin-hadoop3"
os.environ["HADOOP_HOME"] = os.environ["SPARK_HOME"]

os.environ["PATH"] = os.pathsep.join([
    os.path.join(os.environ["JAVA_HOME"], "bin"),
    os.path.join(os.environ["SPARK_HOME"], "bin"),
    os.environ["PATH"]
])

print("JAVA_HOME:", os.environ["JAVA_HOME"])
print("SPARK_HOME:", os.environ["SPARK_HOME"])
print("Probando Java:")
subprocess.run(["java", "-version"])

spark = SparkSession.builder \
    .appName("ParklyticsTest") \
    .master("local[*]") \
    .getOrCreate()

print("✅ SparkSession creada correctamente.")
print(f"Versión de Spark: {spark.version}")

spark.stop()
