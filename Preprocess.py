import sys
from pyspark import SparkContext, SparkConf, sql
from pyspark.ml.classification import LogisticRegression
if __name__ == "__main__":
    # create Spark context with Spark configuration
    conf = SparkConf().setAppName("Practica 4 - O.J.F.")
    sc = SparkContext(conf=conf)
    sqlContext = sql.SQLContext(sc)

    headerFile = sc.textFile("/user/datasets/ecbdl14/ECBDL14_IR2.header").collect()
    headerFiltered = filter(lambda line: line.contains("@attribute"),headerFile)
    headers = list(map(lambda line: line.split()[1], headerFiltered))
    print(headers)

    sc.stop()

    #df = sc.read.csv("/user/datasets/ecbdl14/ECBDL14_IR2.data",header=False,sep=",",inferSchema=True)
    #df.show()
    #df.createOrReplaceTempView("sql_dataset")
    #sqlDF = sc.sql("SELECT campo1, camp3, ... c6 FROM sql_dataset LIMIT 12")
    #sqlDF.show()
    #lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    #lrModel = lr.fit(sqlDF)
    #lrModel.summary()