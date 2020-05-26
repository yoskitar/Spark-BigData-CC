import sys
from pyspark import SparkContext, SparkConf, sql
from pyspark.ml.classification import LogisticRegression
from functools import reduce

if __name__ == "__main__":
    # create Spark context with Spark configuration
    conf = SparkConf().setAppName("Practica 4 - O.J.F.")
    sc = SparkContext(conf=conf)
    sqlContext = sql.SQLContext(sc)

    # Obtenemos una lista con las cabeceras para poder componer 
    # el DF posteriormente y seleccionar las que nos interesen.
    headerFile = sc.textFile("/user/datasets/ecbdl14/ECBDL14_IR2.header").collect()
    headerFiltered = filter(lambda line: "@attribute" in line ,headerFile)
    headers = list(map(lambda line: line.split()[1], headerFiltered))
    
    # Leemos el DF con los datos y renombramos las columnas.
    df = sqlContext.read.csv("/user/datasets/ecbdl14/ECBDL14_IR2.data",header=False,sep=",",inferSchema=True)

    dfRenamed = reduce(lambda data, idx: data.withColumnRenamed(df.schema.names[idx], headers[idx]), range(len(df.schema.names)), df)
    dfRenamed.createOrReplaceTempView("sql_dataset")
    sqlDF = sqlContext.sql("SELECT PredSA_freq_global_0, 'PredSA_central_-2', PSSM_r1_3_V, PSSM_r1_2_I, PSSM_r1_2_W, 'PSSM_r2_-4_Y', class FROM sql_dataset LIMIT 10")
    sqlDF.show()
    #lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    #lrModel = lr.fit(sqlDF)
    #lrModel.summary()


    sc.stop()