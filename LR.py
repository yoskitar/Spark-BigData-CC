import sys
from pyspark import SparkContext, SparkConf, sql
from pyspark.ml.classification import LogisticRegression
from functools import reduce

if __name__ == "__main__":
    # create Spark context with Spark configuration
    conf = SparkConf().setAppName("Practica 4 - O.J.F.")
    sc = SparkContext(conf=conf)
    sqlContext = sql.SQLContext(sc)
    # Leer csv con columnas filtradas
    df_columns = sqlContext.read.csv("./filteredC.small.training", header=True, sep=",", inferSchema=True)
    c0_count = df_columns.filter(df_columns['class']==0).count()
    c1_count = df_columns.filter(df_columns['class']==1).count()
    #print('Class 0 count: ' + str(c0_count))
    #print('Class 1 count: ' + str(c1_count))
    tam_partition = c1_count
    if(c0_count < c1_count):
        tam_partition = c0_count
    # Nos quedamos con el 80% para train y el resto para test
    tam_partition_train = tam_partition*80/100
    tam_partition_test = tam_partition - tam_partition_train
    # Componemos el DF  de train balanceado 
    df_0 = df_columns.filter(df_columns['class']==0).limit(tam_partition_train)
    df_1 = df_columns.filter(df_columns['class']==1).limit(tam_partition_train)

    prueba = df_columns.filter(df_columns['class']==0).limit(10)
    prueba.show()
    prueba.sample(False, 0.2, 5).show()
    #print('DF0 count: ' + str(df_0.select('class').count()))
    #print('DF1 count: ' + str(df_1.select('class').count()))

    #df_columns.createOrReplaceTempView("sql_dataset_columns")
    #sqlDF_0 = sqlContext.sql('SELECT * FROM sql_dataset_columns WHERE class==0 LIMIT 1000')
    #sqlDF_1 = sqlContext.sql('SELECT * FROM sql_dataset_columns WHERE class==1 LIMIT 1000')
 
    #lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    #lrModel = lr.fit(sqlDF)
    #lrModel.summary()
    sc.stop()