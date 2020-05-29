import sys
from pyspark import SparkContext, SparkConf, sql
from pyspark.ml.classification import LogisticRegression
from functools import reduce
from pyspark.ml.linalg import Vectors

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
        
    # Componemos el DF  de train balanceado 
    df_0 = df_columns.filter(df_columns['class']==0).limit(tam_partition)
    df_1 = df_columns.filter(df_columns['class']==1).limit(tam_partition)

    # Unimos las clases y hacemos un shuffle
    df_balanced = df_0.union(df_1)
    # Usaremos el 80% para train y el 20% restante para test
    df_test = df_balanced.sample(False, 0.2, 5)
    df_train = df_balanced.subtract(df_test)
    #print('DF_Balanced count: ' + str(df_balanced.select('class').count()))
    #print('DF_Train count: ' + str(df_train.select('class').count()))
    #print('DF_Test count: ' + str(df_test.select('class').count()))

    print('DF_Train_1 count: ' + str(df_train.filter(df_columns['class']==1).select('class').count()))
    print('DF_Train_0 count: ' + str(df_train.filter(df_columns['class']==0).select('class').count()))

    print('DF_Test_1 count: ' + str(df_test.filter(df_columns['class']==1).select('class').count()))
    print('DF_Test_0 count: ' + str(df_test.filter(df_columns['class']==0).select('class').count()))

    df_train_reduced = df_train.sample(False, 0.01, 5)

    #df_columns.createOrReplaceTempView("sql_dataset_columns")
    #sqlDF_0 = sqlContext.sql('SELECT * FROM sql_dataset_columns WHERE class==0 LIMIT 1000')
    #sqlDF_1 = sqlContext.sql('SELECT * FROM sql_dataset_columns WHERE class==1 LIMIT 1000')
 
    trainingData=df_train_reduced.rdd.map(lambda x:(Vectors.dense(x[0:-1]), x[-1])).toDF(["features", "label"])
    trainingData.show()
    
    sc.stop()