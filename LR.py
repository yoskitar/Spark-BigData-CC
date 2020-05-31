import sys
from pyspark import SparkContext, SparkConf, sql
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from functools import reduce
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def TVS(estimator, paramGrid, dataTrain, dataTest):
    tvs = TrainValidationSplit(estimator=estimator,
                           estimatorParamMaps=paramGrid,
                           evaluator=BinaryClassificationEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)
    model = tvs.fit(dataTrain)
    predictions = model.transform(dataTest)
    return predictions, model

def printResults(model):
    for idx, stage in enumerate(model.getEstimatorParamMaps()):
        print("Stage " + str(idx) + " - AUC: " + str(model.validationMetrics[idx]))
        for param, value in stage.items():
            print("\tParam: " + param.name + " - Value: " +str(value))


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
    #df_test = df_balanced.sample(False, 0.2, 5)
    #df_train = df_balanced.subtract(df_test)

    df_train, df_test = df_balanced.randomSplit([0.9, 0.1], seed=12345)

    #print('DF_Balanced count: ' + str(df_balanced.select('class').count()))
    #print('DF_Train count: ' + str(df_train.select('class').count()))
    #print('DF_Test count: ' + str(df_test.select('class').count()))

    """
    print('DF_Train_1 count: ' + str(df_train.filter(df_columns['class']==1).select('class').count()))
    print('DF_Train_0 count: ' + str(df_train.filter(df_columns['class']==0).select('class').count()))

    print('DF_Test_1 count: ' + str(df_test.filter(df_columns['class']==1).select('class').count()))
    print('DF_Test_0 count: ' + str(df_test.filter(df_columns['class']==0).select('class').count()))

    #df_train_reduced = df_train.sample(False, 1.0, 5)
    """
    # Preparamos el DF para aplicar los algoritmos de MLLib
    assembler = VectorAssembler(inputCols=["PredSA_freq_global_0", "PredSA_central_-2", "PSSM_r1_3_V", "PSSM_r1_2_I", "PSSM_r1_2_W", "PSSM_r2_-4_Y"], outputCol='features')
    trainingData = assembler.transform(df_train).select("features","class").withColumnRenamed("class","label")
    testData = assembler.transform(df_test).select("features","class").withColumnRenamed("class","label")
    """
    # Entrenamos el modelo
    lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8)
    lrModel = lr.fit(trainingData)
    trainingSummary = lrModel.summary

    print("Coefficients: \n" + str(lrModel.coefficientMatrix))
    print("Intercept: " + str(lrModel.interceptVector))
    
    # Obtenemos las metricas del entrenamiento
    trainingSummary.roc.show()
    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))
    
    """
    # Gradient-boosted tree MODEL
    gbt = GBTClassifier(labelCol="label", featuresCol="features", seed=12345)
    paramGridGBT = ParamGridBuilder().addGrid(gbt.maxIter, [10, 15, 20]).addGrid(gbt.maxDepth, [3, 6, 12]).build()
    predictionsGBT, mGBT = TVS(gbt,paramGridGBT,trainingData,testData)

    # Logistic Regression MODEL
    lr = LogisticRegression(maxIter=10)
    paramGridLR = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01, 0.3]).addGrid(lr.fitIntercept, [False, True]).addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]).build()
    predictionsLR, mLR = TVS(lr,paramGridLR,trainingData,testData)
    #predictions.select("features", "label", "prediction").show(100)
    
    # Random Forest MODEL
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=12345)
    paramGridRF = ParamGridBuilder().addGrid(rf.numTrees, [10, 30, 60]).addGrid(rf.maxDepth, [3, 6, 12]).build()
    predictionsRF, mRF = TVS(rf,paramGridRF,trainingData,testData)
    
    # Evaluate model
    evaluator = BinaryClassificationEvaluator()
    auRocLR = evaluator.evaluate(predictionsLR)
    auRocRF = evaluator.evaluate(predictionsRF)
    auRocGBT = evaluator.evaluate(predictionsGBT)

    printResults(mLR)
    printResults(mRF)
    printResults(mGBT)

    print("DF_TEST - Area Under Roc - LR: " + str(auRocLR) )
    print("DF_TEST - Area Under Roc - RF: " + str(auRocRF) )
    print("DF_TEST - Area Under Roc - GBT: " + str(auRocGBT) )
       
    sc.stop()