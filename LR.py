from pyspark import SparkContext, SparkConf, sql
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Funcion para realizar el entrenamiento indicando el grid de parametros
# junto al modelo, la metrica de evaluacion, que por defecto en nuestro 
# caso caso sera AUC, y el porcentaje del conjunto de datos destinados 
# a entrenamiento y validacion.
def TVS(estimator, paramGrid, dataTrain, dataTest):
    # Definimos el TVS
    tvs = TrainValidationSplit(estimator=estimator,
                           estimatorParamMaps=paramGrid,
                           evaluator=BinaryClassificationEvaluator(),
                           # 80% entrenamiento, 20% validacion
                           trainRatio=0.8)
    # Entrenamos el modelo con la mejor combinacion 
    # de parametros del grid por defecto
    model = tvs.fit(dataTrain)
    # Obtenemos predicciones sobre Test
    predictions = model.transform(dataTest)
    return predictions, model

# Funcion para mostrar las parametrizaciones de cada parametrizacion del 
# modelo y el resultado obtenido para el conjunto de validacion.
def printStagesResults(model):
    # Para cada parametrizacion:
    for idx, stage in enumerate(model.getEstimatorParamMaps()):
        # La posicion de la metrica en el vector coincide con la posicion de la 
        # parametrizacion, por lo que accedemos al mismo indice e imprimimos el valor.
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
    
    # Analizamos el desbalanceo de clases
    c0_count = df_columns.filter(df_columns['class']==0).count()
    c1_count = df_columns.filter(df_columns['class']==1).count()

    # Realizamos un undersampling, quedandonos con el tamaño 
    # para cada clase relativo al de la clase en menor proporcion
    tam_partition = c1_count
    if(c0_count < c1_count):
        tam_partition = c0_count

    # Componemos el DF  de train balanceado 
    df_0 = df_columns.filter(df_columns['class']==0).limit(tam_partition)
    df_1 = df_columns.filter(df_columns['class']==1).limit(tam_partition)

    # Unimos las clases y hacemos un shuffle
    df_balanced = df_0.union(df_1)

    # Usaremos el 90% para train y el 10% restante para test
    df_train, df_test = df_balanced.randomSplit([0.9, 0.1], seed=12345)

    # Obtenemos las proporciones del dataset balanceado
    df_balanced_count = df_balanced.select('class').count()
    # Train count
    df_train_count = df_train.select('class').count()
    df_train_1_count = df_train.filter(df_columns['class']==1).select('class').count()
    df_train_0_count = df_train.filter(df_columns['class']==0).select('class').count()
    # Test count
    df_test_count = df_test.select('class').count()
    df_test_1_count = df_test.filter(df_columns['class']==1).select('class').count()
    df_test_0_count = df_test.filter(df_columns['class']==0).select('class').count()

    # Preparamos el DF para aplicar los algoritmos de MLLib,
    # añadiendo una nueva columna como vector de caracteristicas
    assembler = VectorAssembler(inputCols=["PredSA_freq_global_0", "PredSA_central_-2", "PSSM_r1_3_V", "PSSM_r1_2_I", "PSSM_r1_2_W", "PSSM_r2_-4_Y"], outputCol='features')
    trainingData = assembler.transform(df_train).select("features","class").withColumnRenamed("class","label")
    testData = assembler.transform(df_test).select("features","class").withColumnRenamed("class","label")

    # Logistic Regression Model
    lr = LogisticRegression(maxIter=10)
    paramGridLR = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01, 0.3]).addGrid(lr.fitIntercept, [False, True]).addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]).build()
    predictionsLR, mLR = TVS(lr,paramGridLR,trainingData,testData)
    #predictions.select("features", "label", "prediction").show(100)
    
    # Random Forest Model
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=12345)
    paramGridRF = ParamGridBuilder().addGrid(rf.numTrees, [10, 30, 60]).addGrid(rf.maxDepth, [3, 6, 12]).build()

    # Gradient-boosted tree Model
    gbt = GBTClassifier(labelCol="label", featuresCol="features", seed=12345)
    paramGridGBT = ParamGridBuilder().addGrid(gbt.maxIter, [10, 15, 20]).addGrid(gbt.maxDepth, [3, 6, 12]).build()
    predictionsGBT, mGBT = TVS(gbt,paramGridGBT,trainingData,testData)
    predictionsRF, mRF = TVS(rf,paramGridRF,trainingData,testData)
    
    # Evaluamos los modelos con la metrica AUC por defecto
    evaluator = BinaryClassificationEvaluator()
    auRocLR = evaluator.evaluate(predictionsLR)
    auRocRF = evaluator.evaluate(predictionsRF)
    auRocGBT = evaluator.evaluate(predictionsGBT)

    # Mostramos el tamaño de los conjuntos para documentacion
    print("Dataset desbalanceado: ")
    print('\tClass 0 count: ' + str(c0_count))
    print('\tClass 1 count: ' + str(c1_count))
    print("Dataset balanceado: ")
    print('\tTamaño total: ' + str(df_balanced_count))
    print('\tTrain: ' + str(df_train_count))
    print('\t\tTrain class 0: ' + str(df_train_0_count))
    print('\t\tTrain class 1: ' + str(df_train_1_count))
    print('\tTest: ' + str(df_test_count))
    print('\t\tTest class 0: ' + str(df_test_0_count))
    print('\t\tTest class 1: ' + str(df_test_1_count))

    # Mostramos los resultados de las parametrizaciones
    # junto a metrica AUC de validacion
    printStagesResults(mLR)
    printStagesResults(mRF)
    printStagesResults(mGBT)

    # Resultados sobre el conjunto de test
    print("DF_TEST - Area Under Roc - LR: " + str(auRocLR) )
    print("DF_TEST - Area Under Roc - RF: " + str(auRocRF) )
    print("DF_TEST - Area Under Roc - GBT: " + str(auRocGBT) )
       
    sc.stop()