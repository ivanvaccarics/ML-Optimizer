from distutils.command.config import config
import pandas as pd
import numpy as np
from algorithms.randomforest import elaborationrandomforest, optimizationrf
from algorithms.decisiontree import elaborationdecisiontree, optimizationtree
from algorithms.gradientboost import elaborationgradientboost, optimizationgb
from algorithms.logisticregression import elaborationlogisticregression, optimizationlr
from algorithms.knn import elaborationknn, optimizationknn
from algorithms.svm import optimizationsvm,elaborationsvm
from algorithms.commons import calculateroccurve
from config import *
import pathlib
import configparser

CSV_OUTPUT = True
HYPERTUNING = True

# prepare output folder
for folder in FOLDER_LIST:
    path = pathlib.Path(folder)
    path.mkdir(parents=True, exist_ok=True)

# csv output indentation
if CSV_OUTPUT:
    with open(RESULTS_METRICS,'a') as fd:
        fd.write('Algorithm, Optimized, Accuracy, F1 score, FPR, TPR, TNR, FNR, training time (s), test time (s)')


# Import and manipulate your dataset. In case, you can use the function sklearn.model_selection.train_test_split to split a dataset.
#import data
dftrain = pd.read_csv(f"{PATH_TRAIN_DATASET}easa_jsma_7feat_train.csv") 
dftest = pd.read_csv(f"{PATH_TEST_DATASET}easa_jsma_7feat_test.csv")
seed = SEED

#train
dftrain = dftrain.drop('RUL_binary', 1)
print(dftrain)
class_names = dftrain.attack.unique()
#dftrain=dftrain.astype('category')
cat_columns = dftrain.select_dtypes(['category']).columns
dftrain[cat_columns] = dftrain[cat_columns].apply(lambda x: x.cat.codes)
#print(dftrain.loc[125, 'target'])
x_columns = dftrain.columns.drop('attack')
x_train = dftrain[x_columns].values
y_train = dftrain['attack']

#test
dftest = dftest.drop('RUL_binary', 1)
class_names = dftest.attack.unique()
#dftest=dftest.astype('category')
#cat_columns = dftest.select_dtypes(['category']).columns
dftest[cat_columns] = dftest[cat_columns].apply(lambda x: x.cat.codes)
x_columns = dftest.columns.drop('attack')
x_test = dftest[x_columns].values
y_test = dftest['attack']

# From here, the code will perform automatically
print("Ready to generate train and test datasets")
print("x_train, y_train, x_test, y_test" + str(x_train.shape) + "" +str(y_train.shape) + "" +str(x_test.shape) + "" +str(y_test.shape))

svmclassifier = elaborationsvm(x_train,y_train,x_test,y_test,seed,CSV_OUTPUT)
dtclassifier = elaborationdecisiontree(x_train,y_train,x_test,y_test,seed,CSV_OUTPUT)
rfclassifier = elaborationrandomforest(x_train,y_train,x_test,y_test,seed,CSV_OUTPUT)
gbclassifier = elaborationgradientboost(x_train,y_train,x_test,y_test,seed,CSV_OUTPUT)
lrclassifier = elaborationlogisticregression(x_train,y_train,x_test,y_test,seed,CSV_OUTPUT)
knnclassifier = elaborationknn(x_train,y_train,x_test,y_test,seed,CSV_OUTPUT)

calculateroccurve(x_train,x_test,y_train,y_test,dtclassifier,rfclassifier,gbclassifier,lrclassifier,knnclassifier,svmclassifier,0)

if HYPERTUNING:
    dtclassifieropt = optimizationtree(dtclassifier,x_train,y_train,x_test,y_test,seed,CSV_OUTPUT,OPT_ITER)
    rfclassifieropt = optimizationrf(rfclassifier,x_train,y_train,x_test,y_test,seed,CSV_OUTPUT,OPT_ITER)
    knnclassifieropt = optimizationknn(knnclassifier,x_train,y_train,x_test,y_test,seed,CSV_OUTPUT,OPT_ITER)
    lrclassifieropt = optimizationlr(lrclassifier,x_train,y_train,x_test,y_test,seed,CSV_OUTPUT,OPT_ITER)
    gbclassifieropt = optimizationgb(gbclassifier,x_train,y_train,x_test,y_test,seed,CSV_OUTPUT,OPT_ITER)
    svmclassifieropt = optimizationsvm(svmclassifier,x_train,y_train,x_test,y_test,seed,CSV_OUTPUT,OPT_ITER)
    calculateroccurve(x_train,x_test,y_train,y_test,dtclassifieropt,rfclassifieropt,gbclassifieropt,lrclassifieropt,knnclassifieropt,svmclassifier,1)
        