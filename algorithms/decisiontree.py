from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import time
import csv
from sklearn.metrics import accuracy_score
import optuna
import joblib
from utils.business import write_metrics, write_params_json


def optimizationtree(clf,x_train,y_train,x_test,y_test,seed, csvout, OPT_ITER):
    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 2, 300)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 300)
        max_leaf_nodes = int(trial.suggest_int("max_leaf_nodes", 2, 300))
        splitter = trial.suggest_categorical("splitter",["best","random"])
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 300)
        DTC = DecisionTreeClassifier(min_samples_split = min_samples_split, 
                                    max_leaf_nodes = max_leaf_nodes, max_depth=max_depth, random_state=seed, splitter=splitter,min_samples_leaf=min_samples_leaf,
                                    criterion = criterion)
        DTC.fit(x_train, y_train)
        return 1.0 - accuracy_score(y_test, DTC.predict(x_test))
    study = optuna.create_study()
    study.optimize(objective, n_trials = OPT_ITER)
    print(study.best_params)
    print(1.0 - study.best_value)

    start = time.time()
    clfOpt = DecisionTreeClassifier(min_samples_split = study.best_params.get('min_samples_split'), 
                                    max_leaf_nodes = study.best_params.get('max_leaf_nodes'), max_depth=study.best_params.get('max_depth'), random_state=seed, splitter=study.best_params.get('splitter'),min_samples_leaf=study.best_params.get('min_samples_leaf'),
                                    criterion = study.best_params.get('criterion'))
    clfOpt.fit(x_train, y_train)
    end = time.time()
    diff=end-start
    print("Training time: " + str(diff))
    starttest = time.time()
    y_pred_dtOpt = clfOpt.predict(x_test)
    endtest =time.time()
    difftest = endtest-starttest
    print("Test time: " + str(difftest))

    print("Decision Tree optimized, accuracy: " + str(metrics.accuracy_score(y_test, y_pred_dtOpt)) + " F1 score:" + str(metrics.f1_score(y_test, y_pred_dtOpt,average='weighted')))
    matrixdtOpt = confusion_matrix(y_test,y_pred_dtOpt)
    print(matrixdtOpt)
    plot_confusion_matrix(clfOpt, x_test, y_test)
    TN, FP, FN, TP = confusion_matrix(y_test.values, y_pred_dtOpt).ravel()
    plt.grid(False)  
    plt.savefig('./Results/optimizer/confusion_matrix/dtopt',dpi=300)
    plt.clf()
    plt.close()
    joblib.dump(clf, './Results/optimizer/models/dt_opt.sav')
    if csvout:
        write_metrics('Decision tree', 'Yes', metrics.accuracy_score(y_test, y_pred_dtOpt),metrics.f1_score(y_test, y_pred_dtOpt,average='weighted'),FP/(FP+TN),TP/(TP+FN),
        TN/(TN+FP),FN/(FN+TP),diff,difftest)
        write_params_json('./Results/optimizer/params','dt',clfOpt.get_params())

    return clfOpt

def elaborationdecisiontree(x_train,y_train,x_test,y_test,seed,csvout):
    #Decision tree
    print("Starting Decision tree")
    start = time.time()
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train,y_train)
    end = time.time()
    diff=end-start
    print("Training time: " + str(diff))
    starttest = time.time()
    y_pred_dt = clf.predict(x_test)
    endtest =time.time()
    difftest = endtest-starttest
    print("Test time: " + str(difftest))
    

    print("Decision Tree, accuracy: " + str(metrics.accuracy_score(y_test, y_pred_dt)) + " F1 score:" + str(metrics.f1_score(y_test, y_pred_dt,average='weighted')))
    matrixdt = confusion_matrix(y_test,y_pred_dt)
    TN, FP, FN, TP = confusion_matrix(y_test.values, y_pred_dt).ravel()
    print(matrixdt)
    plot_confusion_matrix(clf, x_test, y_test)
    plt.grid(False)  
    plt.savefig('./Results/default/confusion_matrix/dt',dpi=300)
    plt.clf()
    plt.close()
    joblib.dump(clf, './Results/default/models/dt.sav')
    if csvout:
        write_metrics('Decision tree', 'No', metrics.accuracy_score(y_test, y_pred_dt),metrics.f1_score(y_test, y_pred_dt,average='weighted'),FP/(FP+TN),TP/(TP+FN),
        TN/(TN+FP),FN/(FN+TP),diff,difftest)
        write_params_json('./Results/default/params','dt',clf.get_params())
    return clf


