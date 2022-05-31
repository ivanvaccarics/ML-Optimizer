import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import time
import csv
import optuna
from sklearn.metrics import accuracy_score
import joblib
from utils.business import write_metrics, write_params_json

def optimizationlr(clf,x_train,y_train,x_test,y_test,seed,csvout,OPT_ITER):
    # Create hyperparameter options
    # se troppo lento, mettere solo L2
    # tolti saga,sag perchè vanno bene con dataset dove le features with approximately the same scale.

    def objective(trial):
        logreg_c = trial.suggest_float("logreg_c", 1e-10, 1e10, log=True)
        max_iter = trial.suggest_int("max_iter", 50, 300)
        solver = trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"])
        penalty = trial.suggest_categorical("penalty", ["l2"])

        LRC = LogisticRegression(C=logreg_c,max_iter=max_iter,solver=solver,penalty=penalty)
        LRC.fit(x_train, y_train)
        return 1.0 - accuracy_score(y_test, LRC.predict(x_test))
    study = optuna.create_study()
    study.optimize(objective, n_trials = OPT_ITER)
    print(study.best_params)
    print(1.0 - study.best_value)

    hyperparameters = {
    'C':np.logspace(0, 10, 50), 
    'penalty':['l2'],
    'random_state':[seed],
    'max_iter':[200,500,1000],
    'solver':['newton-cg', 'lbfgs', 'liblinear']
    }

    #evaluation with optimization
    start = time.time()
    modellrOpt = LogisticRegression(C=study.best_params.get('logreg_c'),max_iter=study.best_params.get('max_iter'),solver=study.best_params.get('solver'),penalty=study.best_params.get('penalty'))
    modellrOpt.fit(x_train, y_train)
    end = time.time()
    diff=end-start
    print("Training time: " + str(diff))
    starttest = time.time()
    y_pred_lrOpt = modellrOpt.predict(x_test)
    endtest =time.time()
    difftest = endtest-starttest
    print("Test time: " + str(difftest))

    print("Logistic regression optimized, accuracy: " + str(metrics.accuracy_score(y_test, y_pred_lrOpt)) + " F1 score:" + str(metrics.f1_score(y_test, y_pred_lrOpt,average='weighted')))
    matrixrfOpt = confusion_matrix(y_test,y_pred_lrOpt)
    print(matrixrfOpt)
    plot_confusion_matrix(modellrOpt, x_test, y_test)
    TN, FP, FN, TP = confusion_matrix(y_test.values, y_pred_lrOpt).ravel()
    plt.grid(False)  
    plt.savefig('./Results/optimizer/confusion_matrix/lropt',dpi=300)
    plt.clf()
    plt.close()
    joblib.dump(modellrOpt, './Results/optimizer/models/lr_opt.sav')
    if csvout:
        write_metrics('Logistic regression', 'No', metrics.accuracy_score(y_test, y_pred_lrOpt),metrics.f1_score(y_test, y_pred_lrOpt,average='weighted'),FP/(FP+TN),TP/(TP+FN),
        TN/(TN+FP),FN/(FN+TP),diff,difftest)
        write_params_json('./Results/optimizer/params','lr',modellrOpt.get_params())
    return modellrOpt

def elaborationlogisticregression(x_train,y_train,x_test,y_test,seed,csvout):
    #logistic regression, da valutare, risultati non aumentati
    start = time.time()
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    end = time.time()
    diff=end-start
    print("Training time: " + str(diff))
    starttest = time.time()
    y_predlr = logreg.predict(x_test)
    endtest =time.time()
    difftest = endtest-starttest
    print("Test time: " + str(difftest))

    print("Logistic regression, accuracy: " + str(metrics.accuracy_score(y_test, y_predlr)) + " F1 score:" + str(metrics.f1_score(y_test, y_predlr,average='weighted')))
    matrixlr = confusion_matrix(y_test,y_predlr)
    print(matrixlr)
    plot_confusion_matrix(logreg, x_test, y_test)
    TN, FP, FN, TP = confusion_matrix(y_test.values, y_predlr).ravel()
    plt.grid(False)  
    plt.savefig('./Results/default/confusion_matrix/lr',dpi=300)
    plt.clf()
    plt.close()
    joblib.dump(logreg, './Results/default/models/lr.sav')
    if csvout:
        write_metrics('Logistic regression', 'No', metrics.accuracy_score(y_test, y_predlr),metrics.f1_score(y_test, y_predlr,average='weighted'),FP/(FP+TN),TP/(TP+FN),
        TN/(TN+FP),FN/(FN+TP),diff,difftest)
        write_params_json('./Results/default/params','lr',logreg.get_params())
        
    return logreg
