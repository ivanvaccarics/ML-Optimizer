from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import time
import csv
import optuna
from sklearn.metrics import accuracy_score
import joblib
from utils.business import write_metrics, write_params_json

def optimizationgb(clf,x_train,y_train,x_test,y_test,seed,csvout,OPT_ITER):

    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 2, 500)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 500)
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        criterion = trial.suggest_categorical("criterion", ["friedman_mse", "mse"])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 500)
        GBC = GradientBoostingClassifier(min_samples_split = min_samples_split, 
                                    n_estimators=n_estimators, max_depth=max_depth, random_state=seed, min_samples_leaf=min_samples_leaf,
                                    criterion = criterion)
        GBC.fit(x_train, y_train)
        return 1.0 - accuracy_score(y_test, GBC.predict(x_test))
    study = optuna.create_study()
    study.optimize(objective, n_trials = OPT_ITER)
    print(study.best_params)
    print(1.0 - study.best_value)

    #evaluation with optimization
    start = time.time()
    modelgbOpt = GradientBoostingClassifier(min_samples_split = study.best_params.get('min_samples_split'), 
                                    n_estimators=study.best_params.get('n_estimators'), max_depth=study.best_params.get('max_depth'), random_state=seed, min_samples_leaf=study.best_params.get('min_samples_leaf'),
                                    criterion = study.best_params.get('criterion'))
    modelgbOpt.fit(x_train, y_train)
    end = time.time()
    diff=end-start
    print("Training time: " + str(diff))
    starttest = time.time()
    y_pred_gbOpt = modelgbOpt.predict(x_test)
    endtest =time.time()
    difftest = endtest-starttest
    print("Test time: " + str(difftest))

    print("Gradient boost optimized, accuracy: " + str(metrics.accuracy_score(y_test, y_pred_gbOpt)) + " F1 score:" + str(metrics.f1_score(y_test, y_pred_gbOpt,average='weighted')))
    matrixrfOpt = confusion_matrix(y_test,y_pred_gbOpt)
    print(matrixrfOpt)
    plot_confusion_matrix(modelgbOpt, x_test, y_test)
    TN, FP, FN, TP = confusion_matrix(y_test.values, y_pred_gbOpt).ravel()
    plt.grid(False)  
    plt.savefig('./Results/optimizer/confusion_matrix/gbopt',dpi=300)
    plt.clf()
    plt.close()
    joblib.dump(modelgbOpt, './Results/optimizer/models/gb_opt.sav')
    if csvout:
        write_metrics('Gradient Boost', 'No', metrics.accuracy_score(y_test, y_pred_gbOpt),metrics.f1_score(y_test, y_pred_gbOpt,average='weighted'),FP/(FP+TN),TP/(TP+FN),
        TN/(TN+FP),FN/(FN+TP),diff,difftest)
        write_params_json('./Results/optimizer/params','gb',modelgbOpt.get_params())
    return modelgbOpt

def elaborationgradientboost(x_train,y_train,x_test,y_test,seed,csvout):
    #Gradient boost
    print("Starting Gradient boost")
    start = time.time()
    modelgb = GradientBoostingClassifier(n_estimators=20, random_state=seed,verbose=2)
    modelgb.fit(x_train, y_train)
    end = time.time()
    diff=end-start
    print("Training time: " + str(diff))
    starttest = time.time()
    y_pred_gradient = modelgb.predict(x_test)
    endtest =time.time()
    difftest = endtest-starttest
    print("Test time: " + str(difftest))

    print("Gradient Boost, accuracy: " + str(metrics.accuracy_score(y_test, y_pred_gradient)) + " F1 score:" + str(metrics.f1_score(y_test, y_pred_gradient,average='weighted')))
    matrixgb = confusion_matrix(y_test,y_pred_gradient)
    print(matrixgb)
    plot_confusion_matrix(modelgb, x_test, y_test)
    TN, FP, FN, TP = confusion_matrix(y_test.values, y_pred_gradient).ravel()
    plt.grid(False)  
    plt.savefig('./Results/default/confusion_matrix/gb',dpi=300)
    plt.clf()
    plt.close()
    joblib.dump(modelgb, './Results/default/models/gb.sav')
    if csvout:
        write_metrics('Gradient Boost', 'No', metrics.accuracy_score(y_test, y_pred_gradient),metrics.f1_score(y_test, y_pred_gradient,average='weighted'),FP/(FP+TN),TP/(TP+FN),
        TN/(TN+FP),FN/(FN+TP),diff,difftest)
        write_params_json('./Results/default/params','gb',modelgb.get_params())
        
    return modelgb
