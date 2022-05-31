import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import time
import csv
import optuna
from sklearn.metrics import accuracy_score
import joblib
from utils.business import write_metrics,write_params_json

def optimizationrf(clf,x_train,y_train,x_test,y_test,seed,csvout,OPT_ITER):
    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 2, 300)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 300)
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_leaf_nodes = int(trial.suggest_int("max_leaf_nodes", 2, 300))
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 300)
        RFC = RandomForestClassifier(min_samples_split = min_samples_split, 
                                    max_leaf_nodes = max_leaf_nodes, n_estimators=n_estimators, max_depth=max_depth, random_state=seed, min_samples_leaf=min_samples_leaf,
                                    criterion = criterion)
        RFC.fit(x_train, y_train)
        return 1.0 - accuracy_score(y_test, RFC.predict(x_test))
    study = optuna.create_study()
    study.optimize(objective, n_trials = OPT_ITER)
    print(study.best_params)
    print(1.0 - study.best_value)

    start = time.time()
    classifierOpt = RandomForestClassifier(min_samples_split = study.best_params.get('min_samples_split'), 
                                    max_leaf_nodes = study.best_params.get('max_leaf_nodes'), n_estimators=study.best_params.get('n_estimators'), max_depth=study.best_params.get('max_depth'), random_state=seed, min_samples_leaf=study.best_params.get('min_samples_leaf'),
                                    criterion = study.best_params.get('criterion'))
    classifierOpt.fit(x_train, y_train)
    end = time.time()
    diff=end-start
    print("Training time: " + str(diff))
    starttest = time.time()
    y_pred_randomOpt = classifierOpt.predict(x_test)
    endtest =time.time()
    difftest = endtest-starttest
    print("Test time: " + str(difftest))

    print("Random Forest optimized, accuracy: " + str(metrics.accuracy_score(y_test, y_pred_randomOpt)) + " F1 score:" + str(metrics.f1_score(y_test, y_pred_randomOpt,average='weighted')))
    matrixrfOpt = confusion_matrix(y_test,y_pred_randomOpt)
    print(matrixrfOpt)
    plot_confusion_matrix(classifierOpt, x_test, y_test)
    TN, FP, FN, TP = confusion_matrix(y_test.values, y_pred_randomOpt).ravel()
    plt.grid(False)  
    plt.savefig('./Results/optimizer/confusion_matrix/rfopt',dpi=300)
    plt.clf()
    plt.close()
    joblib.dump(classifierOpt, './Results/optimizer/models/rf_opt.sav')
    if csvout:
        write_metrics('Random Forest', 'No', metrics.accuracy_score(y_test, y_pred_randomOpt),metrics.f1_score(y_test, y_pred_randomOpt,average='weighted'),FP/(FP+TN),TP/(TP+FN),
        TN/(TN+FP),FN/(FN+TP),diff,difftest)
        write_params_json('./Results/optimizer/params','rf',classifierOpt.get_params())
    return classifierOpt

def elaborationrandomforest(x_train,y_train,x_test,y_test,seed,csvout):
    print("Starting Random forest")
    start = time.time()
    classifier = RandomForestClassifier(verbose=2,random_state=seed)
    classifier.fit(x_train, y_train)
    end = time.time()
    diff=end-start
    print("Training time: " + str(diff))
    starttest = time.time()
    y_pred_random = classifier.predict(x_test)
    endtest =time.time()
    difftest = endtest-starttest
    print("Test time: " + str(difftest))

    print("Random Forest, accuracy: " + str(metrics.accuracy_score(y_test, y_pred_random)) + " F1 score:" + str(metrics.f1_score(y_test, y_pred_random,average='weighted')))
    matrixrf = confusion_matrix(y_test,y_pred_random)
    print(matrixrf)
    plot_confusion_matrix(classifier, x_test, y_test)
    TN, FP, FN, TP = confusion_matrix(y_test.values, y_pred_random).ravel()
    plt.grid(False)  
    plt.savefig('./Results/default/confusion_matrix/rf',dpi=300)
    plt.clf()
    plt.close()
    joblib.dump(classifier, './Results/default/models/rf.sav')
    if csvout:
        write_metrics('Random Forest', 'No', metrics.accuracy_score(y_test, y_pred_random),metrics.f1_score(y_test, y_pred_random,average='weighted'),FP/(FP+TN),TP/(TP+FN),
        TN/(TN+FP),FN/(FN+TP),diff,difftest)
        write_params_json('./Results/default/params','rf',classifier.get_params())
    return classifier
