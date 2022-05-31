from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import time
import csv
import optuna
from sklearn.metrics import accuracy_score
import joblib
from utils.business import write_metrics, write_params_json

def optimizationknn(clf,x_train,y_train,x_test,y_test,seed, csvout,OPT_ITER):
    #hypertuning

    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 1, 500)
        leaf_size = trial.suggest_int("leaf_size", 1, 500)
        p = trial.suggest_int("p", 1, 2)
        algorithm = trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree"])
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        LRC = KNeighborsClassifier(n_neighbors=n_neighbors,leaf_size=leaf_size,p=p,algorithm=algorithm,weights=weights)
        LRC.fit(x_train, y_train)
        return 1.0 - accuracy_score(y_test, LRC.predict(x_test))
    study = optuna.create_study()
    study.optimize(objective, n_trials = OPT_ITER)
    print(study.best_params)
    print(1.0 - study.best_value)


    #evaluation with optimization
    start = time.time()
    knnOpt = KNeighborsClassifier(n_neighbors=study.best_params.get('n_neighbors'),leaf_size=study.best_params.get('leaf_size'),p=study.best_params.get('p'),algorithm=study.best_params.get('algorithm'),weights=study.best_params.get('weights'))
    knnOpt.fit(x_train, y_train)
    end = time.time()
    diff=end-start
    print("Training time: " + str(diff))
    starttest = time.time()
    y_pred_knnOpt = knnOpt.predict(x_test)
    endtest =time.time()
    difftest = endtest-starttest
    print("Test time: " + str(difftest))

    print("KNN optimized, accuracy: " + str(metrics.accuracy_score(y_test, y_pred_knnOpt)) + " F1 score:" + str(metrics.f1_score(y_test, y_pred_knnOpt,average='weighted')))
    matrixdtOpt = confusion_matrix(y_test,y_pred_knnOpt)
    TN, FP, FN, TP = confusion_matrix(y_test.values, y_pred_knnOpt).ravel()
    print(matrixdtOpt)
    plot_confusion_matrix(knnOpt, x_test, y_test)
    plt.grid(False)  
    plt.savefig('./Results/optimizer/confusion_matrix/knnopt',dpi=300)
    plt.clf()
    plt.close()
    joblib.dump(knnOpt, './Results/optimizer/models/knn_opt.sav')
    if csvout:
        write_metrics('KNN', 'No', metrics.accuracy_score(y_test, y_pred_knnOpt),metrics.f1_score(y_test, y_pred_knnOpt,average='weighted'),FP/(FP+TN),TP/(TP+FN),
        TN/(TN+FP),FN/(FN+TP),diff,difftest)
        write_params_json('./Results/optimizer/params','knn',knnOpt.get_params())
    return knnOpt


def elaborationknn(x_train,y_train,x_test,y_test,seed,csvout):
    print("Starting KNN")
    start = time.time()
    classifier = KNeighborsClassifier()
    classifier.fit(x_train, y_train)
    end = time.time()
    diff=end-start
    print("Training time: " + str(diff))
    starttest = time.time()
    y_pred_knn = classifier.predict(x_test)
    endtest =time.time()
    difftest = endtest-starttest
    print("Test time: " + str(difftest))

    print("KNN, accuracy: " + str(metrics.accuracy_score(y_test, y_pred_knn)) + " F1 score:" + str(metrics.f1_score(y_test, y_pred_knn,average='weighted')))
    matrixdt = confusion_matrix(y_test,y_pred_knn)
    TN, FP, FN, TP = confusion_matrix(y_test.values, y_pred_knn).ravel()
    print(matrixdt)
    plot_confusion_matrix(classifier, x_test, y_test)
    plt.grid(False)  
    plt.savefig('./Results/default/confusion_matrix/knn',dpi=300)
    plt.clf()
    plt.close()
    joblib.dump(classifier, './Results/default/models/knn.sav')
    if csvout:
        write_metrics('KNN', 'No', metrics.accuracy_score(y_test, y_pred_knn),metrics.f1_score(y_test, y_pred_knn,average='weighted'),FP/(FP+TN),TP/(TP+FN),
        TN/(TN+FP),FN/(FN+TP),diff,difftest)
        write_params_json('./Results/default/params','knn',classifier.get_params())
        
    return classifier