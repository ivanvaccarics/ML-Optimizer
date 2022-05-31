import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import time
import csv
import optuna
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
from utils.business import write_metrics,write_params_json

def optimizationsvm(clf,x_train,y_train,x_test,y_test,seed,csvout,OPT_ITER):
    #hypertuning parameters
    def objective(trial):
        logreg_c = trial.suggest_float("logreg_c", 1e-10, 1e10, log=True)
        max_iter = trial.suggest_int("max_iter", 50, 3000)
        loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])
        SVM = LinearSVC(random_state=seed,C=logreg_c,max_iter=max_iter,loss=loss)
        SVM.fit(x_train, y_train)
        return 1.0 - accuracy_score(y_test, SVM.predict(x_test))
    study = optuna.create_study()
    study.optimize(objective, n_trials = OPT_ITER)
    print(study.best_params)
    print(1.0 - study.best_value)

    start = time.time()
    modelsvmOpt = LinearSVC(random_state=seed,C=study.best_params.get('logreg_c'),max_iter=study.best_params.get('max_iter'),loss=study.best_params.get('loss'))
    modelsvmOpt.fit(x_train, y_train)
    end = time.time()
    diff=end-start
    print("Training time: " + str(diff))
    starttest = time.time()
    y_pred_svmOpt = modelsvmOpt.predict(x_test)
    endtest =time.time()
    difftest = endtest-starttest
    print("Test time: " + str(difftest))

    print("Support vector machine optimized, accuracy: " + str(metrics.accuracy_score(y_test, y_pred_svmOpt)) + " F1 score:" + str(metrics.f1_score(y_test, y_pred_svmOpt,average='weighted')))
    matrixsvmOpt = confusion_matrix(y_test,y_pred_svmOpt)
    print(matrixsvmOpt)
    plot_confusion_matrix(modelsvmOpt, x_test, y_test)
    TN, FP, FN, TP = confusion_matrix(y_test.values, y_pred_svmOpt).ravel()
    plt.grid(False)  
    plt.savefig('./Results/optimizer/confusion_matrix/svmopt',dpi=300)
    plt.clf()
    plt.close()
    joblib.dump(modelsvmOpt, './Results/optimizer/models/svm_opt.sav')
    calibrated_svc = CalibratedClassifierCV(base_estimator=modelsvmOpt, cv=2)
    calibrated_svc.fit(x_train, y_train)
    if csvout:
        write_metrics('SVM', 'No', metrics.accuracy_score(y_test, y_pred_svmOpt),metrics.f1_score(y_test, y_pred_svmOpt,average='weighted'),FP/(FP+TN),TP/(TP+FN),
        TN/(TN+FP),FN/(FN+TP),diff,difftest)
        write_params_json('./Results/optimizer/params','svm',modelsvmOpt.get_params())
    return calibrated_svc

def elaborationsvm(x_train,y_train,x_test,y_test,seed,csvout):
    #Create a svm Classifier
    print("Starting SVM")
    start = time.time()
    svm = LinearSVC(random_state=seed,verbose=0) 
    svm.fit(x_train, y_train)
    end = time.time()
    diff=end-start
    print("Training time: " + str(diff))
    starttest = time.time()
    y_predsvm = svm.predict(x_test)
    endtest =time.time()
    difftest = endtest-starttest
    print("Test time: " + str(difftest))
    
    print("Support vector machine, accuracy: " + str(metrics.accuracy_score(y_test, y_predsvm)) + " F1 score:" + str(metrics.f1_score(y_test, y_predsvm,average='weighted')))
    matrixsvm = confusion_matrix(y_test,y_predsvm)
    print(matrixsvm)
    plot_confusion_matrix(svm, x_test, y_test)
    TN, FP, FN, TP = confusion_matrix(y_test.values, y_predsvm).ravel()
    plt.grid(False)  
    plt.savefig('./Results/default/confusion_matrix/svm',dpi=300)
    plt.clf()
    plt.close()
    joblib.dump(svm, './Results/default/models/svm.sav')
    calibrated_svc = CalibratedClassifierCV(base_estimator=svm, cv=2)
    calibrated_svc.fit(x_train, y_train)
    if csvout:
        write_metrics('SVM', 'No', metrics.accuracy_score(y_test, y_predsvm),metrics.f1_score(y_test, y_predsvm,average='weighted'),FP/(FP+TN),TP/(TP+FN),
        TN/(TN+FP),FN/(FN+TP),diff,difftest)
        write_params_json('./Results/default/params','svm',svm.get_params())
    
    return calibrated_svc
