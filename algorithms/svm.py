from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import accuracy_score
import optuna
from algorithms.commons import calculate_fit

def optimizationsvm(clf,x_train,y_train,x_test,y_test,seed,csvout,OPT_ITER):
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

    modelsvmOpt = LinearSVC(random_state=seed,C=study.best_params.get('logreg_c'),max_iter=study.best_params.get('max_iter'),loss=study.best_params.get('loss'))
    calculate_fit(modelsvmOpt, x_train,y_train,x_test,y_test, 'svm_opt',True,csvout)
    calibrated_svc = CalibratedClassifierCV(base_estimator=modelsvmOpt, cv=2)
    calibrated_svc.fit(x_train, y_train) 
    return calibrated_svc

def elaborationsvm(x_train,y_train,x_test,y_test,seed,csvout):
    svm = LinearSVC(random_state=seed,verbose=0) 
    calculate_fit(svm, x_train,y_train,x_test,y_test,'svm',False,csvout)
    calibrated_svc = CalibratedClassifierCV(base_estimator=svm, cv=2)
    calibrated_svc.fit(x_train, y_train)
    return calibrated_svc
