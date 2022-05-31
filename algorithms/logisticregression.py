from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
import optuna
from algorithms.commons import calculate_fit

def optimizationlr(clf,x_train,y_train,x_test,y_test,seed,csvout,OPT_ITER):
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
    modellrOpt = LogisticRegression(C=study.best_params.get('logreg_c'),max_iter=study.best_params.get('max_iter'),solver=study.best_params.get('solver'),penalty=study.best_params.get('penalty'))
    calculate_fit(modellrOpt, x_train,y_train,x_test,y_test, 'logisticregression_opt',True,csvout)
    return modellrOpt

def elaborationlogisticregression(x_train,y_train,x_test,y_test,seed,csvout):
    logreg = LogisticRegression()
    calculate_fit(logreg, x_train,y_train,x_test,y_test, 'logisticregression',False,csvout)
    return logreg
