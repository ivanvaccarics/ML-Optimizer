from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import optuna
from algorithms.commons import calculate_fit

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
    modelgbOpt = GradientBoostingClassifier(min_samples_split = study.best_params.get('min_samples_split'), 
                                    n_estimators=study.best_params.get('n_estimators'), max_depth=study.best_params.get('max_depth'), random_state=seed, min_samples_leaf=study.best_params.get('min_samples_leaf'),
                                    criterion = study.best_params.get('criterion'))
    calculate_fit(modelgbOpt, x_train,y_train,x_test,y_test, 'gradientboost_opt',True,csvout)
    return modelgbOpt

def elaborationgradientboost(x_train,y_train,x_test,y_test,seed,csvout):
    modelgb = GradientBoostingClassifier(n_estimators=20, random_state=seed,verbose=2)
    calculate_fit(modelgb, x_train,y_train,x_test,y_test, 'gradientboost',False,csvout)
    return modelgb
