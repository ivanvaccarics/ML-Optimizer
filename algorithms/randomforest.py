from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import optuna
from algorithms.commons import calculate_fit

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

    rfOpt = RandomForestClassifier(min_samples_split = study.best_params.get('min_samples_split'), 
                                    max_leaf_nodes = study.best_params.get('max_leaf_nodes'), n_estimators=study.best_params.get('n_estimators'), max_depth=study.best_params.get('max_depth'), random_state=seed, min_samples_leaf=study.best_params.get('min_samples_leaf'),
                                    criterion = study.best_params.get('criterion'))
    calculate_fit(rfOpt, x_train,y_train,x_test,y_test, 'randomforest_opt',True,csvout)
    return rfOpt

def elaborationrandomforest(x_train,y_train,x_test,y_test,seed,csvout):
    rf = RandomForestClassifier(verbose=2,random_state=seed)
    calculate_fit(rf, x_train,y_train,x_test,y_test, 'randomforest',False,csvout)
    return rf
