from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
import optuna
from algorithms.commons import calculate_fit

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

    dtOpt = DecisionTreeClassifier(min_samples_split = study.best_params.get('min_samples_split'), 
                                    max_leaf_nodes = study.best_params.get('max_leaf_nodes'), max_depth=study.best_params.get('max_depth'), random_state=seed, splitter=study.best_params.get('splitter'),min_samples_leaf=study.best_params.get('min_samples_leaf'),
                                    criterion = study.best_params.get('criterion'))
    
    calculate_fit(dtOpt, x_train,y_train,x_test,y_test, 'decisiontree_opt',True,csvout)
    return dtOpt

def elaborationdecisiontree(x_train,y_train,x_test,y_test,seed,csvout):
    dt = DecisionTreeClassifier()
    calculate_fit(dt, x_train,y_train,x_test,y_test, 'decisiontree',False,csvout)
    return dt


