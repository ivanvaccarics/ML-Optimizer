from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import optuna
from algorithms.commons import calculate_fit

def optimizationknn(clf,x_train,y_train,x_test,y_test,seed, csvout,OPT_ITER):
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
    knnOpt = KNeighborsClassifier(n_neighbors=study.best_params.get('n_neighbors'),leaf_size=study.best_params.get('leaf_size'),p=study.best_params.get('p'),algorithm=study.best_params.get('algorithm'),weights=study.best_params.get('weights'))
    calculate_fit(knnOpt, x_train,y_train,x_test,y_test, 'knn_opt',True,csvout)
    return knnOpt


def elaborationknn(x_train,y_train,x_test,y_test,seed,csvout):
    knn = KNeighborsClassifier()
    calculate_fit(knn, x_train,y_train,x_test,y_test, 'knn',False,csvout)
    return knn