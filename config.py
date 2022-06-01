# IMPORTANT: put the / as last character of the path!

# default parameters folders
RESULTS_DEFAULT_MODELS = './Results/default/models/'
RESULTS_DEFAULT_MATRIX = './Results/default/confusion_matrix/'
RESULTS_DEFAULT_PARAMS = './Results/default/params/'
ROC_DEFAULTS = './Results/default/ROC'
# optmized parameters folders
RESULTS_OPTIMIZED_MODELS = './Results/optimizer/models/'
RESULTS_OPTIMIZED_MATRIX = './Results/optimizer/confusion_matrix/'
RESULTS_OPTIMIZED_PARAMS = './Results/optimizer/params/'
ROC_OPTIMIZED = './Results/optimizer/ROC_opt'

# metrics path file
RESULTS_METRICS = './Results/metrics.csv'

FOLDER_LIST = [RESULTS_DEFAULT_MODELS,RESULTS_DEFAULT_MATRIX,RESULTS_DEFAULT_PARAMS,RESULTS_OPTIMIZED_MODELS,RESULTS_OPTIMIZED_MATRIX,RESULTS_OPTIMIZED_PARAMS]

# Code config
SEED = 7
PATH_DATASET = 'dataset/'
TEST_SIZE = 0.3

# Optuna iteration to hyperparameters optmization
OPT_ITER = 1