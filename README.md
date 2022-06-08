#  ML-Optimizer, an automatic machine learning tool to classification problems using hyperparameter optimization

This project aims to provide a toolset of machine learning algorithms for classification problems. The toolbox is written in python and it is based on the [scikit-learn](https://scikit-learn.org/stable/) library.

In addition to running with the default parameters, the tool performs a series of iterations to optimize the parameters of the machine learning algorithms using the Optuna tool (for more information on how Optuna works, click [here](https://optuna.org/)).

The tool calculates the classic metrics used in machine learning projects: accuracy, f1 score, FPR, TPR, TNR, FNR, training time (s), test time (s), confusion matrix and ROC curve.

It works both for binary and multiclass classification problems, just enable multiclass in the configuration file. For multiclass problems, the toolbox supports only numeric classes (e.g. 0, 1, 2, 3, 4). 

## Installation

Clone the repo, go to the folder and run.

```bash
pip install -r requirements.txt
```

## Usage

Steps to run the code:
* Configure the config.py file
* In the main.py code, import your data and perform some manipulation in order to obtain the x_train, x_test, y_train, y_test datasets.
* The rest of the code will do all the work :bowtie:

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)