import time
import matplotlib.pyplot as plt
import joblib
from config import *
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import time
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import json
import numpy as np
from sklearn.preprocessing import label_binarize

def calculate_fit(model, x_train, y_train, x_test, y_test, modelname, isOptimized, csvout):
    start = time.time()
    model = model.fit(x_train,y_train)
    end = time.time()
    diff=end-start
    starttest = time.time()
    y_pred = model.predict(x_test)
    endtest =time.time()
    difftest = endtest-starttest
    matrix = confusion_matrix(y_test,y_pred)
    if MULTICLASS:
        FP = matrix.sum(axis=0) - np.diag(matrix) 
        FN = matrix.sum(axis=1) - np.diag(matrix)
        TP = np.diag(matrix)
        TN = matrix.sum() - (FP + FN + TP)
    else:
        TN, FP, FN, TP = confusion_matrix(y_test.values, y_pred).ravel()
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
    if isOptimized:
        path_confusion_matrix = f'{RESULTS_OPTIMIZED_MATRIX}'
        path_models = f'{RESULTS_OPTIMIZED_MODELS}'
        path_params = f'{RESULTS_OPTIMIZED_PARAMS}'
    else:
        path_confusion_matrix = f'{RESULTS_DEFAULT_MATRIX}'
        path_models = f'{RESULTS_DEFAULT_MODELS}'
        path_params = f'{RESULTS_DEFAULT_PARAMS}'
    plt.grid(False)  
    plt.savefig(f'{path_confusion_matrix}{modelname}',dpi=300)
    plt.clf()
    plt.close()
    joblib.dump(model, f'{path_models}{modelname}.sav')
    if csvout:
        write_metrics(modelname, isOptimized, metrics.accuracy_score(y_test, y_pred),metrics.f1_score(y_test, y_pred,average='weighted'),FP/(FP+TN),TP/(TP+FN),
        TN/(TN+FP),FN/(FN+TP),diff,difftest)
        write_params_json(path_params, modelname, model.get_params())

def calculateroccurve(x_train,x_test,y_train,y_test,dtclassifier,rfclassifier,gbclassifier,lrclassifier,knnclassifier,svmclassifier,disp):
#def calculateroccurve(dtclassifier,rfclassifier,gbclassifier,lrclassifier,knnclassifier,disp):
    # predict probabilities
    pred_probdt = dtclassifier.predict_proba(x_test)
    pred_probrf = rfclassifier.predict_proba(x_test)
    pred_probgb = gbclassifier.predict_proba(x_test)
    pred_problr = lrclassifier.predict_proba(x_test)
    pred_probknn = knnclassifier.predict_proba(x_test)
    pred_probsvm = svmclassifier.predict_proba(x_test)
    # roc curve for models
    n_classes = np.shape(np.unique(y_train))[0]
    if MULTICLASS:
        [dict() for _ in range(3)]
        fprdt, tprdt, threshdt, fprrf, tprrf, threshfr, fprgb, tprgb, threshgb, fprlr, tprlr, threshlr, \
        fprknn, tprknn, threshknn, fprsvm, tprsvm, threshsvm, p_fpr, p_tpr = [dict() for _ in range(20)]

        for i in range(n_classes):
            # assume class labels being 1,...,n_classes
            y_test = label_binarize(y_test,classes=[k+1 for k in range(n_classes)])
            
            fprdt[i], tprdt[i], threshdt[i] = roc_curve(y_test[:,i], pred_probdt[:,i])
            fprrf[i], tprrf[i], threshfr[i] = roc_curve(y_test[:,i], pred_probrf[:,i])#, pos_label=i)
            fprgb[i], tprgb[i], threshgb[i] = roc_curve(y_test[:,i], pred_probgb[:,i])#, pos_label=i)
            fprlr[i], tprlr[i], threshlr[i] = roc_curve(y_test[:,i], pred_problr[:,i])#, pos_label=1)
            fprknn[i], tprknn[i], threshknn[i] = roc_curve(y_test[:,i], pred_probknn[:,i])#, pos_label=1)
            fprsvm[i], tprsvm[i], threshsvm[i] = roc_curve(y_test[:,i], pred_probsvm[:,i])#, pos_label=1)
            # roc curve for tpr = fpr 
            random_probs = [0 for _ in range(len(y_test))]
            p_fpr[i], p_tpr[i], _ = roc_curve(y_test[:,i], random_probs)#, pos_label=1)
        auc_scoredt = roc_auc_score(y_test, pred_probdt, multi_class='ovo')
        auc_scorerf = roc_auc_score(y_test, pred_probrf, multi_class='ovo')
        auc_scoregb = roc_auc_score(y_test, pred_probgb, multi_class='ovo')
        auc_scorelr = roc_auc_score(y_test, pred_problr, multi_class='ovo')
        auc_scoreknn = roc_auc_score(y_test, pred_probknn, multi_class='ovo')
        auc_scoresvm = roc_auc_score(y_test, pred_probsvm, multi_class='ovo')
        for i in range(n_classes):
            plt.plot(fprdt[i], tprdt[i], linestyle='--',color='blue', label='Decision tree')
            plt.plot(fprrf[i], tprrf[i], linestyle='--',color='green', label='Random forest')
            plt.plot(fprgb[i], tprgb[i], linestyle='--',color='red', label='Gradient Boost')
            plt.plot(fprlr[i], tprlr[i], linestyle='--',color='cyan', label='Logistic Regression')
            plt.plot(fprknn[i], tprknn[i], linestyle='--',color='magenta', label='KNN')
            plt.plot(fprsvm[i], tprsvm[i], linestyle='--',color='black', label='Support vector machine')
            # title
            if disp == 1: plt.title('ROC curve optimized')
            else: plt.title('ROC curve')
            # x label
            plt.xlabel('False Positive Rate')
            # y label
            plt.ylabel('True Positive rate')

            plt.legend(loc='best')
            if disp == 1: plt.savefig(ROC_OPTIMIZED+"_"+str(i+1),dpi=300)
            else: plt.savefig(ROC_DEFAULTS+"_"+str(i+1),dpi=300)
            plt.clf()
            plt.close()
    else:
        # roc curve for models
        fprdt, tprdt, threshdt = roc_curve(y_test, pred_probdt[:,1], pos_label=1)
        fprrf, tprrf, threshfr = roc_curve(y_test, pred_probrf[:,1], pos_label=1)
        fprgb, tprgb, threshgb = roc_curve(y_test, pred_probgb[:,1], pos_label=1)
        fprlr, tprlr, threshlr = roc_curve(y_test, pred_problr[:,1], pos_label=1)
        fprknn, tprknn, threshknn = roc_curve(y_test, pred_probknn[:,1], pos_label=1)
        fprsvm, tprsvm, threshsvm = roc_curve(y_test, pred_probsvm[:,1], pos_label=1)
        # roc curve for tpr = fpr 
        random_probs = [0 for i in range(len(y_test))]
        p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

        # auc scores
        auc_scoredt = roc_auc_score(y_test, pred_probdt[:,1])
        auc_scorerf = roc_auc_score(y_test, pred_probrf[:,1])
        auc_scoregb = roc_auc_score(y_test, pred_probgb[:,1])
        auc_scorelr = roc_auc_score(y_test, pred_problr[:,1])
        auc_scoreknn = roc_auc_score(y_test, pred_probknn[:,1])
        auc_scoresvm = roc_auc_score(y_test, pred_probsvm[:,1])

        # plot roc curves
        plt.plot(fprdt, tprdt, linestyle='--',color='blue', label='Decision tree')
        plt.plot(fprrf, tprrf, linestyle='--',color='green', label='Random forest')
        plt.plot(fprgb, tprgb, linestyle='--',color='red', label='Gradient Boost')
        plt.plot(fprlr, tprlr, linestyle='--',color='cyan', label='Logistic Regression')
        plt.plot(fprknn, tprknn, linestyle='--',color='magenta', label='KNN')
        plt.plot(fprsvm, tprsvm, linestyle='--',color='black', label='Support vector machine')
        
        # title
        if disp == 1: plt.title('ROC curve optimized')
        else: plt.title('ROC curve')
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')

        plt.legend(loc='best')
        if disp == 1: plt.savefig(ROC_OPTIMIZED,dpi=300)
        else: plt.savefig(ROC_DEFAULTS,dpi=300)
        plt.clf()
        plt.close()

def write_metrics(algorithm, optimized, accuracy, f1score, FPR, TPR, TNR, FNR, training_time, test_time):
    with open(RESULTS_METRICS,'a') as fd:
            fd.write('\n')
            fd.write(f'{algorithm}, {optimized}, {accuracy}, {f1score}, {FPR}, {TPR}, {TNR}, {FNR}, {training_time}, {test_time}')

def write_params_json(path,algorithm,data):
    with open(f'{path}/{algorithm}.json', 'w') as f:
        json.dump(data, f)