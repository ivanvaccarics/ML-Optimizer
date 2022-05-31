from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import csv, time
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import json

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
    if disp == 1: plt.savefig('./Results/optimizer/ROC_opt',dpi=300)
    else: plt.savefig('./Results/default/ROC',dpi=300)
    plt.clf()
    plt.close()

def write_metrics(algorithm, optimized, accuracy, f1score, FPR, TPR, TNR, FNR, training_time, test_time):
    with open('./Results/output.csv','a') as fd:
            fd.write('\n')
            fd.write(f'{algorithm}, {optimized}, {accuracy}, {f1score}, {FPR}, {TPR}, {TNR}, {FNR}, {training_time}, {test_time}')

def write_params_json(path,algorithm,data):
    with open(f'{path}/{algorithm}.json', 'w') as f:
        json.dump(data, f)