import numpy as np
import math


def error_evaluation(sp, act):
    r = act
    x = sp
    points = np.zeros((1, len(x[0])))
    abs_r = np.zeros((1, len(x[0])))
    abs_x = np.zeros((1, len(x[0])))
    abs_r_x = np.zeros((1, len(x[0])))
    abs_x_r = np.zeros((1, len(x[0])))
    abs_r_x__r = np.zeros((1, len(x[0])))
    for j in range(1, len(x[0])):
        points[0][j] = abs(x[0][j] - x[0][j-1])
    for i in range(len(r[0])):
        abs_r[0, i] = abs(r[0][i])
    for i in range(len(r[0])):
        abs_x[0, i] = abs(x[0][i])
    for i in range(len(r[0])):
        abs_r_x[0, i] = abs(r[0][i] - x[0][i])
    for i in range(len(r[0])):
        abs_x_r[0, i] = abs(x[0][i] - r[0][i])
    for i in range(len(r[0])):
        abs_r_x__r[0, i] = abs((r[0][i] - x[0][i]) / r[0][i])
    md = (100/len(x[0])) * sum(abs_r_x__r[0])
    smape = (1/len(x[0])) * sum(abs_r_x[0]/((abs_r[0] + abs_x[0]) / 2))
    mase = sum(abs_r_x[0])/((1 / (len(x[0]) - 1)) * sum(points[0]))
    mae = sum(abs_r_x[0]) / len(r[0])
    rmse = (sum(abs_x_r[0] ** 2) / len(r[0])) ** 0.5
    onenorm = sum(abs_r_x[0])
    twonorm = (sum(abs_r_x[0] ** 2) ** 0.5)
    infinitynorm = max(abs_r_x[0])
    EVAL_ERR = [md, smape, mase, mae, rmse, onenorm, twonorm, infinitynorm]
    return EVAL_ERR


def evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(p)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = sum(Tp)
    fp = sum(Fp)
    tn = sum(Tn)
    fn = sum(Fn)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    FPR = fp / (fp + tn)
    FNR = fn / (tp + fn)
    NPV = tn / (tn + fp)
    FDR = fp / (tp + fp)
    F1_score = (2 * tp) / (2 * tp + fp + fn)
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    EVAL = [tp, tn, fp, fn, accuracy, sensitivity, specificity, precision, FPR, FNR, NPV, FDR, F1_score, MCC]
    return EVAL
