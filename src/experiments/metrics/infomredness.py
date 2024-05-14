def infomredness(cm):
    TPR = cm[1][1]/sum(cm[1])
    TNR = cm[0][0]/sum(cm[0])
    return TPR + TNR - 1