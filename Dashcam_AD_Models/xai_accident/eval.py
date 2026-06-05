import numpy as np

def evaluation_P_R80(all_pred, all_labels, time_of_accidents, fps=10.0):
    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)] # positive video
        else:
            pred = all_pred[idx, :] # negative video
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred[:])
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps
    Precision = np.zeros(n_frames)
    Recall = np.zeros(n_frames)
    Time = np.zeros(n_frames)
    cnt = -1
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        time = 0.0
        counter = 0.0
        for i in range(len(preds_eval)):
            tp = np.where(preds_eval[i] * all_labels[i] >= Th)
            Tp += float(len(tp[0]) > 0)
            if float(len(tp[0]) > 0) > 0:
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter + 1
            Tp_Fp += float(len(np.where(preds_eval[i] >= Th)[0]) > 0)
        if Tp_Fp == 0:
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) == 0:
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1 - time/counter)
        if n_frames-cnt > 1:
            cnt += 1
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    _, rep_index = np.unique(Recall, return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index) - 1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i + 1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i + 1]])
    AP = 0.0
    mTTA = 0.0
    TTA_R80 = 0.0
    P_R80 = 0.0
    try:
        new_Time[-1] = Time[rep_index[-1]]
        new_Precision[-1] = Precision[rep_index[-1]]
        new_Recall = Recall[rep_index]
        if new_Recall[0] != 0:
            AP += new_Precision[0] * (new_Recall[0] - 0)
        for i in range(1, len(new_Precision)):
            AP += (new_Precision[i - 1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i - 1])/2
        mTTA = np.mean(new_Time) * total_seconds
        print("Average Precision= %.4f, mean Time to accident= %.4f" % (AP, mTTA))
        sort_time = new_Time[np.argsort(new_Recall)]
        sort_recall = np.sort(new_Recall)
        a = np.where(new_Recall >= 0.8)
        P_R80 = new_Precision[a[0][0]]
        TTA_R80 = sort_time[np.argmin(np.abs(sort_recall - 0.8))] * total_seconds
        print("Precision at Recall 80: %.4f" % P_R80)
        print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))
    except:
        print('Error in calculating')
    return AP, mTTA, TTA_R80, P_R80