import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline

#calculation of mTTA during training epoch
def evaluation_train(all_pred, all_labels, time_of_accidents, fps=20.0):
    """
    :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
    """

    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]  # positive video
        else:
            pred = all_pred[idx, :]  # negative video
        # find the minimum prediction
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    # iterate a set of thresholds from the minimum predictions
    # temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5)
    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0: # gt of all videos are negative
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    # Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    # new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         # new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    print("mean Time to accident at this training epoch= %.4f"%(mTTA))

    #not necessary
    # sort_time = new_Time[np.argsort(new_Recall)]
    # sort_recall = np.sort(new_Recall)
    # TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    # print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))

    return mTTA


def evaluation(all_pred, all_labels, time_of_accidents, fps=20.0):
    """
    :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
    """

    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]  # positive video
        else:
            pred = all_pred[idx, :]  # negative video
        # find the minimum prediction
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    # iterate a set of thresholds from the minimum predictions
    # temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5)
    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0: # gt of all videos are negative
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    # compute AP (area under P-R curve)
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, mTTA))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    a = np.where(new_Recall>=0.8)
    P_R80 = new_Precision[a[0][0]]
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    print("Precision at Recall 80: %.4f"%(P_R80))
    print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))

    return AP, mTTA, TTA_R80, P_R80

def evaluation_P_R80(all_pred, all_labels, time_of_accidents, fps=20.0):
    """
    :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
    """

    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]  # positive video
        else:
            pred = all_pred[idx, :]  # negative video
        # find the minimum prediction
        #print("hihih", idx, toa, pred.shape)
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    # iterate a set of thresholds from the minimum predictions
    # temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5)
    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0: # gt of all videos are negative
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    # compute AP (area under P-R curve)
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, mTTA))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    a = np.where(new_Recall>=0.8)
    P_R80 = new_Precision[a[0][0]]
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    print("Precision at Recall 80: %.4f"%(P_R80))
    print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))

    return AP, mTTA, TTA_R80, P_R80

# def evaluation_NP_R80(all_pred, all_labels, time_of_accidents, fps=20.0):
#     """
#     :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
#     :param: all_labels (N,)
#     :param: time_of_accidents (N,) int element
#     :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
#     """

#     preds_eval = []
#     min_pred = np.inf
#     n_frames = 0
#     for idx, toa in enumerate(time_of_accidents):
#         if all_labels[idx] > 0:
#             pred = all_pred[idx, :int(toa)]  # positive video
#         else:
#             pred = all_pred[idx, :]  # negative video
#         # find the minimum prediction
#         #print("hihih", idx, toa, pred.shape)
#         min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
#         preds_eval.append(pred)
#         n_frames += len(pred)
#     total_seconds = all_pred.shape[1] / fps

#     # iterate a set of thresholds from the minimum predictions
#     # temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5)
#     Precision = np.zeros((n_frames))
#     Recall = np.zeros((n_frames))
#     Time = np.zeros((n_frames))
#     cnt = 0

#     # Added: Initialize array to store thresholds
#     Thresholds = np.zeros((n_frames)) 

#     for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
#         Tp = 0.0
#         Tp_Fp = 0.0
#         Tp_Tn = 0.0
#         time = 0.0
#         counter = 0.0  # number of TP videos
#         # iterate each video sample
#         for i in range(len(preds_eval)):
#             # true positive frames: (pred->1) * (gt->1)
#             tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
#             Tp += float(len(tp[0])>0)
#             if float(len(tp[0])>0) > 0:
#                 # if at least one TP, compute the relative (1 - rTTA)
#                 time += tp[0][0] / float(time_of_accidents[i])
#                 counter = counter+1
#             # all positive frames
#             Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
#         if Tp_Fp == 0:  # predictions of all videos are negative
#             continue
#         else:
#             Precision[cnt] = Tp/Tp_Fp
#         if np.sum(all_labels) ==0: # gt of all videos are negative
#             continue
#         else:
#             Recall[cnt] = Tp/np.sum(all_labels)
#         if counter == 0:
#             continue
#         else:
#             Time[cnt] = (1-time/counter)
#             Thresholds[cnt] = Th 
#         cnt += 1
 
#     # sort the metrics with recall (ascending)
#     new_index = np.argsort(Recall)
#     Precision = Precision[new_index]
#     Recall = Recall[new_index]
#     Time = Time[new_index]
#     # Added: sort thresholds
#     Thresholds = Thresholds[:cnt][new_index]

#     # unique the recall, and fetch corresponding precisions and TTAs
#     _,rep_index = np.unique(Recall,return_index=1)
#     rep_index = rep_index[1:]
#     new_Time = np.zeros(len(rep_index))
#     new_Precision = np.zeros(len(rep_index))
#     # Added: array for unique thresholds
#     new_Thresholds = np.zeros(len(rep_index))
#     for i in range(len(rep_index)-1):
#          new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
#          new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
#     # sort by descending order
#     new_Time[-1] = Time[rep_index[-1]]
#     new_Precision[-1] = Precision[rep_index[-1]]
#     new_Recall = Recall[rep_index]
#     # compute AP (area under P-R curve)
#     AP = 0.0
#     if new_Recall[0] != 0:
#         AP += new_Precision[0]*(new_Recall[0]-0)
#     for i in range(1,len(new_Precision)):
#         AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

#     # transform the relative mTTA to seconds
#     mTTA = np.mean(new_Time) * total_seconds
#     print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, mTTA))
#     sort_time = new_Time[np.argsort(new_Recall)]
#     sort_recall = np.sort(new_Recall)
#     a = np.where(new_Recall>=0.8)
#     P_R80 = new_Precision[a[0][0]]
#     TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
#     print("Precision at Recall 80: %.4f"%(P_R80))
#     print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))
    
#     # Added: sort thresholds
#     sort_thresholds = new_Thresholds[np.argsort(new_Recall)]  
    
#     # Added: Calculate TTA@0.5
#     time_at_05 = 0.0
#     counter_05 = 0.0
#     for i in range(len(preds_eval)):
#         if all_labels[i] > 0:
#             tp_05 = np.where(preds_eval[i] >= 0.5)[0]
#             if len(tp_05) > 0:
#                 time_at_05 += tp_05[0] / float(time_of_accidents[i])
#                 counter_05 += 1
#     TTA_05 = (1 - time_at_05/counter_05) * total_seconds if counter_05 > 0 else 0

#     # Added: Get threshold at R80
#     threshold_R80 = sort_thresholds[np.argmin(np.abs(sort_recall-0.8))]

#     # Added: Print new metrics
#     print("Time to accident at threshold 0.5= " +"{:.4}".format(TTA_05))
#     print("Threshold at Recall 80%= " +"{:.4}".format(threshold_R80))

#     return AP, mTTA, TTA_R80, P_R80


# def evaluation_kP_R80(all_pred, all_labels, time_of_accidents, fps=20.0):
#     """
#     :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
#     :param: all_labels (N,)
#     :param: time_of_accidents (N,) int element
#     :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
#     """
#     preds_eval = []
#     min_pred = np.inf
#     n_frames = 0
#     for idx, toa in enumerate(time_of_accidents):
#         if all_labels[idx] > 0:
#             pred = all_pred[idx, :int(toa)]  # positive video
#         else:
#             pred = all_pred[idx, :]  # negative video
#         # find the minimum prediction
#         min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
#         preds_eval.append(pred)
#         n_frames += len(pred)
#     total_seconds = all_pred.shape[1] / fps
#     # iterate a set of thresholds from the minimum predictions
#     # temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5)
#     Precision = np.zeros((n_frames))
#     Recall = np.zeros((n_frames))
#     Time = np.zeros((n_frames))
#     cnt = 0
    
#     # Added: Initialize array to store thresholds
#     Thresholds = np.zeros((n_frames))  

#     for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
#         Tp = 0.0
#         Tp_Fp = 0.0
#         Tp_Tn = 0.0
#         time = 0.0
#         counter = 0.0  # number of TP videos
#         # iterate each video sample
#         for i in range(len(preds_eval)):
#             # true positive frames: (pred->1) * (gt->1)
#             tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
#             Tp += float(len(tp[0])>0)
#             if float(len(tp[0])>0) > 0:
#                 # if at least one TP, compute the relative (1 - rTTA)
#                 time += tp[0][0] / float(time_of_accidents[i])
#                 counter = counter+1
#             # all positive frames
#             Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
#         if Tp_Fp == 0:  # predictions of all videos are negative
#             continue
#         else:
#             Precision[cnt] = Tp/Tp_Fp
#         if np.sum(all_labels) ==0: # gt of all videos are negative
#             continue
#         else:
#             Recall[cnt] = Tp/np.sum(all_labels)
#         if counter == 0:
#             continue
#         else:
#             Time[cnt] = (1-time/counter)
#             # Added: store threshold
#             Thresholds[cnt] = Th  
#         cnt += 1
#     # sort the metrics with recall (ascending)
#     new_index = np.argsort(Recall[:cnt])
#     Precision = Precision[:cnt][new_index]
#     Recall = Recall[:cnt][new_index]
#     Time = Time[:cnt][new_index]
#     # Added: sort thresholds
#     Thresholds = Thresholds[:cnt][new_index]
#     # unique the recall, and fetch corresponding precisions and TTAs
#     _, rep_index = np.unique(Recall,return_index=1)
#     rep_index = rep_index[1:]
#     new_Time = np.zeros(len(rep_index))
#     new_Precision = np.zeros(len(rep_index))
#     # Added: array for unique thresholds
#     new_Thresholds = np.zeros(len(rep_index))  
#     for i in range(len(rep_index)-1):
#          new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
#          new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
#          # Added: store threshold
#          new_Thresholds[i] = Thresholds[rep_index[i]]  
#     # sort by descending order
#     new_Time[-1] = Time[rep_index[-1]]
#     new_Precision[-1] = Precision[rep_index[-1]]
#     # Added: store last threshold
#     new_Thresholds[-1] = Thresholds[rep_index[-1]]  
#     new_Recall = Recall[rep_index]
#     # compute AP (area under P-R curve)
#     AP = 0.0
#     if new_Recall[0] != 0:
#         AP += new_Precision[0]*(new_Recall[0]-0)
#     for i in range(1,len(new_Precision)):
#         AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2
#     # transform the relative mTTA to seconds
#     mTTA = np.mean(new_Time) * total_seconds
#     print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, mTTA))
#     sort_time = new_Time[np.argsort(new_Recall)]
#     sort_recall = np.sort(new_Recall)
#     # Added: sort thresholds
#     sort_thresholds = new_Thresholds[np.argsort(new_Recall)]  
#     a = np.where(new_Recall>=0.8)
#     P_R80 = new_Precision[a[0][0]]
#     TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
#     print("Precision at Recall 80: %.4f"%(P_R80))
#     print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))

#     # Added: Calculate TTA@0.5
#     time_at_05 = 0.0
#     counter_05 = 0.0
#     for i in range(len(preds_eval)):
#         if all_labels[i] > 0:
#             tp_05 = np.where(preds_eval[i] >= 0.5)[0]
#             if len(tp_05) > 0:
#                 time_at_05 += tp_05[0] / float(time_of_accidents[i])
#                 counter_05 += 1
#     TTA_05 = (1 - time_at_05/counter_05) * total_seconds if counter_05 > 0 else 0

#     # Added: Get threshold at R80
#     threshold_R80 = sort_thresholds[np.argmin(np.abs(sort_recall-0.8))]

#     # Added: Print new metrics
#     print("Time to accident at threshold 0.5= " +"{:.4}".format(TTA_05))
#     print("Threshold at Recall 80%= " +"{:.4}".format(threshold_R80))

#     return AP, mTTA, TTA_R80, P_R80

def calculate_threshold_at_recall(preds_eval, all_labels, target_recall=0.8):
    """
    Calculate the threshold that achieves the target recall
    
    Args:
    - preds_eval: Predictions for each video
    - all_labels: Ground truth labels
    - target_recall: Target recall threshold (default 0.8)
    
    Returns:
    - Threshold that achieves target recall
    - Actual recall at this threshold
    """
    # Collect all predictions for positive videos
    all_preds = []
    for i, pred in enumerate(preds_eval):
        if all_labels[i] > 0:
            all_preds.extend(pred)
    
    # Sort predictions in descending order
    all_preds = np.sort(all_preds)[::-1]
    
    # Total number of positive samples
    total_positive = np.sum(all_labels > 0)
    
    # Find thresholds
    best_threshold = 0
    best_recall = 0
    
    for threshold in all_preds:
        # Count videos that have any prediction above this threshold
        recalled_videos = 0
        for i, pred in enumerate(preds_eval):
            if all_labels[i] > 0:  # Only check positive videos
                if np.any(pred >= threshold):
                    recalled_videos += 1
        
        # Calculate recall
        recall = recalled_videos / total_positive
        
        # Update best threshold if recall meets or exceeds target
        if recall >= target_recall and threshold > best_threshold:
            best_threshold = threshold
            best_recall = recall
    
    return best_threshold, best_recall

def calculate_precision_recall_curve(all_pred, all_labels, time_of_accidents):
    """
    Calculate Precision and Recall for various thresholds
    
    :param all_pred: Predictions for all videos (N x T)
    :param all_labels: Labels for all videos (N,)
    :param time_of_accidents: Time of accidents for each video (N,)
    :return: 
        - thresholds: Array of thresholds used
        - precisions: Corresponding precision values
        - recalls: Corresponding recall values
    """
    # Define thresholds from 0 to 1 in steps of 0.05
    thresholds = np.arange(0, 1.01, 0.05)
    
    # Initialize arrays to store precision and recall
    precisions = []
    recalls = []
    
    # Calculate precision and recall for each threshold
    for threshold in thresholds:
        # Initialize counters
        TP = 0.0  # True Positives
        FP = 0.0  # False Positives
        FN = 0.0  # False Negatives
        
        # Iterate through each video
        for i in range(len(all_pred)):
            # For positive videos, consider only frames up to time of accident
            if all_labels[i] > 0:
                pred = all_pred[i, :int(time_of_accidents[i])]
            else:
                pred = all_pred[i, :]
            
            # Check if video is predicted positive at this threshold
            pred_pos = np.any(pred >= threshold)
            
            # Update counters based on ground truth and prediction
            if all_labels[i] > 0:  # Positive video
                if pred_pos:
                    TP += 1.0  # Correctly predicted positive
                else:
                    FN += 1.0  # Missed positive
            else:  # Negative video
                if pred_pos:
                    FP += 1.0  # Incorrectly predicted positive
        
        # Calculate precision and recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return thresholds, precisions, recalls

def evaluation_NP_R80(all_pred, all_labels, time_of_accidents, fps=20.0):
    """
    :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%), 
             TTA@Threshold0.5, Threshold@Recall80
    """
    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]  # positive video
        else:
            pred = all_pred[idx, :]  # negative video
        # find the minimum prediction
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps
    # iterate a set of thresholds from the minimum predictions
    # temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5)
    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0: # gt of all videos are negative
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _, rep_index = np.unique(Recall, return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    # compute AP (area under P-R curve)
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2
    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, mTTA))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    a = np.where(new_Recall>=0.8)
    P_R80 = new_Precision[a[0][0]]
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    print("Precision at Recall 80: %.4f"%(P_R80))
    print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))

    # NEW: Calculate Time to Accident at Threshold 0.5
    # Comment: Added calculation for Time to Accident at fixed threshold of 0.5
    TTA_at_threshold_0_5 = 0
    for i in range(len(preds_eval)):
        if all_labels[i] > 0:  # only for positive videos
            threshold_frames = np.where(preds_eval[i] >= 0.5)[0]
            if len(threshold_frames) > 0:
                TTA_at_threshold_0_5 += threshold_frames[0] / float(time_of_accidents[i])
    TTA_at_threshold_0_5 = (1 - TTA_at_threshold_0_5 / np.sum(all_labels > 0)) * total_seconds

    
    # NEW: Calculate Precision and Recall at 0.5 Threshold
    Precision_at_0_5 = 0.0
    Recall_at_0_5 = 0.0
    
    # True Positives, False Positives, and False Negatives at 0.5 threshold
    TP = 0.0  # Correctly predicted positive samples
    FP = 0.0  # Negative samples incorrectly predicted as positive
    FN = 0.0  # Positive samples incorrectly predicted as negative
    
    for i in range(len(preds_eval)):
        # Frames above 0.5 threshold
        pred_pos_frames = np.where(preds_eval[i] >= 0.5)[0]
        
        if all_labels[i] > 0:  # Positive video
            if len(pred_pos_frames) > 0:
                TP += 1.0  # Correctly predicted positive video
            else:
                FN += 1.0  # Missed positive video
        else:  # Negative video
            if len(pred_pos_frames) > 0:
                FP += 1.0  # Incorrectly predicted positive
    
    # Calculate Precision and Recall
    Precision_at_0_5 = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    Recall_at_0_5 = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    print("Precision at Threshold 0.5= {:.4f}".format(Precision_at_0_5))
    print("Recall at Threshold 0.5= {:.4f}".format(Recall_at_0_5))

    thresholds, precisions, recalls = calculate_precision_recall_curve(all_pred, all_labels, time_of_accidents)
    
    # Optional: Add print or plot functionality
    print("Thresholds:", thresholds)
    print("Precisions:", precisions)
    print("Recalls:", recalls)

    # threshold_at_recall_80, actual_recall = calculate_threshold_at_recall(preds_eval, all_labels, 0.8)

    # print("actual_recall", actual_recall) 
    print("Time to Accident at Threshold 0.5= {:.4f}".format(TTA_at_threshold_0_5))
    # print("Threshold at Recall 80%= {:.4f}".format(threshold_at_recall_80))

    return AP, mTTA, TTA_R80, P_R80

def print_results(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, result_dir):
    result_file = os.path.join(result_dir, 'eval_all.txt')
    with open(result_file, 'w') as f:
        for e, APvid, AP, mTTA, TTA_R80 in zip(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all):
            f.writelines('Epoch: %s,'%(e) + ' APvid={:.3f}, AP={:.3f}, mTTA={:.3f}, TTA_R80={:.3f}\n'.format(APvid, AP, mTTA, TTA_R80))
    f.close()


def vis_results(vis_data, batch_size, vis_dir, smooth=False, vis_batchnum=2):
    pass
