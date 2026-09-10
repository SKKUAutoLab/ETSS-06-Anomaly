import os
import numpy as np
import torch
from typing import Dict
from sklearn import metrics
import matplotlib.pyplot as plt


class TAAMetrics:
    """TAA 태스크를 위한 다양한 평가 지표 계산
    
    주요 지표:
    1. AUC-ROC: 전체적인 성능
    2. AP (Average Precision): 정밀도-재현율 곡선 아래 면적
    3. TTA@R80: Recall 80%에서의 Time-to-Accident
    4. mTTA: Mean Time-to-Accident
    """
    
    def __init__(self, dataset: str = 'DAD'):
        """
        Args:
            dataset (str): 데이터셋 이름
        """
        if dataset == 'DAD':
            self.fps = 20.0
            self.sequence_length = 100
            self.accident_frame = 90
        elif dataset == 'CCD':
            self.fps = 10.0
            self.sequence_length = 50
            self.accident_frame = 30
        
        # 결과 저장을 위한 디렉토리 생성
        self.chart_dir = './taa/_charts'
        os.makedirs(os.path.join(self.chart_dir, 'auc'), exist_ok=True)
        os.makedirs(os.path.join(self.chart_dir, 'pr'), exist_ok=True)
    
    def _preprocess_sequences(
        self,
        predictions: np.ndarray,  # (N*T)
        targets: np.ndarray,      # (N*T)
    ) -> tuple[np.ndarray, np.ndarray]:
        """시퀀스 전처리: positive/negative 구분하여 적절한 프레임 수 사용
        
        Args:
            predictions: 예측값 (N*T)
            targets: 정답 레이블 (N*T)
            
        Returns:
            tuple[np.ndarray, np.ndarray]: 처리된 predictions, targets
        """
        # 100프레임 단위로 데이터 재구성

        sequence_length = self.sequence_length
        num_sequences = len(predictions) // sequence_length
        predictions = predictions[:num_sequences * sequence_length].reshape(num_sequences, sequence_length)
        targets = targets[:num_sequences * sequence_length].reshape(num_sequences, sequence_length)
        
        # positive/negative 시퀀스 구분
        positive_sequences = np.where(targets[:, -1] == 1)[0]
        
        # 결과를 저장할 리스트
        processed_predictions = []
        processed_targets = []
        
        for i in range(num_sequences):
            if i in positive_sequences:
                # positive 시퀀스는 90프레임까지만 사용
                processed_predictions.append(predictions[i, :self.accident_frame])
                processed_targets.append(targets[i, :self.accident_frame])
            else:
                # negative 시퀀스는 전체 프레임 사용
                processed_predictions.append(predictions[i])
                processed_targets.append(targets[i])
        
        return np.concatenate(processed_predictions), np.concatenate(processed_targets)
    
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """모든 평가 지표 계산"""
        time_of_accidents = self.accident_frame
        
        print("predictions shape: ", predictions.shape)
        print("targets shape: ", targets.shape)
        
        # CPU로 이동 및 numpy 변환
        predictions = predictions.cpu().numpy()
        sequence_length = self.sequence_length
        num_sequences = len(predictions) // sequence_length
        predictions = predictions[:num_sequences * sequence_length].reshape(num_sequences, sequence_length)
        targets = targets.cpu().numpy()

        print("predictions shape: ", predictions.shape)
        print("targets shape: ", targets.shape)

        # save predictions and targets to csv
        np.savetxt('./taa/_charts/predictions.csv', predictions, delimiter=',')
        np.savetxt('./taa/_charts/targets.csv', targets, delimiter=',')
        
        # pick first and sequence_length interval
        sample_indices = np.arange(0, num_sequences*sequence_length, sequence_length)
        print("sample_indices: ", sample_indices)
        targets = targets[sample_indices]
        
        # if target is 1, time_of_accidents is 90, else 101
        time_of_accidents = np.where(targets == 1, self.accident_frame, sequence_length+1)
        
        preds_eval = []
        min_pred = np.inf
        n_frames = 0
        for idx in range(predictions.shape[0]):
            if targets[idx] > 0:
                pred = predictions[idx, :self.accident_frame]  # positive video
            else:
                pred = predictions[idx, :]  # negative video
            # find the minimum prediction
            min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
            preds_eval.append(pred)
            n_frames += len(pred)
        total_seconds = predictions.shape[1] / self.fps
            # iterate a set of thresholds from the minimum predictions
            
        threholds = np.arange(max(min_pred,0), 1.0, 0.001)
        threholds_num = threholds.shape[0]
        Precision = np.zeros((threholds_num))
        Recall = np.zeros((threholds_num))
        Time = np.zeros((threholds_num)) 
        cnt = 0
        for Th in threholds:
            Tp = 0.0
            Tp_Fp = 0.0
            Tp_Tn = 0.0
            time = 0.0
            counter = 0.0  # number of TP videos
            # iterate each video sample
            for i in range(len(preds_eval)):
                # true positive frames: (pred->1) * (gt->1)
                tp =  np.where(preds_eval[i]*targets[i]>=Th)
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
            if np.sum(targets) ==0: # gt of all videos are negative
                continue
            else:
                Recall[cnt] = Tp/np.sum(targets)
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


        return {
            'ap': AP,
            'mtta': mTTA,
            'tta_r80': TTA_R80
        }