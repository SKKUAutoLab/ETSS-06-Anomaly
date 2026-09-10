import os
import json
import time
from datetime import datetime
from typing import Dict, Any
import torch
import pandas as pd
import numpy as np


class ExpLogger:
    """실험 로깅을 위한 클래스
    
    실험 정보, 학습 로그, 평가 결과를 구조화된 형식으로 저장합니다.
    """
    
    def __init__(self):
        # 실험 시작 시간으로 고유한 실험 ID 생성
        self.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 로그 디렉토리 생성
        self.log_dir = os.path.join('./taa/_experiments', self.exp_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 체크포인트 디렉토리 생성
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 학습 로그를 저장할 DataFrame 초기화
        self.train_log = pd.DataFrame()
        self.eval_log = pd.DataFrame()
        
        # 마지막 저장 시간 초기화
        self.last_save_time = time.time()
        
        # 현재 에포크의 누적 통계를 위한 변수들 추가
        self.current_epoch_stats = {
            'losses': {},
            'steps': 0
        }
        
        # 에포크별 요약 통계를 저장할 DataFrame 추가
        self.epoch_summary = pd.DataFrame()
    
    def save_exp_info(self, config: Dict[str, Any]) -> None:
        """실험 설정 정보를 JSON 형식으로 저장"""
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def log_train_step(self, epoch: int, step: int, losses: Dict[str, float], lr: float) -> None:
        """학습 스텝별 로그 기록 및 현재 에포크 통계 업데이트"""
        # 기존 스텝 로그 기록
        log_entry = {
            'epoch': epoch,
            'step': step,
            'timestamp': time.time(),
            'learning_rate': lr,
            **losses
        }
        
        self.train_log = pd.concat([
            self.train_log,
            pd.DataFrame([log_entry])
        ], ignore_index=True)
        
        # 현재 에포크 통계 업데이트
        if self.current_epoch_stats['steps'] == 0 or self.current_epoch_stats.get('epoch') != epoch:
            # 새로운 에포크 시작
            self.current_epoch_stats = {
                'epoch': epoch,
                'losses': {k: v for k, v in losses.items()},
                'steps': 1,
                'learning_rate': lr
            }
        else:
            # 현재 에포크 통계 업데이트
            for k, v in losses.items():
                if k not in self.current_epoch_stats['losses']:
                    self.current_epoch_stats['losses'][k] = v
                else:
                    # 이동 평균 계산
                    self.current_epoch_stats['losses'][k] = (
                        self.current_epoch_stats['losses'][k] * self.current_epoch_stats['steps'] + v
                    ) / (self.current_epoch_stats['steps'] + 1)
            self.current_epoch_stats['steps'] += 1
        
        # 주기적으로 로그 저장
        if time.time() - self.last_save_time > 60:
            self._save_logs()
            self.last_save_time = time.time()
    
    def log_eval(self, epoch: int, losses: Dict[str, float], metrics: Dict[str, float]) -> None:
        """검증 결과 로그 기록"""
        log_entry = {
            'epoch': epoch,
            'timestamp': time.time(),
            **{f'loss/{k}': v for k, v in losses.items()},
            **{f'metric/{k}': v for k, v in metrics.items()}
        }
        
        self.eval_log = pd.concat([
            self.eval_log,
            pd.DataFrame([log_entry])
        ], ignore_index=True)
        
        self._save_logs()
    
    def get_checkpoint_path(self, name: str) -> str:
        """체크포인트 파일 경로 반환"""
        return os.path.join(self.checkpoint_dir, f'{name}.pth')
    
    def _save_logs(self) -> None:
        """로그를 CSV 파일로 저장"""
        if not self.train_log.empty:
            train_log_path = os.path.join(self.log_dir, 'train_log.csv')
            self.train_log.to_csv(train_log_path, index=False)
        
        if not self.eval_log.empty:
            eval_log_path = os.path.join(self.log_dir, 'eval_log.csv')
            self.eval_log.to_csv(eval_log_path, index=False)
            
        if not self.epoch_summary.empty:
            summary_path = os.path.join(self.log_dir, 'epoch_summary.csv')
            self.epoch_summary.to_csv(summary_path, index=False)
    
    def get_best_metrics(self) -> Dict[str, float]:
        """최고 성능 메트릭 반환"""
        if self.eval_log.empty:
            return {}
        
        metric_cols = [col for col in self.eval_log.columns if col.startswith('metric/')]
        best_metrics = {}
        
        for col in metric_cols:
            metric_name = col.split('/')[-1]
            if metric_name in ['auc_roc', 'ap']:
                # 높을수록 좋은 메트릭
                best_value = self.eval_log[col].max()
            else:
                # 낮을수록 좋은 메트릭 (예: loss)
                best_value = self.eval_log[col].min()
            best_metrics[metric_name] = best_value
        
        return best_metrics
    
    def log_epoch_end(self, epoch: int) -> None:
        """에포크 종료 시 요약 통계 저장"""
        if self.current_epoch_stats['steps'] > 0:
            summary_entry = {
                'epoch': epoch,
                'timestamp': time.time(),
                'total_steps': self.current_epoch_stats['steps'],
                'learning_rate': self.current_epoch_stats['learning_rate'],
                **{f'avg_train_{k}': v for k, v in self.current_epoch_stats['losses'].items()}
            }
            
            self.epoch_summary = pd.concat([
                self.epoch_summary,
                pd.DataFrame([summary_entry])
            ], ignore_index=True)
            
            # 에포크 요약 저장
            self._save_logs()
            
            # 현재 에포크 통계 초기화
            self.current_epoch_stats = {
                'losses': {},
                'steps': 0
            }
    
    def get_current_epoch_stats(self) -> Dict[str, Any]:
        """현재 에포크의 실시간 통계 반환"""
        return {
            'epoch': self.current_epoch_stats.get('epoch'),
            'steps': self.current_epoch_stats['steps'],
            'avg_losses': self.current_epoch_stats['losses'],
            'learning_rate': self.current_epoch_stats.get('learning_rate')
        }
    
    def plot_training_curves(self, save_path: str = None) -> None:
        """학습 곡선 플롯 생성 및 저장 (에포크 요약 포함)"""
        try:
            import matplotlib.pyplot as plt
            
            # 1. Loss 곡선 (검증 손실)
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            loss_cols = [col for col in self.eval_log.columns if col.startswith('loss/')]
            for col in loss_cols:
                plt.plot(self.eval_log['epoch'], self.eval_log[col], label=f'val_{col.split("/")[-1]}')
            
            # 학습 손실 추가
            train_loss_cols = [col for col in self.epoch_summary.columns if col.startswith('avg_train_')]
            for col in train_loss_cols:
                plt.plot(self.epoch_summary['epoch'], self.epoch_summary[col], 
                        label=f'train_{col.split("_")[-1]}', linestyle='--')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Losses')
            plt.legend()
            plt.grid(True)
            
            # 2. 메트릭 곡선
            plt.subplot(2, 1, 2)
            metric_cols = [col for col in self.eval_log.columns if col.startswith('metric/')]
            for col in metric_cols:
                plt.plot(self.eval_log['epoch'], self.eval_log[col], label=col.split('/')[-1])
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.title('Validation Metrics')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            if save_path:
                plt.savefig(os.path.join(save_path, 'training_curves.png'))
            plt.close()
            
        except ImportError:
            print("matplotlib is required for plotting training curves")