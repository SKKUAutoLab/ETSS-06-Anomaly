import os
import yaml
from typing import Dict, Any

class Config:
    """설정 관리를 위한 클래스"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """YAML 설정 파일 로드
        
        Args:
            config_path: YAML 설정 파일 경로
            
        Returns:
            Dict[str, Any]: 설정 딕셔너리
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) 