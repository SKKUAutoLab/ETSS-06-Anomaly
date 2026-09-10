import os
import shutil
from pathlib import Path

def organize_frames(base_dir, output_dir):
    # 입력/출력 경로 설정
    vgg_feature_dir = os.path.join(base_dir, 'vgg16_features')
    frames_dir = os.path.join(base_dir, 'frames')
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # train과 test 데이터셋 정보 읽기
    datasets = {}
    for subset in ['train', 'test']:
        txt_path = os.path.join(vgg_feature_dir, f'{subset}.txt')
        if not os.path.exists(txt_path):
            print(f"Warning: {txt_path} not found")
            continue
            
        with open(txt_path, 'r') as f:
            for line in f:
                # npz 파일 경로에서 비디오 ID 추출
                video_path, label = line.strip().split()
                category, video_id = video_path.split('/')
                video_id = video_id.replace('.npz', '')
                
                # 비디오 정보 저장
                datasets[f"{category}/{video_id}"] = subset
    
    # 프레임 폴더 이동
    for category in ['negative', 'positive']:
        category_dir = os.path.join(frames_dir, category)
        if not os.path.exists(category_dir):
            print(f"Warning: {category_dir} not found")
            continue
            
        for video_id in os.listdir(category_dir):
            video_key = f"{category}/{video_id}"
            if video_key not in datasets:
                print(f"Warning: No subset info for {video_key}")
                continue
                
            # 원본 및 대상 경로
            src_path = os.path.join(frames_dir, category, video_id)
            dst_path = os.path.join(output_dir, datasets[video_key], category, video_id)
            
            if not os.path.exists(src_path):
                print(f"Warning: Source path not found: {src_path}")
                continue
                
            # 대상 디렉토리 생성 및 이동
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            try:
                shutil.move(src_path, dst_path)
                print(f"Moved: {video_key} -> {datasets[video_key]}")
            except Exception as e:
                print(f"Error moving {video_key}: {str(e)}")

if __name__ == "__main__":
    base_dir = "taa/data/CCD"  # 기본 디렉토리 경로
    output_dir = "taa/data/CCD/frames_organized"  # 새로운 구조의 출력 디렉토리
    
    organize_frames(base_dir, output_dir)