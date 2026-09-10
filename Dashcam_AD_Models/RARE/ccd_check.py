import os
from pathlib import Path
from collections import defaultdict

def check_dataset_integrity(root_path: str) -> dict:
    """
    데이터셋의 무결성을 확인하는 함수
    
    Args:
        root_path (str): 데이터셋의 루트 경로
        
    Returns:
        dict: 무결성 검사 결과를 담은 딕셔너리
    """
    results = defaultdict(list)
    expected_frames = 50  # 각 비디오당 예상되는 프레임 수
    
    # training과 testing 폴더 순회
    for split in ['training', 'testing']:
        split_path = os.path.join(root_path, split)
        
        # positive와 negative 폴더 순회
        for label in ['positive', 'negative']:
            label_path = os.path.join(split_path, label)
            
            # 경로가 존재하지 않으면 건너뛰기
            if not os.path.exists(label_path):
                results['missing_directories'].append(label_path)
                continue
                
            # 각 video_id 폴더 순회
            for video_id in os.listdir(label_path):
                video_path = os.path.join(label_path, video_id)
                
                if not os.path.isdir(video_path):
                    continue
                    
                # 프레임 파일 목록 가져오기
                frames = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
                
                # 프레임 개수 확인
                if len(frames) != expected_frames:
                    results['incorrect_frame_count'].append({
                        'path': video_path,
                        'found_frames': len(frames),
                        'expected_frames': expected_frames
                    })
                
                # 프레임 번호 연속성 확인
                frame_numbers = sorted([int(f.split('.')[0]) for f in frames])
                expected_numbers = list(range(1, expected_frames + 1))
                
                missing_frames = set(expected_numbers) - set(frame_numbers)
                if missing_frames:
                    results['missing_frames'].append({
                        'path': video_path,
                        'missing_frame_numbers': sorted(list(missing_frames))
                    })
                
    return dict(results)

def print_integrity_report(results: dict):
    """
    무결성 검사 결과를 출력하는 함수
    
    Args:
        results (dict): 무결성 검사 결과
    """
    print("\n=== 데이터셋 무결성 검사 보고서 ===\n")
    
    if not any(results.values()):
        print("✅ 모든 검사 통과: 데이터셋이 정상입니다.")
        return
        
    if results.get('missing_directories'):
        print("\n❌ 누락된 디렉토리:")
        for dir_path in results['missing_directories']:
            print(f"  - {dir_path}")
            
    if results.get('incorrect_frame_count'):
        print("\n❌ 프레임 개수 불일치:")
        for item in results['incorrect_frame_count']:
            print(f"  - {item['path']}")
            print(f"    예상: {item['expected_frames']}, 실제: {item['found_frames']}")
            
    if results.get('missing_frames'):
        print("\n❌ 누락된 프레임:")
        for item in results['missing_frames']:
            print(f"  - {item['path']}")
            print(f"    누락된 프레임 번호: {item['missing_frame_numbers']}")

# 사용 예시
if __name__ == "__main__":
    dataset_path = "taa/data/CCD/frames"
    results = check_dataset_integrity(dataset_path)
    print_integrity_report(results)