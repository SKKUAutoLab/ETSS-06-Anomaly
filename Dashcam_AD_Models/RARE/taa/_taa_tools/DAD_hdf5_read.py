import h5py
import numpy as np

def analyze_dad_hdf5(file_path):
    """
    DAD 데이터셋의 HDF5 파일 구조를 분석하고 주요 정보를 출력합니다.
    파일은 다음과 같은 계층 구조를 가집니다:
    - frames/
        - positive/
            - video_id1
            - video_id2
            ...
        - negative/
            - video_id1
            - video_id2
            ...
    - labels/
        - positive/
        - negative/
    """
    with h5py.File(file_path, 'r') as f:
        print(f"\n파일 분석: {file_path}")
        
        # 비디오 개수 확인
        n_pos_videos = len(f['frames/positive'])
        n_neg_videos = len(f['frames/negative'])
        total_videos = n_pos_videos + n_neg_videos
        
        print("\n1. 데이터셋 크기:")
        print(f"- 전체 비디오 개수: {total_videos}")
        print(f"  → Positive 비디오: {n_pos_videos}")
        print(f"  → Negative 비디오: {n_neg_videos}")
        
        # 프레임 데이터 형태 확인
        first_pos_key = list(f['frames/positive'].keys())[0] if n_pos_videos > 0 else None
        first_neg_key = list(f['frames/negative'].keys())[0] if n_neg_videos > 0 else None
        
        # positive나 negative 중 존재하는 첫 비디오로 shape 확인
        sample_video = None
        if first_pos_key:
            sample_video = f['frames/positive'][first_pos_key]
        elif first_neg_key:
            sample_video = f['frames/negative'][first_neg_key]
            
        if sample_video is not None:
            frame_shape = sample_video.shape
            print(f"\n2. 비디오 데이터 형태:")
            print(f"- 각 비디오 텐서 shape: {frame_shape}")
            print(f"  → {frame_shape[0]}개 프레임")
            print(f"  → {frame_shape[1]}채널 (RGB)")
            print(f"  → {frame_shape[2]}x{frame_shape[3]} 해상도")
            
            # 압축 정보 확인
            compression = sample_video.compression
            if compression:
                print(f"- 압축 방식: {compression}")
        
        print("\n3. HDF5 파일 구조:")
        print("- frames/")
        print(f"  └─ positive/ ({n_pos_videos} videos)")
        print(f"  └─ negative/ ({n_neg_videos} videos)")
        print("- labels/")
        print("  └─ positive/")
        print("  └─ negative/")

# training과 testing 데이터셋 모두 분석
# print("[Training 데이터셋]")
# analyze_dad_hdf5('taa/data/DAD/hdf5/training.hdf5')

print("\n[Testing 데이터셋]")
analyze_dad_hdf5('taa/data/DAD/hdf5/testing.hdf5')