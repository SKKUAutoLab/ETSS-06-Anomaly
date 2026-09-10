"""
GT annotation을 시각화하여 비디오로 저장하는 도구

사용 예시:
    python visualize_gt.py \
        --root_path ./taa/data/DAD \
        --split training \
        --video_id 000002 \
        --output_dir ./taa/_visualization/gt_videos
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple


def load_annotations(anno_path: str) -> Dict[int, List[Dict]]:
    """프레임별 annotation을 로드합니다.
    
    Args:
        anno_path: annotation 파일 경로
        
    Returns:
        Dict[int, List[Dict]]: {
            frame_id: [{
                'track_id': int,
                'class': str,
                'bbox': [x1, y1, x2, y2],
                'is_related': int
            }, ...]
        }
    """
    frame_annos = {}
    
    with open(anno_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            frame_id = int(parts[0])
            
            # annotation 정보 구성
            anno = {
                'track_id': int(parts[1]),
                'class': parts[2],
                'bbox': [int(float(x)) for x in parts[3:7]],  # [x1, y1, x2, y2]
                'is_related': int(parts[7])
            }
            
            # 프레임별로 리스트에 추가
            if frame_id not in frame_annos:
                frame_annos[frame_id] = []
            frame_annos[frame_id].append(anno)
    
    return frame_annos


def draw_annotations(
    frame: np.ndarray,
    annotations: List[Dict],
    frame_idx: int,
    font_scale: float = 0.5,
    thickness: int = 2
) -> np.ndarray:
    """한 프레임의 annotation을 시각화합니다.
    
    Args:
        frame: BGR 이미지
        annotations: 해당 프레임의 annotation 리스트
        frame_idx: 현재 프레임 번호
        font_scale: 텍스트 크기
        thickness: 선 두께
        
    Returns:
        np.ndarray: 시각화된 이미지
    """
    img = frame.copy()
    
    # 프레임 번호 표시 (오른쪽 위)
    frame_text = f"Frame: {frame_idx:06d}"
    (text_w, text_h), _ = cv2.getTextSize(
        frame_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 1.5,  # 프레임 번호는 좀 더 크게
        thickness
    )
    
    # 텍스트 배경
    margin = 10
    cv2.rectangle(
        img,
        (img.shape[1] - text_w - margin * 2, 0),
        (img.shape[1], text_h + margin * 2),
        (0, 0, 0),
        -1
    )
    
    # 프레임 번호 텍스트
    cv2.putText(
        img,
        frame_text,
        (img.shape[1] - text_w - margin, text_h + margin),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 1.5,
        (255, 255, 255),
        thickness
    )
    
    for anno in annotations:
        # Bbox 좌표
        x1, y1, x2, y2 = anno['bbox']
        
        # 사고 관련 여부에 따른 색상 (BGR)
        color = (0, 255, 0) if anno['is_related'] == 0 else (0, 0, 255)
        
        # Bbox 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Track ID와 클래스 텍스트
        text = f"ID:{anno['track_id']} {anno['class']}"
        
        # 텍스트 크기 계산
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # 텍스트 배경 그리기
        cv2.rectangle(
            img,
            (x1, y1 - text_h - 8),
            (x1 + text_w + 8, y1),
            color,
            -1  # filled
        )
        
        # 텍스트 그리기
        cv2.putText(
            img,
            text,
            (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # white
            thickness
        )
    
    return img


def create_gt_video(
    root_path: str,
    split: str,
    video_id: str,
    output_dir: str,
    fps: int = 20
):
    """GT annotation이 표시된 비디오를 생성합니다.
    
    Args:
        root_path: 데이터셋 루트 경로
        split: 'training' 또는 'testing'
        video_id: 비디오 ID (e.g., '000002')
        output_dir: 출력 디렉토리
        fps: 출력 비디오의 FPS
    """
    # 경로 설정
    frames_dir = os.path.join(root_path, 'frames', split, 'positive', video_id)
    anno_path = os.path.join(root_path, 'annotation', f'{video_id}.txt')
    
    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"프레임 디렉토리를 찾을 수 없습니다: {frames_dir}")
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"Annotation 파일을 찾을 수 없습니다: {anno_path}")
    
    # Annotation 로드
    frame_annos = load_annotations(anno_path)
    
    # 첫 프레임으로 비디오 설정
    first_frame = cv2.imread(os.path.join(frames_dir, '000001.jpg'))
    if first_frame is None:
        raise ValueError(f"프레임을 읽을 수 없습니다: {frames_dir}/000001.jpg")
    
    height, width = first_frame.shape[:2]
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 비디오 writer 초기화
    output_path = os.path.join(output_dir, f'{video_id}_gt.mp4')
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    # 프레임 처리
    n_frames = len(os.listdir(frames_dir))
    print(f"\n비디오 생성 중: {output_path}")
    
    for frame_idx in tqdm(range(1, n_frames + 1)):
        # 프레임 로드
        frame_path = os.path.join(frames_dir, f'{frame_idx:06d}.jpg')
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: 프레임을 읽을 수 없습니다: {frame_path}")
            continue
        
        # Annotation 시각화
        if frame_idx in frame_annos:
            frame = draw_annotations(frame, frame_annos[frame_idx], frame_idx)
        else:
            # annotation이 없는 프레임도 프레임 번호는 표시
            frame = draw_annotations(frame, [], frame_idx)
        
        # 프레임 저장
        writer.write(frame)
    
    # 정리
    writer.release()
    print(f"완료: {output_path}")


def create_gt_videos(
    root_path: str,
    split: str,
    output_dir: str,
    fps: int = 10
):
    """해당 split의 모든 비디오에 대해 GT annotation이 표시된 비디오를 생성합니다."""
    # positive 비디오 디렉토리
    pos_dir = os.path.join(root_path, 'frames', split, 'positive')
    if not os.path.exists(pos_dir):
        raise FileNotFoundError(f"positive 비디오 디렉토리를 찾을 수 없습니다: {pos_dir}")
    
    # 모든 비디오 ID 가져오기
    video_ids = sorted(os.listdir(pos_dir))
    
    print(f"\n{split} 데이터셋의 모든 비디오 처리 중...")
    print(f"총 {len(video_ids)}개의 비디오 발견")
    
    # 각 비디오 처리
    for video_id in video_ids:
        try:
            create_gt_video(
                root_path=root_path,
                split=split,
                video_id=video_id,
                output_dir=output_dir,
                fps=fps
            )
        except Exception as e:
            print(f"\nError processing video {video_id}: {str(e)}")
    
    print(f"\n모든 비디오 처리 완료: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True,
                        help="데이터셋 루트 경로 (e.g., ./taa/data/DAD)")
    parser.add_argument('--split', type=str, choices=['training', 'testing'],
                        default='training', help="데이터셋 split")
    parser.add_argument('--video_id', type=str, required=True,
                        help="비디오 ID (e.g., 000002) 또는 'all'")
    parser.add_argument('--output_dir', type=str, default='./taa/_visualization/gt_videos',
                        help="출력 디렉토리")
    parser.add_argument('--fps', type=int, default=10,
                        help="출력 비디오의 FPS")
    args = parser.parse_args()
    
    if args.video_id.lower() == 'all':
        create_gt_videos(
            root_path=args.root_path,
            split=args.split,
            output_dir=args.output_dir,
            fps=args.fps
        )
    else:
        create_gt_video(
            root_path=args.root_path,
            split=args.split,
            video_id=args.video_id,
            output_dir=args.output_dir,
            fps=args.fps
        )


if __name__ == '__main__':
    main() 