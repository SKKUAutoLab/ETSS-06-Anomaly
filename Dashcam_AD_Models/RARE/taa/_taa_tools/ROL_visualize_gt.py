"""
ROL 데이터셋의 GT annotation을 시각화하여 비디오로 저장하는 도구

사용 예시:
    python ROL_visualize_gt.py \
        --root_path ./taa/data/ROL \
        --split train \
        --video_id video_001 \
        --output_dir ./taa/_visualization/rol_gt_videos
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
        anno_path: npz annotation 파일 경로
        
    Returns:
        Dict[int, List[Dict]]: {
            frame_idx: [{
                'track_id': int,
                'bbox': [x1, y1, x2, y2],
                'is_related': int
            }, ...]
        }
    """
    frame_annos = {}
    
    # npz 파일 로드
    data = np.load(anno_path)
    detections = data['detection']  # [100, 30, 6]
    
    # 각 프레임의 detection 처리
    for frame_idx, frame_detections in enumerate(detections, start=1):
        # 실제 객체가 있는 detection만 필터링 (모든 값이 0인 행 제외)
        valid_detections = frame_detections[~np.all(frame_detections == 0, axis=1)]
        
        if len(valid_detections) > 0:  # 유효한 detection이 있는 경우만 저장
            frame_objects = []
            
            # 각 객체 정보를 딕셔너리로 변환
            for det in valid_detections:
                obj_info = {
                    'track_id': int(det[0]),
                    'bbox': [float(x) for x in det[1:5]],  # [x1, y1, x2, y2]
                    'is_related': int(det[5])  # accident 필드를 is_related로 매핑
                }
                frame_objects.append(obj_info)
            
            frame_annos[frame_idx] = frame_objects
    
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
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # float -> int 변환
        
        # 사고 관련 여부에 따른 색상 (BGR)
        color = (0, 255, 0) if anno['is_related'] == 0 else (0, 0, 255)
        
        # Bbox 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Track ID 텍스트
        text = f"ID:{anno['track_id']}"
        
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
        split: 'train' 또는 'val'
        video_id: 비디오 ID
        output_dir: 출력 디렉토리
        fps: 출력 비디오의 FPS
    """
    # 경로 설정
    frames_dir = os.path.join(root_path, 'frames', split, video_id)
    anno_path = os.path.join(root_path, 'annotations', split, f'{video_id}.npz')
    
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
    fps: int = 20
):
    """해당 split의 모든 비디오에 대해 GT annotation이 표시된 비디오를 생성합니다."""
    # 비디오 디렉토리
    videos_dir = os.path.join(root_path, 'frames', split)
    if not os.path.exists(videos_dir):
        raise FileNotFoundError(f"비디오 디렉토리를 찾을 수 없습니다: {videos_dir}")
    
    # 모든 비디오 ID 가져오기
    video_ids = sorted(os.listdir(videos_dir))
    
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
                        help="데이터셋 루트 경로 (e.g., ./taa/data/ROL)")
    parser.add_argument('--split', type=str, choices=['train', 'val'],
                        default='train', help="데이터셋 split")
    parser.add_argument('--video_id', type=str, required=True,
                        help="비디오 ID 또는 'all'")
    parser.add_argument('--output_dir', type=str, default='./taa/_visualization/rol_gt_videos',
                        help="출력 디렉토리")
    parser.add_argument('--fps', type=int, default=20,
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