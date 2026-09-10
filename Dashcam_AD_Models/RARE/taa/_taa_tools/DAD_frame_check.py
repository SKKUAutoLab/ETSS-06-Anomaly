import os


def check_image_count(root_path):
    # 이미지 확장자 목록
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    problems = 0

    # 모든 하위 디렉토리 순회
    for dirpath, dirnames, filenames in os.walk(root_path):
        # 숫자로만 이루어진 폴더명을 가진 최하위 폴더 확인
        if os.path.basename(dirpath).isdigit():
            # 이미지 파일 수 계산
            image_count = sum(1 for f in filenames if os.path.splitext(f)[1].lower() in image_extensions)

            # 결과 출력
            if image_count != 100:
                print(f"Warning: {dirpath} contains {image_count} images (expected 100)")
                problems += 1

    if problems == 0:
        print(f"OK: all folders contains 100 images")
    print()

# 사용 예시
check_image_count("/home/i2slab/TAA/Datasets/DAD/frames/training/")