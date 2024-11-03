import os

# 데이터셋의 기본 디렉토리 경로
base_dir = 'organized_data'

# 나이 및 성별 그룹을 확인하기 위한 리스트
age_groups = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_groups = ['female', 'male']

# 모든 파일 경로 확인
for age_group in age_groups:
    for gender_group in gender_groups:
        folder_path = os.path.join(base_dir, age_group, gender_group)
        
        if os.path.exists(folder_path):
            print(f"Found folder: {folder_path}")
            
            # 폴더 내 파일 수 세기
            file_count = len(os.listdir(folder_path))
            print(f"Number of images in {folder_path}: {file_count}")
        else:
            print(f"Warning: Folder not found: {folder_path}")
