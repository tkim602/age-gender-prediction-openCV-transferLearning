import os

# organized_data 폴더의 경로를 지정하세요
base_path = "organized_data"

# .DS_Store 파일을 삭제하는 함수
def remove_ds_store_files(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == ".DS_Store":
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed: {file_path}")

# 실행하여 .DS_Store 파일 삭제
remove_ds_store_files(base_path)
