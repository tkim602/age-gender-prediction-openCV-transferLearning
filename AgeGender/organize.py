import os
import shutil
import pandas as pd

data_parent = 'dataset'

# read all fold txt files 
folds = []
for i in range(5):  # fold_0_data.txt to fold_4_data.txt
    fold_data = pd.read_csv(os.path.join(data_parent, f'fold_{i}_data.txt'), sep='\t')
    folds.append(fold_data)

# concat all in one 
total_data = pd.concat(folds, ignore_index=True)

image_folder = os.path.join(data_parent, 'faces')
output_folder = 'organized_data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

age_map = {
    '(0, 2)': '(0, 2)',
    '(4, 6)': '(4, 6)',
    '(8, 12)': '(8, 12)',
    '(15, 20)': '(15, 20)',
    '(25, 32)': '(25, 32)',
    '(38, 43)': '(38, 43)',
    '(48, 53)': '(48, 53)',
    '(60, 100)': '(60, 100)'
}
gender_map = {'f': 'female', 'm': 'male'}

for age, gender in [(age, gender) for age in age_map.keys() for gender in gender_map.values()]:
    path = os.path.join(output_folder, age, gender)
    os.makedirs(path, exist_ok=True)

for _, row in total_data.iterrows():
    age = row['age']
    gender = row['gender']
    user_id = row['user_id']
    target_image = row['original_image']  # ex: '2280.10587826073_6663f5b654_o'

    if age in age_map and gender in gender_map:
        dest_folder = os.path.join(output_folder, age_map[age], gender_map[gender])
        os.makedirs(dest_folder, exist_ok=True)
        user_folder = os.path.join(image_folder, str(user_id))
        if os.path.exists(user_folder):
            matching_files = [f for f in os.listdir(user_folder) if target_image in f]
            if matching_files:
                src_path = os.path.join(user_folder, matching_files[0])
                dest_path = os.path.join(dest_folder, matching_files[0])
                shutil.copy(src_path, dest_path)
                print(f"{src_path} -> {dest_path} copied.")
            else:
                print(f"cannot find the image file: {user_folder}/{target_image}")
        else:
            print(f"cannot find the user_folder: {user_folder}")

print("done!")
