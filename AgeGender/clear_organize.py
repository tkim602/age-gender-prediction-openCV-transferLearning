import os

base_path = "organized_data"


def remove_ds_store_files(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == ".DS_Store":
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed: {file_path}")


remove_ds_store_files(base_path)
