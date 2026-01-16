import os
import pandas as pd
import shutil

xlsx_file = "F:/SI9/split.xlsx"
df = pd.read_excel(xlsx_file)
column_name = df.columns[5]
train_folder = "F:/SI9/Train"
test_folder = "F:/SI9/Test"
for index, row in df.iterrows():
    keyword = row[column_name]
    for folder in os.listdir(train_folder):
        folder_path = os.path.join(train_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and keyword in filename:
                test_subfolder = os.path.join(test_folder, folder)
                os.makedirs(test_subfolder, exist_ok=True)
                shutil.move(file_path, os.path.join(test_subfolder, filename))