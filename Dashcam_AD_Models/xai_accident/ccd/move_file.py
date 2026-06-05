import shutil
import os

destination_negative = 'test/negative/'
destination_positive = 'test/positive/'
if not os.path.exists(destination_negative):
    os.makedirs(destination_negative)
if not os.path.exists(destination_positive):
    os.makedirs(destination_positive)
anno_file = 'test.txt'

with open(anno_file, 'r') as f:
    for line in f.readlines():
        filename = line.split('.')[0]
        file_type = filename.split('/')[0]
        if file_type == 'negative':
            shutil.move(filename, destination_negative)
        else:
            shutil.move(filename, destination_positive)
