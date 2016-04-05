import os
import re

path = 'D:\\Documents\\MyDocuments\\Zare\\FingerPrint\\03082016\\'
files = os.listdir(path)
raw_files = [name for name in files if (name.find('.raw') != -1 and name.find('.cfg') == -1)]
num_pattern = re.compile(r'\d+')
for file in raw_files:
    num = num_pattern.findall(file)[0]
    if len(num) == 2:
        new_name = 'finger_print_0' + num + '.raw'
        os.rename(path + file, path + new_name)

