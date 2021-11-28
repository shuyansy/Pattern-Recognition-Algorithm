# read file in csv format
import csv
import numpy as np

def read_f(current_folder,file_path):
    sample = []
    csv_reader = csv.reader(open(current_folder + file_path, encoding='utf-8'))
    for row in csv_reader:
        new_row = [float(i) for i in row]
        sample.append(new_row)
    return  np.array(sample)




