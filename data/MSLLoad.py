import ast
import csv
import os
import sys
from pickle import dump
import pandas as pd

import numpy as np

#output_folder = 'processed_csv'
#os.makedirs(output_folder, exist_ok=True)

def load_and_save(category, filename, dataset, dataset_folder,output_folder):
    os.makedirs(os.path.join(output_folder, filename.split('.')[0]), exist_ok=True)
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float32,
                         delimiter=',')
    # print(dataset, category, filename, temp.shape)
    fea_len = len(temp[0, :])
    header_list = []
    for i in range(fea_len):
        header_list.append("col_%d"%i)
    data = pd.DataFrame(temp, columns=header_list).reset_index()
    data.rename(columns={'index': 'timestamp'}, inplace=True)
    if category == "test":
        temp1 = np.genfromtxt(os.path.join(dataset_folder, "test_label", filename),
                         dtype=np.float32,
                         delimiter=',')
        data1 = pd.DataFrame(temp1, columns=["label"]).reset_index()
        data1.rename(columns={'index': 'timestamp'}, inplace=True)
        data = pd.merge(data, data1, how="left", on='timestamp')

    print(dataset, category, filename, temp.shape)
    data.to_csv(os.path.join(output_folder,  filename.split('.')[0], dataset + "_" + category + ".csv"), index=False)

def load_data(dataset, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    if dataset == 'SMD':
        dataset_folder = 'SMD'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, filename.strip('.txt'), dataset_folder, output_folder)
                load_and_save('test', filename, filename.strip('.txt'), dataset_folder, output_folder)
    elif dataset == 'SMAP' or dataset == 'MSL':
        dataset_folder = 'MSL/initialData'
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        label_folder = os.path.join(dataset_folder, 'test_label')
        os.makedirs(label_folder, exist_ok=True)
        data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
        labels = []
        for row in data_info:
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.int)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = 1
            labels.extend(label)
        labels = np.asarray(labels)
        print(dataset, 'test_label', labels.shape)

        labels = pd.DataFrame(labels, columns=["label"]).reset_index()
        labels.rename(columns={'index': 'timestamp'}, inplace=True)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                print(os.path.join(dataset_folder, category, filename + '.npy'))
                temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)

            fea_len = len(data[0, :])
            header_list = []
            for i in range(fea_len):
                header_list.append("col_%d" % i)
            data = pd.DataFrame(data, columns=header_list).reset_index()
            data.rename(columns={'index': 'timestamp'}, inplace=True)

            if category == "test":
                data = pd.merge(data, labels, how="left", on='timestamp')
            print(dataset, category, filename, temp.shape)
            data.to_csv(os.path.join(output_folder,  dataset + "_" + category + ".csv"),
                        index=False)

        for c in ['train', 'test']:
            concatenate_and_save(c)


if __name__ == '__main__':
    datasets = ['SMD', 'SMAP', 'MSL']
    outputList = ['./SMD/data', './SMAP/data', './MSL/data']
    load_data('MSL', './MSL/data')
