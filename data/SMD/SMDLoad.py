# 将SMD数据集的保存形式从txt转变成csv，且将测试集的label与测试集关联起来。

import os
import numpy as np
import pandas as pd

# 指定SMD原始数据集的位置
dataset_folder = './intialData'

# 处理后的数据集保存路径
output_folder = './data/SMD'
os.makedirs(output_folder, exist_ok=True)

def load_and_save(category,filename):
    # category用于区分式训练集还是测试集。
    # 函数的功能是将txt文件转换成csv文件，并保存到合适位置。

    # 生成保存数据的文件夹，如果已经存在则此指令忽略，不进行报错。
    os.makedirs(os.path.join(output_folder,category),exist_ok=True)

    # 读取文件，并用" , "进行分隔，以便于后续保存成csv文件
    temp = np.genfromtxt(os.path.join(dataset_folder,category,filename),dtype=np.float32,delimiter=',')

    # 获取特征数量，此处应该是38。
    fea_len = len(temp[0,:])

    # 弄一个空列表以便于，后续修改数据的列名。
    header_list = []

    for i in range(fea_len):
        # 为每一列的标题准备名称 col_i，i是变量。
        header_list.append("col_%d"%i)

    # 把temp转化成DataFrame格式，并且指定列标签。
    data = pd.DataFrame(temp, columns=header_list).reset_index()

    # 在原数据上（inplace=True）把列标签index重命名成timestamp，便于理解
    data.rename(columns={'index': 'timestamp'}, inplace=True)

    # 如果是test还要把数据和label合并到一个文件下。
    if category == "test":

        # 读取标签信息
        temp1 = np.genfromtxt(os.path.join(dataset_folder, "test_label", filename),
                         dtype=np.float32,
                         delimiter=',')

        # 将temp1转化成DataFrame格式，并给它这一列命名成“label”。
        data1 = pd.DataFrame(temp1, columns=["label"]).reset_index()

        # 同样的，在原数据上（inplace=True）把列标签index重命名成timestamp
        data1.rename(columns={'index': 'timestamp'}, inplace=True)

        # 将标签信息与数据信息组合起来
        data = pd.merge(data, data1, how="left", on='timestamp')

    print(category,",",filename,",",data.shape)
    data.to_csv(os.path.join(output_folder,category,filename.split('.')[0]+".csv"),index=False) # 输出成csv文件




def load_data():
    for category in ["train","test"]:
        # 生成新的数据集
        file_list = os.listdir(os.path.join(dataset_folder, category))  # 确定原始数据集路径
        for filename in file_list: # 遍历原始数据集路径下的全部文件
            if filename.endswith('.txt'):# 由前置内容不难知所有的数据都是采用.txt文件的形式保存，因此要找到以.txt形式结尾的文件
                load_and_save(category, filename)  # 将数据集的txt文件保存成csv文件


if __name__ == "__main__":
    load_data()
