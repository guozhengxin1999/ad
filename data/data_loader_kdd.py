import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='network_traffic.csv',
                 target='label', scale=True,
                 cols=None, categorical_cols=None):
        """
        参数说明：
        - root_path: 数据文件所在目录
        - flag: 'train', 'val', 或 'test'
        - size: [seq_len, label_len, pred_len]，序列长度配置
        - features: 'M' 表示多变量输入
        - data_path: 数据文件路径，默认为 'network_traffic.csv'
        - target: 目标列名，默认为 'label'
        - scale: 是否标准化数值型特征
        - cols: 要使用的特征列列表（不包括 'target'）
        - categorical_cols: 类别型特征列列表
        """
        # 设置序列长度
        self.seq_len = size[0] if size else 96
        self.label_len = size[1] if size else 48  # 分类任务中可忽略
        self.pred_len = size[2] if size else 24

        # 初始化参数
        assert flag in ['train', 'test', 'val']
        self.type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = self.type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.cols = cols
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """读取和预处理数据"""
        # 读取原始数据
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 处理类别型特征
        self.encoders = {}
        for col in self.categorical_cols:
            if col in df_raw.columns:
                le = LabelEncoder()
                df_raw[col] = le.fit_transform(df_raw[col].astype(str))
                self.encoders[col] = le

        # 处理分类标签
        self.label_encoder = LabelEncoder()
        df_raw[self.target] = self.label_encoder.fit_transform(df_raw[self.target].astype(str))

        # 选择特征列（包括 'date'，即 duration）
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = [col for col in df_raw.columns if col != self.target]

        df_features = df_raw[cols]
        df_labels = df_raw[[self.target]]

        # 标准化数值型特征（包括 'date'，即 duration）
        self.scaler = StandardScaler()
        if self.scale:
            numerical_cols = [col for col in cols if col not in self.categorical_cols]
            if numerical_cols:
                train_features = df_features[numerical_cols].iloc[:int(len(df_raw) * 0.7)]
                self.scaler.fit(train_features.values)
                df_features[numerical_cols] = self.scaler.transform(df_features[numerical_cols].values)

        # 数据集分割
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 设置输入和目标数据
        self.data_x = df_features.values[border1:border2]
        self.data_y = df_labels.values[border1:border2].ravel()  # 展平为一维

    def __getitem__(self, index):
        """获取单条数据"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]  # 输入序列
        seq_y = self.data_y[s_end:r_end]  # 预测标签

        return torch.FloatTensor(seq_x), torch.LongTensor(seq_y)

    def __len__(self):
        """数据集长度"""
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """反标准化数值型特征"""
        return self.scaler.inverse_transform(data)

    def inverse_label_transform(self, labels):
        """将数值标签转换回原始标签"""
        return self.label_encoder.inverse_transform(labels.astype(int))