import os
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
import re
from signal_process import signal_preprocess, signal_normalize, signal_resize, signal_slice
from tqdm import tqdm


def data_slice_norm(data,window_size,stride, start = 0):
    segments = []
    for i in range(start, len(data) - window_size + 1, stride):
        slice = data[i:i + window_size]
        signal_norm =(slice - slice.min())/(slice.max() - slice.min()+1e-5)
        segments.append(signal_norm)
    return segments

def sci_reader(root_dir):
    signal_list = []
    pul_list =[]
    rsp_list=[]
    ecg_list=[]
    bp_list=[]
    y_list = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # 获取当前文件的路径
            if filename.endswith('.mat'):
                try:
                    mat_data = loadmat(file_path)
                    radar_signal = mat_data['radar_signal']
                    #reversed_arr = radar_signal[::-1]
                    #radar_signal = reversed_arr
                    rsp = mat_data['rsp']
                    ecg = mat_data['ecg']
                    bp = mat_data['bp']
                    fs = int(mat_data['fs'][0,0])
                    window_size = 1024
                    radar_segments = data_slice_norm(radar_signal, window_size=window_size, stride=int(window_size/2))
                    signal_list.extend(radar_segments)
                    # y_list.extend(ecg_segments)
                    #y_list.extend(bp_segments)
                except KeyError as e:
                    print(f"Warning: Missing variable in {filename} - {e}")
    else:
        print(f"共读取到 {len(signal_list)} 个有效数据。")
        return signal_list

def occupancy_dataset_reader(root_dir):
    signal_list = []
    y_list = []
    for dirpath,dirnames,filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.mat'):
                file_path = os.path.join(dirpath,filename)
                mat_data = loadmat(file_path)['data']
                label = os.path.basename(os.path.dirname(file_path))
                label = int(label.split('_')[0])
                signal_processed = signal_preprocess( mat_data, fs=23.328e9, fc=7.29e9, bw=1.4e9)
                fps = 100
                segments = signal_slice(signal_processed, window_size=fps,stride=int(fps*1.5),start=0)
                for segment in segments:
                    signal_norm = signal_normalize(segment)
                    signal_list.append(torch.tensor(signal_norm,dtype=torch.float32))
                    y_list.append(torch.tensor(label,dtype=torch.long))
    else:
        data_tensor = torch.stack(signal_list)
        label_tensor = torch.stack(y_list)
        print(f"共读取到 {len(signal_list)} 个有效数据。")
        return data_tensor,label_tensor

def drowsiness_dataset_reader(root_dir):
    signal_list = []
    y_list = []
    drosiness_dict = {'normal':0,'excited':1,'drowsy':2}
    for dirpath,dirnames,filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.mat'):
                file_path = os.path.join(dirpath,filename)
                mat_data = loadmat(file_path)['data']
                label = os.path.basename(os.path.dirname(file_path))
                label = int(drosiness_dict[label])
                signal_processed = signal_preprocess( mat_data, fs=23.328e9, fc=7.29e9, bw=1.4e9)
                fps = 100
                segments = signal_slice(signal_processed, window_size=fps,stride=int(fps),start=0)
                for segment in segments:
                    signal_norm = signal_normalize(segment)
                    signal_list.append(torch.tensor(signal_norm,dtype=torch.float32))
                    y_list.append(torch.tensor(label,dtype=torch.long))
    else:
        data_tensor = torch.stack(signal_list)
        label_tensor = torch.stack(y_list)
        print(f"共读取到 {len(signal_list)} 个有效数据。")
        return data_tensor,label_tensor
    


def vital_sign_dataset_reader(root_dir):
    all_data = []
    all_labels = []
    pattern_dict = {'Resting':0,'Valsalva':1,'Apnea':2,'TiltUp':3,'TiltDown':4}
    for file_name in os.listdir(root_dir):
        if file_name.endswith('.mat'):
            file_path = os.path.join(root_dir, file_name)
            mat_data = loadmat(file_path)
            signal = mat_data['radar_signal']
            label = file_name.split('_')[1].split('.')[0]
            label = pattern_dict[label]
            window_size = 1024
            radar_segments = data_slice_norm(signal, window_size=window_size, stride=int(window_size))
            for i in range(len(radar_segments)):
                all_data.append(torch.tensor(radar_segments[i],dtype=torch.float32))
                all_labels.append(torch.tensor(label,dtype=torch.long))
    data_tensor = torch.stack(all_data)
    label_tensor = torch.stack(all_labels)
    print(f"成功保存 {len(all_data)} 个样本")
    return data_tensor,label_tensor

def gesture_dataset_reader(root_dir):
    all_data = []
    all_labels = []
    def normalize_range(matrix):
        min_vals = np.min(matrix, axis=1, keepdims=True)
        max_vals = np.max(matrix, axis=1, keepdims=True) 
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        normalized_matrix = (matrix - min_vals) / range_vals
        return normalized_matrix
    
    for file_name in tqdm(os.listdir(root_dir)):
        if file_name.endswith('.mat'):
            file_path = os.path.join(root_dir, file_name)
            mat_data = loadmat(file_path)
            
            parts = file_name.split('_')
            label = int(parts[2][1:])

            data_id = f'HV{int(parts[1])}_{parts[2]}'
            signal_left = mat_data[data_id +'_RadarRight_ClutterRemoved_100samples']
            signal_right = mat_data[data_id + '_RadarRight_ClutterRemoved_100samples']
            signal_top = mat_data[data_id+ '_RadarTop_ClutterRemoved_100samples']
            for i in range(100):
                try:
                    sample_left = normalize_range(signal_left)[90*i+14:90*i+90,:]
                    sample_top = normalize_range(signal_top)[90*i+14:90*i+90,:]
                    sample_right = normalize_range(signal_right)[90*i+14:90*i+90,:]
                    three_channel_sample = np.stack([sample_left, sample_top, sample_right], axis=-1)

                    all_data.append(torch.tensor(three_channel_sample,dtype=torch.float32))
                    all_labels.append(torch.tensor(label,dtype=torch.long))
                except IndexError:
                    print(f"Error: {file_name} has an unexpected shape.")
    data_tensor = torch.stack(all_data)
    label_tensor = torch.stack(all_labels)
    print(f"成功保存 {len(all_data)} 个样本")
    return data_tensor,label_tensor
    

if __name__ == '__main__':
    data,label = gesture_dataset_reader('./UWB/uwb-gesture-recognition/processed')
    torch.save({'data':data,'label':label}, 'gesture.pt')