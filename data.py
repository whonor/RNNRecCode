
import pandas as pd
import pickle
from config import args

def splits(data):
    data = data.strip().split(',')
    data = [i.strip() for i in data]
    data = [i.strip('[') for i in data]
    data = [i.strip(']') for i in data]
    data = [i.strip('/') for i in data]
    data = [i.strip("'") for i in data]
    data = [i.strip('\\') for i in data]
    return data

def device_Eng_split(data):
    data = [i[:-5] for i in data]
    return data

def device_Num_split(data):
    data = [i[-5:] for i in data]
    return data

def load_data(dataPath):
    data = pd.read_csv(dataPath, encoding='utf-8', usecols=[0, 2, 3, 4, 5],
                       names=['ID', 'Device', 'Event', 'Desc', 'label'])
    data['ID'].iloc[1:] = data['ID'].iloc[1:].map(int)
    data['Device'][1:] = data['Device'][1:].map(splits)
    data['Event'][1:] = data['Event'][1:].map(splits)
    data['Desc'][1:] = data['Desc'][1:].map(splits)
    data['label'][1:] = data['label'][1:].map(int)
    data['label'][1:] = data['label'][1:].map(lambda x: x-1)
    data['DeviceEng'] = data['Device'].map(device_Eng_split)
    data['DeviceNum'] = data['Device'].map(device_Num_split)
    data = data.drop(index=0)

    id = data['ID'].tolist()
    deviceEng = data['DeviceEng'].tolist()
    deviceNum = data['DeviceNum'].tolist()
    deviceLen = [len(i) for i in deviceEng]
    event = data['Event'].tolist()
    eventLen = [len(i) for i in event]
    desc = data['Desc'].tolist()
    label = data['label'].tolist()
    return id, deviceEng, deviceNum, deviceLen, event, eventLen, desc, label




# def constructDict(data, save_path):
#     dic = {'pad':0}
#     for line in data:
#         for word in line:
#             if word not in dic:
#                 dic[word] = len(dic)
#     with open(save_path, 'wb') as fwb:
#         pickle.dump(dic, fwb)
#
# constructDict(deviceEng, 'deviceEng_dic.pkl')
# constructDict(deviceNum, 'deviceNum_dic.pkl')
# constructDict(event, 'event_dic.pkl')
# constructDict(desc, 'desc_dic.pkl')

def load_dict(path):
    with open(path, 'rb') as frb:
        data = pickle.load(frb)
    return data




class MyDataCorpus:
    def __init__(self, args, dataPath, deviceEng_dict, deviceNum_dict, event_dict, desc_dict):
        id, deviceEng, deviceNum, deviceLen, event, eventLen, desc, label = load_data(dataPath)
        deviceEng_dict = load_dict(deviceEng_dict)
        deviceNum_dict = load_dict(deviceNum_dict)
        event_dict = load_dict(event_dict)
        desc_dict = load_dict(desc_dict)
        self.args = args

        self.id = id
        self.deviceEng = self.get_data(deviceEng, deviceEng_dict)
        self.deviceNum = self.get_data(deviceNum, deviceNum_dict)
        self.deviceLen = deviceLen
        self.event = self.get_data(event, event_dict)
        self.eventLen = eventLen
        self.desc = self.get_data(desc, desc_dict)
        self.label = label

    def get_data(self, data, dic):
        #字母变数字，补零+截断
        data_num = [[dic[word] if word in dic else 0 for word in line] for line in data]
        for i in range(len(data_num)):
            if len(data_num[i]) > self.args.maxLength:
                data_num[i] = data_num[i][:self.args.maxLength]
            elif len(data_num[i]) < self.args.maxLength:
                data_num[i] += [0]*(self.args.maxLength-len(data_num[i]))
        return data_num












