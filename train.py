import torch
import torch.nn as nn
from model import Model
from config import args
from mydataset import MyDataset
from torch.utils.data import DataLoader
from data import  MyDataCorpus
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@torch.no_grad()
def dev(model, dev_dataloader, test=False):
    model.eval()
    labels = []
    preds = []
    for batch in tqdm(dev_dataloader, desc='valid'):
        idx, deviceEng, deviceNum, deviceLen, event, eventLen, desc, label = batch
        idx = idx.to(device)
        deviceEng = deviceEng.to(device)
        deviceNum = deviceNum.to(device)
        deviceLen = deviceLen.to(device)
        event = event.to(device)
        eventLen = eventLen.to(device)
        desc = desc.to(device)
        label = label.to(device).tolist()
        pred = model(deviceEng, deviceNum, event, deviceLen, eventLen, desc).tolist()
        labels.extend(label)
        preds.extend(pred)
    preds = [1 if i>=0.5 else 0 for i in preds]
    acc = accuracy_score(labels, preds)
    if test==True:
        matrixes = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=matrixes, display_labels=[1,2])
        disp.plot()
        plt.show()

        return acc, matrixes
    return acc


def train():
    path = "Sysetic_data.csv"
    deviceEng_dict = 'deviceEng_dic.pkl'
    deviceNum_dict = 'deviceNum_dic.pkl'
    event_dict = 'event_dic.pkl'
    desc_dict = 'desc_dic.pkl'
    deviceEngEmbedding = torch.from_numpy(
        np.load("devicehead_word_embedding_vectors.npy")
    ).float()
    deviceNumEmbedding = torch.from_numpy(
        np.load("devicenum_word_embedding_vectors.npy")).float()
    eventEmbedding = torch.from_numpy(
        np.load("eventID_word_embedding_vectors.npy")).float()
    descEmbedding = torch.from_numpy(
        np.load("tfidf_descEmbedding.npy")).float()


    train_dataset = MyDataset(args, MyDataCorpus, path, deviceEng_dict, deviceNum_dict, event_dict, desc_dict)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    dev_dataset = MyDataset(args, MyDataCorpus, path, deviceEng_dict, deviceNum_dict, event_dict, desc_dict)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

    test_dataset = MyDataset(args, MyDataCorpus, path, deviceEng_dict, deviceNum_dict, event_dict, desc_dict)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    #加载参数
    model = Model(args,deviceEngEmbedding, deviceNumEmbedding, eventEmbedding, descEmbedding)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    bestacc = 0
    for i in range(1, args.epoch+1):
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch{i}'):
            idx, deviceEng, deviceNum, deviceLen, event, eventLen, desc, label = batch
            idx = idx.to(device)
            deviceEng = deviceEng.to(device)
            deviceNum = deviceNum.to(device)
            deviceLen = deviceLen.to(device)
            event = event.to(device)
            eventLen = eventLen.to(device)
            desc = desc.to(device)
            label = label.to(device)
            pred = model(deviceEng, deviceNum, event, deviceLen, eventLen, desc)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
        print('loss=',epoch_loss/len(train_dataset))
        acc = dev(model, dev_loader)
        model.train()
        if acc>bestacc:
            bestacc=acc
            torch.save(model.state_dict(), 'best.pkl')
        print(f'acc:{acc}, best_acc:{bestacc}')

    new_model = Model(args,deviceEngEmbedding, deviceNumEmbedding, eventEmbedding, descEmbedding)
    new_model.load_state_dict(torch.load("best.pkl"))
    acc, matrix = dev(new_model, test_loader, test=True)

    print(f'test_acc:{acc}')
    print(f'confusion matrix:{matrix}')
    #画图


train()








