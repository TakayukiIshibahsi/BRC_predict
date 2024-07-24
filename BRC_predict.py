
import os
import re
import csv
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import multilabel_confusion_matrix
from sentence_transformers import SentenceTransformer
from transformers import RobertaModel, RobertaTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

##################################################################################################################################initialize

code_size=768
hidden_size1=512
hidden_size2=256
description_size=384
hidden_size3=640
hidden_size4=256
num_layers=2

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv0 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv1 = nn.Conv1d(in_channels=code_size, out_channels=hidden_size1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size1, out_channels=hidden_size2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_size3, out_channels=hidden_size4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=hidden_size4, out_channels=1, kernel_size=3, padding=1)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x,sentence_inputs):
        x = self.conv0(x)
        x = x.transpose(1, 2)  # (batch_size, 768, 512)
        x = self.conv1(x)  # (batch_size, 512, 512)
        x = self.conv2(x)  # (batch_size, 256, 512)
        x = x.transpose(1, 2) # (batch_size, 512, 256)
# 新しい次元を追加
        tensor = sentence_inputs.unsqueeze(1)  # batch_size x 1 x 384

# 新しい次元を512に繰り返す
        tensor = tensor.repeat(1, 512, 1)  # batch_size x 512 x 384
        x = torch.cat((x,tensor),dim=2)  #(batch_size, 512, 384+256=640)
        x = x.transpose(1, 2)  # (batch_size, 640, 512)
        x = self.conv3(x)  # (batch_size, 256, 512)
        x = self.conv4(x)
        x = x.squeeze(1)  # (batch_size, 512)
        return x

class MyFullyConnected(nn.Module):
    def __init__(self):
        super(MyFullyConnected, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.cnn = MyCNN()
        self.fc = MyFullyConnected()

    def forward(self, x, sentence_inputs):
        x = self.cnn(x,sentence_inputs)  # (batch_size, 512)
        x = self.fc(x)  # (batch_size, 512)
        return x

class BugReportDataset(Dataset):
    def __init__(self, inputs, labels):
        self.data = inputs
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
######################################################################################################################################oldModels

class FirstCNN(nn.Module):
    def __init__(self):
        super(FirstCNN, self).__init__()
        self.conv0 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv1 = nn.Conv1d(in_channels=code_size, out_channels=hidden_size1, kernel_size=3, padding=1)
    
    def forward(self, x,sentence_inputs):
        x = self.conv0(x)
        x = x.transpose(1, 2)  # (batch_size, 768, 512)
        x = self.conv1(x)  # (batch_size, 512, 512)
    
class FirstLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FirstLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        return x

class MiddleCNN(nn.Module):
    def __init__(self):
        super(MiddleCNN, self).__init__()
        self.conv2 = nn.Conv1d(in_channels=hidden_size1, out_channels=hidden_size2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_size3, out_channels=hidden_size4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=hidden_size4, out_channels=1, kernel_size=3, padding=1)
        self.flatten = nn.Flatten(start_dim=1)
                                  
    def forward(self, x, sentence_inputs):
        x = self.conv2(x)  # (batch_size, 256, 512)
        x = x.transpose(1, 2) # (batch_size, 512, 256)
        # 新しい次元を追加
        tensor = sentence_inputs.unsqueeze(1)  # batch_size x 1 x 384
        # 新しい次元を512に繰り返す
        tensor = tensor.repeat(1, 512, 1)  # batch_size x 512 x 384
        x = torch.cat((x,tensor),dim=2)  #(batch_size, 512, 384+256=640)
        x = x.transpose(1, 2)  # (batch_size, 640, 512)
        x = self.conv3(x)  # (batch_size, 256, 512)
        x = self.conv4(x)
        x = x.squeeze(1)  # (batch_size, 512)
        return x


class CombinedModel_CNN(nn.Module):
    def __init__(self):
        super(CombinedModel_CNN,self).__init__()
        self.first=FirstCNN()
        self.middle=MiddleCNN()
        self.fc=MyFullyConnected()

    def forward(self, x, sentence_inputs):
        x=self.first(x)
        x=self.middle(x,sentence_inputs)
        x=self.fc(x)
        return x
    
class CombinedModel_LSTM(nn.Module):
    def __init__(self):
        super(CombinedModel_LSTM,self).__init__()
        self.first=FirstLSTM(code_size,hidden_size1,num_layers)
        self.middle=MiddleCNN()
        self.fc=MyFullyConnected()

    def forward(self, x, sentence_inputs):
        x=self.first(x) #input_size:768, hidden_size=512, layer=2
        x=self.middle(x,sentence_inputs)
        x=self.fc(x)
        return x
######################################################################################################################################newModels
def pos_weight(labels):
    negative=0
    positive=0
    for label in labels:
        for l in label:
            if l==0:
                negative+=1
            else:
                positive+=1
    return negative/positive 
def train_model(model, train_loader, sentence_inputs, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            pos = pos_weight(labels)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos))
            inputs, labels = inputs.to(device), labels.to(device)  
            optimizer.zero_grad()
            outputs = model(inputs, sentence_inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        total_loss=running_loss/len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')

        return total_loss

def evaluate_model(model, data_loader,sentence_inputs):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    total_loss=0
    with torch.no_grad():
        k=0
        running_loss=0
        for inputs, labels in data_loader:
            pos = pos_weight(labels)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs,sentence_inputs)
            loss = criterion(outputs, labels.float())
            running_loss += loss.item()
            threshold=0.5
            predicted = (torch.sigmoid(outputs) > threshold).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 予測結果と実際のラベルを保存
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            k += 1
        total_loss += running_loss/len(data_loader)

    accuracy = correct / total

    # 多ラベル混合行列を計算
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    mcm = multilabel_confusion_matrix(all_labels, all_preds)

    # 各ラベルごとに混合行列をテキスト形式で表示
    tn, fp, fn, tp =0,0,0,0
    for idx, cm in enumerate(mcm):
        ctn, cfp, cfn, ctp = cm.ravel()
        tn+=ctn
        fp+=cfp
        fn+=cfn
        tp+=ctp
    total=tn+fp+fn+tp
    tn_rate=tn/total
    fp_rate=fp/total
    fn_rate=fn/total
    tp_rate=tp/total
    print(f'Confusion Matrix for Label {idx+1}:')
    print(f'True Negative (TN): {tn_rate}')
    print(f'False Positive (FP): {fp_rate}')
    print(f'False Negative (FN): {fn_rate}')
    print(f'True Positive (TP): {tp_rate}')
    print(f'loss: {total_loss}')
    print()

    return accuracy,tn_rate,fp_rate,fn_rate,tp_rate,total_loss

def train_valid(li,threshold):
    train=li[:threshold]
    valid=li[threshold:]
    return train,valid

def description_to_vec(sentences,model):
    embeddings = model.encode(sentences)
    return embeddings

def code_to_vec(codes,model,tokenizer):
    tokenized_data = tokenizer(codes, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    data_tensor = model(**tokenized_data).last_hidden_state
    return data_tensor


def Learning(codes,descriptions,labels,epoch,model):
    labels = torch.tensor(labels)
    data_num=len(codes)
    rate=2 / 3
    threshold = int(data_num * rate)
    train_codes,valid_codes=train_valid(codes,threshold)
    train_descriptions,valid_descriptions=train_valid(descriptions,threshold)
    train_labels,valid_labels=train_valid(labels,threshold)
    divideNumber=4
    for i in range(epoch):
        i+=1
        partLearning(model,train_codes,train_descriptions,train_labels,divideNumber,True,i)
        partLearning(model,valid_codes,valid_descriptions,valid_labels,divideNumber,False,i)
        
    
    


def partLearning(model,codes,descriptions,labels,divideNumber,train,order):
    total=int(len(codes)/divideNumber)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 学習率を下げる
    sum_tn=0
    sum_fp=0
    sum_fn=0
    sum_tp=0
    sum_loss=0
    st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 
    model_name = "microsoft/graphcodebert-base"
    codeBERTmodel = RobertaModel.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    for i in range(total):
        before=(i*divideNumber)
        after=before+divideNumber
        mini_codes=code_to_vec(codes[before:after],codeBERTmodel,tokenizer)
        mini_descriptions=torch.from_numpy(description_to_vec(descriptions[before:after],st))
        mini_labels=labels[before:after]
        dataset = BugReportDataset(mini_codes, mini_labels)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        if train:
            print(f'cycle : {order}, running : {i+1}/{total}\n')
            train_model(model, loader, mini_descriptions, optimizer, num_epochs=1)
            # 正答率の評価
            train_accuracy,tn,fp,fn,tp,loss= evaluate_model(model, loader, mini_descriptions)
            print(f'Training Accuracy: {train_accuracy:.4f}\n\n')
        else:
            valid_accuracy,tn,fp,fn,tp,loss= evaluate_model(model, loader, mini_descriptions)
            print(f'Valid Accuracy: {valid_accuracy:.4f}\n\n')
        sum_tn+=tn
        sum_fn+=fn
        sum_fp+=fp
        sum_tp+=tp
        sum_loss+=loss

    model_path=f'./model_{order}'
    torch.save(model.to('cpu').state_dict(), model_path)
    save_data(sum_tn,sum_fn,sum_fp,sum_tp,total,sum_loss,order,train)
            

def save_data(tn,fn,fp,tp,count,loss,order,train):
    path='./result.txt'
    if train:
        mode="train : "
    else:
        mode="valid : "
    text=f"{mode}{order}\naverage_tn : {tn/count} \naverage_fp : {fp/count}\naverage_fn : {fn/count}\naverage_tp : {tp/count}\nloss : {loss/count}\n\n"
    with open(path,mode='a') as f:
        f.write(text)
##############################################################################################################################################learning

def sentence_to_list(sentence):
    label=[]
    nums=re.findall(r'\d',sentence)
    for num in nums:
        label.append(int(num))
    return label


def load_data(name):
    file="./"+name
    codes=[]
    descriptions=[]
    labels=[]
    with open(file,'r',encoding='utf-8') as file:
        reader=csv.reader(file)
        next(reader)
        for row in reader:
            code,description,label=row
            codes.append(code)
            descriptions.append(description)
            labels.append(sentence_to_list(label))
            
    return codes,descriptions,labels

###########################################################################################################################################data

def predict(model):
    print("please input your code")
    code=input()
    print("please input your description")
    description=input()
    outputs = model([code],[description])
    threshold=0.5
    predicted = (torch.sigmoid(outputs) > threshold).float()
    return predicted

###########################################################################################################################################use
name='inputs_2024-7-21-2.csv'
codes,descriptions,labels=load_data(name)
model = CombinedModel_LSTM().to(device)
Learning(codes,descriptions,labels,4,model)


############################################################################################################################################code