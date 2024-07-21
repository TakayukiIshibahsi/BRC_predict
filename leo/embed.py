import argparse
import os
import csv
import re
import torch
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='データセットの名前')
    parser.add_argument('--output', type=str, default='data', help='結果の出力先ディレクトリ')
    parser.add_argument('--batch_size', type=int, default=512, help='埋め込み計算のバッチサイズ')
    parser.add_argument('--num_gpus', type=int, default=4, help='使用するGPUの数')
    return parser.parse_args()

class BugData(Dataset):
    def __init__(self, codes, descriptions, labels):
        self.codes = codes # 512 * 718
        self.descriptions = descriptions # 384
        self.labels = labels # 512

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'codes': self.codes[idx],
            'descriptions': self.descriptions[idx],
            'labels': self.labels[idx]
        }

    @staticmethod
    def load_pt(filepath):
        data = torch.load(filepath)
        codes, descriptions, labels = data
        return BugData(codes, descriptions, labels)
    
    @staticmethod
    def load_npz(filepath):
        data = np.load(filepath, mmap_mode='r')
        codes = torch.from_numpy(data['codes'])
        descriptions = torch.from_numpy(data['descriptions'])
        labels = torch.from_numpy(data['labels'])
        return BugData(codes, descriptions, labels)
    
# 512
def sentence_to_list(sentence):
    sentence = sentence[1:-1]
    return [float(elm.strip()) for elm in sentence.split(',')]

# 384
def description_to_vec(sentences, model, batch_size):
    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="description_to_vec"):
        batch_sentences = sentences[i:i+batch_size]
        batch_embeddings = model.encode(batch_sentences, batch_size=batch_size, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# 512 * 768
def code_to_vec(codes, model, tokenizer, device, batch_size, num_gpus):
    if num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
    model = model.to(device)
    data_tensors = []

    for i in tqdm(range(0, len(codes), batch_size), desc="code_to_vec"):
        batch_codes = codes[i:i+batch_size]
        tokenized_data = tokenizer(batch_codes, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        tokenized_data = {k: v.to(device) for k, v in tokenized_data.items()}
        with torch.no_grad():
            data_tensor = model(**tokenized_data).last_hidden_state
        data_tensors.append(data_tensor.cpu().detach())

    return torch.cat(data_tensors, dim=0)


def clean_description(description):
    # URLの削除
    cleaned_description = re.sub(r'http\S+', '', description)
    # 余分なスペースの削除
    cleaned_description = re.sub(r'\s+', ' ', cleaned_description).strip()
    return cleaned_description


def load_and_preprocess_data(name, device, batch_size, num_gpus):
    file = "./" + name
    codes = []
    descriptions = []
    labels = []
    with open(file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in tqdm(reader, desc="reading csv"):
            code, description, label = row
            codes.append(code)
            descriptions.append(clean_description(description))
            labels.append(sentence_to_list(label))
    
    STmodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model_name = "microsoft/graphcodebert-base"
    codeBERTmodel = RobertaModel.from_pretrained(model_name)
    codeBERTtokenizer = RobertaTokenizer.from_pretrained(model_name)

    print("finished reading csv file")
    
    codes = code_to_vec(codes, codeBERTmodel, codeBERTtokenizer, device, batch_size, num_gpus)
    print("embedded codes")
    descriptions = description_to_vec(descriptions, STmodel, batch_size)
    descriptions = torch.tensor(descriptions, dtype=torch.float32)
    print("embedded descriptions")
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return codes, descriptions, labels

def save_data_splits_npz(datas, output_dir):
    (train, valid, test) = datas
    os.makedirs(output_dir, exist_ok=True)
    
    np.savez(os.path.join(output_dir, 'train.npz'), 
             codes=train.codes.numpy(),
             descriptions=train.descriptions.numpy(),
             labels=train.labels.numpy())
    
    np.savez(os.path.join(output_dir, 'valid.npz'), 
             codes=valid.codes.numpy(),
             descriptions=valid.descriptions.numpy(),
             labels=valid.labels.numpy())
    
    np.savez(os.path.join(output_dir, 'test.npz'), 
             codes=test.codes.numpy(),
             descriptions=test.descriptions.numpy(),
             labels=test.labels.numpy())

def load_data(input, batch_size, num_gpus):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    codes, descriptions, labels = load_and_preprocess_data(input, device, batch_size, num_gpus)

    print("finish loading data")
    
    codes_train, codes_temp, descriptions_train, descriptions_temp, labels_train, labels_temp = train_test_split(
        codes, descriptions, labels, test_size=0.3, random_state=42
    )
    codes_valid, codes_test, descriptions_valid, descriptions_test, labels_valid, labels_test = train_test_split(
        codes_temp, descriptions_temp, labels_temp, test_size=0.5, random_state=42
    )

    print("finish splitting data")
    train = BugData(codes_train, descriptions_train, labels_train)
    valid = BugData(codes_valid, descriptions_valid, labels_valid)
    test = BugData(codes_test, descriptions_test, labels_test)
    return (train, valid, test)

if __name__ == "__main__":
    args = parse_args()
    name = args.input
    
    datas = load_data(args.input, args.batch_size, args.num_gpus)
    
    save_data_splits_npz(datas, args.output)

    print("finish saving embeddings")
