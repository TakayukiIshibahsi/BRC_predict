import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from embed import load_data
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="data", help='データディレクトリ')
    parser.add_argument('--output', type=str, default='out', help='結果の出力先')
    parser.add_argument('--model_name', type=str, default=f'{datetime.now().strftime("%Y-%m-%d-%H-%M")}')
    parser.add_argument('--epoch', type=int, default=5, help='学習回数')
    parser.add_argument('--batch_size', type=int, default=256, help='バッチサイズ')
    parser.add_argument('--num_gpus', type=int, default=4, help='使用するGPUの数')
    parser.add_argument('--input', type=str, default='inputs_2024-7-21-2.csv', help='入力ファイル名')
    return parser.parse_args()

last_logged = time.time()
def log(message=""):
    global last_logged
    print(message, end = "")
    print(f"\t elapsed: {time.time() - last_logged:.2f}s", end="\n\n")
    last_logged = time.time()



class TransformerModel(nn.Module):
    def __init__(self, code_dim, desc_dim, label_dim, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=code_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.desc_linear = nn.Linear(desc_dim, code_dim)
        self.fc = nn.Linear(code_dim, label_dim)  # This needs to be modified

    def forward(self, code, desc):
        code_transformed = self.transformer_encoder(code)
        desc_transformed = self.desc_linear(desc)
        combined = code_transformed + desc_transformed.unsqueeze(1)
        output = self.fc(combined)
        output = output.mean(dim=1)  # Modify output shape to match (batch_size, label_dim)
        return output
    

def train_model(model, train_loader, test_loader, epochs, device, result):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    with open(result, 'w') as rs:
        rs.write("tp\ttn\tfp\tfn\tloss")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            with tqdm(train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Train Epoch {epoch+1}")

                    codes = batch['codes'].to(device)
                    descriptions = batch['descriptions'].to(device)
                    labels = batch['labels'].to(device).float()  # Convert labels to float

                    optimizer.zero_grad()
                    outputs = model(codes, descriptions)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                    tepoch.set_postfix(loss=epoch_loss / (tepoch.n + 1))
            
            log(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

            # Confusion matrix calculation
            with torch.no_grad():
                tp, tn, fp, fn = 0, 0, 0, 0
                with tqdm(test_loader, unit='batch') as tepoch:
                    for batch in tepoch:
                        tepoch.set_description("Validation")
                        codes = batch['codes'].to(device)
                        descriptions = batch['descriptions'].to(device)
                        labels = batch['labels'].to(device)
                        
                        outputs = model(codes, descriptions)
                        preds = torch.sigmoid(outputs).round()
                        true_labels = labels
                        tp += (preds * true_labels).sum().item()
                        tn += ((1 - preds) * (1 - true_labels)).sum().item()
                        fp += (preds * (1 - true_labels)).sum().item()
                        fn += ((1 - preds) * true_labels).sum().item()
                sum_tfpn = tp + tn + fp + fn
                print(f'TP: {tp/sum_tfpn:.4}, TN: {tn/sum_tfpn:.4}')
                print(f'FP: {fp/sum_tfpn:.4}, FN: {fn/sum_tfpn:.4}')
                log()
                rs.write(f"{tp}\t{tn}\t{fp}\t{fn}\t{epoch_loss / len(train_loader):.4f}")
                rs.flush()
    return model

if __name__ == "__main__":
    args = parse_args()
    train_data, valid_data, test_data = load_data(args.input, args.batch_size, args.num_gpus)
    # train_data = BugData.load_npz(f"{args.data}/train.npz")
    # test_data = BugData.load_npz(f"{args.data}/test.npz")
    # valid_data = BugData.load_npz(f"{args.data}/valid.npz")    
    print(f"train;code:{train_data.codes.shape},desc:{train_data.descriptions.shape},label:{train_data.labels.shape}")
    print(f"test;code:{test_data.codes.shape},desc:{test_data.descriptions.shape},label:{test_data.labels.shape}")
    print(f"valid;code:{valid_data.codes.shape},desc:{valid_data.descriptions.shape},label:{valid_data.labels.shape}")
    log("finish loading data")

    # データローダーの作成
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(code_dim=768, desc_dim=384, label_dim=512, nhead=8, num_layers=2)
    
    # Use multiple GPUs if available and specified
    if torch.cuda.device_count() > 1 and args.num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
    
    log("finish preparing model")
    os.makedirs(args.output, exist_ok=True)
    trained_model = train_model(model, train_loader, test_loader, args.epoch, device, f"{args.output}/{args.model_name}_log.tsv")
    log("finish training.")

    # Save the trained model
    
    torch.save(trained_model.state_dict(), f"{args.output}/{args.model_name}_model.pt")
    log("saved model")
