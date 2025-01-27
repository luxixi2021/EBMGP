import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import PearsonCorrCoef
import torch.optim as optim
import os
import random
import math
from sklearn.metrics import mean_squared_error
from FSdata import DNADataset
from model import EBMGP


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, train_loader, optimizer, loss_fn, scheduler, device):
    model.train()
    train_loss = []
    for batch_idx, (seqs, type_ids, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        seqs, type_ids, labels = seqs.to(device), type_ids.to(device), labels.to(device)
        predict = model(seqs, type_ids)
        loss = loss_fn(predict, labels)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
    return np.mean(train_loss)


def evaluate_epoch(model, test_loader, loss_fn, pearson, device):
    model.eval()
    valid_loss = []
    vp, vt = [], []
    with torch.no_grad():
        for seqs, type_ids, labels in test_loader:
            seqs, type_ids, labels = seqs.to(device), type_ids.to(device), labels.to(device)
            pred = model(seqs, type_ids)
            val_loss = loss_fn(pred, labels)
            valid_loss.append(val_loss.item())
            vp.extend(pred.squeeze().cpu().numpy())
            vt.extend(labels.squeeze().cpu().numpy())
    return np.mean(valid_loss), vp, vt


def train_and_evaluate(trait, data_path, label_path, geno_path, device, learning_rate, epochs,seed, sel_num):
    setup_seed(3407)

    loss_fn = nn.L1Loss()
    bs = 32
    traindataset = DNADataset(data_path, label_path, geno_path, trait, seed,sel_num, is_training=True)
    testdataset = DNADataset(data_path, label_path, geno_path, trait, seed, sel_num,is_training=False)

    train_loader = DataLoader(traindataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(testdataset, batch_size=bs, shuffle=False)


    model = EBMGP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    steps = math.ceil(len(traindataset) / bs) * epochs - 1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    pearson = PearsonCorrCoef().to(device)

    corrs, RMSE, pred, obser = [], [], [], []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scheduler, device)
        #print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}')
        if epoch == epochs:
            valid_loss, vp, vt = evaluate_epoch(model, test_loader, loss_fn, pearson, device)
            valMSE = mean_squared_error(vp, vt)
            v_corr = pearson(torch.tensor(vp).to(device), torch.tensor(vt).to(device))
            RMSE.append(valMSE)
            corrs.append(v_corr.item())
            pred.extend(vp)
            obser.extend(vt)
    return np.mean(corrs), np.mean(RMSE), pred, obser


def main():
    learning_rate = 0.0005
    epochs = 100
    sel_num = 5000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    label_path = "./data/rice_pheno.csv"
    geno_path = './data/ricerawgeno.csv' 
    #geno_path = './data/soybeanrawgeno.csv'
    #label_path = "./data/soybean_pheno.csv" 
    #label_path = "./data/sorghum_pheno.csv"
    #geno_path = './data/sorghumrawgeno.csv'    
    #label_path = "./data/bulls_pheno.csv"
    #geno_path = "./data/bullsgeno.csv"
    traits = ['SW', 'FLW', 'AC', 'PH', 'SNPP']
    #traits = ['HT-IL','FL-IL','HT-MX']
    #traits = ['protein','Steartic','R8','SdWgt','Yield']
    #traits = ['MS','NMSP','VE']
    mean_corrs, mean_RMSE = {}, {}


    for trait in range(5):
        data_path = f"./EN/rice/T5000{traits[trait]}"
        corrs,RMSE = [],[]
        for seed in range(5):
            fold_corrs, fold_RMSE, fold_pred, fold_obser = train_and_evaluate(
                trait, data_path, label_path, geno_path, device, learning_rate, epochs,seed, sel_num
            )
            corrs.append(fold_corrs)
            RMSE.append(fold_RMSE)
            print(traits[trait],fold_corrs,fold_RMSE)

    mean_corrs[traits[trait]] = np.mean(corrs)
    mean_RMSE[traits[trait]] = np.mean(RMSE)

    print(mean_corrs)
    print(mean_RMSE)

if __name__ == "__main__":
    main()
