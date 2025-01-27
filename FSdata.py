import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from itertools import accumulate

class DNADataset(Dataset):
    def __init__(self, data_path, label_path, geno_path, trait, seed, sel_num,is_training=True):
        cs = pd.read_csv(f"{data_path}{seed}.csv").sort_values(by='cs', ascending=False)
        Top = sorted(cs.index[:sel_num])  
        Rawgeno = pd.read_csv(geno_path)
        geno = Rawgeno.loc[Top]
        LD = self.calculate_LD(geno)
        geno['gap'] = self.assign_gap_labels(LD)
        geno = geno.drop(columns=geno.columns[[-3, -2]])  
        lines = self.generate_geno_sequences(geno)
        annos = pd.read_csv(label_path, index_col=0).iloc[:, [trait]]
        annos = annos.fillna(annos.mean()) 
        annos = StandardScaler().fit_transform(annos).astype(np.float32)

        kfold = KFold(n_splits=5, shuffle=True, random_state=27)
        for i, (train_idx, val_idx) in enumerate(kfold.split(lines, annos)):
            if i == seed:
                train_lines, val_lines = lines[train_idx], lines[val_idx]
                train_annos, val_annos = annos[train_idx], annos[val_idx]                
                break

        train_seqs, train_type_ids = self.process_sequences(train_lines)
        val_seqs, val_type_ids = self.process_sequences(val_lines)

        if is_training:
            self.seqs, self.type_ids, self.annos = train_seqs, train_type_ids, train_annos
        else:
            self.seqs, self.type_ids, self.annos = val_seqs, val_type_ids, val_annos

    def calculate_LD(self, geno):
        geno_values = np.select(
            [geno.iloc[:, :-2].values == 'H', geno.iloc[:, :-2].values == 'M', geno.iloc[:, :-2].values == 'L'],
            [0, 1, 2],
            default=-1
        ) 

        a, b = geno_values[:-1], geno_values[1:]  
        var_a, var_b = np.var(a, axis=1), np.var(b, axis=1)
        mean_a, mean_b = np.mean(a, axis=1), np.mean(b, axis=1)
        d = np.mean((a - mean_a[:, None]) * (b - mean_b[:, None]), axis=1)
        
        LD = np.where((var_a == 0) | (var_b == 0), 0, (d ** 2) / (var_a * var_b))
        LD = np.append(LD, -1)  

        chrom = sorted(set(geno['chrom']))
        index = list(accumulate([len(geno.groupby('chrom').get_group(i)) for i in chrom])) 
        for idx in index:
            LD[idx - 1] = -1        
        return LD

    def assign_gap_labels(self, LD):
        return np.where(LD == -1, 'N', np.where(LD >= 0.8, 'J', 'Y'))

    def generate_geno_sequences(self, geno):
        lines = []
        for i in range(geno.shape[1] - 1):
            geno.iloc[:, i] = geno.iloc[:, i] + geno['gap']
            lines.append(''.join(geno.iloc[:, i]))
        return np.stack(lines, axis=0)

    def process_sequences(self, lines):
        vocabs = {f"{a}{b}": i + 1 for i, (a, b) in enumerate([("H", "J"), ("H", "Y"), ("H", "N"), ("L", "J"), ("L", "Y"), ("L", "N"), ("M", "J"), ("M", "Y"), ("M", "N")])}
        type_vocabs = {"J": 1, "Y": 2, "N": 3}

        seqs, type_ids = [], []
        for raw_seq in lines:
            seq, type_id = [], []
            for i in range(0, len(raw_seq), 2):
                seq.append(vocabs[raw_seq[i:i + 2]])
                type_id.append(type_vocabs[raw_seq[i + 1]])
            seqs.append(seq)
            type_ids.append(type_id)

        return np.asarray(seqs), np.asarray(type_ids)

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        seq = torch.tensor(self.seqs[index], dtype=torch.float32)
        type_ids = torch.tensor(self.type_ids[index], dtype=torch.float32)
        annos = torch.tensor(self.annos[index], dtype=torch.float32)
        return seq, type_ids, annos
