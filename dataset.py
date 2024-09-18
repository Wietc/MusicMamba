import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from midi_tokenize.remi.vocab import Vocab

from utilities.constants import *
from utilities.device import cpu_device

SEQUENCE_START = 0
myvocab = Vocab()
TOKEN_PAD = myvocab.token2id["padding"]


class EPianoDataset(Dataset):

    def __init__(self, root, max_seq=2048, random_seq=True):
        self.root       = root
        self.max_seq    = max_seq
        self.random_seq = random_seq

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        i_stream    = open(self.data_files[idx], "rb")
        raw_mid = pickle.load(i_stream)
        remi_tokens = torch.tensor(raw_mid["remi_tokens"], dtype=TORCH_LABEL_TYPE, device=cpu_device())
        mode_mask = torch.tensor(raw_mid["mode_mask"], dtype=TORCH_LABEL_TYPE, device=cpu_device())
        i_stream.close()

        x, tgt, x_mask = process_midi(remi_tokens, mode_mask, self.max_seq, self.random_seq)

        return x, tgt, x_mask
    
    
def process_midi(remi_tokens, mode_mask, max_seq, random_seq):
    x          = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())
    x_mask     = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())
    tgt        = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())
    
    raw_len     = len(remi_tokens)
    full_seq    = max_seq + 1 # Performing seq2seq

    if(raw_len == 0):
        return x, tgt, x_mask

    if(raw_len < full_seq):
        x[:raw_len]         = remi_tokens
        tgt[:raw_len-1]     = remi_tokens[1:]
        tgt[raw_len-1]      = TOKEN_PAD
        x_mask[:raw_len]    = mode_mask
    else:
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = remi_tokens[start:end]
        mode_mask = mode_mask[start:end]
        
        x = data[:max_seq]
        x_mask = mode_mask[:max_seq]
        tgt = data[1:full_seq]

    return x, tgt, x_mask


def create_epiano_datasets(dataset_root, max_seq, random_seq=True):
    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = EPianoDataset(train_root, max_seq, random_seq)
    val_dataset = EPianoDataset(val_root, max_seq, random_seq)
    test_dataset = EPianoDataset(test_root, max_seq, random_seq)

    return train_dataset, val_dataset, test_dataset


def compute_epiano_accuracy(out, tgt):
    out = torch.argmax(F.softmax(out, dim=-1), dim=-1)

    out = out.flatten()
    tgt = tgt.flatten()

    mask = (tgt != TOKEN_PAD)

    out = out[mask]
    tgt = tgt[mask]

    if(len(tgt) == 0):
        return 1.0

    num_right = (out == tgt)
    num_right = torch.sum(num_right).type(TORCH_FLOAT)

    acc = num_right / len(tgt)

    return acc
