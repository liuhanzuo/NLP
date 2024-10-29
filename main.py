from dataset import LexMTurk, NNSeval, BenchLS
from evaluation import evaluate
from model import BaseModel
import torch
from torch.utils.data import DataLoader

def main():
    model = BaseModel()
    dataloader = DataLoader(LexMTurk(), batch_size=1, shuffle=True)
    sg, sr = evaluate(model, dataloader)
    print(sg, sr)

if __name__ == '__main__':
    main()