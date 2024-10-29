import itertools
from torch.utils.data import Dataset

class BenchLS(Dataset):
    def __init__(self, file_path = 'datasets/BenchLS.txt'):
        self.data, self.target = list(), list()
        with open(file_path,'rt') as f:
            for line in f:
                data = line.strip().split('\t')
                replacement = list()
                for rep in data[3:]:
                    rank_rep = rep.split(':')
                    replacement.append((rank_rep[1], int(rank_rep[0])))
                
                self.data.append(data[0])
                self.target.append((data[1], replacement))

        print('[INFO] Loaded dataset {}'.format(file_path))
        print('[INFO] Database has {} samples'.format(len(self.data)))
        print('[INFO] Sample for example:\n  ',self.data[0], '\n')
        print('[INFO] Candidate for replacement and options ranking:\n  ', self.target[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
class LexMTurk(Dataset):
    def __init__(self, file_path = 'datasets/lex.mturk.txt'):
        self.data, self.target = list(), list()
        with open(file_path, 'rt', encoding='iso8859-1') as f:
            for line in itertools.islice(f, 1, 501):
                data = line.strip().split('\t')
                replacement = dict()
                for rep in data[3:]:
                    if rep not in replacement:
                        replacement[rep] = 1
                    else:
                        replacement[rep] += 1
                replacement = [(k, v) for k, v in replacement.items()]

                self.data.append(data[0])
                self.target.append((data[1], replacement))

        print('[INFO] Loaded dataset {}'.format(file_path))
        print('[INFO] Database has {} samples'.format(len(self.data)))
        print('[INFO] Sample for example:\n  ',self.data[0], '\n')
        print('[INFO] Candidate for replacement and options ranking:\n  ', self.target[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
class NNSeval(Dataset):
    def __init__(self, file_path = 'datasets/NNSeval.txt'):
        self.data, self.target = list(), list()
        with open(file_path,'rt') as f:
            for line in f:
                data = line.strip().split('\t')
                replacement = list()
                for rep in data[3:]:
                    rank_rep = rep.split(':')
                    replacement.append((rank_rep[1], int(rank_rep[0])))
                
                self.data.append(data[0])
                self.target.append((data[1], replacement))

        print('[INFO] Loaded dataset {}'.format(file_path))
        print('[INFO] Database has {} samples'.format(len(self.data)))
        print('[INFO] Sample for example:\n  ',self.data[0], '\n')
        print('[INFO] Candidate for replacement and options ranking:\n  ', self.target[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
