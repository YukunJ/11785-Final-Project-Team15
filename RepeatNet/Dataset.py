from torch.utils.data import Dataset
import sys
import csv
import codecs
import torch

class RepeatNetDataset(Dataset):
    def __init__(self, sample_file, side_file):
        super(RepeatNetDataset, self).__init__()

        self.sample_file=sample_file
        self.side_file=side_file

        self.item_atts=dict()
        self.samples=[]

        self.side_atts=dict()
        self.sizes=[]
        self.load()

    def load(self):
        clean = lambda l: [int(x) for x in l.strip('[]').split(',')]

        id=0
        with codecs.open(self.sample_file, encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter='|')
            for row in csv_reader:
                id+=1
                self.samples.append([torch.tensor([id]), torch.tensor(clean(row[0])), torch.tensor(clean(row[1]))])

        with codecs.open(self.side_file, encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter='|')
            for row in csv_reader:
                id+=1
                self.sizes.append([torch.tensor([id]), torch.tensor(clean(row[0])), torch.tensor(clean(row[1]))])

        self.len=len(self.samples)
        self.size_len=len(self.sizes)
        print('data size: ', self.len)
        print('side data size: ', self.size_len)

    def __getitem__(self, index):
        return self.samples[index], self.sizes[index]

    def __len__(self):
        return self.len

def collate_fn(data):
    item, side = [i for i, s in data], [s for i, s in data]
    id, item_seq, item_tgt = zip(*item)
    _, side_seq, _ = zip(*side)

    return {
            'id': torch.cat(id),
            'item_seq': torch.stack(item_seq),
            'side_seq': torch.stack(side_seq),
            'item_tgt': torch.stack(item_tgt)
            }
