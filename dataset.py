

import numpy as np
import torch
from torch.utils.data import Dataset

class TileDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        config_feat = torch.tensor(row['config_feat'].astype(np.float32))
        node_feat = torch.tensor(row['node_feat'].astype(np.float32))
        node_opcode = torch.tensor(row['node_opcode'].astype(np.int64))
        edge_index = torch.tensor(np.swapaxes(row['edge_index'],0,1).astype(np.int64))
        target = (row['config_runtime']/(row['config_runtime_normalizers']+1e-5)).astype(np.float32) 
        #/row['config_runtime_normalizers']
        # minmax scale the target, we only care about order
        target = (target-np.mean(target))/(np.std(target)+1e-5)

#         target = (target-np.mean(target))/(np.std(target))
        target = torch.tensor(target)
        return config_feat,node_feat,node_opcode,edge_index,target