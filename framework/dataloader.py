import os, random
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class RatMixData(object):
    # The class fits in those rating datas, with ratings ranging from different level of values.
    # For other datasets, such as CTR datasets, refer to other class.
    def __init__(self, dir:str, data:str):
        self.file_path = os.path.join(dir, data + '.mat')
        
    def get_train_test(self, split_ratio:float=0.8):   
        mat = sio.loadmat(self.file_path)['data'] # the rating matrix

        # Split train/test data
        train_mat, test_mat = self.split_matrix(mat, split_ratio)
        return train_mat, test_mat

    def split_matrix(self, mat, ratio=0.8):
        # TODO : whether only use the positive samples
        mat = mat.tocsr()  #followed by rows(users)
        m,n = mat.shape
        train_data_indices = []
        train_indptr = [0] * (m+1)
        test_data_indices = []
        test_indptr = [0] * (m+1)
        for i in range(m):
            row = [(mat.indices[j], mat.data[j]) for j in range(mat.indptr[i], mat.indptr[i+1])]
            train_idx = random.sample(range(len(row)), round(ratio * len(row)))
            train_binary_idx = np.full(len(row), False)
            train_binary_idx[train_idx] = True
            test_idx = (~train_binary_idx).nonzero()[0]
            for idx in train_idx:
                train_data_indices.append(row[idx]) 
            train_indptr[i+1] = len(train_data_indices)
            for idx in test_idx:
                test_data_indices.append(row[idx])
            test_indptr[i+1] = len(test_data_indices)

        [train_indices, train_data] = zip(*train_data_indices)
        [test_indices, test_data] = zip(*test_data_indices)

        train_mat = sp.csr_matrix((train_data, train_indices, train_indptr), (m,n))
        test_mat = sp.csr_matrix((test_data, test_indices, test_indptr), (m,n))
        return train_mat, test_mat    

    
class UserHisData(Dataset):
    def __init__(self, train_mat:sp.spmatrix):
        super().__init__()
        self.train = train_mat.tocoo()
    
    def __len__(self):
        return self.train.nnz
    
    def __getitem__(self, idx):
        return self.train.row[idx].astype(np.int64), self.train.col[idx].astype(np.int64) + 1

class UserTestData(Dataset):
    def __init__(self, train_mat, test_mat):
        # the max_test_num is always smaller than the number of users. Maybe modified.
        super().__init__()
        self.train, self.test = train_mat, test_mat

    def __len__(self):
        return  self.train.shape[0]
    
    def __getitem__(self, index):
        user_id = torch.LongTensor([index])
        user_his = self.train[index].nonzero()[1] + 1
        
        start, end = self.test.indptr[index], self.test.indptr[index + 1]
        user_test = self.test.indices[start:end] + 1
        user_rating = self.test.data[start:end] 

        return user_id, torch.LongTensor(user_his), torch.LongTensor(user_test), torch.Tensor(user_rating)

def pad_collate_valid(batch):
    (user, user_his, items, user_rating) = zip(*batch)
    return torch.LongTensor(user), pad_sequence(user_his, batch_first=True), pad_sequence(items, batch_first=True), pad_sequence(user_rating, batch_first=True)


if __name__ == '__main__':
    dir = 'datasets/clean_data'
    data = 'gowalla'


    mldata = RatMixData(dir, data)
    train_mat, test_mat = mldata.get_train_test()
    
    train_data = UserHisData(train_mat)
    
    train_loader = DataLoader(train_data, batch_size=2048, num_workers=8, shuffle=True, pin_memory=True)
    import time
    start_time = time.time()

    # for _ in range(1):
    #     for batch_data in train_loader:
    #         user_id, neg_id = batch_data
    #         # if neg_items is None:
    #             # if neg_ is false, neg_items is None
    #             # if neg_items is not None, add the negatives into the negative set
    #             # pass

    # end_time = time.time()
    # print(end_time - start_time)

    test_data = UserTestData(train_mat, test_mat)
    test_loader = DataLoader(test_data, batch_size=11, collate_fn=pad_collate_valid)
    for batch in test_loader:
        user_id, user_his, user_cand, user_rating = batch
        import pdb; pdb.set_trace()

