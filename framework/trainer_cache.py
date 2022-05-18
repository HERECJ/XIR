import os, datetime, torch
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from framework.model import MFModel
from framework.dataloader import UserHisData, UserTestData, pad_collate_valid
from framework.trainer import Trainer, get_logger
from framework.debias import Pop_Debias
from torch.utils.tensorboard import SummaryWriter

class Cache(Pop_Debias):
    def __init__(self, pop_count, device, size, mode=1, **kwargs):
        super().__init__(pop_count, device, mode, **kwargs)

        self.size = size
        self.occurence = torch.ones(self.item_num, device=device)

        self.sample_pool = torch.randint(1, self.item_num, (size,), device=self.device) # self.item_num: item_num + 1

    def update_pool(self, current_items, current_scores, cached_scores, batch_idx=0, ratio=0.5, **kwargs):
        num_item = int(self.size / 2) 
        # check the dtype of num_item

        P_cache = self.occurence[self.sample_pool]


        cache_weights = F.softmax(cached_scores - torch.log(self.pop_prob[self.sample_pool]), dim=-1)
        current_weights = F.softmax(current_scores - torch.log(self.pop_prob[current_items]), dim=-1)

        indices_cache = torch.multinomial(cache_weights, num_item, replacement=True)
        indices_current = torch.multinomial(current_weights, num_item, replacement=True)

        sampled_items = torch.cat([self.sample_pool[indices_cache], current_items[indices_current]], dim=-1)

        values, counts = torch.unique(sampled_items, return_counts=True)

        # if batch_idx == 0: 
            # self.occurence = torch.zeros(self.item_num, device=self.device)
        
        self.occurence[values] += counts

        cache_indices = torch.multinomial(self.occurence[values], self.size, replacement=True)

        self.sample_pool = values[cache_indices]

        return torch.gather(cached_scores, -1, indices_cache), torch.gather(current_scores, -1, indices_current)

    def get_items_from_cache(self):
        return self.sample_pool


class Trainer_Cache(Trainer):
    def __init__(self, config):
        super().__init__(config)
        assert self.config['lambda'] <= 1.0 and self.config['lambda'] >=  0.0

    def init_logger(self):
        if not os.path.exists(self.config['log_path']):
            os.makedirs(self.config['log_path'])
        
        ISOTIMEFORMAT = '%m%d-%H%M%S'
        timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
        seed = 'seed' + str(self.config['seed']) 

        log_name = '_'.join((self.config['data_name'], str(self.config['batch_size']), str(self.config['debias']), str(self.config['lambda']), str(self.config['learning_rate']), seed, timestamp))
        os.makedirs(os.path.join(self.config['log_path'], log_name))
        log_file_name = os.path.join(self.config['log_path'], log_name)
        self.writer = SummaryWriter(log_dir=log_file_name)
        
        logname = log_file_name + '/log.txt'
        self.logger = get_logger(logname)
        self.logger.info(self.config)
    
    def fit(self, train_mat, test_mat):
        train_data = UserHisData(train_mat=train_mat)
        train_loader = DataLoader(train_data, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], shuffle=True)
        test_data = UserTestData(train_mat=train_mat, test_mat=test_mat)
        test_loader = DataLoader(test_data, batch_size=self.config['eval_batch_size'], collate_fn=pad_collate_valid, num_workers=self.config['num_workers'])
        model = self.model_init(train_mat=train_mat)

        pop_count = train_mat.sum(axis=0).A.squeeze()
        if self.config['debias'] in [8]:
            debias_module = Cache(pop_count, self.device, self.config['batch_size'], mode=self.config['pop_mode'])
        else:
            raise NotImplementedError

        debias_module = debias_module.to(self.device)
        self._fit(model, debias_module, train_loader, test_loader)
    
    def _train_step(self, user_id, item_id, model: MFModel, debias: Cache, batch_idx=0, **kwargs):
        query = model.construct_query(user_id)
        item_emb = model.item_encoder(item_id)

        scores = torch.matmul(query, item_emb.T)
        pos_rat = torch.diag(scores)

        cache_items = debias.get_items_from_cache()
        cache_item_emb = model.item_encoder(cache_items)
        cache_score = torch.matmul(query, cache_item_emb.T)

        neg_rat_cache, neg_rat_current = debias.update_pool(item_id, scores, cache_score, batch_idx=batch_idx, ratio=self.config['lambda'])

        loss_cache = model.loss_(pos_rat, neg_rat_cache)
        loss_current = model.loss_(pos_rat, neg_rat_current)

        return self.config['lambda'] * loss_cache + (1-self.config['lambda']) * loss_current