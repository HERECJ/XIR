import logging, os, datetime, time
from framework.dataloader import RatMixData, UserHisData, UserTestData, pad_collate_valid
from framework.model import TowerModel, MFModel
from framework.debias1 import Base_Debias, Pop_Debias, RePop_Debias, RePop_Debias_pop, RePop_Debias_uni
import framework.eval as eval
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def get_logger(filename, verbosity=1, name=None):
    filename = filename + '.txt'
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.init_logger()
        self.set_seed()
        
    
    def init_logger(self):
        if not os.path.exists(self.config['log_path']):
            os.makedirs(self.config['log_path'])
        
        ISOTIMEFORMAT = '%m%d-%H%M%S'
        timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))


        sampled_flag = 'sampled_' + str(self.config['sample_size']) if self.config['sample_from_batch'] is True else 'full'
        log_name = '_'.join((self.config['data_name'], str(self.config['debias']), sampled_flag, timestamp))
        os.makedirs(os.path.join(self.config['log_path'], log_name))
        log_file_name = os.path.join(self.config['log_path'], log_name)
        self.writer = SummaryWriter(log_dir=log_file_name)
        
        logname = log_file_name + '/log.txt'
        self.logger = get_logger(logname)
        self.logger.info(self.config)

    def set_seed(self):
        if self.config['fix_seed']:
            import os
            seed = self.config['fix_seed']
            os.environ['PYTHONHASHSEED']=str(seed)

            import random
            random.seed(seed)
            np.random.seed(seed)
            
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    def load_dataset(self):
        mldata = RatMixData(self.config['data_dir'],self.config['data_name'])

        train_mat, test_mat = mldata.get_train_test()
        (M,N) = train_mat.shape
        self.logger.info('Number of Users/Items, {}/{}'.format(M,N))
        self.item_num = N + 1
        return train_mat, test_mat
    
    def model_init(self, train_mat):
        (user_num, item_num) = train_mat.shape
        if self.config['model'].lower() == 'mf':
            return MFModel(user_num, item_num, self.config['emb_dim']).to(self.device)
        else:
            raise ValueError('Not supported model types')
    
    def config_optimizers(self, parameters, lr, wd):
        if self.config['optim'].lower() == 'adam':
            return optim.Adam(parameters, lr=lr, weight_decay=wd) 
        elif self.config['optim'].lower() == 'sgd':
            return optim.SGD(parameters, lr=lr, weight_decay=wd)
        else:
            raise NotImplementedError

    def topk(self, model, query, k, user_h=None):
        more = user_h.size(1) if user_h is not None else 0
        score, topk_items = torch.topk(model.scorer(query, model.item_encoder.weight[1:]), k + more)
        if user_h is not None:
            topk_items += 1
            existing, _ = user_h.sort()
            idx_ = torch.searchsorted(existing, topk_items)
            idx_[idx_ == existing.size(1)] = existing.size(1) - 1
            score[torch.gather(existing, 1, idx_) == topk_items] = -float('inf')
            score1, idx = score.topk(k)
            return score1, torch.gather(topk_items, 1, idx)
        else:
            return score, topk_items

    def _test_step(self, model, test_data, eval_metric, cutoffs):
        user_id, user_his, user_cand, user_rating = test_data
        user_id, user_his, user_cand, user_rating = user_id.to(self.device), user_his.to(self.device), user_cand.to(self.device), user_rating.to(self.device)
        rank_m = eval.get_rank_metrics(eval_metric)
        topk = self.config['topk']
        bs = user_id.size(0)
        query = model.construct_query(user_id)
        score, topk_items = self.topk(model, query, topk, user_his)
        if user_cand.dim() > 1:
            target, _ = user_cand.sort()
            idx_ = torch.searchsorted(target, topk_items)
            idx_[idx_ == target.size(1)] = target.size(1) - 1
            label = torch.gather(target, 1, idx_) == topk_items
            pos_rating = user_rating
        else:
            label = user_cand.view(-1, 1) == topk_items
            pos_rating = user_rating.view(-1, 1)
        return [func(label, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m], bs

    def evaluate(self, model, test_loader):
        model.eval()
        eval_metric = self.config['metrics']
        cutoffs = self.config['cutoffs']
        out_res = []
        for batch_idx, test_data in enumerate(test_loader):
            outputs = self._test_step(model, test_data, eval_metric, cutoffs)
            out_res.append(outputs)

        metric, bs = zip(*out_res)
        metric = torch.tensor(metric)
        bs = torch.tensor(bs)
        out = (metric * bs.view(-1, 1)).sum(0) / bs.sum()
        metrics = [f"{v}@{c}" for c in cutoffs for v in eval_metric]
        out = dict(zip(metrics, out))
        return out

    def _train_step(self, user_id, item_id, model:TowerModel, debias:Base_Debias):
        """
            Support not resample based method
        """
        # TODO : item features (optional)
        query = model.construct_query(user_id)
        item_emb = model.item_encoder(item_id)


        # generate the index matrix of items
        B = user_id.shape[0]
        if self.config['sample_from_batch']:
            sample_size = self.config['sample_size']
            assert sample_size > 1 and sample_size<B, ValueError('The number of samples must be greater than 1 and smaller than batch_size')
            # Actually, 'replacement=True'
            IndM = torch.randint(B, size=(B,sample_size), device=self.device)  # B x S
        else:
            IndM = torch.arange(B, device=self.device).view(1,-1).repeat(B,1)  # B x B
        
        neg_items = item_id[IndM]
        # neg_items_emb = item_emb[IndM]

        log_pos_prob, log_neg_prob = debias(item_id), debias(neg_items)


        scores = torch.matmul(query, item_emb.T)
        # neg_items = item_id[IndM]
        ##### training step
        pos_rat = torch.diag(scores)
        neg_rat = torch.gather(scores, 1, IndM)

        ##### training step
        # pos_rat = model.scorer(query, item_emb)
        # neg_rat = model.scorer(query, neg_items_emb) 
        loss = model.loss(pos_rat, log_pos_prob, neg_rat, log_neg_prob)
        return loss


    def _fit(self, model:TowerModel, debias:Base_Debias, train_loader:DataLoader, test_loader=None):
        num_epoch = self.config['epoch']
        optimizer = self.config_optimizers(model.parameters(), self.config['learning_rate'], self.config['weight_decay'])
        
        if self.config['steprl'] :
            scheduler = optim.lr_scheduler.StepLR(optimizer, self.config['step_size'], self.config['step_gamma'])


        for epoch in range(num_epoch):
            loss_ = 0.0
            
            for batch_idx, batch_data in enumerate(train_loader):
                model.train()
                debias.train()

                optimizer.zero_grad()

                user_id, item_id = batch_data
                user_id, item_id = user_id.to(self.device), item_id.to(self.device)

                loss = self._train_step(user_id, item_id, model, debias)

                loss_ += loss.item()
                loss.backward()
                optimizer.step()

            if self.config['steprl'] :
                scheduler.step()
            self.writer.add_scalar("Train/Loss", loss_/(batch_idx+1.0), epoch)
            self.logger.info('Epoch {}'.format(epoch))
            self.logger.info('***************Train loss {:.8f}'.format(loss_))

            if ((epoch % self.config['valid_interval']) == 0) or (epoch >= num_epoch - 1):
                with torch.no_grad():
                    out = self.evaluate(model, test_loader)

                for k in out.keys():
                    self.writer.add_scalar("Evaluate/{}".format(k), out[k], epoch)
                ress = (', ').join(["{} : {:.6f}".format(k, out[k]) for k in out.keys()])
                    
                self.logger.info('***************Eval_Res ' + ress)
        
            self.writer.flush()


    def fit(self, train_mat, test_mat):
        train_data = UserHisData(train_mat=train_mat)
        train_loader = DataLoader(train_data, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], shuffle=True, pin_memory=True)
        test_data = UserTestData(train_mat=train_mat, test_mat=test_mat)
        test_loader = DataLoader(test_data, batch_size=self.config['eval_batch_size'], collate_fn=pad_collate_valid, num_workers=self.config['num_workers'])
        model = self.model_init(train_mat=train_mat)

        #=========================================
        # Define bias mmodule
        # Base debias : uniform, Pop debias : pop, RePop debias : resampling + pop debias
        if self.config['debias'] == 1 :
            """ base debias, uniform sampling  """
            debias_module = Base_Debias(train_mat.shape[1], self.device)
        elif self.config['debias'] == 2:
            """ debias with popularity   """
            pop_count = train_mat.sum(axis=0).A.squeeze()
            debias_module = Pop_Debias(pop_count, self.device)
        elif self.config['debias'] == 3:
            pop_count = train_mat.sum(axis=0).A.squeeze()
            debias_module = RePop_Debias(pop_count, self.device)
        elif self.config['debias'] == 4:
            pop_count = train_mat.sum(axis=0).A.squeeze()
            debias_module = RePop_Debias_pop(pop_count, self.device)
        elif self.config['debias'] == 5:
            pop_count = train_mat.sum(axis=0).A.squeeze()
            debias_module = RePop_Debias_uni(pop_count, self.device)
        else:
             raise NotImplementedError
        
        debias_module = debias_module.to(self.device)
        #=========================================
        self._fit(model, debias_module, train_loader, test_loader)


class Trainer_Full(Trainer):
    def __init__(self, config):
        super().__init__(config)
    
    def _train_step(self, user_id, item_id, model: TowerModel, debias: Base_Debias):
        query = model.construct_query(user_id)
        pos_score = model.scorer(query, model.item_encoder(item_id))
        all_score = model.scorer(query, model.item_encoder.weight[1:])
        loss = model.loss_full_softmax(pos_score, all_score)
        return loss

class Trainer_BPR(Trainer):
    def __init__(self, config):
        super().__init__(config)
    
    def _train_step(self, user_id, item_id, model: TowerModel, debias: Base_Debias):
        B = item_id.shape[0]
        query = model.construct_query(user_id)
        pos_score = model.scorer(query, model.item_encoder(item_id))

        neg_items = torch.randint(self.item_num, size=(B, 5), device=self.device)
        neg_score =  model.scorer(query, model.item_encoder(neg_items))
        loss = -torch.mean(F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score))
        return loss

class Trainer_Resample(Trainer):
    def __init__(self, config):
        super().__init__(config)
    
    def _train_step(self, user_id, item_id, model: TowerModel, debias: Base_Debias):
        """
            Support resample        
        """
        # TODO : item features (optional)
        query = model.construct_query(user_id)
        item_emb = model.item_encoder(item_id)

        # generate the index matrix of items
        B = user_id.shape[0]
        if self.config['sample_from_batch']:
            sample_size = min(B, self.config['sample_size'])
        else:
            sample_size = B
        log_pop_bias = debias.get_pop_bias(item_id)

        
        scores = torch.matmul(query, item_emb.T)
        log_pos_prob, IndM, log_neg_prob = debias.resample(scores, log_pop_bias, sample_size)

        # neg_items = item_id[IndM]
        ##### training step
        pos_rat = torch.diag(scores)
        neg_rat = torch.gather(scores, 1, IndM)
        loss = model.loss(pos_rat, log_pos_prob, neg_rat, log_neg_prob)
        return loss