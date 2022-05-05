import torch
import torch.nn as nn
import torch.nn.functional as F


class Base_Debias(nn.Module):
    def __init__(self, item_num, device, **kwargs):
        super().__init__()
        self.item_num = item_num + 1
        self.device = device
    
    def forward(self, items):
        """
            calculate the sampled weights
            Base_Debias utilizes the uniform sampling
        """
        # pos_items : B x 1  (N_p is the length of padded [sequence] postive items)
        # neg_items : B x B
        # select other training examples as negatives
        return -torch.log(self.item_num * torch.ones_like(items))
    
class Pop_Debias(Base_Debias):
    """
        debias the weights according to the popularity
    """
    def __init__(self, pop_count, device, mode=1, **kwargs):
        super().__init__(pop_count.shape[0], device)
        pop_count = torch.from_numpy(pop_count).to(self.device) # check whether use the device parameter
        if mode == 1:
            pop_count = pop_count
        elif mode == 2:
            pop_count = torch.log( 1 + pop_count )
        elif mode == 3:
            pop_count = pop_count ** 0.75
        else:
            raise ValueError
        
        pop_count = torch.cat([torch.zeros(1, device=self.device), pop_count])
        self.pop_prob = pop_count / pop_count.sum() # other normalization strategy can be satisfied
        self.pop_prob[0] = torch.ones(1, device=self.device) # padding values
    
    def forward(self, items):
        return torch.log(self.pop_prob[items])

class ReSample_Debias(Pop_Debias):
    def __init__(self, pop_count, device, mode=1, **kwargs):
        super().__init__(pop_count, device, mode, **kwargs)
    
    def get_pop_bias(self, items):
        return torch.log(self.pop_prob[items])
    
    def resample(self, score, log_prob, sample_size):
        # score : B x B
        # log_prob: B 
        sample_weight = F.softmax(score - log_prob, dim=-1)
        indices = torch.multinomial(sample_weight, sample_size, replacement=True)
        return -torch.log(self.item_num * torch.ones_like(log_prob)), indices, -torch.log(self.item_num * torch.ones_like(log_prob[indices]))

class MixNeg_Debias(Pop_Debias):
    def __init__(self, pop_count, device, mode=1, **kwargs):
        super().__init__(pop_count, device, mode, **kwargs)
    
    def get_pop_bias(self, items):
        return torch.log(self.pop_prob[items])
    
    def forward(self, items, ratio=0.5):
        return torch.log(ratio * self.pop_prob[items] + (1 - ratio) * (1.0 / self.item_num))