import torch

def update_statistics_1d_v5(local_mean:torch.Tensor, local_var:torch.Tensor, global_mean:torch.Tensor, global_var:torch.Tensor, momentum:float, data:torch.Tensor, label:torch.Tensor, gamma:float, training:bool=False):
    if local_mean.size(0) != local_var.size(0):
        raise RuntimeError("the sizes of local_mean and local_var are inequal.")
    
    if data.size(0) != label.size(0):
        raise RuntimeError("the values of the first dimension of data and label are inequal.")
    
    if training:
        # for example, label: [2, 2, 2, 3, 3, 4] (N = 6)
        l:torch.Tensor = label.unique()  # [2, 3, 4]
        l_mapping = torch.arange(l.size(0)).to(l.device)  # [0, 1, 2]
        l_reverse_mapping = l.new_full((local_mean.size(0), ), -1)
        l_reverse_mapping[l] = l_mapping  # [-1, -1, 0, 1, 2]
        label_local = l_reverse_mapping[label]  # (N,) [0, 0, 0, 1, 1, 2]
        lm = local_mean[label]  # (N, C)
            # [local_mean(0), local_mean(0), local_mean(0), local_mean(1), local_mean(1), local_mean(2)]
        
        label_num = torch.zeros_like(l_mapping)  # (K',) [0, 0, 0]
        label_num.scatter_add_(0, label_local, torch.ones_like(label_local))  # (K',) [3, 2, 1]
        mask = label_num > (1.0 / momentum)
        m = torch.where(mask, 1.0 / label_num.float(), momentum)

        delta_pre = data - lm[..., None]   # (N, C, L)

        delta_k = delta_pre.new_zeros((l.size(0), delta_pre.size(1)))  # (K', C)
        delta_k.scatter_add_(0, label_local.unsqueeze(-1).expand(-1, delta_k.size(1)), delta_pre.mean(2))  # (K', C)
        delta_k *= m.view(-1, 1)
        local_mean[l] = (1.0 - gamma) * delta_k + local_mean[l]
        local_mean = gamma * delta_k.mean(0, keepdim=True) + local_mean

        delta_square_k = delta_pre.new_zeros((l.size(0), delta_pre.size(1)))
        delta_square_k.scatter_add_(0, label_local.unsqueeze(-1).expand(-1, delta_k.size(1)), delta_pre.pow(2).mean(2))
        local_var[l] = local_var[l] + (1.0 - gamma) * (m.view(-1, 1) * (delta_square_k - label_num.view(-1, 1) * local_var[l]) - delta_k.pow(2))

        var_gap = (m.view(-1, 1) * delta_square_k - delta_k.pow(2)).mean(0, keepdim=True)
        local_var[...] = local_var + gamma * (var_gap - (m * label_num).mean() * local_var)

        global_mean[...] = local_mean.mean(0)
        global_var[...] = local_var.mean(0) + local_mean.var(0)
    
    return


def update_statistics_2d_v5(local_mean:torch.Tensor, local_var:torch.Tensor, global_mean:torch.Tensor, global_var:torch.Tensor, momentum:float, data:torch.Tensor, label:torch.Tensor, gamma:float, training:bool=False):
    if local_mean.size(0) != local_var.size(0):
        raise RuntimeError("the sizes of local_mean and local_var are inequal.")
    
    if data.size(0) != label.size(0):
        raise RuntimeError("the values of the first dimension of data and label are inequal.")
    
    if training:
        # for example, label: [2, 2, 2, 3, 3, 4] (N = 6)
        l:torch.Tensor = label.unique()  # [2, 3, 4]
        l_mapping = torch.arange(l.size(0)).to(l.device)  # [0, 1, 2]
        l_reverse_mapping = l.new_full((local_mean.size(0), ), -1)
        l_reverse_mapping[l] = l_mapping  # [-1, -1, 0, 1, 2]
        label_local = l_reverse_mapping[label]  # (N,) [0, 0, 0, 1, 1, 2]
        lm = local_mean[label]  # (N, C)
            # [local_mean(0), local_mean(0), local_mean(0), local_mean(1), local_mean(1), local_mean(2)]
        
        label_num = torch.zeros_like(l_mapping)  # (K',) [0, 0, 0]
        label_num.scatter_add_(0, label_local, torch.ones_like(label_local))  # (K',) [3, 2, 1]
        mask = label_num > (1.0 / momentum)
        m = torch.where(mask, 1.0 / label_num.float(), momentum)

        delta_pre = data - lm[..., None, None]   # (N, C, H, W)

        delta_k = delta_pre.new_zeros((l.size(0), delta_pre.size(1)))  # (K', C)
        delta_k.scatter_add_(0, label_local.unsqueeze(-1).expand(-1, delta_k.size(1)), delta_pre.mean((2, 3)))  # (K', C)
        delta_k *= m.view(-1, 1)
        local_mean[l] = (1.0 - gamma) * delta_k + local_mean[l]
        local_mean[...] = gamma * delta_k.mean(0, keepdim=True) + local_mean

        delta_square_k = delta_pre.new_zeros((l.size(0), delta_pre.size(1)))
        delta_square_k.scatter_add_(0, label_local.unsqueeze(-1).expand(-1, delta_k.size(1)), delta_pre.pow(2).mean((2, 3)))
        local_var[l] = local_var[l] + (1.0 - gamma) * (m.view(-1, 1) * (delta_square_k - label_num.view(-1, 1) * local_var[l]) - delta_k.pow(2))

        var_gap = (m.view(-1, 1) * delta_square_k - delta_k.pow(2)).mean(0, keepdim=True)
        local_var[...] = local_var + gamma * (var_gap - (m * label_num).mean() * local_var)

        global_mean[...] = local_mean.mean(0)
        global_var[...] = local_var.mean(0) + local_mean.var(0)
    
    return



def update_statistics_2d_ema(local_mean:torch.Tensor, local_var:torch.Tensor, global_mean:torch.Tensor, global_var:torch.Tensor, momentum:float, data:torch.Tensor, label:torch.Tensor, gamma:float, training:bool=False):
    if local_mean.size(0) != local_var.size(0):
        raise RuntimeError("the sizes of local_mean and local_var are inequal.")
    
    if data.size(0) != label.size(0):
        raise RuntimeError("the values of the first dimension of data and label are inequal.")
    
    if training:
        # for example, label: [2, 2, 2, 3, 3, 4] (N = 6)
        l:torch.Tensor = label.unique()  # [2, 3, 4]
        l_mapping = torch.arange(l.size(0)).to(l.device)  # [0, 1, 2]
        l_reverse_mapping = l.new_full((local_mean.size(0), ), -1)
        l_reverse_mapping[l] = l_mapping  # [-1, -1, 0, 1, 2]
        label_local = l_reverse_mapping[label]  # (N,) [0, 0, 0, 1, 1, 2]
        lm = local_mean[label]  # (N, C)
            # [local_mean(0), local_mean(0), local_mean(0), local_mean(1), local_mean(1), local_mean(2)]
        
        label_num = torch.zeros_like(l_mapping)  # (K',) [0, 0, 0]
        label_num.scatter_add_(0, label_local, torch.ones_like(label_local))  # (K',) [3, 2, 1]

        m = 1 / (label_num.float() + 1e-5) * momentum  # this is the main difference between EMA and the above.

        delta_pre = data - lm[..., None, None]   # (N, C, H, W)

        delta_k = delta_pre.new_zeros((l.size(0), delta_pre.size(1)))  # (K', C)
        delta_k.scatter_add_(0, label_local.unsqueeze(-1).expand(-1, delta_k.size(1)), delta_pre.mean((2, 3)))  # (K', C)
        delta_k *= m.view(-1, 1)

        local_mean[l] = delta_k + local_mean[l]

        delta_square_k = delta_pre.new_zeros((l.size(0), delta_pre.size(1)))
        delta_square_k.scatter_add_(0, label_local.unsqueeze(-1).expand(-1, delta_k.size(1)), delta_pre.pow(2).mean((2, 3)))
        local_var[l] = local_var[l] + (m.view(-1, 1) * (delta_square_k - label_num.view(-1, 1) * local_var[l]) - delta_k.pow(2))

        global_mean[...] = local_mean.mean(0)
        global_var[...] = local_var.mean(0) + local_mean.var(0)
    
    return








