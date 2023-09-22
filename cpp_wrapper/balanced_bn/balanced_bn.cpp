#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include "utils.h"
#include <thrust/for_each.h>
#include <thrust/device_vector.h>


namespace py = pybind11;

at::Tensor create()
{
    return torch::rand({2, 3});
}


void update_statistics_1d_v5(at::Tensor local_mean, at::Tensor local_var, at::Tensor global_mean, at::Tensor global_var, float momentum, at::Tensor data, at::Tensor label, float gamma, bool training=false){
    if (local_mean.size(0) != local_var.size(0)){
        throw std::runtime_error("the sizes of local_mean and local_var are inequal.");
    }

    if (data.size(0) != label.size(0)){
        throw std::runtime_error("the values of the first dimension of data and label are inequal.");
    }

    if (training){

        auto ret = torch::unique_dim(label, 0);
        // label: [2, 2, 2, 3, 3, 4] (N = 6)
        auto l = std::get<0>(ret);  // [2, 3, 4]
        auto l_mapping = torch::arange(l.size(0)).to(l.device());  // [0, 1, 2]
        auto l_reverse_mapping = torch::full({local_mean.size(0),}, -1).to(l.device());
        l_reverse_mapping.index_put_({l}, l_mapping); // [-1, -1, 0, 1, 2]
        auto label_ = l_reverse_mapping.index({label});  // (N,)  [0, 0, 0, 1, 1, 2]  
        auto lm = local_mean.index({label});  // (N, C) 
            // [local_mean(0), local_mean(0), local_mean(0), local_mean(1), local_mean(1), local_mean(2)]

        auto label_num = torch::zeros_like(l_mapping);  // (K',)  [0, 0, 0]
        label_num.scatter_add_(0, label_, torch::ones_like(label_));  // (K',)  [3, 2, 1]
        auto mask = label_num > 1.0 / momentum;
        auto m = torch::where(mask, 1.0 / label_num.to(torch::kFloat32), momentum);  // (K',)  [0.005, 0.005, 0.005]

        auto delta_pre = data - lm.index({"...", torch::indexing::None});  // (N, C, L)

        auto delta_k = torch::zeros({l.size(0), delta_pre.size(1)}).to(l.device());  // (K', C)
        delta_k.scatter_add_(0, label_.unsqueeze(-1).expand({label_.size(0), delta_k.size(1)}), delta_pre.mean({2}));  // (K', C)
        delta_k *= m.view({-1, 1});
        local_mean.index_put_({l}, (1.0 - gamma) * delta_k + local_mean.index({l}));
        local_mean.index_put_({"..."}, local_mean.index({"..."}) + gamma * delta_k.mean({0}, true));

        auto delta_square_k = torch::zeros({l.size(0), delta_pre.size(1)}).to(l.device());
        delta_square_k.scatter_add_(0, label_.unsqueeze(-1).expand({label_.size(0), delta_k.size(1)}), delta_pre.pow(2).mean({2}));
        local_var.index_put_({l}, local_var.index({l}) + (1.0 - gamma) * (m.view({-1, 1}) * (delta_square_k - label_num.view({-1, 1}) * local_var.index({l})) - delta_k.pow(2)));

        auto var_gap = (m.view({-1, 1}) * delta_square_k - delta_k.pow(2)).mean(0, true);
        local_var.index_put_({"..."}, local_var.index({"..."}) + gamma * (var_gap - (m * label_num).mean() * local_var.index({"..."})));
        
        global_mean.index_put_({"..."}, local_mean.mean({0}));
        global_var.index_put_({"..."}, local_var.mean({0}) + local_mean.var({0}));
    }
    
    return;
}


void update_statistics_2d_v5(at::Tensor local_mean, at::Tensor local_var, at::Tensor global_mean, at::Tensor global_var, float momentum, at::Tensor data, at::Tensor label, float gamma, bool training=false){
    if (local_mean.size(0) != local_var.size(0)){
        throw std::runtime_error("the sizes of local_mean and local_var are inequal.");
    }

    if (data.size(0) != label.size(0)){
        throw std::runtime_error("the values of the first dimension of data and label are inequal.");
    }

    if (training){

        auto ret = torch::unique_dim(label, 0);
        // label: [2, 2, 2, 3, 3, 4] (N = 6)
        auto l = std::get<0>(ret);  // [2, 3, 4]
        auto l_mapping = torch::arange(l.size(0)).to(l.device());  // [0, 1, 2]
        auto l_reverse_mapping = torch::full({local_mean.size(0),}, -1).to(l.device());
        l_reverse_mapping.index_put_({l}, l_mapping); // [-1, -1, 0, 1, 2]
        auto label_ = l_reverse_mapping.index({label});  // (N,)  [0, 0, 0, 1, 1, 2]  
        auto lm = local_mean.index({label});  // (N, C) 
            // [local_mean(0), local_mean(0), local_mean(0), local_mean(1), local_mean(1), local_mean(2)]

        auto label_num = torch::zeros_like(l_mapping);  // (K',)  [0, 0, 0]
        label_num.scatter_add_(0, label_, torch::ones_like(label_));  // (K',)  [3, 2, 1]
        auto mask = label_num > 1.0 / momentum;
        auto m = torch::where(mask, 1.0 / label_num.to(torch::kFloat32), momentum);  // (K',)  [0.005, 0.005, 0.005]

        auto delta_pre = data - lm.index({"...", torch::indexing::None, torch::indexing::None});  // (N, C, H, W)

        auto delta_k = torch::zeros({l.size(0), delta_pre.size(1)}).to(l.device());  // (K', C)
        delta_k.scatter_add_(0, label_.unsqueeze(-1).expand({label_.size(0), delta_k.size(1)}), delta_pre.mean({2, 3}));  // (K', C)
        delta_k *= m.view({-1, 1});
        local_mean.index_put_({l}, (1.0 - gamma) * delta_k + local_mean.index({l}));
        local_mean.index_put_({"..."}, local_mean.index({"..."}) + gamma * delta_k.mean({0}, true));

        auto delta_square_k = torch::zeros({l.size(0), delta_pre.size(1)}).to(l.device());
        delta_square_k.scatter_add_(0, label_.unsqueeze(-1).expand({label_.size(0), delta_k.size(1)}), delta_pre.pow(2).mean({2, 3}));
        local_var.index_put_({l}, local_var.index({l}) + (1.0 - gamma) * (m.view({-1, 1}) * (delta_square_k - label_num.view({-1, 1}) * local_var.index({l})) - delta_k.pow(2)));

        auto var_gap = (m.view({-1, 1}) * delta_square_k - delta_k.pow(2)).mean(0, true);
        local_var.index_put_({"..."}, local_var.index({"..."}) + gamma * (var_gap - (m * label_num).mean() * local_var.index({"..."})));
        
        global_mean.index_put_({"..."}, local_mean.mean({0}));
        global_var.index_put_({"..."}, local_var.mean({0}) + local_mean.var({0}));
    }
    
    return;
}


void update_statistics_2d_ema(at::Tensor local_mean, at::Tensor local_var, at::Tensor global_mean, at::Tensor global_var, float momentum, at::Tensor data, at::Tensor label, float gamma, bool training=false){
    if (local_mean.size(0) != local_var.size(0)){
        throw std::runtime_error("the sizes of local_mean and local_var are inequal.");
    }

    if (data.size(0) != label.size(0)){
        throw std::runtime_error("the values of the first dimension of data and label are inequal.");
    }

    if (training){

        auto ret = torch::unique_dim(label, 0);
        // label: [2, 2, 2, 3, 3, 4] (N = 6)
        auto l = std::get<0>(ret);  // [2, 3, 4]
        auto l_mapping = torch::arange(l.size(0)).to(l.device());  // [0, 1, 2]
        auto l_reverse_mapping = torch::full({local_mean.size(0),}, -1).to(l.device());
        l_reverse_mapping.index_put_({l}, l_mapping); // [-1, -1, 0, 1, 2]
        auto label_ = l_reverse_mapping.index({label});  // (N,)  [0, 0, 0, 1, 1, 2]  
        auto lm = local_mean.index({label});  // (N, C) 
            // [local_mean(0), local_mean(0), local_mean(0), local_mean(1), local_mean(1), local_mean(2)]

        auto label_num = torch::zeros_like(l_mapping);  // (K',)  [0, 0, 0]
        label_num.scatter_add_(0, label_, torch::ones_like(label_));  // (K',)  [3, 2, 1]

        auto m = 1.0 / (label_num.to(torch::kFloat32) + 1e-5) * momentum;

        auto delta_pre = data - lm.index({"...", torch::indexing::None, torch::indexing::None});  // (N, C, H, W)

        auto delta_k = torch::zeros({l.size(0), delta_pre.size(1)}).to(l.device());  // (K', C)
        delta_k.scatter_add_(0, label_.unsqueeze(-1).expand({label_.size(0), delta_k.size(1)}), delta_pre.mean({2, 3}));  // (K', C)
        delta_k *= m.view({-1, 1});

        local_mean.index_put_({l}, delta_k + local_mean.index({l}));

        auto delta_square_k = torch::zeros({l.size(0), delta_pre.size(1)}).to(l.device());
        delta_square_k.scatter_add_(0, label_.unsqueeze(-1).expand({label_.size(0), delta_k.size(1)}), delta_pre.pow(2).mean({2, 3}));
        local_var.index_put_({l}, local_var.index({l}) + (m.view({-1, 1}) * (delta_square_k - label_num.view({-1, 1}) * local_var.index({l})) - delta_k.pow(2)));

        global_mean.index_put_({"..."}, local_mean.mean({0}));
        global_var.index_put_({"..."}, local_var.mean({0}) + local_mean.var({0}));
    }
    
    return;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "libtorch tensor test";
    m.def("create", &create, "create tensor");
    
    m.def("update_statistics_1d_v5", &update_statistics_1d_v5, "update statistics for 1d data v5.");
    m.def("update_statistics_2d_v5", &update_statistics_2d_v5, "update statistics for 2d data v5.");
    m.def("update_statistics_2d_ema", &update_statistics_2d_ema, "update statistics for 2d data ema.");

        
}