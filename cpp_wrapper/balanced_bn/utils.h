#ifndef _UTILS
#define _UTILS

#include <ATen/cuda/CUDAContext.h>
#include <torch/serialize/tensor.h>
#include <c10/core/ScalarType.h>


void _print_array(at::Tensor A, int show_num, const char* format="%.4g", int offset=1);
void print_array(at::Tensor A, int show_num=2);
std::string get_device(at::Tensor A);


#endif