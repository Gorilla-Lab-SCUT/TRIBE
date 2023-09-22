#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "utils.h"
#include <string.h>
#include <c10/core/ScalarType.h>

// void _print_array(at::Tensor A, int show_num, const char* format="%.4g", int offset=1);
// void print_array(at::Tensor A, int show_num=2);
// std::string get_device(at::Tensor A);

void print_array(at::Tensor A, int show_num){
    _print_array(A, show_num);
    printf("\n");
    std::cout << "(" 
              << "Shape: " << A.sizes()
              << "; "
              << "Dtype: " << A.scalar_type()
              << "; "
              << "Device: " << get_device(A)
              << "; Type: at::Tensor)" << std::endl;
}

void _print_array(at::Tensor A, int show_num, const char* format, int offset){
    const unsigned int ndim = A.dim();
    std::string a(offset, ' ');
    int curr_show_num;
    if (show_num == -1){
        curr_show_num = A.size(0);
    }else{
        curr_show_num = show_num;
    }
    printf("[");
    if (ndim == 1){
        printf(" ");
        for (int i=0; i<A.size(0); i++){
            if (i < curr_show_num || i >= A.size(0) - curr_show_num){
                auto type = toString(A.scalar_type());
                if (type == "Float" || type == "Double" || type == "BFloat16"){
                    printf("%.4g", A[i].item<double>());
                }else if (type == "Int" || type == "Short" || type == "Byte" || type == "Char" || type == "Long" || type == "Half"){
                    printf("%d", A[i].item<int>());
                }else if (type == "ComplexFloat" || type == "ComplexDouble"){
                    printf("-");
                }else if (type == "Bool"){
                    if (A[i].item<bool>()){
                        printf("True");
                    }else{
                        printf("False");
                    }
                }
                printf(" ");
            }else if (i == curr_show_num && A.size(0) > curr_show_num * 2){
                printf("... ");
            }
        }
    }else{
        for (int i=0; i<A.size(0); i++){
            if (i < curr_show_num || i >= A.size(0) - curr_show_num){
                _print_array(A[i], show_num, format, offset + 1);
                if (i != A.size(0) - 1){
                    printf(",\n%s", a.c_str());
                }
            }else if (i == curr_show_num && A.size(0) > curr_show_num * 2){
                printf("...\n%s", a.c_str());
            }
        }
    }
    printf("]");
}




std::string get_device(at::Tensor A){
    std::stringstream info;
    if (A.is_cuda()){
        info << "CUDA";
    }else{
        info << "CPU";
    }
    if (A.device().has_index()){
        info << ":" << (int)A.device().index();
    }
    return info.str();
}