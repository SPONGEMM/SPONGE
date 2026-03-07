#pragma once

#include "../third_party/dlpack.h"

struct SpongeTensor
{
    using MTensor = DLManagedTensor;
    using Tensor = DLTensor;
    MTensor* mtensor;
    static void delete_dltensor(MTensor* mtensor)
    {
        Tensor* tensor = &mtensor->dl_tensor;
        free(tensor->shape);
        if (tensor->strides != NULL) free(tensor->strides);
    }
    SpongeTensor() { mtensor = NULL; }
    SpongeTensor(MTensor* mtensor) { this->mtensor = mtensor; }
    // strides虽然dlpack支持，但是很多引擎不支持，所以尽量给NULL
    // 使用例：
    // SpongeTensor tensor = SpongeTensor(crd, 2, {atom_numbers, 3}, NULL,
    // kDLFloat, false);
    SpongeTensor(void* data, int N_dim, int64_t* shape, int64_t* strides,
                 DLDataTypeCode type_code, bool on_host)
    {
        mtensor = (MTensor*)malloc(sizeof(MTensor));
        memset(mtensor, 0, sizeof(MTensor));
        mtensor->deleter = delete_dltensor;
        Tensor* tensor = &mtensor->dl_tensor;
        tensor->data = data;
        if (!on_host)
        {
#ifdef USE_HIP
            tensor->device = {kDLROCM, 0};
#elif defined(USE_CUDA)
            tensor->device = {kDLCUDA, 0};
#else
            tensor->device = {kDLCPU, 0};
#endif
        }
        else
        {
            tensor->device = {kDLCPU, 0};
        }
        tensor->ndim = N_dim;
        tensor->dtype = {(uint8_t)type_code, 32, 1};
        tensor->shape = (int64_t*)malloc(sizeof(int64_t) * N_dim);
        memcpy(tensor->shape, shape, sizeof(int64_t) * N_dim);
        if (strides != NULL)
        {
            tensor->strides = (int64_t*)malloc(sizeof(int64_t) * N_dim);
            memcpy(tensor->strides, strides, sizeof(int64_t) * N_dim);
        }
    }
    ~SpongeTensor()
    {
        if (mtensor != NULL)
        {
            mtensor->deleter(mtensor);
            free(mtensor);
        }
    }
    void* data() { return mtensor->dl_tensor.data; }
    int64_t shape(int i) { return mtensor->dl_tensor.shape[i]; }
};
