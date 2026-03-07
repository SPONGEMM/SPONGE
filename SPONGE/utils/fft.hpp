/*
CUFFT for cuda backend - license: Nvidia SDK License
MKL for x86 backend - license: Intel Simplified Software License
KML for kunpeng backend - license: 鲲鹏应用使能套件BoostKit用户许可协议 2.0
ARMPL for general arm backend - license: Arm Simplified End User License
Agreement HCFFT for hpcc backend -  cooperated with 沐曦MetaX
*/

#pragma once
#include <string.h>

#ifdef USE_HIP
#include "../third_party/device_backend/hip_api.h"
#elif defined(USE_CUDA)
#include "../third_party/device_backend/cuda_api.h"
#else
#include "../third_party/device_backend/cpu_api.h"
#endif

__device__ __host__ __forceinline__ FFT_COMPLEX expc(FFT_COMPLEX z);
__device__ __host__ __forceinline__ FFT_COMPLEX divc(FFT_COMPLEX a,
                                                     FFT_COMPLEX b);

struct SPONGE_FFT_WRAPPER
{
    friend __device__ __host__ __forceinline__ FFT_COMPLEX expc(FFT_COMPLEX z)
    {
        FFT_COMPLEX res;
        float t = expf(REAL(z));
#ifdef _WIN32
        IMAGINARY(res) = sinf(IMAGINARY(z));
        REAL(res) = cosf(IMAGINARY(z));
#else
        sincosf(IMAGINARY(z), &IMAGINARY(res), &REAL(res));
#endif
        REAL(res) *= t;
        IMAGINARY(res) *= t;
        return res;
    }

    friend __device__ __host__ __forceinline__ FFT_COMPLEX divc(FFT_COMPLEX a,
                                                                FFT_COMPLEX b)
    {
        FFT_COMPLEX result;
        float denom = REAL(b) * REAL(b) + IMAGINARY(b) * IMAGINARY(b);
        REAL(result) =
            (REAL(a) * REAL(b) + IMAGINARY(a) * IMAGINARY(b)) / denom;
        IMAGINARY(result) =
            (IMAGINARY(a) * REAL(b) - REAL(a) * IMAGINARY(b)) / denom;
        return result;
    }

    static FFT_RESULT Make_FFT_Plan(FFT_HANDLE* handle, int batch,
                                    int dimension, FFT_SIZE_t* length,
                                    FFT_TYPE type)
    {
#ifdef USE_GPU
        return deviceFFTPlanMany(handle, dimension, length, NULL, 0, 0, NULL, 0,
                                 0, type, batch);
#else
        int* c_length = (int*)malloc(sizeof(int) * dimension);
        memcpy(c_length, length, sizeof(int) * dimension);
        c_length[dimension - 1] = c_length[dimension - 1] / 2 + 1;
        int r_dim = 1, c_dim = 1;
        for (int i = 0; i < dimension; i++)
        {
            r_dim *= length[i];
            c_dim *= c_length[i];
        }
        float* tmp_in = (float*)fftwf_malloc(sizeof(float) * r_dim * batch);
        fftwf_complex* tmp_out =
            (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * c_dim * batch);

        if (type == FFT_R2C)
        {
            handle[0] = fftwf_plan_many_dft_r2c(
                dimension, length, batch, tmp_in, length, 1, r_dim, tmp_out,
                c_length, 1, c_dim, FFTW_ESTIMATE);
        }
        else
        {
            handle[0] = fftwf_plan_many_dft_c2r(
                dimension, length, batch, tmp_out, c_length, 1, c_dim, tmp_in,
                length, 1, r_dim, FFTW_ESTIMATE);
        }
        fftwf_free(tmp_in);
        fftwf_free(tmp_out);
        free(c_length);
        return handle[0] == NULL;
#endif
    }

    static void R2C(FFT_HANDLE handle, float* input, FFT_COMPLEX* output)
    {
#ifdef USE_GPU
        deviceFFTExecR2C(handle, input, output);
#else
        fftwf_execute_dft_r2c(handle, input, (fftwf_complex*)output);
#endif
    }

    static void C2R(FFT_HANDLE handle, FFT_COMPLEX* input, float* output)
    {
#ifdef USE_GPU
        deviceFFTExecC2R(handle, input, output);
#else
        fftwf_execute_dft_c2r(handle, (fftwf_complex*)input, output);
#endif
    }

    static void Destroy_FFT_Plan(FFT_HANDLE* handle)
    {
#ifdef USE_GPU
        deviceFFTDestroy(handle[0]);
#else
        fftwf_destroy_plan(handle[0]);
#endif
    }
};
