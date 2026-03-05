value += R"JIT(
#define foreach_2nd_create(EQUATION, START) \
        for (int i = 0; i < N2l; i++) \
        { \
            int index = i * N; \
            for (int j = START; j < N; j++) \
            { \
                EQUATION; \
            } \
        } \
        for (int i = N2l; i < N; i++) \
        { \
            int index = (2 * N - i + N2l) * (i - N2l) / 2 + N2l * N; \
            for (int j = 0; j < N2l && index + j < N2; j++) \
            { \
                EQUATION; \
            } \
            index -= i - N2l; \
            for (int j = i; j < N && index + j < N2; j++) \
            { \
                EQUATION; \
            } \
        }

/*使用方法：
1. 确定该部分需要求偏微分的数量N
2. 确定该部分需要求二阶偏微分的最大序数N2，第i个变量和第j个变量(i <= j)的二阶偏微分序数为2 * N - i - 1) * i / 2 + j
3. 将包含微分的变量和过程使用变量SA2Dfloat<N,N2>，其中对于想求的变量初始化时需要两个参数：本身的值和第i个变量，SA2Dfloat<1,1> x(1.0f, 0);
4. 正常计算，那么最后结果中的dval[i]即为第i个变量的微分，ddval[(2 * N - i - 1) * i / 2 + j]即为第i个变量和第j个变量(i <= j)的二阶偏微分。
样例：
已知f(x, y) = x ^ y + x * y， x=1.2, y = 2, 求df/dx和ddf/dx/dy
可以直接将下面的代码和common.h复制到一个新文件test.cu，再使用nvcc test.cu -o SAD_TEST编译，最后使用./SAD_TEST运行

#include "common.h"
#include "stdio.h"

int main()
{
    SA2Dfloat<2, 2> x(1.2f, 0);
    SA2Dfloat<2, 2> y(2.0f, 1);
    SA2Dfloat<2, 2> f = powf(x, y) + x * y;
    printf("df/dx = %f, ddf/dx/dy = %f\n", f.dval[0], f.ddval[1]);
}
*/
template<int N, int N2, int N2l = 0>
struct SA2Dfloat
{
    SADfloat<N> sad;
    float& val = sad.val;
    float* dval = sad.dval;
    float ddval[N2] = { 0.0f };
    __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l>()
    {
        sad = SADfloat<N>();
    }
    __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l>(float f, int id = -1)
    {
        sad = SADfloat<N>(f, id);
    }
    __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l>(const SA2Dfloat<N, N2, N2l>& f)
    {
        this->sad = f.sad;
        for (int i = 0; i < N2; i++)
        {
            this->ddval[i] = f.ddval[i];
        }
    }
    __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> operator-()
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = -this->sad;
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = -this->ddval[i];
        }
        return f;
    }
    __device__ __host__ __forceinline__ void operator=(const SA2Dfloat<N, N2, N2l>& f1)
    {
        this->sad = f1.sad;
        for (int i = 0; i < N2; i++)
        {
            this->ddval[i] = f1.ddval[i];
        }
    }
    __device__ __host__ __forceinline__ void operator+=(const SA2Dfloat<N, N2, N2l>& f1)
    {
        this->sad += f1.sad;
        for (int i = 0; i < N2; i++)
        {
            this->ddval[i] += f1.ddval[i];
        }
    }
    __device__ __host__ __forceinline__ void operator-=(const SA2Dfloat<N, N2, N2l>& f1)
    {
        this->sad -= f1.sad;
        for (int i = 0; i < N2; i++)
        {
            this->ddval[i] -= f1.ddval[i];
        }
    }
    __device__ __host__ __forceinline__ void operator*=(const SA2Dfloat<N, N2, N2l>& f1)
    {
        for (int i = 0; i < N2; i++)
        {
            this->ddval[i] = this->val * f1.ddval[i] + f1.val * this->ddval[i];
        }
        foreach_2nd_create(this->ddval[index + j] += this->dval[i] * f1.dval[j] + this->dval[j] * f1.dval[i], N2l)
            this->sad *= f1.sad;
    }
    __device__ __host__ __forceinline__ void operator/=(const SA2Dfloat<N, N2, N2l>& f1)
    {
        for (int i = 0; i < N2; i++)
        {
            this->ddval[i] = (this->ddval[i] * f1.val - this->val * f1.ddval[i]) / f1.val / f1.val;
        }
        foreach_2nd_create(this->ddval[index + j] += (2.0f * f1.dval[j] * f1.dval[i] * this->val / f1.val - (this->dval[i] * f1.dval[j] + this->dval[j] * f1.dval[i])) / f1.val / f1.val, N2l)
            this->sad /= f1.sad;
    }
    friend __device__ __host__ __forceinline__ bool operator==(const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1.val == f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1.val != f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>(const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1.val > f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<(const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1.val < f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1.val >= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1.val <= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator==(const SA2Dfloat<N, N2, N2l>& f1, const float& f2)
    {
        return f1.val == f2;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(const SA2Dfloat<N, N2, N2l>& f1, const float& f2)
    {
        return f1.val != f2;
    }
    friend __device__ __host__ __forceinline__ bool operator>(const SA2Dfloat<N, N2, N2l>& f1, const float& f2)
    {
        return f1.val > f2;
    }
    friend __device__ __host__ __forceinline__ bool operator<(const SA2Dfloat<N, N2, N2l>& f1, const float& f2)
    {
        return f1.val < f2;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(const SA2Dfloat<N, N2, N2l>& f1, const float& f2)
    {
        return f1.val >= f2;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(const SA2Dfloat<N, N2, N2l>& f1, const float& f2)
    {
        return f1.val <= f2;
    }
    friend __device__ __host__ __forceinline__ bool operator==(const float& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1 == f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(const float& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1 != f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>(const float& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1 > f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<(const float& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1 < f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(const float& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1 >= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(const float& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1 <= f2.val;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> operator+ (const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = f1.sad + f2.sad;
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = f1.ddval[i] + f2.ddval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> operator- (const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = f1.sad - f2.sad;
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = f1.ddval[i] - f2.ddval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> operator* (const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = f1.sad * f2.sad;
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = f1.val * f2.ddval[i] + f2.val * f1.ddval[i];
        }
        foreach_2nd_create(f.ddval[index + j] += f1.sad.dval[i] * f2.sad.dval[j] + f1.sad.dval[j] * f2.sad.dval[i], N2l)
            return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> operator/ (const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = f1.sad / f2.sad;
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = (f1.ddval[i] * f2.val - f1.val * f2.ddval[i]) / f2.val / f2.val;
        }
        foreach_2nd_create(f.ddval[index + j] += (2.0f * f2.dval[j] * f2.dval[i] * f1.val / f2.val - (f1.dval[i] * f2.dval[j] + f1.dval[j] * f2.dval[i])) / f2.val / f2.val, N2l)
            return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> powf(const SA2Dfloat<N, N2, N2l>& x, const SA2Dfloat<N, N2, N2l>& y)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = powf(x.sad, y.sad);
        float df_dx = y.val * powf(x.val, y.val - 1.0f);
        float df_dy = f.val * logf(x.val);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i] + df_dy * y.ddval[i];
        }
        float ddf_dxdx = y.val * (y.val - 1.0f) * powf(x.val, y.val - 2.0f);
        float ddf_dxdy = powf(x.val, y.val - 1.0f) * (1.0f + y.val * logf(x.val));
        float ddf_dydy = df_dy * logf(x.val);
        foreach_2nd_create(f.ddval[index + j] += ddf_dxdx * x.dval[i] * x.dval[j] + ddf_dxdy * (x.dval[i] * y.dval[j] + x.dval[j] * y.dval[i]) + ddf_dydy * y.dval[i] * y.dval[j], N2l)
            return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> expf(const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = expf(x.sad);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = f.val * x.ddval[i];
        }
        foreach_2nd_create(f.ddval[index + j] += f.val * x.dval[i] * x.dval[j], N2l)
            return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> erfcf(const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = erfcf(x.sad);
        float df_dx = -2.0f / sqrtf(CONSTANT_Pi) * expf(-x.val * x.val);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        df_dx *= -2.0f * x.val;
        foreach_2nd_create(f.ddval[index + j] += df_dx * x.dval[i] * x.dval[j], N2l)
            return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> logf(const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = logf(x.sad);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = x.ddval[i] / x.val;
        }
        foreach_2nd_create(f.ddval[index + j] -= x.dval[i] * x.dval[j] / x.val / x.val, N2l)
            return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> sqrtf(const SA2Dfloat<N, N2, N2l>& x)
    {
        return powf(x, 0.5f);
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> norm3df(
        const SA2Dfloat<N, N2, N2l>& x, const SA2Dfloat<N, N2, N2l>& y,
        const SA2Dfloat<N, N2, N2l>& z)
    {
        return sqrtf(x * x + y * y + z * z);
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> cbrtf(const SA2Dfloat<N, N2, N2l>& x)
    {
        return powf(x, 0.33333333f);
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> cosf(const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = cosf(x.sad);
        float df_dx = -sinf(x.val);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        foreach_2nd_create(f.ddval[index + j] -= f.val * x.dval[i] * x.dval[j], N2l)
            return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> sinf(const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = sinf(x.sad);
        float df_dx = cosf(x.val);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        foreach_2nd_create(f.ddval[index + j] -= f.val * x.dval[i] * x.dval[j], N2l)
            return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> tanf(const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = tanf(x.sad);
        float df_dx = 2.0f / (1.0f + cosf(2.0f * x.val));
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        df_dx *= 2.0f * f.val;
        foreach_2nd_create(f.ddval[index + j] += df_dx * x.dval[i] * x.dval[j], N2l)
            return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> acosf(const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = acosf(x.sad);
        float df_dx = -1.0f / sqrtf(1.0f - x.val * x.val);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        df_dx *= x.val / (1.0f - x.val * x.val);
        foreach_2nd_create(f.ddval[index + j] += df_dx * x.dval[i] * x.dval[j], N2l)
            return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> asinf(const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = asinf(x.sad);
        float df_dx = 1.0f / sqrtf(1.0f - x.val * x.val);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        df_dx *= x.val / (1.0f - x.val * x.val);
        foreach_2nd_create(f.ddval[index + j] += df_dx * x.dval[i] * x.dval[j], N2l)
            return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> atanf(const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = atanf(x.sad);
        float df_dx = 1.0f / (1.0f + x.val * x.val);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        df_dx *= -2.0f * x.val / (1.0f + x.val * x.val);
        foreach_2nd_create(f.ddval[index + j] += df_dx * x.dval[i] * x.dval[j], N2l)
            return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> fabsf(const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = fabsf(x.sad);
        float df_dx = copysignf(1.0f, x.val);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> fmaxf(const SA2Dfloat<N, N2, N2l>& x, const SA2Dfloat<N, N2, N2l>& y)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = fmaxf(x.sad, y.sad);
        float df_dx = fmaxf(copysignf(1.0f, x.val - y.val), 0.0f);
        float df_dy = fmaxf(copysignf(1.0f, y.val - x.val), 0.0f);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i] + df_dy * y.ddval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> fminf(const SA2Dfloat<N, N2, N2l>& x, const SA2Dfloat<N, N2, N2l>& y)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = fminf(x.sad, y.sad);
        float df_dx = fmaxf(copysignf(1.0f, y.val - x.val), 0.0f);
        float df_dy = fmaxf(copysignf(1.0f, x.val - y.val), 0.0f);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i] + df_dy * y.ddval[i];
        }
        return f;
    }
    __device__ __host__ __forceinline__ int get_ddval_index(int i, int j)
    {
        if (i < N2l)
            return i * N + j;
        if (j < N2l)
            return (2 * N - i + N2l) * (i - N2l) / 2 + N2l * N + j;
        if (i > j)
        {
            int temp = i;
            i = j;
            j = temp;
        }
        return (2 * N - i + N2l) * (i - N2l) / 2 + N2l * N + j - i + N2l;
    }
};

)JIT";
