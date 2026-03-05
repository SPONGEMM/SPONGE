value += R"JIT(
/*XYJ备注：SAD=simple/sponge auto diff，简单/SPONGE自动微分
SA2D: 2阶微分
实现原理：利用操作符重载，将f(x,y)的关系同时用链式法则链接到df(x,y)和ddf(x,y)上。效率会有影响，但影响较小，因为主要成本在通讯上，在每个线程的内部利用cache计算不是决速步
使用方法：1. 确定该部分需要求偏微分的数量，假设有1个，则后面使用的类就为SADfloat<1>或SADvector<1>，2个则为SADfloat<2>或SADvector<2>
2. 将包含微分的变量和过程用上面确定的类声明变量，其中对于想求的变量初始化时需要两个参数：本身的值和第i个变量
3. 正常计算，那么最后结果中的dval[i]即为第i个变量的微分。
使用样例：（均在No_PNC/generalized_Born.cu中）
1. 求有效伯恩半径对距离的导数：不求导数的函数为Effective_Born_Radii_Factor_CUDA，求导数的函数为GB_accumulate_Force_Energy_CUDA
2. 求GB能量对距离和有效伯恩半径的导数：不求导数的函数为GB_inej_Energy_CUDA，求导数的函数为GB_inej_Force_Energy_CUDA
*/
template<int N>
struct SADfloat
{
    float val;
    float dval[N];
    __device__ __host__ __forceinline__ SADfloat<N>()
    {
        this->val = 0.0f;
    }
    __device__ __host__ __forceinline__ SADfloat<N>(float f, int id = -1)
    {
        this->val = f;
        for (int i = 0; i < N; i++)
        {
            if (i != id)
                this->dval[i] = 0.0f;
            else
                this->dval[i] = 1.0f;
        }
    }
    __device__ __host__ __forceinline__ SADfloat<N>(const SADfloat<N>& f)
    {
        this->val = f.val;
        for (int i = 0; i < N; i++)
        {
            this->dval[i] = f.dval[i];
        }
    }
    __device__ __host__ __forceinline__ SADfloat<N> operator-()
    {
        SADfloat<N> f;
        f.val = -this->val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = -this->dval[i];
        }
        return f;
    }
    __device__ __host__ __forceinline__ void operator=(const SADfloat<N>& f1)
    {
        val = f1.val;
        for (int i = 0; i < N; i++)
        {
            dval[i] = f1.dval[i];
        }
    }
    __device__ __host__ __forceinline__ void operator+=(const SADfloat<N>& f1)
    {
        val += f1.val;
        for (int i = 0; i < N; i++)
        {
            dval[i] += f1.dval[i];
        }
    }
    __device__ __host__ __forceinline__ void operator-=(const SADfloat<N>& f1)
    {
        val -= f1.val;
        for (int i = 0; i < N; i++)
        {
            dval[i] -= f1.dval[i];
        }
    }
    __device__ __host__ __forceinline__ void operator*=(const SADfloat<N>& f1)
    {
        for (int i = 0; i < N; i++)
        {
            this->dval[i] = this->dval[i] * f1.val + this->val * f1.dval[i];
        }
        this->val *= f1.val;
    }
    __device__ __host__ __forceinline__ void operator/=(const SADfloat<N>& f1)
    {
        for (int i = 0; i < N; i++)
        {
            this->dval[i] = dval[i] * f1.val - f1.dval[i] * this->val;
            this->dval[i] /= f1.val * f1.val;
        }
        this->val /= f1.val;
    }
    friend __device__ __host__ __forceinline__ bool operator==(const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val == f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val != f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>(const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val > f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<(const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val < f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val >= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val <= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator==(const SADfloat<N>& f1, const float& f2)
    {
        return f1.val == f2;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(const SADfloat<N>& f1, const float& f2)
    {
        return f1.val != f2;
    }
    friend __device__ __host__ __forceinline__ bool operator>(const SADfloat<N>& f1, const float& f2)
    {
        return f1.val > f2;
    }
    friend __device__ __host__ __forceinline__ bool operator<(const SADfloat<N>& f1, const float& f2)
    {
        return f1.val < f2;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(const SADfloat<N>& f1, const float& f2)
    {
        return f1.val >= f2;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(const SADfloat<N>& f1, const float& f2)
    {
        return f1.val <= f2;
    }
    friend __device__ __host__ __forceinline__ bool operator==(const float& f1, const SADfloat<N>& f2)
    {
        return f1 == f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(const float& f1, const SADfloat<N>& f2)
    {
        return f1 != f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>(const float& f1, const SADfloat<N>& f2)
    {
        return f1 > f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<(const float& f1, const SADfloat<N>& f2)
    {
        return f1 < f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(const float& f1, const SADfloat<N>& f2)
    {
        return f1 >= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(const float& f1, const SADfloat<N>& f2)
    {
        return f1 <= f2.val;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator+ (const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        SADfloat<N> f;
        f.val = f1.val + f2.val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f1.dval[i] + f2.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator- (const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        SADfloat<N> f;
        f.val = f1.val - f2.val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f1.dval[i] - f2.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator* (const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        SADfloat<N> f;
        f.val = f1.val * f2.val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f2.val * f1.dval[i] + f1.val * f2.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator/ (const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        SADfloat<N> f;
        f.val = f1.val / f2.val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f1.dval[i] * f2.val - f2.dval[i] * f1.val;
            f.dval[i] /= f2.val * f2.val;
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> powf(const SADfloat<N>& x, const SADfloat<N>& y)
    {
        SADfloat<N> f;
        f.val = powf(x.val, y.val);
        float df_dx = y.val * powf(x.val, y.val - 1.0f);
        float df_dy = 0.0f;
        if (x.val > 0.0f)
        {
            df_dy = f.val * logf(x.val);
        }
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i] + df_dy * y.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> expf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = expf(x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f.val * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> erfcf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = erfcf(x.val);
        float df_dx = -2.0f / sqrtf(CONSTANT_Pi) * expf(-x.val * x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> logf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = logf(x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = x.dval[i] / x.val;
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> sqrtf(const SADfloat<N>& x)
    {
        return powf(x, 0.5f);
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> norm3df(
        const SADfloat<N>& x, const SADfloat<N>& y, const SADfloat<N>& z)
    {
        return sqrtf(x * x + y * y + z * z);
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> cbrtf(const SADfloat<N>& x)
    {
        return powf(x, 0.33333333f);
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> cosf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = cosf(x.val);
        float df_dx = -sinf(x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> sinf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = sinf(x.val);
        float df_dx = cosf(x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> tanf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = tanf(x.val);
        float df_dx = 2.0f / (1.0f + cosf(2.0f * x.val));
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> acosf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = acosf(x.val);
        float df_dx = -1.0f / sqrtf(1.0f - x.val * x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> asinf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = asinf(x.val);
        float df_dx = 1.0f / sqrtf(1.0f - x.val * x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> atanf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = atanf(x.val);
        float df_dx = 1.0f / (1.0f + x.val * x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> fabsf(const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = fabsf(x.val);
        float df_dx = copysignf(1.0f, x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> fmaxf(const SADfloat<N>& x, const SADfloat<N>& y)
    {
        SADfloat<N> f;
        f.val = fmaxf(x.val, y.val);
        float df_dx = fmaxf(copysignf(1.0f, x.val - y.val), 0.0f);
        float df_dy = fmaxf(copysignf(1.0f, y.val - x.val), 0.0f);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i] + df_dy * y.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> fminf(const SADfloat<N>& x, const SADfloat<N>& y)
    {
        SADfloat<N> f;
        f.val = fminf(x.val, y.val);
        float df_dx = fmaxf(copysignf(1.0f, y.val - x.val), 0.0f);
        float df_dy = fmaxf(copysignf(1.0f, x.val - y.val), 0.0f);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i] + df_dy * y.dval[i];
        }
        return f;
    }
};

)JIT";
