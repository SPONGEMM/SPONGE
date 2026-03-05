#ifndef SPONGE_COMMON_AUTO_DIFF_HPP
#define SPONGE_COMMON_AUTO_DIFF_HPP

/*XYJ备注：SAD=simple/sponge auto diff，简单/SPONGE自动微分
SA2D: 2阶微分
实现原理：利用操作符重载，将f(x,y)的关系同时用链式法则链接到df(x,y)和ddf(x,y)上。效率会有影响，但影响较小，因为主要成本在通讯上，在每个线程的内部利用cache计算不是决速步
使用方法：1.
确定该部分需要求偏微分的数量，假设有1个，则后面使用的类就为SADfloat<1>或SADvector<1>，2个则为SADfloat<2>或SADvector<2>
2.
将包含微分的变量和过程用上面确定的类声明变量，其中对于想求的变量初始化时需要两个参数：本身的值和第i个变量
3. 正常计算，那么最后结果中的dval[i]即为第i个变量的微分。
使用样例：（均在No_PNC/generalized_Born.cu中）
1.
求有效伯恩半径对距离的导数：不求导数的函数为Effective_Born_Radii_Factor_CUDA，求导数的函数为GB_accumulate_Force_Energy_CUDA
2.
求GB能量对距离和有效伯恩半径的导数：不求导数的函数为GB_inej_Energy_CUDA，求导数的函数为GB_inej_Force_Energy_CUDA
*/
template <int N>
struct SADfloat
{
    float val;
    float dval[N];
    __device__ __host__ __forceinline__ SADfloat<N>() { this->val = 0.0f; }
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
        float quotient = this->val / f1.val;
        for (int i = 0; i < N; i++)
        {
            this->dval[i] = (this->dval[i] - quotient * f1.dval[i]) / f1.val;
        }
        this->val = quotient;
    }
    friend __device__ __host__ __forceinline__ bool operator==(
        const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val == f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(
        const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val != f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>(
        const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val > f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<(
        const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val < f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(
        const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val >= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(
        const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        return f1.val <= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator==(
        const SADfloat<N>& f1, const float& f2)
    {
        return f1.val == f2;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(
        const SADfloat<N>& f1, const float& f2)
    {
        return f1.val != f2;
    }
    friend __device__ __host__ __forceinline__ bool operator>(
        const SADfloat<N>& f1, const float& f2)
    {
        return f1.val > f2;
    }
    friend __device__ __host__ __forceinline__ bool operator<(
        const SADfloat<N>& f1, const float& f2)
    {
        return f1.val < f2;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(
        const SADfloat<N>& f1, const float& f2)
    {
        return f1.val >= f2;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(
        const SADfloat<N>& f1, const float& f2)
    {
        return f1.val <= f2;
    }
    friend __device__ __host__ __forceinline__ bool operator==(
        const float& f1, const SADfloat<N>& f2)
    {
        return f1 == f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(
        const float& f1, const SADfloat<N>& f2)
    {
        return f1 != f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>(
        const float& f1, const SADfloat<N>& f2)
    {
        return f1 > f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<(
        const float& f1, const SADfloat<N>& f2)
    {
        return f1 < f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(
        const float& f1, const SADfloat<N>& f2)
    {
        return f1 >= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(
        const float& f1, const SADfloat<N>& f2)
    {
        return f1 <= f2.val;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator+(
        const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        SADfloat<N> f;
        f.val = f1.val + f2.val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f1.dval[i] + f2.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator-(
        const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        SADfloat<N> f;
        f.val = f1.val - f2.val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f1.dval[i] - f2.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator*(
        const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        SADfloat<N> f;
        f.val = f1.val * f2.val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f2.val * f1.dval[i] + f1.val * f2.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator*(
        const SADfloat<N>& f1, const float& f2)
    {
        SADfloat<N> f;
        f.val = f1.val * f2;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f1.dval[i] * f2;
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator*(
        const float& f1, const SADfloat<N>& f2)
    {
        return f2 * f1;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator-(
        const SADfloat<N>& f1, const float& f2)
    {
        SADfloat<N> f = f1;
        f.val -= f2;
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator+(
        const SADfloat<N>& f1, const float& f2)
    {
        SADfloat<N> f = f1;
        f.val += f2;
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator+(
        const float& f1, const SADfloat<N>& f2)
    {
        return f2 + f1;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator/(
        const SADfloat<N>& f1, const SADfloat<N>& f2)
    {
        SADfloat<N> f;
        f.val = f1.val / f2.val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = (f1.dval[i] - f.val * f2.dval[i]) / f2.val;
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator/(
        const float& f1, const SADfloat<N>& f2)
    {
        SADfloat<N> f;
        f.val = f1 / f2.val;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = -f.val * f2.dval[i] / f2.val;
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> powf(
        const SADfloat<N>& x, const SADfloat<N>& y)
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
    friend __device__ __host__ __forceinline__ SADfloat<N> expf(
        const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = expf(x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = f.val * x.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> erfcf(
        const SADfloat<N>& x)
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
    friend __device__ __host__ __forceinline__ SADfloat<N> logf(
        const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = logf(x.val);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = x.dval[i] / x.val;
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> Log_Sum_Exp(
        const SADfloat<N>& x, const SADfloat<N>& y)
    {
        SADfloat<N> f;
        float max_xy = fmaxf(x.val, y.val);
        float ex = expf(x.val - max_xy);
        float ey = expf(y.val - max_xy);
        float sum = ex + ey;
        float inv_sum = 1.0f / sum;
        float wx = ex * inv_sum;
        float wy = ey * inv_sum;
        f.val = max_xy + logf(sum);
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = wx * x.dval[i] + wy * y.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> sqrtf(
        const SADfloat<N>& x)
    {
        return powf(x, 0.5f);
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> norm3df(
        const SADfloat<N>& x, const SADfloat<N>& y, const SADfloat<N>& z)
    {
        return sqrtf(x * x + y * y + z * z);
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> cbrtf(
        const SADfloat<N>& x)
    {
        return powf(x, 0.33333333f);
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> cosf(
        const SADfloat<N>& x)
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
    friend __device__ __host__ __forceinline__ SADfloat<N> sinf(
        const SADfloat<N>& x)
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
    friend __device__ __host__ __forceinline__ SADfloat<N> tanf(
        const SADfloat<N>& x)
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
    friend __device__ __host__ __forceinline__ SADfloat<N> acosf(
        const SADfloat<N>& x)
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
    friend __device__ __host__ __forceinline__ SADfloat<N> asinf(
        const SADfloat<N>& x)
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
    friend __device__ __host__ __forceinline__ SADfloat<N> atanf(
        const SADfloat<N>& x)
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
    friend __device__ __host__ __forceinline__ SADfloat<N> atan2f(
        const SADfloat<N>& y, const SADfloat<N>& x)
    {
        SADfloat<N> f;
        f.val = atan2f(y.val, x.val);
        float r2 = x.val * x.val + y.val * y.val;
        float df_dy = x.val / r2;
        float df_dx = -y.val / r2;
        for (int i = 0; i < N; i++)
        {
            f.dval[i] = df_dx * x.dval[i] + df_dy * y.dval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> fabsf(
        const SADfloat<N>& x)
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
    friend __device__ __host__ __forceinline__ SADfloat<N> fmaxf(
        const SADfloat<N>& x, const SADfloat<N>& y)
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
    friend __device__ __host__ __forceinline__ SADfloat<N> fminf(
        const SADfloat<N>& x, const SADfloat<N>& y)
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

template <int N>
struct SADvector
{
    SADfloat<N> x, y, z;
    __device__ __host__ __forceinline__ SADvector<N>()
    {
        this->x = SADfloat<N>(0);
        this->y = SADfloat<N>(0);
        this->z = SADfloat<N>(0);
    }
    __device__ __host__ __forceinline__ SADvector<N>(VECTOR v, int idx = -1,
                                                     int idy = -1, int idz = -1)
    {
        this->x = SADfloat<N>(v.x, idx);
        this->y = SADfloat<N>(v.y, idy);
        this->z = SADfloat<N>(v.z, idz);
    }
    friend __device__ __host__ __forceinline__ SADvector<N> operator+(
        const SADvector<N>& veca, const SADvector<N>& vecb)
    {
        SADvector<N> vec;
        vec.x = veca.x + vecb.x;
        vec.y = veca.y + vecb.y;
        vec.z = veca.z + vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ SADvector<N> operator+(
        const SADvector<N>& veca, const SADfloat<N>& b)
    {
        SADvector<N> vec;
        vec.x = veca.x + b;
        vec.y = veca.y + b;
        vec.z = veca.z + b;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SADvector<N> operator*(
        const SADfloat<N>& a, const SADvector<N>& vecb)
    {
        SADvector<N> vec;
        vec.x = a * vecb.x;
        vec.y = a * vecb.y;
        vec.z = a * vecb.z;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SADfloat<N> operator*(
        const SADvector<N>& veca, const SADvector<N>& vecb)
    {
        return veca.x * vecb.x + veca.y * vecb.y + veca.z * vecb.z;
    }
    friend __device__ __host__ __forceinline__ SADvector<N> operator-(
        const SADvector<N>& veca, const SADvector<N>& vecb)
    {
        SADvector<N> vec;
        vec.x = veca.x - vecb.x;
        vec.y = veca.y - vecb.y;
        vec.z = veca.z - vecb.z;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SADvector<N> operator*(
        const SADvector<N>& veca, const float& b)
    {
        SADvector<N> vec;
        vec.x = veca.x * b;
        vec.y = veca.y * b;
        vec.z = veca.z * b;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SADvector<N> operator*(
        const float& a, const SADvector<N>& vecb)
    {
        return vecb * a;
    }
    __device__ __host__ __forceinline__ void operator-=(
        const SADvector<N>& vecb)
    {
        this->x -= vecb.x;
        this->y -= vecb.y;
        this->z -= vecb.z;
    }
    __device__ __host__ __forceinline__ void operator+=(
        const SADvector<N>& vecb)
    {
        this->x += vecb.x;
        this->y += vecb.y;
        this->z += vecb.z;
    }
    friend __device__ __host__ __forceinline__ SADvector<N> operator^(
        const SADvector<N>& veca, const SADvector<N>& vecb)
    {
        SADvector<N> vec;
        vec.x = veca.y * vecb.z - veca.z * vecb.y;
        vec.y = veca.z * vecb.x - veca.x * vecb.z;
        vec.z = veca.x * vecb.y - veca.y * vecb.x;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SADvector<N> cross(
        const SADvector<N>& veca, const SADvector<N>& vecb)
    {
        return veca ^ vecb;
    }
    friend __device__ __host__ __forceinline__ SADvector<N>
    Get_Periodic_Displacement(const SADvector<N> vec_a,
                              const SADvector<N> vec_b,
                              const SADvector<N> box_length)
    {
        SADvector<N> dr;
        dr = vec_a - vec_b;
        dr.x.val = dr.x.val -
                   floorf(dr.x.val / box_length.x.val + 0.5) * box_length.x.val;
        dr.y.val = dr.y.val -
                   floorf(dr.y.val / box_length.y.val + 0.5) * box_length.y.val;
        dr.z.val = dr.z.val -
                   floorf(dr.z.val / box_length.z.val + 0.5) * box_length.z.val;
        for (int i = 0; i < N; i++)
        {
            dr.x.dval[i] = dr.x.val / box_length.x.val * box_length.x.dval[i] +
                           dr.x.dval[i];
            dr.y.dval[i] = dr.y.val / box_length.y.val * box_length.y.dval[i] +
                           dr.y.dval[i];
            dr.z.dval[i] = dr.z.val / box_length.z.val * box_length.z.dval[i] +
                           dr.z.dval[i];
        }
        return dr;
    }
    friend __device__ __host__ __forceinline__ SADvector<N>
    Get_Periodic_Displacement(const SADvector<N> a, const SADvector<N> b,
                              const LTMatrix3 cell, const LTMatrix3 rcell)
    {
        SADvector<N> dr = a - b;
        SADvector<N> scaled_dr;
        scaled_dr.x = dr.x * rcell.a11 + dr.y * rcell.a21 + dr.z * rcell.a31;
        scaled_dr.y = dr.y * rcell.a22 + dr.z * rcell.a32;
        scaled_dr.z = dr.z * rcell.a33;

        SADvector<N> shift;
        shift.x = floorf(scaled_dr.x.val + 0.5f);
        shift.y = floorf(scaled_dr.y.val + 0.5f);
        shift.z = floorf(scaled_dr.z.val + 0.5f);

        SADvector<N> result = dr;
        result.x -=
            shift.x * cell.a11 + shift.y * cell.a21 + shift.z * cell.a31;
        result.y -= shift.y * cell.a22 + shift.z * cell.a32;
        result.z -= shift.z * cell.a33;
        return result;
    }
};

#define foreach_2nd_create(EQUATION, START)                      \
    for (int i = 0; i < N2l; i++)                                \
    {                                                            \
        int index = i * N;                                       \
        for (int j = START; j < N; j++)                          \
        {                                                        \
            EQUATION;                                            \
        }                                                        \
    }                                                            \
    for (int i = N2l; i < N; i++)                                \
    {                                                            \
        int index = (2 * N - i + N2l) * (i - N2l) / 2 + N2l * N; \
        for (int j = 0; j < N2l && index + j < N2; j++)          \
        {                                                        \
            EQUATION;                                            \
        }                                                        \
        index -= i / 2 - N2l;                                    \
        for (int j = i; j < N && index + j < N2; j++)            \
        {                                                        \
            EQUATION;                                            \
        }                                                        \
    }

/*使用方法：
1. 确定该部分需要求偏微分的数量N
2. 确定该部分需要求二阶偏微分的最大序数N2，第i个变量和第j个变量(i <=
j)的二阶偏微分序数为2 * N - i - 1) * i / 2 + j
3.
将包含微分的变量和过程使用变量SA2Dfloat<N,N2>，其中对于想求的变量初始化时需要两个参数：本身的值和第i个变量，SA2Dfloat<1,1>
x(1.0f, 0);
4. 正常计算，那么最后结果中的dval[i]即为第i个变量的微分，ddval[(2 * N - i - 1) *
i / 2 + j]即为第i个变量和第j个变量(i <= j)的二阶偏微分。 样例： 已知f(x, y) = x
^ y + x * y， x=1.2, y = 2, 求df/dx和ddf/dx/dy
可以直接将下面的代码和common.h复制到一个新文件test.cu，再使用nvcc test.cu -o
SAD_TEST编译，最后使用./SAD_TEST运行

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
template <int N, int N2, int N2l = 0>
struct SA2Dfloat
{
    SADfloat<N> sad;
    float& val = sad.val;
    float* dval = sad.dval;
    float ddval[N2] = {0.0f};
    __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l>()
    {
        sad = SADfloat<N>();
    }
    __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l>(float f,
                                                              int id = -1)
    {
        sad = SADfloat<N>(f, id);
    }
    __device__ __host__ __forceinline__
    SA2Dfloat<N, N2, N2l>(const SA2Dfloat<N, N2, N2l>& f)
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
    __device__ __host__ __forceinline__ void operator=(
        const SA2Dfloat<N, N2, N2l>& f1)
    {
        this->sad = f1.sad;
        for (int i = 0; i < N2; i++)
        {
            this->ddval[i] = f1.ddval[i];
        }
    }
    __device__ __host__ __forceinline__ void operator+=(
        const SA2Dfloat<N, N2, N2l>& f1)
    {
        this->sad += f1.sad;
        for (int i = 0; i < N2; i++)
        {
            this->ddval[i] += f1.ddval[i];
        }
    }
    __device__ __host__ __forceinline__ void operator-=(
        const SA2Dfloat<N, N2, N2l>& f1)
    {
        this->sad -= f1.sad;
        for (int i = 0; i < N2; i++)
        {
            this->ddval[i] -= f1.ddval[i];
        }
    }
    __device__ __host__ __forceinline__ void operator*=(
        const SA2Dfloat<N, N2, N2l>& f1)
    {
        for (int i = 0; i < N2; i++)
        {
            this->ddval[i] = this->val * f1.ddval[i] + f1.val * this->ddval[i];
        }
        foreach_2nd_create(
            this->ddval[index + j] +=
            this->dval[i] * f1.dval[j] + this->dval[j] * f1.dval[i],
            N2l) this->sad *= f1.sad;
    }
    __device__ __host__ __forceinline__ void operator/=(
        const SA2Dfloat<N, N2, N2l>& f1)
    {
        for (int i = 0; i < N2; i++)
        {
            this->ddval[i] =
                (this->ddval[i] * f1.val - this->val * f1.ddval[i]) / f1.val /
                f1.val;
        }
        foreach_2nd_create(
            this->ddval[index + j] +=
            (2.0f * f1.dval[j] * f1.dval[i] * this->val / f1.val -
             (this->dval[i] * f1.dval[j] + this->dval[j] * f1.dval[i])) /
            f1.val / f1.val,
            N2l) this->sad /= f1.sad;
    }
    friend __device__ __host__ __forceinline__ bool operator==(
        const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1.val == f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(
        const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1.val != f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>(
        const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1.val > f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<(
        const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1.val < f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(
        const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1.val >= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(
        const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1.val <= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator==(
        const SA2Dfloat<N, N2, N2l>& f1, const float& f2)
    {
        return f1.val == f2;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(
        const SA2Dfloat<N, N2, N2l>& f1, const float& f2)
    {
        return f1.val != f2;
    }
    friend __device__ __host__ __forceinline__ bool operator>(
        const SA2Dfloat<N, N2, N2l>& f1, const float& f2)
    {
        return f1.val > f2;
    }
    friend __device__ __host__ __forceinline__ bool operator<(
        const SA2Dfloat<N, N2, N2l>& f1, const float& f2)
    {
        return f1.val < f2;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(
        const SA2Dfloat<N, N2, N2l>& f1, const float& f2)
    {
        return f1.val >= f2;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(
        const SA2Dfloat<N, N2, N2l>& f1, const float& f2)
    {
        return f1.val <= f2;
    }
    friend __device__ __host__ __forceinline__ bool operator==(
        const float& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1 == f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator!=(
        const float& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1 != f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>(
        const float& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1 > f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<(
        const float& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1 < f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator>=(
        const float& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1 >= f2.val;
    }
    friend __device__ __host__ __forceinline__ bool operator<=(
        const float& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        return f1 <= f2.val;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> operator+(
        const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = f1.sad + f2.sad;
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = f1.ddval[i] + f2.ddval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> operator-(
        const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = f1.sad - f2.sad;
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = f1.ddval[i] - f2.ddval[i];
        }
        return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> operator*(
        const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = f1.sad * f2.sad;
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = f1.val * f2.ddval[i] + f2.val * f1.ddval[i];
        }
        foreach_2nd_create(
            f.ddval[index + j] +=
            f1.sad.dval[i] * f2.sad.dval[j] + f1.sad.dval[j] * f2.sad.dval[i],
            N2l) return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> operator/(
        const SA2Dfloat<N, N2, N2l>& f1, const SA2Dfloat<N, N2, N2l>& f2)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = f1.sad / f2.sad;
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] =
                (f1.ddval[i] * f2.val - f1.val * f2.ddval[i]) / f2.val / f2.val;
        }
        foreach_2nd_create(
            f.ddval[index + j] +=
            (2.0f * f2.dval[j] * f2.dval[i] * f1.val / f2.val -
             (f1.dval[i] * f2.dval[j] + f1.dval[j] * f2.dval[i])) /
            f2.val / f2.val,
            N2l) return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> powf(
        const SA2Dfloat<N, N2, N2l>& x, const SA2Dfloat<N, N2, N2l>& y)
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
        float ddf_dxdy =
            powf(x.val, y.val - 1.0f) * (1.0f + y.val * logf(x.val));
        float ddf_dydy = df_dy * logf(x.val);
        foreach_2nd_create(
            f.ddval[index + j] +=
            ddf_dxdx * x.dval[i] * x.dval[j] +
            ddf_dxdy * (x.dval[i] * y.dval[j] + x.dval[j] * y.dval[i]) +
            ddf_dydy * y.dval[i] * y.dval[j],
            N2l) return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> expf(
        const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = expf(x.sad);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = f.val * x.ddval[i];
        }
        foreach_2nd_create(f.ddval[index + j] += f.val * x.dval[i] * x.dval[j],
                           N2l) return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> erfcf(
        const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = erfcf(x.sad);
        float df_dx = -2.0f / sqrtf(CONSTANT_Pi) * expf(-x.val * x.val);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        df_dx *= -2.0f * x.val;
        foreach_2nd_create(f.ddval[index + j] += df_dx * x.dval[i] * x.dval[j],
                           N2l) return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> logf(
        const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = logf(x.sad);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = x.ddval[i] / x.val;
        }
        foreach_2nd_create(
            f.ddval[index + j] -= x.dval[i] * x.dval[j] / x.val / x.val,
            N2l) return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> sqrtf(
        const SA2Dfloat<N, N2, N2l>& x)
    {
        return powf(x, 0.5f);
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> norm3df(
        const SA2Dfloat<N, N2, N2l>& x, const SA2Dfloat<N, N2, N2l>& y,
        const SA2Dfloat<N, N2, N2l>& z)
    {
        return sqrtf(x * x + y * y + z * z);
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> cbrtf(
        const SA2Dfloat<N, N2, N2l>& x)
    {
        return powf(x, 0.33333333f);
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> cosf(
        const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = cosf(x.sad);
        float df_dx = -sinf(x.val);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        foreach_2nd_create(f.ddval[index + j] -= f.val * x.dval[i] * x.dval[j],
                           N2l) return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> sinf(
        const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = sinf(x.sad);
        float df_dx = cosf(x.val);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        foreach_2nd_create(f.ddval[index + j] -= f.val * x.dval[i] * x.dval[j],
                           N2l) return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> tanf(
        const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = tanf(x.sad);
        float df_dx = 2.0f / (1.0f + cosf(2.0f * x.val));
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        df_dx *= 2.0f * f.val;
        foreach_2nd_create(f.ddval[index + j] += df_dx * x.dval[i] * x.dval[j],
                           N2l) return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> acosf(
        const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = acosf(x.sad);
        float df_dx = -1.0f / sqrtf(1.0f - x.val * x.val);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        df_dx *= x.val / (1.0f - x.val * x.val);
        foreach_2nd_create(f.ddval[index + j] += df_dx * x.dval[i] * x.dval[j],
                           N2l) return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> asinf(
        const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = asinf(x.sad);
        float df_dx = 1.0f / sqrtf(1.0f - x.val * x.val);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        df_dx *= x.val / (1.0f - x.val * x.val);
        foreach_2nd_create(f.ddval[index + j] += df_dx * x.dval[i] * x.dval[j],
                           N2l) return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> atanf(
        const SA2Dfloat<N, N2, N2l>& x)
    {
        SA2Dfloat<N, N2, N2l> f;
        f.sad = atanf(x.sad);
        float df_dx = 1.0f / (1.0f + x.val * x.val);
        for (int i = 0; i < N2; i++)
        {
            f.ddval[i] = df_dx * x.ddval[i];
        }
        df_dx *= -2.0f * x.val / (1.0f + x.val * x.val);
        foreach_2nd_create(f.ddval[index + j] += df_dx * x.dval[i] * x.dval[j],
                           N2l) return f;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> fabsf(
        const SA2Dfloat<N, N2, N2l>& x)
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
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> fmaxf(
        const SA2Dfloat<N, N2, N2l>& x, const SA2Dfloat<N, N2, N2l>& y)
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
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> fminf(
        const SA2Dfloat<N, N2, N2l>& x, const SA2Dfloat<N, N2, N2l>& y)
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
        if (i < N2l) return i * N + j;
        if (j < N2l) return (2 * N - i + N2l) * (i - N2l) / 2 + N2l * N + j;
        if (i > j)
        {
            int temp = i;
            i = j;
            j = temp;
        }
        return (2 * N - i + N2l) * (i - N2l) / 2 + N2l * N + j - i + N2l;
    }
};

template <int N, int N2, int N2l = 0>
struct SA2Dvector
{
    SA2Dfloat<N, N2, N2l> x, y, z;
    __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l>()
    {
        this->x = SA2Dfloat<N, N2, N2l>(0);
        this->y = SA2Dfloat<N, N2, N2l>(0);
        this->z = SA2Dfloat<N, N2, N2l>(0);
    }
    __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l>(VECTOR v,
                                                               int idx = -1,
                                                               int idy = -1,
                                                               int idz = -1)
    {
        this->x = SA2Dfloat<N, N2, N2l>(v.x, idx);
        this->y = SA2Dfloat<N, N2, N2l>(v.y, idy);
        this->z = SA2Dfloat<N, N2, N2l>(v.z, idz);
    }
    friend __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l> operator+(
        const SA2Dvector<N, N2, N2l>& veca, const SA2Dvector<N, N2, N2l>& vecb)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = veca.x + vecb.x;
        vec.y = veca.y + vecb.y;
        vec.z = veca.z + vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l> operator+(
        const SA2Dvector<N, N2, N2l>& veca, const SA2Dfloat<N, N2, N2l>& b)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = veca.x + b;
        vec.y = veca.y + b;
        vec.z = veca.z + b;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l> operator*(
        const SA2Dfloat<N, N2, N2l>& a, const SA2Dvector<N, N2, N2l>& vecb)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = a * vecb.x;
        vec.y = a * vecb.y;
        vec.z = a * vecb.z;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SA2Dfloat<N, N2, N2l> operator*(
        const SA2Dvector<N, N2, N2l>& veca, const SA2Dvector<N, N2, N2l>& vecb)
    {
        return veca.x * vecb.x + veca.y * vecb.y + veca.z * vecb.z;
    }
    friend __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l> operator-(
        const SA2Dvector<N, N2, N2l>& veca, const SA2Dvector<N, N2, N2l>& vecb)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = veca.x - vecb.x;
        vec.y = veca.y - vecb.y;
        vec.z = veca.z - vecb.z;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l> operator-(
        const SA2Dvector<N, N2, N2l>& veca, const SA2Dfloat<N, N2, N2l>& b)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = veca.x - b;
        vec.y = veca.y - b;
        vec.z = veca.z - b;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l> operator/(
        const SA2Dvector<N, N2, N2l>& veca, const SA2Dvector<N, N2, N2l>& vecb)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = veca.x / vecb.x;
        vec.y = veca.y / vecb.y;
        vec.z = veca.z / vecb.z;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l> operator/(
        const float& a, const SA2Dvector<N, N2, N2l>& vecb)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = a / vecb.x;
        vec.y = a / vecb.y;
        vec.z = a / vecb.z;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l> operator^(
        const SA2Dvector<N, N2, N2l>& veca, const SA2Dvector<N, N2, N2l>& vecb)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = veca.y * vecb.z - veca.z * vecb.y;
        vec.y = veca.z * vecb.x - veca.x * vecb.z;
        vec.z = veca.x * vecb.y - veca.y * vecb.x;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l>
    Get_Periodic_Displacement(const SA2Dvector<N, N2, N2l> vec_a,
                              const SA2Dvector<N, N2, N2l> vec_b,
                              const SA2Dvector<N, N2, N2l> box_length)
    {
        SA2Dvector<N, N2, N2l> dr;
        dr = vec_a - vec_b;
        dr.x.val = dr.x.val -
                   floorf(dr.x.val / box_length.x.val + 0.5) * box_length.x.val;
        dr.y.val = dr.y.val -
                   floorf(dr.y.val / box_length.y.val + 0.5) * box_length.y.val;
        dr.z.val = dr.z.val -
                   floorf(dr.z.val / box_length.z.val + 0.5) * box_length.z.val;
        foreach_2nd_create(
            dr.x.ddval[index + j] =
                (dr.x.dval[j] * box_length.x.dval[i] -
                 dr.x.val / box_length.x.val * box_length.x.dval[i] *
                     box_length.x.dval[j]) /
                box_length.x.val;
            dr.y.ddval[index + j] =
                (dr.y.dval[j] * box_length.y.dval[i] -
                 dr.y.val / box_length.y.val * box_length.y.dval[i] *
                     box_length.y.dval[j]) /
                box_length.y.val;
            dr.z.ddval[index + j] =
                (dr.z.dval[j] * box_length.z.dval[i] -
                 dr.z.val / box_length.z.val * box_length.z.dval[i] *
                     box_length.z.dval[j]) /
                box_length.z.val,
            0) for (int i = 0; i < N; i++)
        {
            dr.x.dval[i] = dr.x.val / box_length.x.val * box_length.x.dval[i] +
                           dr.x.dval[i];
            dr.y.dval[i] = dr.y.val / box_length.y.val * box_length.y.dval[i] +
                           dr.y.dval[i];
            dr.z.dval[i] = dr.z.val / box_length.z.val * box_length.z.dval[i] +
                           dr.z.dval[i];
        }
        return dr;
    }
};

#endif
