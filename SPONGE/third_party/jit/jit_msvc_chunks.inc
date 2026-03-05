value += R"JIT(
//存储各种常数
//圆周率
#define CONSTANT_Pi 3.1415926535897932f
//自然对数的底
#define CONSTANT_e 2.7182818284590452f
//玻尔兹曼常量（kcal.mol^-1.K^ -1）
//使用kcal为能量单位，因此kB=8.31441(J.mol^-1.K^-1)/4.18407(J/cal)/1000
#define CONSTANT_kB 0.00198716f
//程序中使用的单位时间与物理时间的换算1/20.455*dt=1 ps
#define CONSTANT_TIME_CONVERTION 20.455f
//程序中使用的单位压强与物理压强的换算
// 压强单位: bar -> kcal/mol/A^3 
// (1 kcal/mol) * (4.184074e3 J/kcal) / (6.023e23 mol^-1) * (1e30 m^3/A^3) * (1e-5 bar/pa)
// 程序的压强/(kcal/mol/A^3 ) * CONSTANT_PRES_CONVERTION = 物理压强/bar
#define CONSTANT_PRES_CONVERTION 6.946827162543585e4f
// 物理压强/bar * CONSTANT_PRES_CONVERTION_INVERSE = 程序的压强/(kcal/mol/A^3 )
#define CONSTANT_PRES_CONVERTION_INVERSE 0.00001439506089041446f
//角度制到弧度制的转换系数
#define CONSTANT_RAD_TO_DEG 57.2957795f
//弧度制到角度制的转换系数
#define CONSTANT_DEG_TO_RAD 0.0174532925f

#define CHAR_LENGTH_MAX 512
#define FULL_MASK 0xffffffff

//用于计算边界循环所定义的结构体
struct UNSIGNED_INT_VECTOR
{
    unsigned int uint_x;
    unsigned int uint_y;
    unsigned int uint_z;
};

//用于计算边界循环或者一些三维数组大小所定义的结构体
struct INT_VECTOR
{
    int int_x;
    int int_y;
    int int_z;
};

__device__ __forceinline__ void Warp_Sum_To(float* y, float& x, int delta = 32);
__host__ __device__ __forceinline__ float BSpline_4_1(float x);
__host__ __device__ __forceinline__ float BSpline_4_2(float x);
__host__ __device__ __forceinline__ float BSpline_4_3(float x);
__host__ __device__ __forceinline__ float BSpline_4_4(float x);
__host__ __device__ __forceinline__ float dBSpline_4_1(float x);
__host__ __device__ __forceinline__ float dBSpline_4_2(float x);
__host__ __device__ __forceinline__ float dBSpline_4_3(float x);
__host__ __device__ __forceinline__ float dBSpline_4_4(float x);
//用于存储各种三维float矢量而定义的结构体
struct VECTOR
{
    float x;
    float y;
    float z;

    friend __device__ __host__ __forceinline__ VECTOR operator+ (const VECTOR& veca, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = veca.x + vecb.x;
        vec.y = veca.y + vecb.y;
        vec.z = veca.z + vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR operator+ (const VECTOR& veca, const float& b)
    {
        VECTOR vec;
        vec.x = veca.x + b;
        vec.y = veca.y + b;
        vec.z = veca.z + b;
        return vec;
    }

    friend __device__ __host__  __forceinline__ float operator* (const VECTOR& veca, const VECTOR& vecb)
    {
        return veca.x * vecb.x + veca.y * vecb.y + veca.z * vecb.z;
    }
    friend __device__ __host__  __forceinline__ VECTOR operator* (const float& a, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = a * vecb.x;
        vec.y = a * vecb.y;
)JIT";
value += R"JIT(
        vec.z = a * vecb.z;
        return vec;
    }
    friend __device__ __host__  __forceinline__ VECTOR operator- (const VECTOR& veca, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = veca.x - vecb.x;
        vec.y = veca.y - vecb.y;
        vec.z = veca.z - vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR operator- (const VECTOR& veca, const float& b)
    {
        VECTOR vec;
        vec.x = veca.x - b;
        vec.y = veca.y - b;
        vec.z = veca.z - b;
        return vec;
    }

    friend __device__ __host__  __forceinline__ VECTOR operator- (const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = -vecb.x;
        vec.y = -vecb.y;
        vec.z = -vecb.z;
        return vec;
    }

    friend __device__ __host__  __forceinline__ VECTOR operator/ (const VECTOR& veca, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = veca.x / vecb.x;
        vec.y = veca.y / vecb.y;
        vec.z = veca.z / vecb.z;
        return vec;
    }

    friend __device__ __host__  __forceinline__ VECTOR operator/ (const float& a, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = a / vecb.x;
        vec.y = a / vecb.y;
        vec.z = a / vecb.z;
        return vec;
    }

    friend __device__ __host__  __forceinline__ VECTOR operator^ (const VECTOR& veca, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = veca.y * vecb.z - veca.z * vecb.y;
        vec.y = veca.z * vecb.x - veca.x * vecb.z;
        vec.z = veca.x * vecb.y - veca.y * vecb.x;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR Get_Periodic_Displacement(const UNSIGNED_INT_VECTOR uvec_a, const UNSIGNED_INT_VECTOR uvec_b, const VECTOR scaler)
    {
        VECTOR dr;
        dr.x = ((int)(uvec_a.uint_x - uvec_b.uint_x)) * scaler.x;
        dr.y = ((int)(uvec_a.uint_y - uvec_b.uint_y)) * scaler.y;
        dr.z = ((int)(uvec_a.uint_z - uvec_b.uint_z)) * scaler.z;
        return dr;
    }


    friend __device__ __host__ __forceinline__ VECTOR Get_Periodic_Displacement(const VECTOR vec_a, const VECTOR vec_b, const VECTOR box_length)
    {
        VECTOR dr;
        dr = vec_a - vec_b;
        dr.x = dr.x - floorf(dr.x / box_length.x + 0.5f) * box_length.x;
        dr.y = dr.y - floorf(dr.y / box_length.y + 0.5f) * box_length.y;
        dr.z = dr.z - floorf(dr.z / box_length.z + 0.5f) * box_length.z;
        return dr;
    }

    friend __device__ __host__ __forceinline__ VECTOR Get_Periodic_Displacement(const VECTOR vec_a, const VECTOR vec_b, const VECTOR box_length, const VECTOR box_length_inverse)
    {
        VECTOR dr;
        dr = vec_a - vec_b;
        dr.x = dr.x - floorf(dr.x * box_length_inverse.x + 0.5f) * box_length.x;
        dr.y = dr.y - floorf(dr.y * box_length_inverse.y + 0.5f) * box_length.y;
        dr.z = dr.z - floorf(dr.z * box_length_inverse.z + 0.5f) * box_length.z;
        return dr;
    }

)JIT";
value += R"JIT(
    friend __host__ __device__ __forceinline__ VECTOR floorf(VECTOR v)
    {
        return { floorf(v.x), floorf(v.y), floorf(v.z) };
    }

    friend __device__ __forceinline__ VECTOR Make_Vector_Not_Exceed_Value(VECTOR vector, const float value)
    {
        return fminf(1.0, value * rnorm3df(vector.x, vector.y, vector.z)) * vector;
    }

    friend __device__ __forceinline__ void atomicAdd(VECTOR* a, const VECTOR b)
    {
        atomicAdd(&a->x, b.x);
        atomicAdd(&a->y, b.y);
        atomicAdd(&a->z, b.z);
    }
    friend __device__ __forceinline__ void Warp_Sum_To(float* y, float& x, int delta)
    {
        unsigned int mask = __ballot_sync(FULL_MASK, threadIdx.x < delta);
        for (delta >>= 1; delta > 0; delta >>= 1)
        {
            x += __shfl_down_sync(FULL_MASK, x, delta);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(y, x);
        }
    }
    friend __device__ __forceinline__ void Warp_Sum_To(VECTOR* y, VECTOR& x, int delta = 32)
    {
        for (delta >>= 1; delta > 0; delta >>= 1)
        {
            x.x += __shfl_down_sync(FULL_MASK, x.x, delta);
            x.y += __shfl_down_sync(FULL_MASK, x.y, delta);
            x.z += __shfl_down_sync(FULL_MASK, x.z, delta);
        }
        if (threadIdx.x == 0)
        {
            atomicAdd(y, x);
        }
    }
    friend __host__ __device__ __forceinline__ float BSpline_4_1(float x)
    {
        return 0.1666667f * x * x * x;
    }
    friend __host__ __device__ __forceinline__ float BSpline_4_2(float x)
    {
        return -0.5f * x * x * x + 0.5f * x * x + 0.5f * x + 0.16666667f;
    }
    friend __host__ __device__ __forceinline__ float BSpline_4_3(float x)
    {
        return 0.5f * x * x * x - x * x + 0.66666667f;
    }
    friend __host__ __device__ __forceinline__ float BSpline_4_4(float x)
    {
        return  -0.16666667f * x * x * x + 0.5f * x * x - 0.5f * x + 0.16666667f;
    }
    friend __host__ __device__ __forceinline__ float dBSpline_4_1(float x)
    {
        return -0.5f * x * x;
    }
    friend __host__ __device__ __forceinline__ float dBSpline_4_2(float x)
    {
        return  1.5f * x * x - x - 0.5f;
    }
    friend __host__ __device__ __forceinline__ float dBSpline_4_3(float x)
    {
        return -1.5f * x * x + 2.0f * x;
    }
    friend __host__ __device__ __forceinline__ float dBSpline_4_4(float x)
    {
        return  -0.5f * x * x + x - 0.5f;
    }
    //Reference: Eigenvalues and Eigenvectors for 3x3 Symmetric Matrices: An Analytical Approach
    friend __device__ __host__ __forceinline__ void get_single_eigen_vector(const float* m, const float eigen_value, VECTOR* eigen_vector)
    {
        float b11 = m[0] - eigen_value;
#define b12 m[1]
#define b13 m[2]
        float b22 = m[3] - eigen_value;
#define b23 m[4]
        float b33 = m[5] - eigen_value;
        float Q, P, n;
        VECTOR local_VECTOR;
        if (fabsf(b12 * b12 - b11 * b22) > 1e-6 && fabsf(b13) > 1e-6f)
        {
)JIT";
value += R"JIT(
            Q = (b11 * b23 - b13 * b12) / (b12 * b12 - b11 * b22);
            P = -(b23 * Q + b33) / b13;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = Q * n;
            local_VECTOR.z = n;
        }
        else if (fabsf(b12 * b13 - b11 * b23) > 1e-6 && fabsf(b12) > 1e-6f)
        {
            Q = (b11 * b33 - b13 * b13) / (b12 * b13 - b11 * b23);
            P = -(b22 * Q + b23) / b12;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = Q * n;
            local_VECTOR.z = n;
        }
        else if (fabsf(b22 * b13 - b12 * b23) > 1e-6 && fabsf(b11) > 1e-6f)
        {
            Q = (b12 * b33 - b23 * b13) / (b22 * b13 - b12 * b23);
            P = -(b12 * Q + b13) / b11;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = Q * n;
            local_VECTOR.z = n;
        }
        else if (fabsf(b11 * b22 - b12 * b12) > 1e-6 && fabsf(b23) > 1e-6f)
        {
            P = (b12 * b23 - b13 * b22) / (b11 * b22 - b12 * b12);
            Q = -(b13 * P + b33) / b23;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = Q * n;
            local_VECTOR.z = n;
        }
        else if (fabsf(b11 * b23 - b12 * b13) > 1e-6 && fabsf(b22) > 1e-6f)
        {
            P = (b12 * b33 - b13 * b23) / (b11 * b23 - b12 * b13);
            Q = -(b12 * P + b23) / b22;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = Q * n;
            local_VECTOR.z = n;
        }
        else if (fabsf(b12 * b23 - b22 * b13) > 1e-6 && fabsf(b12) > 1e-6f)
        {
            P = (b22 * b33 - b23 * b23) / (b12 * b23 - b22 * b13);
            Q = -(b11 * P + b13) / b12;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = Q * n;
            local_VECTOR.z = n;
        }
        else if (fabsf(b11 * b23 - b13 * b12) > 1e-6 && fabsf(b33) > 1e-6f)
        {
            P = (b13 * b22 - b12 * b23) / (b11 * b23 - b13 * b12);
            Q = -(b13 * P + b23) / b33;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = n;
            local_VECTOR.z = Q * n;
        }
        else if (fabsf(b11 * b33 - b13 * b13) > 1e-6 && fabsf(b23) > 1e-6f)
        {
            P = (b13 * b23 - b12 * b33) / (b11 * b33 - b13 * b13);
            Q = -(b12 * P + b22) / b23;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
            local_VECTOR.y = n;
            local_VECTOR.z = Q * n;
        }
        else
        {
            P = (b23 * b23 - b22 * b33) / (b12 * b33 - b23 * b13);
            Q = -(b11 * P + b12) / b13;
            n = 1.0f / sqrtf(P * P + Q * Q + 1);
            local_VECTOR.x = P * n;
)JIT";
value += R"JIT(
            local_VECTOR.y = n;
            local_VECTOR.z = Q * n;
        }
#undef b12
#undef b13
#undef b23
        eigen_vector[0] = local_VECTOR;
    }

//Reference: 1. Eigenvalues and Eigenvectors for 3x3 Symmetric Matrices: An Analytical Approach
//Reference: 2. github: svd_3x3_cuda
    friend __device__ __host__ __forceinline__ void get_eigen(const float* m, float* eigen_values, VECTOR* eigen_vector)
    {
        float t1, t2;
        VECTOR v1, v2, v3;
        if (m[1] == 0 && m[2] == 0 && m[4] == 0)
        {
            eigen_values[0] = m[0];
            eigen_values[1] = m[3];
            eigen_values[2] = m[5];
            v1.x = 1;
            v1.y = 0;
            v1.z = 0;
            v2.x = 0;
            v2.y = 1;
            v2.z = 0;
        }
        else if (m[1] == 0 && m[2] == 0)
        {
            t1 = m[3] - m[5];
            t1 = sqrtf(t1 * t1 / 4 + m[4] * m[4]);
            t2 = (m[3] + m[5]) / 2;
            eigen_values[0] = m[0];
            eigen_values[1] = t2 + t1;
            eigen_values[2] = t2 - t1;
            v1.x = 1;
            v1.y = 0;
            v1.z = 0;
            t1 = m[3] - eigen_values[2];
            t1 = sqrtf(t1 * t1 + m[4] * m[4]);
            v2.x = 0;
            v2.y = -m[4] / t1;
            v2.z = (m[3] - eigen_values[2]) / t1;
        }
        else if (m[1] == 0 && m[4] == 0)
        {
            t1 = m[0] - m[5];
            t1 = sqrtf(t1 * t1 / 4 + m[2] * m[2]);
            t2 = (m[0] + m[5]) / 2;
            eigen_values[0] = t2 + t1;
            eigen_values[1] = m[3];
            eigen_values[2] = t2 - t1;

            v2.x = 0;
            v2.y = 1;
            v2.z = 0;
            t1 = m[0] - eigen_values[0];
            t1 = sqrtf(t1 * t1 + m[2] * m[2]);
            v1.x = -m[2] / t1;
            v1.y = 0;
            v1.z = (m[0] - eigen_values[0]) / t1;
        }
        else if (m[2] == 0 && m[4] == 0)
        {
            t1 = m[3] - m[0];
            t1 = sqrtf(t1 * t1 / 4 + m[1] * m[1]);
            t2 = (m[3] + m[0]) / 2;
            eigen_values[0] = t2 + t1;
            eigen_values[1] = t2 - t1;
            eigen_values[2] = m[5];
            t1 = m[0] - eigen_values[0];
            t1 = sqrtf(t1 * t1 + m[1] * m[1]);
            v1.x = -m[1] / t1;
            v1.y = (m[0] - eigen_values[0]) / t1;
            v1.z = 0;
            v2.x = -(m[0] - eigen_values[0]) / t1;
            v2.y = -m[1] / t1;
            v2.z = 0;
        }
        else
        {
            float m_ = 1.0f / 3.0f * (m[0] + m[3] + m[5]);
            float a11 = m[0] - m_;
            float a22 = m[3] - m_;
            float a33 = m[5] - m_;
            float a12_sqr = m[1] * m[1];
            float a13_sqr = m[2] * m[2];
            float a23_sqr = m[4] * m[4];
            float p = 1.0f / 6.0f * (a11 * a11 + a22 * a22 + a33 * a33 + 2 * (a12_sqr + a13_sqr + a23_sqr));
            float q = 0.5f * (a11 * (a22 * a33 - a23_sqr) - a22 * a13_sqr - a33 * a12_sqr) + m[1] * m[2] * m[4];
)JIT";
value += R"JIT(
            float sqrt_p = sqrtf(p);
            float disc = p * p * p - q * q;
            float phi = 1.0f / 3.0f * atan2f(sqrtf(fmaxf(0.0f, disc)), q);
            float c = cosf(phi);
            float s = sinf(phi);
            float sqrt_p_cos = sqrt_p * c;
            float root_three_sqrt_p_sin = sqrtf(3.0f) * sqrt_p * s;
            eigen_values[0] = m_ + 2.0f * sqrt_p;
            eigen_values[1] = m_ - sqrt_p_cos - root_three_sqrt_p_sin;
            eigen_values[2] = m_ - sqrt_p_cos + root_three_sqrt_p_sin;
            get_single_eigen_vector(m, eigen_values[0], &v1);
            get_single_eigen_vector(m, eigen_values[1], &v2);
        }
        v3 = v2 ^ v1;
        if (eigen_values[0] < eigen_values[1])
        {
            t1 = eigen_values[0];
            eigen_values[0] = eigen_values[1];
            eigen_values[1] = t1;
            t1 = v1.x;
            v1.x = v2.x;
            v2.x = t1;
            t1 = v1.y;
            v1.y = v2.y;
            v2.y = t1;
            t1 = v1.z;
            v1.z = v2.z;
            v2.z = t1;
        }
        if (eigen_values[0] < eigen_values[2])
        {
            t1 = eigen_values[0];
            eigen_values[0] = eigen_values[2];
            eigen_values[2] = t1;
            t1 = v1.x;
            v1.x = v3.x;
            v3.x = t1;
            t1 = v1.y;
            v1.y = v3.y;
            v3.y = t1;
            t1 = v1.z;
            v1.z = v3.z;
            v3.z = t1;
        }
        if (eigen_values[1] < eigen_values[2])
        {
            t1 = eigen_values[1];
            eigen_values[1] = eigen_values[2];
            eigen_values[2] = t1;
            t1 = v2.x;
            v2.x = v3.x;
            v3.x = t1;
            t1 = v2.y;
            v2.y = v3.y;
            v3.y = t1;
            t1 = v2.z;
            v2.z = v3.z;
            v3.z = t1;
        }
        eigen_vector[0] = v1;
        eigen_vector[1] = v2;
        eigen_vector[2] = v3;
    }
};

//默认的所有VECTOR都是行矢量
//三阶下三角矩阵，用于存储盒子信息
struct LTMatrix3
{
    float a11, a21, a22, a31, a32, a33;
    friend __host__ __device__ __forceinline__ LTMatrix3 operator+(LTMatrix3 m1, LTMatrix3 m2)
    {
        return { m1.a11 + m2.a11, 
                 m1.a21 + m2.a21, m1.a22 + m2.a22, 
                 m1.a31 + m2.a31, m1.a32 + m2.a32, m1.a33 + m2.a33 };
    }
    friend __host__ __device__ __forceinline__ LTMatrix3 operator-(LTMatrix3 m1, LTMatrix3 m2)
    {
        return { m1.a11 - m2.a11,
                 m1.a21 - m2.a21, m1.a22 - m2.a22,
                 m1.a31 - m2.a31, m1.a32 - m2.a32, m1.a33 - m2.a33 };
    }
    friend __host__ __device__ __forceinline__ LTMatrix3 operator*(float m1, LTMatrix3 m2)
    {
        return { m1 * m2.a11, 
                 m1 * m2.a21, m1 * m2.a22,
                 m1 * m2.a31, m1 * m2.a32, m1 * m2.a33 };
    }
    friend __host__ __device__ __forceinline__ VECTOR MultiplyTranspose(VECTOR vec, LTMatrix3 mat)
    {
)JIT";
value += R"JIT(
        return { vec.x * mat.a11,
            vec.x * mat.a21 + vec.y * mat.a22,
            vec.x * mat.a31 + vec.y * mat.a32 + vec.z * mat.a33};
    }
    friend __host__ __device__ __forceinline__ VECTOR operator*(VECTOR vec, LTMatrix3 mat)
    {
        return { vec.x * mat.a11 + vec.y * mat.a21 + vec.z * mat.a31,
            vec.y * mat.a22 + vec.z * mat.a32,
            vec.z * mat.a33 };
    }
    friend __host__ __device__ __forceinline__
        VECTOR Get_Periodic_Displacement(VECTOR a, VECTOR b, LTMatrix3 cell, LTMatrix3 rcell)
    {
        VECTOR dr = a - b;
        return dr - floorf(dr * rcell + 0.5f) * cell;
    }
    friend __device__ __host__ __forceinline__ LTMatrix3 Get_Virial_From_Force_Dis(const VECTOR& veca, const VECTOR& vecb)
    {
        LTMatrix3 mat;
        mat.a11 = veca.x * vecb.x;
        mat.a21 = veca.x * vecb.y + veca.y * vecb.x;
        mat.a22 = veca.y * vecb.y;
        mat.a31 = veca.x * vecb.z + veca.z * vecb.x;
        mat.a32 = veca.y * vecb.z + veca.z * vecb.y;
        mat.a33 = veca.z * vecb.z;
        return mat;
    }
    friend __device__ __forceinline__
        void atomicAdd(LTMatrix3* a, LTMatrix3 L)
    {
        atomicAdd(&a->a11, L.a11);
        atomicAdd(&a->a21, L.a21);
        atomicAdd(&a->a22, L.a22);
        atomicAdd(&a->a31, L.a31);
        atomicAdd(&a->a32, L.a32);
        atomicAdd(&a->a33, L.a33);
    }
};
__device__ __host__ __forceinline__ LTMatrix3 Get_Virial_From_Force_Dis(const VECTOR& veca, const VECTOR& vecb);

//用于记录原子组
struct ATOM_GROUP
{
    int atom_numbers;
    int *atom_serial;
};

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
)JIT";
value += R"JIT(
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
)JIT";
value += R"JIT(
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
)JIT";
value += R"JIT(
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
)JIT";
value += R"JIT(
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

template<int N>
struct SADvector
{
    SADfloat<N> x, y, z;
    __device__ __host__ __forceinline__ SADvector<N>()
    {
        this->x = SADfloat<N>(0);
        this->y = SADfloat<N>(0);
        this->z = SADfloat<N>(0);
    }
    __device__ __host__ __forceinline__ SADvector<N>(VECTOR v, int idx = -1, int idy = -1, int idz = -1)
    {
        this->x = SADfloat<N>(v.x, idx);
        this->y = SADfloat<N>(v.y, idy);
        this->z = SADfloat<N>(v.z, idz);
    }
    friend __device__ __host__ __forceinline__ SADvector<N> operator+ (const SADvector<N>& veca, const SADvector<N>& vecb)
    {
        SADvector<N> vec;
        vec.x = veca.x + vecb.x;
        vec.y = veca.y + vecb.y;
        vec.z = veca.z + vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ SADvector<N> operator+ (const SADvector<N>& veca, const SADfloat<N>& b)
    {
        SADvector<N> vec;
        vec.x = veca.x + b;
        vec.y = veca.y + b;
        vec.z = veca.z + b;
        return vec;
    }
    friend __device__ __host__  __forceinline__ SADvector<N> operator* (const SADfloat<N>& a, const SADvector<N>& vecb)
    {
        SADvector<N> vec;
        vec.x = a * vecb.x;
        vec.y = a * vecb.y;
        vec.z = a * vecb.z;
        return vec;
    }
)JIT";
value += R"JIT(
    friend __device__ __host__  __forceinline__ SADfloat<N> operator* (const SADvector<N>& veca, const SADvector<N>& vecb)
    {
        return veca.x * vecb.x + veca.y * vecb.y + veca.z * vecb.z;
    }
    friend __device__ __host__  __forceinline__ SADvector<N> operator- (const SADvector<N>& veca, const SADvector<N>& vecb)
    {
        SADvector<N> vec;
        vec.x = veca.x - vecb.x;
        vec.y = veca.y - vecb.y;
        vec.z = veca.z - vecb.z;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SADvector<N> operator- (const SADvector<N>& veca, const SADfloat<N>& b)
    {
        SADvector<N> vec;
        vec.x = veca.x - b;
        vec.y = veca.y - b;
        vec.z = veca.z - b;
        return vec;
    }
    friend __device__ __host__  __forceinline__ SADvector<N> operator/ (const SADvector<N>& veca, const SADvector<N>& vecb)
    {
        SADvector<N> vec;
        vec.x = veca.x / vecb.x;
        vec.y = veca.y / vecb.y;
        vec.z = veca.z / vecb.z;
        return vec;
    }
    friend __device__ __host__  __forceinline__ SADvector<N> operator/ (const float& a, const SADvector<N>& vecb)
    {
        SADvector<N> vec;
        vec.x = a / vecb.x;
        vec.y = a / vecb.y;
        vec.z = a / vecb.z;
        return vec;
    }
    friend __device__ __host__  __forceinline__ SADvector<N> operator^ (const SADvector<N>& veca, const SADvector<N>& vecb)
    {
        SADvector<N> vec;
        vec.x = veca.y * vecb.z - veca.z * vecb.y;
        vec.y = veca.z * vecb.x - veca.x * vecb.z;
        vec.z = veca.x * vecb.y - veca.y * vecb.x;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SADvector<N> Get_Periodic_Displacement(const SADvector<N> vec_a, const SADvector<N> vec_b, const SADvector<N> box_length)
    {
        SADvector<N> dr;
        dr = vec_a - vec_b;
        dr.x.val = dr.x.val - floorf(dr.x.val / box_length.x.val + 0.5) * box_length.x.val;
        dr.y.val = dr.y.val - floorf(dr.y.val / box_length.y.val + 0.5) * box_length.y.val;
        dr.z.val = dr.z.val - floorf(dr.z.val / box_length.z.val + 0.5) * box_length.z.val;
        for (int i = 0; i < N; i++)
        {
            dr.x.dval[i] = dr.x.val / box_length.x.val * box_length.x.dval[i] + dr.x.dval[i];
            dr.y.dval[i] = dr.y.val / box_length.y.val * box_length.y.dval[i] + dr.y.dval[i];
            dr.z.dval[i] = dr.z.val / box_length.z.val * box_length.z.dval[i] + dr.z.dval[i];
        }
        return dr;
    }
};

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
)JIT";
value += R"JIT(
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
)JIT";
value += R"JIT(
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
)JIT";
value += R"JIT(
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
)JIT";
value += R"JIT(
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
)JIT";
value += R"JIT(
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
)JIT";
value += R"JIT(
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

template<int N, int N2, int N2l = 0>
struct SA2Dvector
{
    SA2Dfloat<N, N2, N2l> x, y, z;
    __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l>()
    {
        this->x = SA2Dfloat<N, N2, N2l>(0);
        this->y = SA2Dfloat<N, N2, N2l>(0);
        this->z = SA2Dfloat<N, N2, N2l>(0);
    }
    __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l>(VECTOR v, int idx = -1, int idy = -1, int idz = -1)
    {
        this->x = SA2Dfloat<N, N2, N2l>(v.x, idx);
        this->y = SA2Dfloat<N, N2, N2l>(v.y, idy);
        this->z = SA2Dfloat<N, N2, N2l>(v.z, idz);
    }
    friend __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l> operator+ (const SA2Dvector<N, N2, N2l>& veca, const SA2Dvector<N, N2, N2l>& vecb)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = veca.x + vecb.x;
        vec.y = veca.y + vecb.y;
        vec.z = veca.z + vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l> operator+ (const SA2Dvector<N, N2, N2l>& veca, const SA2Dfloat<N, N2, N2l>& b)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = veca.x + b;
        vec.y = veca.y + b;
        vec.z = veca.z + b;
        return vec;
    }
    friend __device__ __host__  __forceinline__ SA2Dvector<N, N2, N2l> operator* (const SA2Dfloat<N, N2, N2l>& a, const SA2Dvector<N, N2, N2l>& vecb)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = a * vecb.x;
        vec.y = a * vecb.y;
        vec.z = a * vecb.z;
        return vec;
    }
    friend __device__ __host__  __forceinline__ SA2Dfloat<N, N2, N2l> operator* (const SA2Dvector<N, N2, N2l>& veca, const SA2Dvector<N, N2, N2l>& vecb)
    {
        return veca.x * vecb.x + veca.y * vecb.y + veca.z * vecb.z;
    }
    friend __device__ __host__  __forceinline__ SA2Dvector<N, N2, N2l> operator- (const SA2Dvector<N, N2, N2l>& veca, const SA2Dvector<N, N2, N2l>& vecb)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = veca.x - vecb.x;
        vec.y = veca.y - vecb.y;
        vec.z = veca.z - vecb.z;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l> operator- (const SA2Dvector<N, N2, N2l>& veca, const SA2Dfloat<N, N2, N2l>& b)
    {
        SA2Dvector<N, N2, N2l> vec;
)JIT";
value += R"JIT(
        vec.x = veca.x - b;
        vec.y = veca.y - b;
        vec.z = veca.z - b;
        return vec;
    }
    friend __device__ __host__  __forceinline__ SA2Dvector<N, N2, N2l> operator/ (const SA2Dvector<N, N2, N2l>& veca, const SA2Dvector<N, N2, N2l>& vecb)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = veca.x / vecb.x;
        vec.y = veca.y / vecb.y;
        vec.z = veca.z / vecb.z;
        return vec;
    }
    friend __device__ __host__  __forceinline__ SA2Dvector<N, N2, N2l> operator/ (const float& a, const SA2Dvector<N, N2, N2l>& vecb)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = a / vecb.x;
        vec.y = a / vecb.y;
        vec.z = a / vecb.z;
        return vec;
    }
    friend __device__ __host__  __forceinline__ SA2Dvector<N, N2, N2l> operator^ (const SA2Dvector<N, N2, N2l>& veca, const SA2Dvector<N, N2, N2l>& vecb)
    {
        SA2Dvector<N, N2, N2l> vec;
        vec.x = veca.y * vecb.z - veca.z * vecb.y;
        vec.y = veca.z * vecb.x - veca.x * vecb.z;
        vec.z = veca.x * vecb.y - veca.y * vecb.x;
        return vec;
    }
    friend __device__ __host__ __forceinline__ SA2Dvector<N, N2, N2l> Get_Periodic_Displacement(const SA2Dvector<N, N2, N2l> vec_a, const SA2Dvector<N, N2, N2l> vec_b, const SA2Dvector<N, N2, N2l> box_length)
    {
        SA2Dvector<N, N2, N2l> dr;
        dr = vec_a - vec_b;
        dr.x.val = dr.x.val - floorf(dr.x.val / box_length.x.val + 0.5) * box_length.x.val;
        dr.y.val = dr.y.val - floorf(dr.y.val / box_length.y.val + 0.5) * box_length.y.val;
        dr.z.val = dr.z.val - floorf(dr.z.val / box_length.z.val + 0.5) * box_length.z.val;
        foreach_2nd_create(dr.x.ddval[index + j] = (dr.x.dval[j] * box_length.x.dval[i] - dr.x.val / box_length.x.val * box_length.x.dval[i] * box_length.x.dval[j]) / box_length.x.val;
        dr.y.ddval[index + j] = (dr.y.dval[j] * box_length.y.dval[i] - dr.y.val / box_length.y.val * box_length.y.dval[i] * box_length.y.dval[j]) / box_length.y.val;
        dr.z.ddval[index + j] = (dr.z.dval[j] * box_length.z.dval[i] - dr.z.val / box_length.z.val * box_length.z.dval[i] * box_length.z.dval[j]) / box_length.z.val, 0)
            for (int i = 0; i < N; i++)
            {
                dr.x.dval[i] = dr.x.val / box_length.x.val * box_length.x.dval[i] + dr.x.dval[i];
                dr.y.dval[i] = dr.y.val / box_length.y.val * box_length.y.dval[i] + dr.y.dval[i];
                dr.z.dval[i] = dr.z.val / box_length.z.val * box_length.z.dval[i] + dr.z.dval[i];
            }
        return dr;
    }
};

)JIT";
