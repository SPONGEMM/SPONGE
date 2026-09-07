#ifndef SPONGE_COMMON_VECTOR_HPP
#define SPONGE_COMMON_VECTOR_HPP

// 用于计算边界循环所定义的结构体
struct UNSIGNED_INT_VECTOR
{
    unsigned int uint_x;
    unsigned int uint_y;
    unsigned int uint_z;
};

// 用于计算边界循环或者一些三维数组大小所定义的结构体
struct INT_VECTOR
{
    int int_x;
    int int_y;
    int int_z;
};

__host__ __device__ __forceinline__ float BSpline_4_1(float x);
__host__ __device__ __forceinline__ float BSpline_4_2(float x);
__host__ __device__ __forceinline__ float BSpline_4_3(float x);
__host__ __device__ __forceinline__ float BSpline_4_4(float x);
__host__ __device__ __forceinline__ float dBSpline_4_1(float x);
__host__ __device__ __forceinline__ float dBSpline_4_2(float x);
__host__ __device__ __forceinline__ float dBSpline_4_3(float x);
__host__ __device__ __forceinline__ float dBSpline_4_4(float x);

// 用于存储各种三维float矢量而定义的结构体
struct VECTOR
{
    float x;
    float y;
    float z;

    __host__ __device__ __forceinline__ VECTOR() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ __forceinline__ explicit VECTOR(float v)
        : x(v), y(v), z(v)
    {
    }  // 单值构造: x=y=z=v
    __host__ __device__ __forceinline__ VECTOR(float x_, float y_, float z_)
        : x(x_), y(y_), z(z_)
    {
    }

    __host__ __device__ __forceinline__ VECTOR& operator=(float v)
    {
        x = v;
        y = v;
        z = v;
        return *this;
    }

    friend __device__ __host__ __forceinline__ VECTOR
    wiseproduct(const VECTOR& veca, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = veca.x * vecb.x;
        vec.y = veca.y * vecb.y;
        vec.z = veca.z * vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR
    wisediv(const VECTOR& veca, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = veca.x / vecb.x;
        vec.y = veca.y / vecb.y;
        vec.z = veca.z / vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR
    operator+(const VECTOR& veca, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = veca.x + vecb.x;
        vec.y = veca.y + vecb.y;
        vec.z = veca.z + vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR
    operator+(const VECTOR& veca, const float& b)
    {
        VECTOR vec;
        vec.x = veca.x + b;
        vec.y = veca.y + b;
        vec.z = veca.z + b;
        return vec;
    }

    friend __device__ __host__ __forceinline__ float operator*(
        const VECTOR& veca, const VECTOR& vecb)
    {
        return veca.x * vecb.x + veca.y * vecb.y + veca.z * vecb.z;
    }
    friend __device__ __host__ __forceinline__ VECTOR
    operator*(const float& a, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = a * vecb.x;
        vec.y = a * vecb.y;
        vec.z = a * vecb.z;
        return vec;
    }
    friend __device__ __host__ __forceinline__ VECTOR
    operator-(const VECTOR& veca, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = veca.x - vecb.x;
        vec.y = veca.y - vecb.y;
        vec.z = veca.z - vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR
    operator-(const VECTOR& veca, const float& b)
    {
        VECTOR vec;
        vec.x = veca.x - b;
        vec.y = veca.y - b;
        vec.z = veca.z - b;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR
    operator-(const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = -vecb.x;
        vec.y = -vecb.y;
        vec.z = -vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR
    operator/(const VECTOR& veca, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = veca.x / vecb.x;
        vec.y = veca.y / vecb.y;
        vec.z = veca.z / vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR
    operator/(const float& a, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = a / vecb.x;
        vec.y = a / vecb.y;
        vec.z = a / vecb.z;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR
    operator^(const VECTOR& veca, const VECTOR& vecb)
    {
        VECTOR vec;
        vec.x = veca.y * vecb.z - veca.z * vecb.y;
        vec.y = veca.z * vecb.x - veca.x * vecb.z;
        vec.z = veca.x * vecb.y - veca.y * vecb.x;
        return vec;
    }

    friend __device__ __host__ __forceinline__ VECTOR
    Get_Mesh_Index_Displacement(const UNSIGNED_INT_VECTOR uvec_a,
                                const UNSIGNED_INT_VECTOR uvec_b,
                                const VECTOR scaler)
    {
        VECTOR dr;
        dr.x = ((int)(uvec_a.uint_x - uvec_b.uint_x)) * scaler.x;
        dr.y = ((int)(uvec_a.uint_y - uvec_b.uint_y)) * scaler.y;
        dr.z = ((int)(uvec_a.uint_z - uvec_b.uint_z)) * scaler.z;
        return dr;
    }

    friend __device__ __host__ __forceinline__ VECTOR Get_Displacement(
        const VECTOR vec_a, const VECTOR vec_b, const VECTOR box_length)
    {
        VECTOR dr;
        dr = vec_a - vec_b;
        dr.x = dr.x - floorf(dr.x / box_length.x + 0.5f) * box_length.x;
        dr.y = dr.y - floorf(dr.y / box_length.y + 0.5f) * box_length.y;
        dr.z = dr.z - floorf(dr.z / box_length.z + 0.5f) * box_length.z;
        return dr;
    }

    friend __device__ __host__ __forceinline__ VECTOR
    Get_Displacement(const VECTOR vec_a, const VECTOR vec_b,
                     const VECTOR box_length, const VECTOR box_length_inverse)
    {
        VECTOR dr;
        dr = vec_a - vec_b;
        dr.x = dr.x - floorf(dr.x * box_length_inverse.x + 0.5f) * box_length.x;
        dr.y = dr.y - floorf(dr.y * box_length_inverse.y + 0.5f) * box_length.y;
        dr.z = dr.z - floorf(dr.z * box_length_inverse.z + 0.5f) * box_length.z;
        return dr;
    }

    friend __host__ __device__ __forceinline__ VECTOR floorf(VECTOR v)
    {
        return {floorf(v.x), floorf(v.y), floorf(v.z)};
    }

    friend __device__ __forceinline__ VECTOR
    Make_Vector_Not_Exceed_Value(VECTOR vector, const float value)
    {
        return fminf(1.0, value * rnorm3df(vector.x, vector.y, vector.z)) *
               vector;
    }

    friend __device__ __forceinline__ void atomicAdd(VECTOR* a, const VECTOR b)
    {
        atomicAdd(&a->x, b.x);
        atomicAdd(&a->y, b.y);
        atomicAdd(&a->z, b.z);
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
        return -0.16666667f * x * x * x + 0.5f * x * x - 0.5f * x + 0.16666667f;
    }
    friend __host__ __device__ __forceinline__ float dBSpline_4_1(float x)
    {
        return -0.5f * x * x;
    }
    friend __host__ __device__ __forceinline__ float dBSpline_4_2(float x)
    {
        return 1.5f * x * x - x - 0.5f;
    }
    friend __host__ __device__ __forceinline__ float dBSpline_4_3(float x)
    {
        return -1.5f * x * x + 2.0f * x;
    }
    friend __host__ __device__ __forceinline__ float dBSpline_4_4(float x)
    {
        return 0.5f * x * x - x + 0.5f;
    }
};

// 默认的所有VECTOR都是行矢量
// 三阶下三角矩阵，用于存储盒子信息
struct LTMatrix3
{
    float a11, a21, a22, a31, a32, a33;

    __host__ __device__ __forceinline__ LTMatrix3()
        : a11(0.0f), a21(0.0f), a22(0.0f), a31(0.0f), a32(0.0f), a33(0.0f)
    {
    }
    __host__ __device__ __forceinline__ explicit LTMatrix3(float v)
        : a11(v), a21(v), a22(v), a31(v), a32(v), a33(v)
    {
    }  // 单值构造: 所有元素都等于v
    __host__ __device__ __forceinline__ LTMatrix3(float a11_, float a21_,
                                                  float a22_, float a31_,
                                                  float a32_, float a33_)
        : a11(a11_), a21(a21_), a22(a22_), a31(a31_), a32(a32_), a33(a33_)
    {
    }

    __host__ __device__ __forceinline__ LTMatrix3& operator=(float v)
    {
        a11 = a21 = a22 = a31 = a32 = a33 = v;
        return *this;
    }

    friend __host__ __device__ __forceinline__ LTMatrix3 operator+(LTMatrix3 m1,
                                                                   LTMatrix3 m2)
    {
        return {m1.a11 + m2.a11, m1.a21 + m2.a21, m1.a22 + m2.a22,
                m1.a31 + m2.a31, m1.a32 + m2.a32, m1.a33 + m2.a33};
    }
    friend __host__ __device__ __forceinline__ LTMatrix3 operator-(LTMatrix3 m1,
                                                                   LTMatrix3 m2)
    {
        return {m1.a11 - m2.a11, m1.a21 - m2.a21, m1.a22 - m2.a22,
                m1.a31 - m2.a31, m1.a32 - m2.a32, m1.a33 - m2.a33};
    }
    friend __host__ __device__ __forceinline__ LTMatrix3 operator*(float m1,
                                                                   LTMatrix3 m2)
    {
        return {m1 * m2.a11, m1 * m2.a21, m1 * m2.a22,
                m1 * m2.a31, m1 * m2.a32, m1 * m2.a33};
    }
    friend __host__ __device__ __forceinline__ LTMatrix3 operator*(LTMatrix3 m1,
                                                                   LTMatrix3 m2)
    {
        return {m1.a11 * m2.a11,
                m1.a21 * m2.a11 + m1.a22 * m2.a21,
                m1.a22 * m2.a22,
                m1.a31 * m2.a11 + m1.a32 * m2.a21 + m1.a33 * m2.a31,
                m1.a32 * m2.a22 + m1.a33 * m2.a32,
                m1.a33 * m2.a33};
    }

    friend __host__ __device__ __forceinline__ VECTOR
    MultiplyTranspose(VECTOR vec, LTMatrix3 mat)
    {
        return {vec.x * mat.a11, vec.x * mat.a21 + vec.y * mat.a22,
                vec.x * mat.a31 + vec.y * mat.a32 + vec.z * mat.a33};
    }
    friend __host__ __device__ __forceinline__ VECTOR operator*(VECTOR vec,
                                                                LTMatrix3 mat)
    {
        return {vec.x * mat.a11 + vec.y * mat.a21 + vec.z * mat.a31,
                vec.y * mat.a22 + vec.z * mat.a32, vec.z * mat.a33};
    }
    friend __device__ __host__ __forceinline__ LTMatrix3
    Get_Virial_From_Force_Dis(const VECTOR& veca, const VECTOR& vecb)
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
    friend __device__ __host__ __forceinline__ LTMatrix3
    Get_Virial_From_Force_Dis(const VECTOR& veca)
    {
        LTMatrix3 mat;
        mat.a11 = veca.x * veca.x;
        mat.a21 = veca.x * veca.y + veca.y * veca.x;
        mat.a22 = veca.y * veca.y;
        mat.a31 = veca.x * veca.z + veca.z * veca.x;
        mat.a32 = veca.y * veca.z + veca.z * veca.y;
        mat.a33 = veca.z * veca.z;
        return mat;
    }
    friend __device__ __forceinline__ void atomicAdd(LTMatrix3* a, LTMatrix3 L)
    {
        atomicAdd(&a->a11, L.a11);
        atomicAdd(&a->a21, L.a21);
        atomicAdd(&a->a22, L.a22);
        atomicAdd(&a->a31, L.a31);
        atomicAdd(&a->a32, L.a32);
        atomicAdd(&a->a33, L.a33);
    }
    friend __device__ __host__ __forceinline__ LTMatrix3 inv(LTMatrix3 mat)
    {
        LTMatrix3 invmat;
        invmat.a33 = 1.0f / mat.a33;
        invmat.a32 = -mat.a32 * invmat.a33 / mat.a22;
        invmat.a31 = (mat.a32 * mat.a21 * invmat.a33 - mat.a31 * mat.a22) /
                     (mat.a11 * mat.a22 * mat.a33);
        invmat.a22 = 1.0f / mat.a22;
        invmat.a21 = -mat.a21 * invmat.a22 / mat.a11;
        invmat.a11 = 1.0f / mat.a11;
        return invmat;
    }
};

__device__ __host__ __forceinline__ LTMatrix3
Get_Virial_From_Force_Dis(const VECTOR& veca, const VECTOR& vecb);
__device__ __host__ __forceinline__ LTMatrix3
Get_Virial_From_Force_Dis(const VECTOR& veca);

#endif  // SPONGE_COMMON_VECTOR_HPP
