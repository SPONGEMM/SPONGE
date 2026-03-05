value += R"JIT(
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

)JIT";
