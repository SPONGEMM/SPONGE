value += R"JIT(
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

)JIT";
