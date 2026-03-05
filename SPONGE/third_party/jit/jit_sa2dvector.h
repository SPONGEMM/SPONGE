value += R"JIT(
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
