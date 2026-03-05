value += R"JIT(
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

)JIT";
