#pragma once

// Debug kernel: compute ssss ERI via simplified method AND via generic path,
// write both values to a comparison buffer.
// Buffer layout: [task_id*2] = ssss_eri, [task_id*2+1] = generic_eri

static __global__ void QC_Verify_ssss_Kernel(
    const int n_tasks, const QC_ERI_TASK* __restrict__ tasks,
    const int* __restrict__ atm, const int* __restrict__ bas,
    const float* __restrict__ env, const int* __restrict__ ao_offsets_cart,
    const int* __restrict__ ao_offsets_sph, const float* __restrict__ norms,
    const int is_spherical, const float* __restrict__ cart2sph_mat,
    const int nao_sph,
    float* __restrict__ global_hr_pool, int hr_base, int hr_size,
    int shell_buf_size, float prim_screen_tol,
    float* __restrict__ out_cmp)  // size: n_tasks * 2
{
    SIMPLE_DEVICE_FOR(task_id, n_tasks)
    {
        const QC_ERI_TASK t = tasks[task_id];
        const int si = t.x, sj = t.y, sk = t.z, sl = t.w;
        const int sh[4] = {si, sj, sk, sl};

        // --- Method 1: simplified ssss ---
        int np[4], p_exp[4], p_cof[4];
        float R[4][3];
        for (int i = 0; i < 4; i++)
        {
            np[i] = bas[sh[i] * 8 + 2];
            p_exp[i] = bas[sh[i] * 8 + 5];
            p_cof[i] = bas[sh[i] * 8 + 6];
            const int ptr_R = atm[bas[sh[i] * 8 + 0] * 6 + 1];
            R[i][0] = env[ptr_R + 0];
            R[i][1] = env[ptr_R + 1];
            R[i][2] = env[ptr_R + 2];
        }

        const float dx_ab = R[0][0] - R[1][0], dy_ab = R[0][1] - R[1][1],
                    dz_ab = R[0][2] - R[1][2];
        const float rab2 = dx_ab*dx_ab + dy_ab*dy_ab + dz_ab*dz_ab;
        const float dx_cd = R[2][0] - R[3][0], dy_cd = R[2][1] - R[3][1],
                    dz_cd = R[2][2] - R[3][2];
        const float rcd2 = dx_cd*dx_cd + dy_cd*dy_cd + dz_cd*dz_cd;

        float ssss_eri = 0.0f;
        for (int ip = 0; ip < np[0]; ip++)
        {
            float ai = env[p_exp[0]+ip], ci = env[p_cof[0]+ip];
            for (int jp = 0; jp < np[1]; jp++)
            {
                float aj = env[p_exp[1]+jp], cj = env[p_cof[1]+jp];
                float p_val = ai + aj;
                float inv_p = 1.0f / p_val;
                float kab = expf(-(ai*aj*inv_p)*rab2);
                float n_ab = ci * cj * kab;
                if (fabsf(n_ab) < prim_screen_tol) continue;
                float Px = (ai*R[0][0]+aj*R[1][0])*inv_p;
                float Py = (ai*R[0][1]+aj*R[1][1])*inv_p;
                float Pz = (ai*R[0][2]+aj*R[1][2])*inv_p;
                for (int kp = 0; kp < np[2]; kp++)
                {
                    float ak = env[p_exp[2]+kp], ck = env[p_cof[2]+kp];
                    for (int lp = 0; lp < np[3]; lp++)
                    {
                        float al = env[p_exp[3]+lp], cl = env[p_cof[3]+lp];
                        float q_val = ak + al;
                        float inv_q = 1.0f / q_val;
                        float kcd = expf(-(ak*al*inv_q)*rcd2);
                        float pref = 2.0f * PI_25 / (p_val*q_val*sqrtf(p_val+q_val));
                        float n_abcd = n_ab * ck * cl * kcd * pref;
                        if (fabsf(n_abcd) < prim_screen_tol) continue;
                        float Qx = (ak*R[2][0]+al*R[3][0])*inv_q;
                        float Qy = (ak*R[2][1]+al*R[3][1])*inv_q;
                        float Qz = (ak*R[2][2]+al*R[3][2])*inv_q;
                        float alpha = p_val*q_val/(p_val+q_val);
                        float dpx=Px-Qx, dpy=Py-Qy, dpz=Pz-Qz;
                        float T = alpha*(dpx*dpx+dpy*dpy+dpz*dpz);
                        double F_d[1];
                        compute_boys_double(F_d, T, 0);
                        ssss_eri += n_abcd * (float)F_d[0];
                    }
                }
            }
        }

        // Apply norms
        const int ao_p = is_spherical ? ao_offsets_sph[si] : ao_offsets_cart[si];
        const int ao_q = is_spherical ? ao_offsets_sph[sj] : ao_offsets_cart[sj];
        const int ao_r = is_spherical ? ao_offsets_sph[sk] : ao_offsets_cart[sk];
        const int ao_s = is_spherical ? ao_offsets_sph[sl] : ao_offsets_cart[sl];
        ssss_eri *= norms[ao_p] * norms[ao_q] * norms[ao_r] * norms[ao_s];

        // --- Method 2: generic path ---
#ifdef GPU_ARCH_NAME
        const int scratch_id = task_id;
#else
        const int scratch_id = omp_get_thread_num();
#endif
        float* task_pool = global_hr_pool +
                           (int)scratch_id * (hr_size + 2 * shell_buf_size);
        float* HR = task_pool;
        float* shell_eri = task_pool + hr_size;
        float* shell_tmp = shell_eri + shell_buf_size;
        int dims_eff[4], off_eff[4];
        int sh_arr[4] = {si, sj, sk, sl};
        float generic_eri = 0.0f;
        if (QC_Compute_Shell_Quartet_ERI_Buffer(
                sh_arr, atm, bas, env, ao_offsets_cart, ao_offsets_sph, norms,
                is_spherical, cart2sph_mat, nao_sph, HR, shell_eri, shell_tmp,
                hr_base, shell_buf_size, prim_screen_tol, dims_eff, off_eff))
        {
            generic_eri = shell_eri[0];
        }

        out_cmp[task_id * 2 + 0] = ssss_eri;
        out_cmp[task_id * 2 + 1] = generic_eri;
    }
}
