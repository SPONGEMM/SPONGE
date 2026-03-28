#include "minao.h"

#include "../quantum_chemistry.h"

// H-Ar 基态电子构型：每个子壳层的角动量和电子数
// 子壳层按 1s, 2s, 2p, 3s, 3p 排列
static const int MAX_SUBSHELLS = 5;

struct AtomConfig
{
    int n_subshells;
    int l[MAX_SUBSHELLS];
    int occ[MAX_SUBSHELLS];
};

// clang-format off
static const AtomConfig ATOM_CONFIGS[] = {
    {0, {}, {}},                             // Z=0  dummy
    {1, {0},             {1}},               // Z=1  H
    {1, {0},             {2}},               // Z=2  He
    {2, {0, 0},          {2, 1}},            // Z=3  Li
    {2, {0, 0},          {2, 2}},            // Z=4  Be
    {3, {0, 0, 1},       {2, 2, 1}},         // Z=5  B
    {3, {0, 0, 1},       {2, 2, 2}},         // Z=6  C
    {3, {0, 0, 1},       {2, 2, 3}},         // Z=7  N
    {3, {0, 0, 1},       {2, 2, 4}},         // Z=8  O
    {3, {0, 0, 1},       {2, 2, 5}},         // Z=9  F
    {3, {0, 0, 1},       {2, 2, 6}},         // Z=10 Ne
    {4, {0, 0, 1, 0},    {2, 2, 6, 1}},      // Z=11 Na
    {4, {0, 0, 1, 0},    {2, 2, 6, 2}},      // Z=12 Mg
    {5, {0, 0, 1, 0, 1}, {2, 2, 6, 2, 1}},   // Z=13 Al
    {5, {0, 0, 1, 0, 1}, {2, 2, 6, 2, 2}},   // Z=14 Si
    {5, {0, 0, 1, 0, 1}, {2, 2, 6, 2, 3}},   // Z=15 P
    {5, {0, 0, 1, 0, 1}, {2, 2, 6, 2, 4}},   // Z=16 S
    {5, {0, 0, 1, 0, 1}, {2, 2, 6, 2, 5}},   // Z=17 Cl
    {5, {0, 0, 1, 0, 1}, {2, 2, 6, 2, 6}},   // Z=18 Ar
};
// clang-format on

static const int MAX_SUPPORTED_Z =
    (int)(sizeof(ATOM_CONFIGS) / sizeof(ATOM_CONFIGS[0])) - 1;

// 设备核函数：将 AO 占据数写入密度矩阵对角线
static __global__ void QC_Set_Diagonal_Kernel(const int count, const int nao,
                                              const int* ao_indices,
                                              const float* ao_occs, float* P)
{
    SIMPLE_DEVICE_FOR(i, count)
    {
        P[ao_indices[i] * nao + ao_indices[i]] = ao_occs[i];
    }
}

void QC_Build_Minao_Guess(const QC_MOLECULE& mol,
                          const QC_SCF_Runtime_State& runtime, float* d_P,
                          float* d_P_beta)
{
    const int nao = mol.nao;
    const bool unrestricted = runtime.unrestricted;
    const bool is_spherical = mol.is_spherical;

    // 按原子分组壳层
    std::vector<std::vector<int>> atom_shells(mol.natm);
    for (int ish = 0; ish < mol.nbas; ish++)
    {
        int iatm = mol.h_bas[ish * 8 + 0];
        atom_shells[iatm].push_back(ish);
    }

    // 在主机侧计算每个 AO 的占据数
    std::vector<int> h_ao_idx_alpha, h_ao_idx_beta;
    std::vector<float> h_occ_alpha, h_occ_beta;

    for (int iatm = 0; iatm < mol.natm; iatm++)
    {
        int Z = mol.h_Z[iatm];
        if (Z <= 0 || Z > MAX_SUPPORTED_Z) continue;
        const AtomConfig& config = ATOM_CONFIGS[Z];

        // 按角动量统计该原子在当前基组中的 AO 总数
        const int l_max = 3;
        int nao_per_l[4] = {0, 0, 0, 0};
        for (int ish : atom_shells[iatm])
        {
            int l = mol.h_l_list[ish];
            if (l > l_max) continue;
            nao_per_l[l] +=
                is_spherical ? (2 * l + 1) : ((l + 1) * (l + 2) / 2);
        }

        // 按角动量汇总原子构型中的电子数
        int elec_per_l[4] = {0, 0, 0, 0};
        for (int s = 0; s < config.n_subshells; s++)
        {
            int l = config.l[s];
            if (l <= l_max) elec_per_l[l] += config.occ[s];
        }

        // 对每个壳层的每个 AO 均匀分配
        for (int ish : atom_shells[iatm])
        {
            int l = mol.h_l_list[ish];
            if (l > l_max || nao_per_l[l] == 0) continue;
            int nao_shell =
                is_spherical ? (2 * l + 1) : ((l + 1) * (l + 2) / 2);
            int ao_start = is_spherical ? mol.h_ao_offsets_sph[ish]
                                        : mol.h_ao_offsets[ish];
            float total_occ = (float)elec_per_l[l] / (float)nao_per_l[l];

            for (int k = 0; k < nao_shell; k++)
            {
                int ao = ao_start + k;
                if (!unrestricted)
                {
                    h_ao_idx_alpha.push_back(ao);
                    h_occ_alpha.push_back(total_occ / runtime.occ_factor);
                }
                else
                {
                    float a_occ = (total_occ + 1.0f) * 0.5f;
                    if (a_occ > 1.0f) a_occ = 1.0f;
                    float b_occ = total_occ - a_occ;
                    if (b_occ < 0.0f) b_occ = 0.0f;
                    h_ao_idx_alpha.push_back(ao);
                    h_occ_alpha.push_back(a_occ);
                    h_ao_idx_beta.push_back(ao);
                    h_occ_beta.push_back(b_occ);
                }
            }
        }
    }

    // 用设备核函数写入密度矩阵对角线
    const int threads = 256;

    if (!h_ao_idx_alpha.empty())
    {
        const int n = (int)h_ao_idx_alpha.size();
        int* d_idx = nullptr;
        float* d_occ = nullptr;
        Device_Malloc_Safely((void**)&d_idx, sizeof(int) * n);
        Device_Malloc_Safely((void**)&d_occ, sizeof(float) * n);
        deviceMemcpy(d_idx, h_ao_idx_alpha.data(), sizeof(int) * n,
                     deviceMemcpyHostToDevice);
        deviceMemcpy(d_occ, h_occ_alpha.data(), sizeof(float) * n,
                     deviceMemcpyHostToDevice);
        Launch_Device_Kernel(QC_Set_Diagonal_Kernel,
                             (n + threads - 1) / threads, threads, 0, 0, n, nao,
                             d_idx, d_occ, d_P);
        deviceFree(d_idx);
        deviceFree(d_occ);
    }

    if (unrestricted && d_P_beta && !h_ao_idx_beta.empty())
    {
        const int n = (int)h_ao_idx_beta.size();
        int* d_idx = nullptr;
        float* d_occ = nullptr;
        Device_Malloc_Safely((void**)&d_idx, sizeof(int) * n);
        Device_Malloc_Safely((void**)&d_occ, sizeof(float) * n);
        deviceMemcpy(d_idx, h_ao_idx_beta.data(), sizeof(int) * n,
                     deviceMemcpyHostToDevice);
        deviceMemcpy(d_occ, h_occ_beta.data(), sizeof(float) * n,
                     deviceMemcpyHostToDevice);
        Launch_Device_Kernel(QC_Set_Diagonal_Kernel,
                             (n + threads - 1) / threads, threads, 0, 0, n, nao,
                             d_idx, d_occ, d_P_beta);
        deviceFree(d_idx);
        deviceFree(d_occ);
    }
}
