#pragma once

#include <initializer_list>
#include <map>
#include <string>
#include <vector>

// ECP 用户选项 ("auto" 按基组自动选、"none" 禁用、显式指定集合)
enum class QC_ECP_TYPE
{
    AUTO,
    NONE,
    DEF2_ECP,
    LANL2DZ,
};

// ECP 单项: d_k · r^(n_k-2) · exp(-ζ_k · r²)
struct QC_ECP_TERM
{
    float d_k;     // 收缩系数
    int n_k;       // r 幂次 (实际幂为 n_k - 2)
    float zeta_k;  // 高斯指数
};

// ECP 角动量通道
struct QC_ECP_CHANNEL
{
    int l;  // 角动量 (-1 表示 local / ul)
    std::vector<QC_ECP_TERM> terms;
};

// 单个原子的 ECP 参数
struct QC_ECP_ATOM_DATA
{
    int n_core = 0;  // 被替代的核心电子数
    int l_max = -1;  // 最大角动量 (local channel 的 l 值)
    std::vector<QC_ECP_CHANNEL> channels;
    // channels[l_max] = local (U_L)
    // channels[0..l_max-1] = semi-local (U_l)
};

// ECP 参数集 (类比 QC_BASIS_SET)
using QC_ECP_MAP = std::map<std::string, QC_ECP_ATOM_DATA>;

struct QC_ECP_SET
{
    virtual ~QC_ECP_SET() = default;
    virtual void Initialize() = 0;
    const char* name = "";
    QC_ECP_MAP data;
};

// Helper to build ECP channels from inline data
inline QC_ECP_CHANNEL make_channel(int l,
                                   std::initializer_list<QC_ECP_TERM> terms)
{
    QC_ECP_CHANNEL ch;
    ch.l = l;
    ch.terms.assign(terms.begin(), terms.end());
    return ch;
}
