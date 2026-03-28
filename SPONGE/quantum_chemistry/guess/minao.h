#ifndef QC_GUESS_MINAO_H
#define QC_GUESS_MINAO_H

#include "../structure/molecule.h"
#include "../structure/scf_workspace.h"

// 基于原子占据数构造 minao 初始密度矩阵
// 对每个原子，按 aufbau 原则将电子分配到各角动量壳层，
// 在密度矩阵的对角位置填入 occ / n_ao_per_l 的均匀占据
void QC_Build_Minao_Guess(const QC_MOLECULE& mol,
                          const QC_SCF_Runtime_State& runtime,
                          float* d_P,
                          float* d_P_beta);

#endif
