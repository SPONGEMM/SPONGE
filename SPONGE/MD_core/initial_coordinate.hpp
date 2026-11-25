#pragma once

#include "../common.h"

// Initialization only: use the represented cell, rather than its rounded FP32
// inverse, to classify points near a face. Preserve coordinates already in the
// primary cell exactly; only subtract an integer lattice translation outside
// it.
static VECTOR Wrap_Initial_Coordinate(VECTOR crd, const LTMatrix3& cell)
{
    const double fz = double(crd.z) / cell.a33;
    const double fy = (double(crd.y) - fz * cell.a32) / cell.a22;
    const double fx =
        (double(crd.x) - fy * cell.a21 - fz * cell.a31) / cell.a11;
    const double nx = std::floor(fx);
    const double ny = std::floor(fy);
    const double nz = std::floor(fz);
    if (nx == 0 && ny == 0 && nz == 0) return crd;
    return {
        float(double(crd.x) - nx * cell.a11 - ny * cell.a21 - nz * cell.a31),
        float(double(crd.y) - ny * cell.a22 - nz * cell.a32),
        float(double(crd.z) - nz * cell.a33)};
}

static void Move_Crd_Nearest_From_Connectivity(
    const CPP_ATOM_GROUP& mol_atoms, const CONECT& connectivity, VECTOR* crd,
    const Boundary boundary, std::vector<int> periodic_molecules)
{
    // 复制原始坐标供最小镜像计算
    int max_atom_idx = -1;
    for (auto& mol : mol_atoms)
    {
        for (int idx : mol)
        {
            if (idx > max_atom_idx) max_atom_idx = idx;
        }
    }
    if (max_atom_idx < 0) return;
    std::vector<VECTOR> crd_orig(max_atom_idx + 1);
    for (int i = 0; i <= max_atom_idx; ++i) crd_orig[i] = crd[i];

    std::vector<int> mark(max_atom_idx + 1, 0);
    std::vector<int> visited(max_atom_idx + 1, 0);
    std::deque<int> queue;

    for (int i = 0; i < mol_atoms.size(); i++)
    {
        auto& atoms = mol_atoms[i];
        if (atoms.empty()) continue;
        for (int idx : atoms)
        {
            mark[idx] = 1;
            visited[idx] = 0;
        }
        // 先把第一个原子映射到主盒子
        int anchor = atoms[0];
        crd[anchor] = Wrap_Initial_Coordinate(crd_orig[anchor], boundary.cell);

        visited[anchor] = 1;
        queue.clear();
        queue.push_back(anchor);
        while (!queue.empty())
        {
            int atom = queue.front();
            queue.pop_front();
            auto itMap = connectivity.find(atom);
            if (itMap == connectivity.end())
            {
                continue;
            }
            for (auto it = itMap->second.begin(); it != itMap->second.end();
                 ++it)
            {
                int nb = *it;
                if (nb < 0 || nb > max_atom_idx || mark[nb] == 0 || visited[nb])
                    continue;
                VECTOR dr = Get_Displacement<BoundaryPolicy::Periodic>(
                    crd_orig[nb], crd_orig[atom], boundary);
                crd[nb] = crd[atom] + dr;
                visited[nb] = 1;
                queue.push_back(nb);
            }
        }
        for (int idx : atoms) mark[idx] = 0;
    }
}
