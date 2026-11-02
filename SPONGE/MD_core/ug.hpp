#pragma once

void MD_INFORMATION::update_group_information::Initial_Edge(int atom_numbers)
{
    for (int i = 0; i < atom_numbers; i++)
    {
        std::set<int> s;
        connectivity[i] = s;
    }
}

void MD_INFORMATION::update_group_information::Read_Update_Group(
    int atom_numbers)
{
    int edge_numbers = 0;
    CPP_ATOM_GROUP h_update_groups;
    for (int i = 0; i < atom_numbers; i++)
    {
        edge_numbers += connectivity[i].size();
    }
    edge_numbers *= 2;
    int* first_edge = NULL;  // 每个原子的第一个边（链表的头）
    int* edges = NULL;       // 每个边的序号
    int* edge_next = NULL;   // 每个原子的边（链表结构）
    Malloc_Safely((void**)&first_edge, sizeof(int) * atom_numbers);
    Malloc_Safely((void**)&edges, sizeof(int) * edge_numbers);
    Malloc_Safely((void**)&edge_next, sizeof(int) * edge_numbers);
    // 初始化链表
    for (int i = 0; i < atom_numbers; i++)
    {
        first_edge[i] = -1;
    }
    int atom_i, atom_j, edge_count = 0;
    for (atom_i = 0; atom_i < atom_numbers; atom_i++)
    {
        const std::set<int>& conect_i = connectivity[atom_i];
        for (auto iter = conect_i.begin(); iter != conect_i.end(); iter++)
        {
            atom_j = *iter;
            edge_next[edge_count] = first_edge[atom_i];
            first_edge[atom_i] = edge_count;
            edges[edge_count] = atom_j;
            edge_count++;
        }
    }
    std::deque<int> queue;
    std::vector<int> visited(atom_numbers, 0);
    int atom;
    for (int i = 0; i < atom_numbers; i++)
    {
        if (!visited[i])
        {
            std::vector<int> atoms;
            visited[i] = 1;
            queue.push_back(i);
            while (!queue.empty())
            {
                atom = queue[0];
                atoms.push_back(atom);
                queue.pop_front();
                edge_count = first_edge[atom];
                while (edge_count != -1)
                {
                    atom = edges[edge_count];
                    if (!visited[atom])
                    {
                        queue.push_back(atom);
                        visited[atom] = 1;
                    }
                    edge_count = edge_next[edge_count];
                }
            }
            h_update_groups.push_back(atoms);
        }
    }
    ug_numbers = h_update_groups.size();
    int* atom_index = (int*)malloc(sizeof(int) * atom_numbers);
    edge_count = 0;
    ug = (ATOM_GROUP*)malloc(sizeof(ATOM_GROUP) * ug_numbers);
    for (int j = 0; j < h_update_groups.size(); j++)
    {
        std::vector<int> vec = h_update_groups[j];
        std::sort(vec.begin(), vec.end());
        ug[j].atom_numbers = vec.size();
        ug[j].ghost_numbers = 0;
        ug[j].atom_serial = atom_index + edge_count;
        for (auto i : vec)
        {
            atom_index[edge_count] = i;
            edge_count += 1;
        }
    }
    Copy_UG_To_Device();
}

void MD_INFORMATION::update_group_information::Copy_UG_To_Device()
{
    // 分配设备内存用于存储 ATOM_GROUP 数组
    Device_Malloc_Safely((void**)&d_ug, sizeof(ATOM_GROUP) * ug_numbers);

    // 分配设备内存用于存储所有 atom_serial 数据
    int total_atom_serials = 0;
    for (int i = 0; i < ug_numbers; i++)
    {
        total_atom_serials += ug[i].atom_numbers;
    }
    int* d_atom_serials;
    Device_Malloc_Safely((void**)&d_atom_serials,
                         sizeof(int) * total_atom_serials);

    // 创建临时数组用于存储设备上的 ATOM_GROUP 数据
    ATOM_GROUP* h_ug_tmp = (ATOM_GROUP*)malloc(sizeof(ATOM_GROUP) * ug_numbers);

    int offset = 0;
    for (int i = 0; i < ug_numbers; i++)
    {
        h_ug_tmp[i].atom_numbers = ug[i].atom_numbers;
        h_ug_tmp[i].ghost_numbers = ug[i].ghost_numbers;
        h_ug_tmp[i].atom_serial = d_atom_serials + offset;
        offset += ug[i].atom_numbers;
    }

    // ug[i].atom_serial 指向同一块连续主机内存且按组序排列，一次拷贝即可
    deviceMemcpy(d_atom_serials, ug[0].atom_serial,
                 sizeof(int) * total_atom_serials, deviceMemcpyHostToDevice);

    // 将 ATOM_GROUP 数组复制到设备
    deviceMemcpy(d_ug, h_ug_tmp, sizeof(ATOM_GROUP) * ug_numbers,
                 deviceMemcpyHostToDevice);

    // 释放主机临时内存
    free(h_ug_tmp);
}
