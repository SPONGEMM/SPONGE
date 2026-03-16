#pragma once

void MD_INFORMATION::non_bond_information::Initial(CONTROLLER* controller,
                                                   MD_INFORMATION* md_info)
{
    if (controller[0].Command_Exist("skin"))
    {
        controller->Check_Float(
            "skin", "MD_INFORMATION::non_bond_information::Initial");
        skin = atof(controller[0].Command("skin"));
    }
    else
    {
        skin = 2.0;
    }
    controller->printf("    skin set to %.2f Angstrom\n", skin);

    if (controller[0].Command_Exist("cutoff"))
    {
        controller->Check_Float(
            "cutoff", "MD_INFORMATION::non_bond_information::Initial");
        cutoff = atof(controller[0].Command("cutoff"));
    }
    else
    {
        cutoff = 10.0;
    }
    controller->printf("    cutoff set to %.2f Angstrom\n", cutoff);
    /*===========================
    读取排除表相关信息
    ============================*/
    if (!Xponge::system.exclusions.excluded_atoms.empty())
    {
        int atom_numbers = md_info->atom_numbers;
        excluded_atom_numbers = 0;
        controller->printf("    Start reading excluded list from Xponge:\n");
        for (const auto& excluded : Xponge::system.exclusions.excluded_atoms)
        {
            excluded_atom_numbers += (int)excluded.size();
        }

        Malloc_Safely((void**)&h_excluded_list_start,
                      sizeof(int) * atom_numbers);
        Malloc_Safely((void**)&h_excluded_numbers, sizeof(int) * atom_numbers);
        Malloc_Safely((void**)&h_excluded_list,
                      sizeof(int) * excluded_atom_numbers * 2);
        int count = 0;
        for (int i = 0; i < atom_numbers; i++)
        {
            h_excluded_list_start[i] = count;
            h_excluded_numbers[i] =
                (int)Xponge::system.exclusions.excluded_atoms[i].size();
            for (int excluded_atom :
                 Xponge::system.exclusions.excluded_atoms[i])
            {
                h_excluded_list[count] = excluded_atom;
                count++;
            }
        }
        Device_Malloc_And_Copy_Safely((void**)&d_excluded_list_start,
                                      h_excluded_list_start,
                                      sizeof(int) * atom_numbers);
        Device_Malloc_And_Copy_Safely((void**)&d_excluded_numbers,
                                      h_excluded_numbers,
                                      sizeof(int) * atom_numbers);
        Device_Malloc_And_Copy_Safely((void**)&d_excluded_list, h_excluded_list,
                                      sizeof(int) * excluded_atom_numbers * 2);
        controller->printf("    End reading excluded list from Xponge\n\n");
    }
    else
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMissingCommand,
            "MD_INFORMATION::non_bond_information::Initial",
            "Reason:\n\tno exclusion information found in Xponge::system\n");
    }
}

void MD_INFORMATION::non_bond_information::Excluded_List_Reform(
    int atom_numbers)
{
    std::map<int, std::vector<int>> temp;
    int *new_list, *new_list_tail;
    Malloc_Safely((void**)&new_list, sizeof(int) * excluded_atom_numbers * 2);
    excluded_atom_numbers = 0;
    new_list_tail = new_list;
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
    {
        std::vector<int> initializer;
        temp[atom_i] = initializer;
    }
    for (int atom_i = 0; atom_i < atom_numbers; atom_i++)
    {
        int* excluded_list_j = h_excluded_list + h_excluded_list_start[atom_i];
        for (int atom_j : temp[atom_i])
        {
            new_list_tail[0] = atom_j;
            new_list_tail += 1;
        }
        for (int j = 0; j < h_excluded_numbers[atom_i]; j++)
        {
            int atom_j = excluded_list_j[j];
            new_list_tail[0] = atom_j;
            new_list_tail += 1;
            temp[atom_j].push_back(atom_i);
        }
        h_excluded_numbers[atom_i] += temp[atom_i].size();

        h_excluded_list_start[atom_i] = excluded_atom_numbers;

        excluded_atom_numbers += h_excluded_numbers[atom_i];
    }
    // 释放host上的旧排除表
    free(h_excluded_list);
    h_excluded_list = new_list;

#ifndef USE_CPU
    // 释放device上的旧排除表
    if (d_excluded_list != NULL) deviceFree(d_excluded_list);
    if (d_excluded_list_start != NULL) deviceFree(d_excluded_list_start);
    if (d_excluded_numbers != NULL) deviceFree(d_excluded_numbers);
#endif
    // 重新分配device上的排除表
    Device_Malloc_And_Copy_Safely((void**)&d_excluded_list, h_excluded_list,
                                  sizeof(int) * excluded_atom_numbers * 2);
    // 重新分配device上的排除表起点
    Device_Malloc_And_Copy_Safely((void**)&d_excluded_list_start,
                                  h_excluded_list_start,
                                  sizeof(int) * atom_numbers);
    // 重新分配device上的排除表每个原子排除数
    Device_Malloc_And_Copy_Safely((void**)&d_excluded_numbers,
                                  h_excluded_numbers,
                                  sizeof(int) * atom_numbers);
}
