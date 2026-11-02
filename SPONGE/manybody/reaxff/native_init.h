#ifndef REAXFF_NATIVE_INIT_H
#define REAXFF_NATIVE_INIT_H

struct CONTROLLER;
struct REAXFF;

namespace SpongeH5MD
{
struct NativeReaxFFDefinition;
}

void Initial_ReaxFF_From_Native(
    REAXFF* reaxff, CONTROLLER* controller,
    const SpongeH5MD::NativeReaxFFDefinition& definition, int atom_numbers,
    float cutoff, float* cutoff_full, bool* need_full_nl_flag);

#endif
