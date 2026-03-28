#ifndef QC_STRUCTURE_METHOD_H
#define QC_STRUCTURE_METHOD_H

#include "../../common.h"

enum class QC_METHOD
{
    HF = 0,
    LDA,
    PBE,
    BLYP,
    PBE0,
    B3LYP
};

enum class QC_INITIAL_GUESS
{
    NONE = 0,
    MINAO,
    SAP
};

#endif
