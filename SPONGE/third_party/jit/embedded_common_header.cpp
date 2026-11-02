#include "embedded_common_header.hpp"

namespace sponge_jit_detail
{
const std::string& Embedded_Common_Header()
{
    static const std::string header = []()
    {
        std::string value;
        value.reserve(65536);
#include "jit.h"
        return value;
    }();
    return header;
}
}  // namespace sponge_jit_detail
