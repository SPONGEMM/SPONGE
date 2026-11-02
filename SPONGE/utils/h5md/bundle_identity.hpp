#pragma once

#include <array>
#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>

namespace SpongeH5MD
{
inline std::string Generate_Uuid_V4()
{
    std::random_device source;
    std::array<std::uint8_t, 16> bytes{};
    for (auto& byte : bytes)
    {
        byte = static_cast<std::uint8_t>(source());
    }
    bytes[6] = static_cast<std::uint8_t>((bytes[6] & 0x0fU) | 0x40U);
    bytes[8] = static_cast<std::uint8_t>((bytes[8] & 0x3fU) | 0x80U);

    std::ostringstream value;
    value << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < bytes.size(); ++i)
    {
        if (i == 4 || i == 6 || i == 8 || i == 10) value << '-';
        value << std::setw(2) << static_cast<unsigned int>(bytes[i]);
    }
    return value.str();
}
}  // namespace SpongeH5MD
