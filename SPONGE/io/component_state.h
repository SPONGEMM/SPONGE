#pragma once
#include <map>
#include <string>
namespace SpongeDynamics
{
inline std::map<std::string, std::string> State_Prefix(
    const std::map<std::string, std::string>& s, const std::string& p)
{
    std::map<std::string, std::string> out;
    for (const auto& kv : s) out.emplace(p + kv.first, kv.second);
    return out;
}
inline std::map<std::string, std::string> State_Select(
    const std::map<std::string, std::string>& s, const std::string& p)
{
    std::map<std::string, std::string> out;
    for (const auto& kv : s)
        if (kv.first.rfind(p, 0) == 0)
            out.emplace(kv.first.substr(p.size()), kv.second);
    return out;
}
}  // namespace SpongeDynamics
