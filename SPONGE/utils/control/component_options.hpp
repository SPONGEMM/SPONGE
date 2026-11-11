#pragma once
#include <cmath>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>

#include "../../control.h"
#include "../float_classification.hpp"
namespace SpongeDynamics
{
inline double Option_Number(CONTROLLER* c, const std::string& key,
                            double fallback)
{
    if (!c->Command_Exist(key.c_str())) return fallback;
    std::istringstream in(c->Command(key.c_str()));
    in.imbue(std::locale::classic());
    double x;
    if (!(in >> x) || !SpongeFloat::Is_Finite(x))
        throw std::runtime_error(key + " must be finite numeric");
    in >> std::ws;
    if (!in.eof()) throw std::runtime_error(key + " has trailing data");
    return x;
}
inline int Option_Integer(CONTROLLER* c, const std::string& key, int fallback,
                          int lo = 1, int hi = 2147483647)
{
    double x = Option_Number(c, key, fallback);
    if (x < lo || x > hi || x != std::floor(x))
        throw std::runtime_error(key + " is outside its integer range");
    return int(x);
}
inline int Option_Axis(CONTROLLER* c, const std::string& key, int fallback)
{
    if (!c->Command_Exist(key.c_str())) return fallback;
    std::string x = c->Command(key.c_str());
    if (x == "x") return 0;
    if (x == "y") return 1;
    if (x == "z") return 2;
    throw std::runtime_error(key + " must be x/y/z");
}
inline bool Option_Bool(CONTROLLER* c, const std::string& key,
                        bool fallback = false)
{
    if (!c->Command_Exist(key.c_str())) return fallback;
    std::string x = c->Command(key.c_str());
    if (x == "true" || x == "1") return true;
    if (x == "false" || x == "0") return false;
    throw std::runtime_error(key + " must be boolean");
}
inline void Component_Keys(CONTROLLER* c, const std::string& prefix,
                           const std::set<std::string>& fields)
{
    if (c->Command_Exist(prefix.c_str()))
        throw std::runtime_error(prefix + " must be a configuration table");
    for (const auto& kv : c->commands)
        if (kv.first.rfind(prefix + "_", 0) == 0 &&
            !fields.count(kv.first.substr(prefix.size() + 1)))
            throw std::runtime_error("unknown " + prefix +
                                     " parameter: " + kv.first);
}
inline void Orthogonal_Box(VECTOR box, VECTOR angles, bool periodic)
{
    if (!periodic || !SpongeFloat::Is_Finite(angles.x) ||
        !SpongeFloat::Is_Finite(angles.y) ||
        !SpongeFloat::Is_Finite(angles.z) || std::abs(angles.x - 90) > 1e-4 ||
        std::abs(angles.y - 90) > 1e-4 || std::abs(angles.z - 90) > 1e-4 ||
        !SpongeFloat::Is_Finite(box.x) || !SpongeFloat::Is_Finite(box.y) ||
        !SpongeFloat::Is_Finite(box.z) || box.x <= 0 || box.y <= 0 ||
        box.z <= 0)
        throw std::runtime_error(
            "component requires a periodic orthogonal fixed box");
}
}  // namespace SpongeDynamics
