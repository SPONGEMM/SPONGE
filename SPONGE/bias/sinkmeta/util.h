#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace sinkmeta
{

inline int split_sentence(const std::string& line,
                          std::vector<std::string>& words)
{
    words.clear();
    std::istringstream iss(line);
    std::string word;
    while (iss >> word)
    {
        words.push_back(word);
    }
    return static_cast<int>(words.size());
}

inline int split_sentence(const char* line, std::vector<std::string>& words)
{
    if (line == nullptr) return 0;
    return split_sentence(std::string(line), words);
}

}  // namespace sinkmeta
