#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace SpongeH5MD
{
namespace CanonicalHashDetail
{
class Sha256
{
   public:
    void Update(const void* data, std::size_t size)
    {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        total_size_ += size;
        while (size > 0)
        {
            const std::size_t count =
                std::min(size, block_.size() - block_size_);
            std::copy_n(bytes, count, block_.begin() + block_size_);
            block_size_ += count;
            bytes += count;
            size -= count;
            if (block_size_ == block_.size())
            {
                Transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    void Update(const std::string& value)
    {
        Update(value.data(), value.size());
    }

    std::string Hex_Digest() const
    {
        Sha256 copy = *this;
        const std::uint64_t bit_size =
            static_cast<std::uint64_t>(copy.total_size_) * 8U;
        const std::uint8_t marker = 0x80;
        copy.Update(&marker, 1);
        const std::uint8_t zero = 0;
        while (copy.block_size_ != 56) copy.Update(&zero, 1);
        std::array<std::uint8_t, 8> encoded_size{};
        for (std::size_t index = 0; index < encoded_size.size(); ++index)
        {
            encoded_size[index] =
                static_cast<std::uint8_t>(bit_size >> (56U - index * 8U));
        }
        copy.Update(encoded_size.data(), encoded_size.size());

        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const std::uint32_t word : copy.state_)
        {
            output << std::setw(8) << word;
        }
        return output.str();
    }

   private:
    static std::uint32_t Rotate_Right(std::uint32_t value, std::uint32_t count)
    {
        return (value >> count) | (value << (32U - count));
    }

    void Transform(const std::uint8_t* block)
    {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index)
        {
            const std::size_t offset = index * 4;
            words[index] =
                (static_cast<std::uint32_t>(block[offset]) << 24U) |
                (static_cast<std::uint32_t>(block[offset + 1]) << 16U) |
                (static_cast<std::uint32_t>(block[offset + 2]) << 8U) |
                static_cast<std::uint32_t>(block[offset + 3]);
        }
        for (std::size_t index = 16; index < words.size(); ++index)
        {
            const std::uint32_t s0 = Rotate_Right(words[index - 15], 7) ^
                                     Rotate_Right(words[index - 15], 18) ^
                                     (words[index - 15] >> 3U);
            const std::uint32_t s1 = Rotate_Right(words[index - 2], 17) ^
                                     Rotate_Right(words[index - 2], 19) ^
                                     (words[index - 2] >> 10U);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t index = 0; index < words.size(); ++index)
        {
            const std::uint32_t sum1 =
                Rotate_Right(e, 6) ^ Rotate_Right(e, 11) ^ Rotate_Right(e, 25);
            const std::uint32_t choice = (e & f) ^ (~e & g);
            const std::uint32_t temporary1 =
                h + sum1 + choice + constants_[index] + words[index];
            const std::uint32_t sum0 =
                Rotate_Right(a, 2) ^ Rotate_Right(a, 13) ^ Rotate_Right(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    inline static constexpr std::array<std::uint32_t, 64> constants_{{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
        0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
        0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
        0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
        0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
        0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
        0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
        0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
        0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    }};
    std::array<std::uint32_t, 8> state_{{
        0x6a09e667U,
        0xbb67ae85U,
        0x3c6ef372U,
        0xa54ff53aU,
        0x510e527fU,
        0x9b05688cU,
        0x1f83d9abU,
        0x5be0cd19U,
    }};
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_ = 0;
    std::size_t total_size_ = 0;
};

inline std::string Shape_Text(const std::vector<std::size_t>& dimensions)
{
    if (dimensions.empty()) return "()";
    std::ostringstream output;
    output << '(';
    for (std::size_t index = 0; index < dimensions.size(); ++index)
    {
        if (index != 0) output << ", ";
        output << dimensions[index];
    }
    if (dimensions.size() == 1) output << ',';
    output << ')';
    return output.str();
}
}  // namespace CanonicalHashDetail

class CanonicalDatasetHash
{
   public:
    template <typename T>
    void Add_Numeric(const std::string& path,
                     const std::vector<std::size_t>& dimensions,
                     const T* values, std::size_t count)
    {
        static_assert(std::is_arithmetic<T>::value,
                      "numeric dataset type required");
        Dataset dataset;
        dataset.dtype = Dtype_Name<T>();
        dataset.dimensions = dimensions;
        dataset.bytes.resize(count * sizeof(T));
        if (count != 0 && values != nullptr)
        {
            std::memcpy(dataset.bytes.data(), values, dataset.bytes.size());
        }
        datasets_[path] = std::move(dataset);
    }

    template <typename T>
    void Add_Scalar(const std::string& path, const T& value)
    {
        Add_Numeric(path, {}, &value, 1);
    }

    void Add_String(const std::string& path, const std::string& value)
    {
        Dataset dataset;
        dataset.dtype = "object";
        dataset.bytes.assign(value.begin(), value.end());
        datasets_[path] = std::move(dataset);
    }

    void Add_Strings(const std::string& path,
                     const std::vector<std::size_t>& dimensions,
                     const std::vector<std::string>& values)
    {
        Dataset dataset;
        dataset.dtype = "object";
        dataset.dimensions = dimensions;
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            if (index != 0) dataset.bytes.push_back(0);
            dataset.bytes.insert(dataset.bytes.end(), values[index].begin(),
                                 values[index].end());
        }
        datasets_[path] = std::move(dataset);
    }

    std::string Digest(const std::string& bundle_file) const
    {
        CanonicalHashDetail::Sha256 digest;
        digest.Update(bundle_file);
        const std::uint8_t separator = 0;
        for (const auto& entry : datasets_)
        {
            digest.Update(&separator, 1);
            digest.Update(entry.first);
            digest.Update(&separator, 1);
            digest.Update(entry.second.dtype);
            digest.Update(&separator, 1);
            digest.Update(
                CanonicalHashDetail::Shape_Text(entry.second.dimensions));
            digest.Update(&separator, 1);
            if (!entry.second.bytes.empty())
            {
                digest.Update(entry.second.bytes.data(),
                              entry.second.bytes.size());
            }
        }
        return "sha256:" + digest.Hex_Digest();
    }

   private:
    struct Dataset
    {
        std::string dtype;
        std::vector<std::size_t> dimensions;
        std::vector<std::uint8_t> bytes;
    };

    template <typename T>
    static const char* Dtype_Name();

    std::map<std::string, Dataset> datasets_;
};

template <>
inline const char* CanonicalDatasetHash::Dtype_Name<float>()
{
    return "float32";
}
template <>
inline const char* CanonicalDatasetHash::Dtype_Name<double>()
{
    return "float64";
}
template <>
inline const char* CanonicalDatasetHash::Dtype_Name<std::int32_t>()
{
    return "int32";
}
template <>
inline const char* CanonicalDatasetHash::Dtype_Name<std::uint32_t>()
{
    return "uint32";
}
template <>
inline const char* CanonicalDatasetHash::Dtype_Name<std::int64_t>()
{
    return "int64";
}
template <>
inline const char* CanonicalDatasetHash::Dtype_Name<std::uint64_t>()
{
    return "uint64";
}
template <>
inline const char* CanonicalDatasetHash::Dtype_Name<std::int8_t>()
{
    return "int8";
}
template <>
inline const char* CanonicalDatasetHash::Dtype_Name<std::uint8_t>()
{
    return "uint8";
}
}  // namespace SpongeH5MD
