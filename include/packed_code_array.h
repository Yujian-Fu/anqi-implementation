#pragma once

#include <cstddef>
#include <cstdint>

namespace anqi {

inline bool supported_packed_code_bits(int bits) {
    return bits == 2 || bits == 4 || bits == 8 || bits == 16;
}

inline std::size_t packed_code_bytes(std::size_t count, int bits) {
    return supported_packed_code_bits(bits)
        ? (count * static_cast<std::size_t>(bits) + 7) / 8
        : 0;
}

inline void set_packed_code(
    std::uint8_t* data,
    std::size_t index,
    int bits,
    std::uint32_t code
) {
    if (bits == 2) {
        const unsigned shift = static_cast<unsigned>(index & 3u) * 2u;
        std::uint8_t& dst = data[index >> 2];
        dst = static_cast<std::uint8_t>(
            (dst & static_cast<std::uint8_t>(~(0x03u << shift))) |
            static_cast<std::uint8_t>((code & 0x03u) << shift));
    } else if (bits == 4) {
        std::uint8_t& dst = data[index >> 1];
        if (index & 1u)
            dst = static_cast<std::uint8_t>((dst & 0x0fu) | ((code & 0x0fu) << 4));
        else
            dst = static_cast<std::uint8_t>((dst & 0xf0u) | (code & 0x0fu));
    } else if (bits == 8) {
        data[index] = static_cast<std::uint8_t>(code);
    } else if (bits == 16) {
        data[index * 2] = static_cast<std::uint8_t>(code & 0xffu);
        data[index * 2 + 1] = static_cast<std::uint8_t>((code >> 8) & 0xffu);
    }
}

inline std::uint32_t get_packed_code(
    const std::uint8_t* data,
    std::size_t index,
    int bits
) {
    if (bits == 2) {
        const unsigned shift = static_cast<unsigned>(index & 3u) * 2u;
        return static_cast<std::uint32_t>((data[index >> 2] >> shift) & 0x03u);
    }
    if (bits == 4) {
        const std::uint8_t src = data[index >> 1];
        return (index & 1u) ? static_cast<std::uint32_t>(src >> 4)
                            : static_cast<std::uint32_t>(src & 0x0fu);
    }
    if (bits == 8) return static_cast<std::uint32_t>(data[index]);
    if (bits == 16)
        return static_cast<std::uint32_t>(data[index * 2]) |
               (static_cast<std::uint32_t>(data[index * 2 + 1]) << 8);
    return 0;
}

}  // namespace anqi
