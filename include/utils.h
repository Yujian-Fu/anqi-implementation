#pragma once
// Shared utility functions for graph construction and search.

#include <omp.h>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include <random>
#include <thread>
#include <cfloat>
#include <iostream>
#include <mutex>
#include <unordered_set>
#include <queue>
#include <cstring>
#include <fstream>
#include <sys/time.h>
#include <map>
#include <atomic>
#include <sstream>
#include <iomanip>
#include <immintrin.h>
#include <dirent.h>
#include <sys/stat.h>
#include <xmmintrin.h>

#include "distance.h"

namespace nndgraph
{

const float INF     =  std::numeric_limits<float>::infinity();
const float NEG_INF = -std::numeric_limits<float>::infinity();

// 若目录不存在则创建（仅当前用户可读写执行）。
inline void prepare_folder(const char * FilePath)
{
    if (NULL == opendir(FilePath))
        mkdir(FilePath, S_IRWXU);
}

// 优先队列比较器：按 first（距离）比较。配合 std::priority_queue 使用时，
// 默认得到“大顶堆”（堆顶 = 最大距离），用于维护“当前最远的候选”。
struct CompareByFirst {
    constexpr bool operator()(std::pair<float, uint32_t> const& a,
                              std::pair<float, uint32_t> const& b) const noexcept {
        return a.first < b.first;
    }
};

// Cumulative construction-stage timer.
struct performance_recorder {
private:
    std::chrono::steady_clock::time_point start_time;
    std::string label;

public:
    performance_recorder(const std::string& label)
        : start_time(std::chrono::steady_clock::now()), label(label) {
        std::cout << "Initialization - " << label << "\n";
    }

    float get_time_elapsed() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now() - start_time).count() / 1000000.0f;
    }

    void print_performance(const std::string& label) {
        float time_elapsed = get_time_elapsed();
        std::cout << label << " - Elapsed: " << time_elapsed << " seconds\n";
    }

    ~performance_recorder() {
        print_performance("Destruction - " + label);
    }
};

inline bool ends_with(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

// Load float vectors into a preallocated buffer. Supported inputs are vecs
// records and fbin files with a uint32 [count, dimension] header.
inline void read_data(const std::string & file_path, float* data, size_t & n, size_t & dim,
                      bool Check = true, bool Process = true)
{
    std::ifstream file_input(file_path, std::ios::binary | std::ios::ate);
    if (!file_input.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    const std::streamoff stream_size = file_input.tellg();
    if (stream_size < 0 || n == 0 || dim == 0 ||
        n > std::numeric_limits<size_t>::max() / dim) {
        throw std::runtime_error("Invalid vector input shape or size");
    }
    const uint64_t actual_size = static_cast<uint64_t>(stream_size);
    file_input.seekg(0, std::ios::beg);
    const size_t print_every = std::max<size_t>(1, n / 10);
    const uint64_t values = static_cast<uint64_t>(n) * dim;

    if (ends_with(file_path, "vecs")) {
        const uint64_t expected_size = static_cast<uint64_t>(n) *
            (sizeof(uint32_t) + static_cast<uint64_t>(dim) * sizeof(float));
        if (actual_size != expected_size) {
            std::cout << "The actual size is not the same with expected size, actual size: "
                      << actual_size << " expected size: " << expected_size << "\n";
            throw std::runtime_error("Size mismatch");
        }
        uint32_t load_dim = 0;
        std::vector<float> vec(dim);
        for (size_t i = 0; i < n; i++) {
            if (!file_input.read(reinterpret_cast<char *>(&load_dim), sizeof(uint32_t)) ||
                load_dim != dim) {
                std::cout << "The loaded dim of dataset not the same with given: " << load_dim << "\n";
                throw std::runtime_error("Dimension mismatch");
            }
            if (!file_input.read(reinterpret_cast<char *>(vec.data()),
                                 dim * sizeof(float))) {
                throw std::runtime_error("Truncated vecs input");
            }
            std::copy(vec.begin(), vec.end(), data + i * dim);
            if (Process && (i + 1) % print_every == 0)
                std::cout << "[Finished loading " << i + 1 << " / " << n << "]" << std::endl;
        }
    } else if (ends_with(file_path, "bin")) {
        uint32_t load_dim = 0, load_n = 0;
        if (!file_input.read(reinterpret_cast<char *>(&load_n), sizeof(uint32_t)) ||
            !file_input.read(reinterpret_cast<char *>(&load_dim), sizeof(uint32_t))) {
            throw std::runtime_error("Truncated fbin header");
        }
        if (load_n != n || load_dim != dim) {
            std::cout << "The loaded dim and size of dataset not the same with given: "
                      << load_n << " " << load_dim << " " << n << " " << dim << "\n";
            throw std::runtime_error("Dimension or size mismatch");
        }
        const uint64_t expected_size = 2 * sizeof(uint32_t) +
            values * sizeof(float);
        const uint64_t expected_gt_size = 2 * sizeof(uint32_t) +
            values * (sizeof(uint32_t) + sizeof(float));
        if (actual_size != expected_size && actual_size != expected_gt_size) {
            std::cout << "The actual size is not the same with expected size, actual size: "
                      << actual_size << " expected size: " << expected_size << "\n";
            throw std::runtime_error("Size mismatch");
        }
        if (!file_input.read(reinterpret_cast<char *>(data),
                             static_cast<size_t>(values) * sizeof(float))) {
            throw std::runtime_error("Truncated fbin values");
        }
    } else {
        std::cout << "The given file name " << file_path << " does not end with vecs or bin, format not supported\n";
        throw std::runtime_error("Unsupported file format");
    }

    if (Check) {
        std::cout << "The first element: \n";
        for (uint32_t i = 0; i < dim; i++) std::cout << data[i] << " ";
        std::cout << "\nThe last element: \n";
        for (uint32_t i = 0; i < dim; i++) std::cout << data[(n - 1) * dim + i] << " ";
        std::cout << "\n";
    }
}

// 线程安全的均匀随机整数，区间为闭区间 [min, max]。seed==0 时用真随机源播种。
// 每线程各有独立 generator（thread_local），避免竞争。
inline uint64_t generate_random_integer(uint64_t min = 0, uint64_t max = UINT64_MAX, uint64_t seed = 0) {
    thread_local std::mt19937 generator(seed == 0 ? std::random_device{}() : seed);
    std::uniform_int_distribution<uint64_t> distribution(min, max);
    return distribution(generator);
}

} // namespace nndgraph
