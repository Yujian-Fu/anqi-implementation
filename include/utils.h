#pragma once
// =============================================================================
//  utils.h —— 基础设施层
// -----------------------------------------------------------------------------
//  通用工具：标准库聚合 include、目录创建、性能（时间 + 峰值内存）记录、
//  二进制数据集读取（.bin / .vecs 两种格式）、线程安全随机数。
//  原文件里未被主链路使用的 Range 区间运算、kmeans、reservoir 采样等已删除。
// =============================================================================

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
#include <sys/resource.h>
#include <sys/time.h>
#include <map>
#include <atomic>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <immintrin.h>
#include <dirent.h>
#include <sys/stat.h>
#include <boost/dynamic_bitset.hpp>
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

// RAII 性能记录器：构造时记起始时刻 + 峰值内存，析构（或手动 print）时打印耗时与内存增量。
struct performance_recorder {
private:
    std::chrono::steady_clock::time_point start_time;
    long initial_memory;
    std::string label;

    long get_peak_memory() {
        struct rusage r_usage;
        getrusage(RUSAGE_SELF, &r_usage);
        return r_usage.ru_maxrss; // KB（Linux）
    }

public:
    performance_recorder(const std::string& label)
        : start_time(std::chrono::steady_clock::now()), initial_memory(get_peak_memory()), label(label) {
        std::time_t now = std::time(0);
        std::cout << "Initialization - " << label << " - Time: " << std::ctime(&now)
                  << "Peak Memory: " << initial_memory / 1000 << " MB\n";
    }

    float get_time_elapsed() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now() - start_time).count() / 1000000.0f;
    }

    void print_performance(const std::string& label) {
        float time_elapsed = get_time_elapsed();
        long peak_memory = get_peak_memory();
        std::time_t now = std::time(0);
        std::cout << label << " - Time: " << std::ctime(&now)
                  << "Elapsed: " << time_elapsed << " seconds, Peak Memory: "
                  << (peak_memory - initial_memory) / 1000 << " MB\n";
    }

    ~performance_recorder() {
        print_performance("Destruction - " + label);
    }
};

// 判断字符串是否以 suffix 结尾（用于按扩展名分派读取格式）。
inline bool ends_with(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

// -----------------------------------------------------------------------------
//  读取 float 向量数据集到预分配的 data 缓冲区。
//  支持两种格式：
//    *.vecs : 每条向量前缀一个 uint32 维度，后跟 dim 个 float。
//    *.bin  : 文件头是 [uint32 n][uint32 dim]，后跟 n*dim 个 float（连续）。
//  会校验文件实际大小与 (n, dim) 是否吻合（.bin 还兼容带 ground-truth 的更大尺寸）。
// -----------------------------------------------------------------------------
inline void read_data(const std::string & file_path, float* data, size_t & n, size_t & dim,
                      bool Check = true, bool Process = true)
{
    std::ifstream file_input(file_path, std::ios::binary | std::ios::ate);
    if (!file_input.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    uint32_t actual_size = file_input.tellg();
    file_input.seekg(0, std::ios::beg);
    uint32_t print_every = n / 10;

    if (ends_with(file_path, "vecs")) {
        uint32_t expected_size = (n * sizeof(uint32_t)) + (n * dim * sizeof(float));
        if (actual_size != expected_size) {
            std::cout << "The actual size is not the same with expected size, actual size: "
                      << actual_size << " expected size: " << expected_size << "\n";
            throw std::runtime_error("Size mismatch");
        }
        uint32_t load_dim;
        std::vector<float> vec(dim);
        for (size_t i = 0; i < n; i++) {
            file_input.read(reinterpret_cast<char *>(&load_dim), sizeof(uint32_t));
            if (load_dim != dim) {
                std::cout << "The loaded dim of dataset not the same with given: " << load_dim << "\n";
                throw std::runtime_error("Dimension mismatch");
            }
            file_input.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
            std::copy(vec.begin(), vec.end(), data + i * dim);
            if (Process && (i + 1) % print_every == 0)
                std::cout << "[Finished loading " << i + 1 << " / " << n << "]" << std::endl;
        }
    } else if (ends_with(file_path, "bin")) {
        uint32_t load_dim, load_n;
        file_input.read(reinterpret_cast<char *>(&load_n), sizeof(uint32_t));
        file_input.read(reinterpret_cast<char *>(&load_dim), sizeof(uint32_t));
        if (load_n != n || load_dim != dim) {
            std::cout << "The loaded dim and size of dataset not the same with given: "
                      << load_n << " " << load_dim << " " << n << " " << dim << "\n";
            throw std::runtime_error("Dimension or size mismatch");
        }
        uint32_t expected_size    = (2 * sizeof(uint32_t)) + (n * dim * sizeof(float));
        uint32_t expected_gt_size = (2 * sizeof(uint32_t)) + (n * dim * sizeof(uint32_t)) + (n * dim * sizeof(float));
        if (actual_size != expected_size && actual_size != expected_gt_size) {
            std::cout << "The actual size is not the same with expected size, actual size: "
                      << actual_size << " expected size: " << expected_size << "\n";
            throw std::runtime_error("Size mismatch");
        }
        size_t total_size = n * dim;
        file_input.read(reinterpret_cast<char *>(data), total_size * sizeof(float));
    } else {
        std::cout << "The given file name " << file_path << " does not end with vecs or bin, format not supported\n";
        throw std::runtime_error("Unsupported file format");
    }

    if (Check) { // 打印首尾向量，肉眼核对读取是否正确
        std::cout << "The first element: \n";
        for (uint32_t i = 0; i < dim; i++) std::cout << data[i] << " ";
        std::cout << "\nThe last element: \n";
        for (uint32_t i = 0; i < dim; i++) std::cout << data[(n - 1) * dim + i] << " ";
        std::cout << "\n";
    }
}

// read_data 的模板版本：读任意类型 T 的数据（典型用于 ground-truth 的 uint32 近邻 id）。
template<typename T>
void readDataT(std::string FilePath, std::vector<T> & data, size_t n, size_t Dim, bool Check, bool Process)
{
    data.resize(n * Dim);
    std::ifstream FileInput(FilePath, std::ios::binary | std::ios::ate);
    size_t actualSize = FileInput.tellg();
    FileInput.seekg(0, std::ios::beg);
    size_t printevery = n / 10;

    if (ends_with(FilePath, "vecs")) {
        size_t expectedSize = (n * sizeof(uint32_t)) + (n * Dim * sizeof(T));
        if (actualSize != expectedSize) {
            std::cout << "The actual size is not the same with expected size, actual size: "
                      << actualSize << " expected size: " << expectedSize << "\n";
            throw std::runtime_error("");
        }
        uint32_t loadDim;
        std::vector<T> vec(Dim);
        for (size_t i = 0; i < n; i++) {
            FileInput.read(reinterpret_cast<char *>(&loadDim), sizeof(uint32_t));
            if (loadDim != Dim) {
                std::cout << "The loaded dim of dataset not the same with given: " << loadDim << "\n";
                throw std::runtime_error("");
            }
            FileInput.read(reinterpret_cast<char *>(vec.data()), Dim * sizeof(T));
            for (size_t j = 0; j < Dim; j++) data[i * Dim + j] = vec[j];
            if (Process && (i + 1) % printevery == 0)
                std::cout << "[Finished loading " << i + 1 << " / " << n << "]" << std::endl;
        }
    } else if (ends_with(FilePath, "bin")) {
        size_t expectedSize   = (2 * sizeof(uint32_t)) + (n * Dim * sizeof(T));
        size_t expectedgtsize = (2 * sizeof(uint32_t)) + (n * Dim * sizeof(uint32_t)) + (n * Dim * sizeof(float));
        if (actualSize != expectedSize && actualSize != expectedgtsize) {
            std::cout << "The actual size is not the same with expected size, actual size: "
                      << actualSize << " expected size: " << expectedSize << "\n";
            throw std::runtime_error("");
        }
        uint32_t loadDim, loadn;
        FileInput.read(reinterpret_cast<char *>(&loadn), sizeof(uint32_t));
        FileInput.read(reinterpret_cast<char *>(&loadDim), sizeof(uint32_t));
        if (loadn != n || loadDim != Dim) {
            std::cout << "The loaded dim and size of dataset not the same with given: "
                      << loadn << " " << loadDim << "\n";
            throw std::runtime_error("");
        }
        std::vector<T> vec(Dim);
        for (size_t i = 0; i < n; i++) {
            FileInput.read(reinterpret_cast<char *>(vec.data()), Dim * sizeof(T));
            for (size_t j = 0; j < Dim; j++) data[i * Dim + j] = vec[j];
            if (Process && (i + 1) % printevery == 0)
                std::cout << "[Finished loading " << i + 1 << " / " << n << "]" << std::endl;
        }
    } else {
        std::cout << "The given file name" << FilePath << " is not ends with vecs or bin, format not supported\n";
    }

    if (Check) {
        std::cout << "The first element: \n";
        for (size_t i = 0; i < Dim; i++) std::cout << data[i] << " ";
        std::cout << "\nThe last element: \n";
        for (size_t i = 0; i < Dim; i++) std::cout << data[(n - 1) * Dim + i] << " ";
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
