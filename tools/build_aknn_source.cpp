// Build the approximate base-space top-k source shared by ANQI's graph and verifier.

#include "index.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <sys/resource.h>
#include <vector>

using namespace nndgraph;

struct TopKRecord {
    uint32_t id;
    float key;
};
static_assert(sizeof(TopKRecord) == 8, "unexpected top-k record layout");

static double peak_gb() {
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1e6;
}

static void require_output(const std::ofstream& stream, const std::string& path) {
    if (!stream) {
        std::cerr << "[aknn] failed writing " << path << "\n";
        std::exit(6);
    }
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 10) {
        std::cerr << "usage: " << argv[0]
                  << " BASE.fbin OUTPUT_PREFIX [K=100] [leaf=500]"
                     " [trees=64] [ef=600] [threads=64] [seed=10] [metric=l2]\n";
        return 2;
    }
    const std::string base_path = argv[1];
    const std::string prefix = argv[2];
    const size_t k = argc > 3 ? std::stoull(argv[3]) : 100;
    const size_t leaf = argc > 4 ? std::stoull(argv[4]) : 500;
    const size_t trees = argc > 5 ? std::stoull(argv[5]) : 64;
    const size_t ef = argc > 6 ? std::stoull(argv[6]) : 600;
    const size_t threads = argc > 7 ? std::stoull(argv[7]) : 64;
    const size_t seed = argc > 8 ? std::stoull(argv[8]) : 10;
    const std::string metric = argc > 9 ? argv[9] : "l2";
    if (metric != "l2" && metric != "ip") {
        std::cerr << "[aknn] metric must be l2 or ip\n";
        return 2;
    }

    std::ifstream header(base_path, std::ios::binary);
    uint32_t n = 0, d = 0;
    if (!header.read(reinterpret_cast<char*>(&n), 4) ||
        !header.read(reinterpret_cast<char*>(&d), 4) || n <= k || d == 0 ||
        k > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "[aknn] invalid base file or K: " << base_path << "\n";
        return 3;
    }

    Parms parms;
    parms.n = n;
    parms.dim = d;
    parms.meric = metric;
    parms.random_seed = seed;
    parms.nn_k = k;
    parms.M = 32;
    parms.leaf_size = leaf;
    parms.max_depth = 100;
    parms.n_trees = trees;
    parms.ef_alpha = 8;
    parms.prune_alpha = 1.2f;
    parms.explore_range = ef;
    parms.n_threads = threads;
    parms.folder_path = prefix + "_aknn_work/";

    const auto started = std::chrono::steady_clock::now();
    nndindex index(base_path, parms);
    std::vector<std::vector<uint32_t>> ids;
    std::vector<std::vector<float>> distances;
    index.build_approx_knn(ids, distances);
    if (ids.size() != n || distances.size() != n) {
        std::cerr << "[aknn] row-count mismatch\n";
        return 4;
    }

    for (size_t row = 0; row < n; ++row) {
        if (ids[row].size() < k || distances[row].size() < k) {
            std::cerr << "[aknn] undersized row=" << row << "\n";
            return 5;
        }
        for (size_t rank = 0; rank < k; ++rank) {
            if (ids[row][rank] >= n || ids[row][rank] == row) {
                std::cerr << "[aknn] invalid neighbor at row=" << row
                          << " rank=" << rank << "\n";
                return 5;
            }
            for (size_t previous = 0; previous < rank; ++previous) {
                if (ids[row][previous] == ids[row][rank]) {
                    std::cerr << "[aknn] duplicate neighbor at row=" << row
                              << " rank=" << rank << "\n";
                    return 5;
                }
            }
            if (rank != 0 && distances[row][rank] < distances[row][rank - 1]) {
                std::cerr << "[aknn] unsorted row=" << row << "\n";
                return 5;
            }
        }
    }

    const std::string record_path = prefix + "_baseknn_aknn.rec";
    const std::string record_tmp = record_path + ".part";
    std::ofstream records(record_tmp, std::ios::binary | std::ios::trunc);
    std::vector<TopKRecord> row_records(k);
    for (size_t row = 0; row < n; ++row) {
        for (size_t rank = 0; rank < k; ++rank) {
            const float key = metric == "ip" ? distances[row][rank] - 1.0f
                                               : std::max(0.0f, distances[row][rank]);
            row_records[rank] = {ids[row][rank], key};
        }
        records.write(reinterpret_cast<const char*>(row_records.data()),
                      static_cast<std::streamsize>(k * sizeof(TopKRecord)));
    }
    records.close();
    require_output(records, record_tmp);
    if (std::rename(record_tmp.c_str(), record_path.c_str()) != 0) {
        std::perror("rename");
        return 6;
    }

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const std::string timing_path = prefix + "_aknn_build_time.meta";
    std::ofstream timing(timing_path);
    timing << "ANQI_AKNN_BUILD_V1\n"
           << "seconds " << std::setprecision(17) << elapsed << "\n"
           << "metric " << metric << "\n"
           << "n " << n << "\n"
           << "d " << d << "\n"
           << "k " << k << "\n"
           << "leaf " << leaf << "\n"
           << "trees " << trees << "\n"
           << "ef " << ef << "\n"
           << "threads " << threads << "\n"
           << "seed " << seed << "\n";
    timing.close();
    require_output(timing, timing_path);

    std::cout << "[aknn] DONE n=" << n << " d=" << d << " K=" << k
              << " leaf=" << leaf << " trees=" << trees << " ef=" << ef
              << " threads=" << threads << " seed=" << seed
              << " metric=" << metric << " elapsed_s=" << elapsed
              << " peak_gb=" << peak_gb() << " source=" << record_path << "\n";
    return 0;
}
