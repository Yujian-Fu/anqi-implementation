#pragma once
// Shared construction and benchmark parameters.

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace nndgraph
{

// Environment helpers keep benchmark overrides out of the source tree.
inline std::string env_str(const char* k, const std::string& def) {
    const char* v = std::getenv(k); return v ? std::string(v) : def;
}
inline size_t env_sz(const char* k, size_t def) {
    const char* v = std::getenv(k); return v ? (size_t)std::strtoull(v, nullptr, 10) : def;
}
inline float env_flt(const char* k, float def) {
    const char* v = std::getenv(k); return v ? std::strtof(v, nullptr) : def;
}

struct Parms
{
    size_t      n           = 1000000;
    size_t      dim         = 128;
    // The field name is retained for compatibility with the index code.
    std::string meric       = "l2";     // l2, ip, or ip_origin
    size_t      nn_k        = env_sz("ANQI_NNK", 50); // Initial AKNN degree.
    size_t      M           = env_sz("ANQI_M", 32);   // Maximum output degree.
    size_t      n_trees     = 0;        // 0 selects a size-dependent default.
    size_t      leaf_size   = 0;        // 0 selects max(10, nn_k).
    size_t      max_depth   = 0;        // 0 selects 100.
    uint64_t    random_seed = 0;        // 0 uses random_device.
    size_t      n_threads   = 0;        // 0 uses the available hardware threads.

    float       ef_alpha       = 0.0;
    float       prune_alpha    = 0.0;   // 0 selects 1.2.
    size_t      explore_range  = 0;

    std::string folder_path   = "";
};

// Optional driver configuration retained for downstream benchmark programs.
struct RunConfig
{
    std::string data_name   = env_str("ANQI_DATASET", "SIFT1M");
    std::string dataset_dir = env_str("ANQI_DATADIR", "data/");
    size_t      n           = env_sz("ANQI_N",   1000000);
    size_t      dim         = env_sz("ANQI_DIM", 128);
    size_t      nq          = env_sz("ANQI_NQ",  10000);
    size_t      ngt         = 100;
    size_t      k           = 10;

    bool rerun     = false;
    bool show_edge = true;

    // ANQI_EF accepts a comma-separated search-width list.
    std::vector<uint32_t> ef_list = env_ef_list();
    static std::vector<uint32_t> env_ef_list() {
        const char* v = std::getenv("ANQI_EF");
        if (!v) return {10,20,30,40,50,60,70,80,90,100,110,120,150,200,250,300,400,
                         500,1000,1100,1200,1300,1400,1500,2000,2500,3000,3500,4000,4500,5000};
        std::vector<uint32_t> out; std::string s(v), cur;
        for (char c : s) { if (c==',') { if(!cur.empty()){out.push_back((uint32_t)std::stoul(cur));cur.clear();} } else cur+=c; }
        if (!cur.empty()) out.push_back((uint32_t)std::stoul(cur));
        return out;
    }

    std::string base_path()  const { return dataset_dir + data_name + "/" + data_name + "_base.bin"; }
    std::string query_path() const { return dataset_dir + data_name + "/" + data_name + "_query.bin"; }
    std::string gt_path()    const { return dataset_dir + data_name + "/" + data_name + "_gt_l2.bin"; }
    std::string folder()     const { return dataset_dir + data_name + "/"; }

    Parms build_params() const
    {
        Parms par;
        par.n             = n;
        par.dim           = dim;
        par.random_seed   = 10;
        par.nn_k          = env_sz("ANQI_NNK", 50);
        par.M             = env_sz("ANQI_M", 32);
        par.leaf_size     = 100;
        par.max_depth     = 100;
        par.n_trees       = env_sz("ANQI_NTREES", 16);
        par.ef_alpha      = 8;
        par.prune_alpha   = env_flt("ANQI_ALPHA", 1.2f);
        par.explore_range = env_sz("ANQI_ER", 100);
        par.n_threads     = 64;
        par.folder_path   = folder();
        return par;
    }
};

} // namespace nndgraph
