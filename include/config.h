#pragma once
// =============================================================================
//  config.h —— 全局参数中枢
// -----------------------------------------------------------------------------
//  本文件集中存放所有“可调参数”，分两类：
//    1. Parms      —— 索引构建参数（喂给 nndindex 构造函数）
//    2. RunConfig  —— 数据集 / 查询 / 搜索运行参数（driver Vista.cpp 用）
//  需要换数据集、调超参、改搜索 ef 序列时，只改这一个文件即可，无需动算法代码。
// =============================================================================

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace nndgraph
{

// 环境变量读取助手（换数据集时用 env 覆盖，不必改代码重编）。
inline std::string env_str(const char* k, const std::string& def) {
    const char* v = std::getenv(k); return v ? std::string(v) : def;
}
inline size_t env_sz(const char* k, size_t def) {
    const char* v = std::getenv(k); return v ? (size_t)std::strtoull(v, nullptr, 10) : def;
}
inline float env_flt(const char* k, float def) {
    const char* v = std::getenv(k); return v ? std::strtof(v, nullptr) : def;
}

// -----------------------------------------------------------------------------
//  索引构建参数
//  这些值控制“图怎么建”。nndindex 构造时若某项为 0，会在 set_parameters() 里
//  按数据规模推一个默认值（见 index.h）。
// -----------------------------------------------------------------------------
struct Parms
{
    size_t      n           = 1000000;  // 数据点数
    size_t      dim         = 128;      // 向量维度
    std::string meric       = "l2";     // 距离度量："l2" / "ip" / "ip_origin"（注意拼写沿用原代码）
    // —— 两个【独立】的度数参数（不要混用）——
    size_t      nn_k        = env_sz("ANQI_NNK", 50); // 【建图】初始 kNN 图的 k：NN-descent/RP森林造初始近邻图，每点 k 个近邻（中间产物）
    size_t      M           = env_sz("ANQI_M", 32);   // 【最终图】固定最大度数：每点最终存的边数上限（决定 stride/搜索成本）。与 nn_k 无关
    size_t      n_trees     = 0;        // 随机投影树棵数（0 = 按 n^0.25 自动推，封顶 32）
    size_t      leaf_size   = 0;        // RP 树叶子容量（0 = max(10, nn_k)）
    size_t      max_depth   = 0;        // RP 树最大深度（0 = 100）
    uint64_t    random_seed = 0;        // 随机种子（0 = 用 random_device）
    size_t      n_threads   = 0;        // OpenMP 线程数（0 = 硬件并发数）

    // —— 边选择相关 ——
    float       ef_alpha       = 0.0;   // 边候选范围相对探索范围的倍率（trajectory 大小）
    float       prune_alpha    = 0.0;   // RNG 剪枝松弛 α（0=默认1.2；Vamana 式 >1 图更密）
    size_t      explore_range  = 0;     // beam search 基础探索宽度

    std::string folder_path   = "";     // 索引落盘 / 中间产物目录
};

// -----------------------------------------------------------------------------
//  数据集 / 查询 / 搜索运行参数（driver 专用）
//  默认值对应原 Vista.cpp 里的 DEEP1M 设置。
// -----------------------------------------------------------------------------
struct RunConfig
{
    // —— 数据集 ——（默认 SIFT1M；可用环境变量覆盖，不必改代码重编：
    //    ANQI_DATASET=GAUSS96 ANQI_DIM=96 ANQI_N=1000000 ./anqi_base）
    std::string data_name   = env_str("ANQI_DATASET", "SIFT1M");
    std::string dataset_dir = env_str("ANQI_DATADIR", "/data00/home/yujian.fu/Dataset/"); // 末尾带 '/'
    size_t      n           = env_sz("ANQI_N",   1000000);  // 库向量数
    size_t      dim         = env_sz("ANQI_DIM", 128);      // 维度（SIFT=128, GAUSS96=96）
    size_t      nq          = env_sz("ANQI_NQ",  10000);    // 查询数
    size_t      ngt         = 100;      // ground-truth 每条的近邻数
    size_t      k           = 10;       // 评测 recall@k 的 k

    // —— 构建开关 ——
    bool rerun     = false; // true = 忽略已落盘索引、强制重建
    bool show_edge = true;  // true = 每阶段打印度数统计

    // —— 搜索 ef 扫描序列（recall–QPS 曲线的横轴）——
    // 可用 env 覆盖（逗号分隔），如 ANQI_EF=10,50,90,200,500，避免单线程跑到 ef5000 巨慢。
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

    // —— 派生路径 ——（基于 dataset_dir + data_name 拼出三个 .bin 文件）
    std::string base_path()  const { return dataset_dir + data_name + "/" + data_name + "_base.bin"; }
    std::string query_path() const { return dataset_dir + data_name + "/" + data_name + "_query.bin"; }
    std::string gt_path()    const { return dataset_dir + data_name + "/" + data_name + "_gt_l2.bin"; }
    std::string folder()     const { return dataset_dir + data_name + "/"; }

    // 由运行配置生成一份索引构建参数（默认值对应原 Vista.cpp）。
    // 需要调构建超参时改这里或在 driver 里覆盖返回值的字段。
    Parms build_params() const
    {
        Parms par;
        par.n             = n;
        par.dim           = dim;
        par.random_seed   = 10;
        par.nn_k          = env_sz("ANQI_NNK", 50);  // 【参数1】初始 kNN 图的 k（建图中间产物）
        par.M             = env_sz("ANQI_M", 32);    // 【参数2】最终图最大度数（独立于 nn_k）
        par.leaf_size     = 100;
        par.max_depth     = 100;
        par.n_trees       = env_sz("ANQI_NTREES", 16);
        par.ef_alpha      = 8;
        par.prune_alpha   = env_flt("ANQI_ALPHA", 1.2f); // Vamana α；env 可调
        par.explore_range = env_sz("ANQI_ER", 100);      // 构建 beam 宽度
        par.n_threads     = 64;
        par.folder_path   = folder();
        return par;
    }
};

} // namespace nndgraph
