#pragma once
// Fixed-degree, co-located proximity graph used by ANQI.

#include "config.h"
#include "rptrees.h"
#include "heap.h"
#include "neighbor.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <unistd.h>

namespace nndgraph
{

class nndindex
{
public:
    struct WarmStartStream {
        std::string command;
        std::string file_path;
        size_t stride = 0;
        size_t width = 0;
        bool ids_only = false;
        bool unlink_file_after_read = false;
    };

    struct QueryVisited {
        uint16_t* stamps = nullptr;
        uint16_t version = 0;
        uint64_t* bits = nullptr;
        std::vector<uint32_t>* touched = nullptr;

        bool first_visit(uint32_t id) {
            if (stamps) {
                if (stamps[id] == version) return false;
                stamps[id] = version;
                return true;
            }
            const uint64_t mask = 1ull << (id & 63u);
            uint64_t& word = bits[id >> 6];
            if (word & mask) return false;
            word |= mask;
            touched->push_back(id);
            return true;
        }
    };

    std::string metric;
    size_t nn_k;           // 【参数1】初始 kNN 图的 k：NN-descent/RP-森林构建初始近邻图时每点的近邻数（仅建图中间产物用）
    size_t M;              // 【参数2】最终图固定最大度数：co-located block 里每点存的边数上限（决定 stride/搜索）。与 nn_k 完全独立
    size_t n_trees, leaf_size, max_depth;
    uint64_t random_seed;
    size_t n_threads;

    size_t n, dim;
    bool   angular;
    DistCal dist;

    // ---- co-located 定长存储 ----
    //   block_i (stride_ 个 float, 64B 对齐) = [vector(dim) | norm | deg(u32) | links(M, u32) | pad]
    float * store_ = nullptr;
    size_t  stride_ = 0;            // = round_up(dim + 2 + M, 16)
    size_t  off_norm_, off_deg_, off_links_; // 块内 float 偏移

    rptree search_tree;            // = forest[0]，查询入口
    uint32_t graph_entry_ = 0;      // Vamana/DiskANN-style 单入口；默认与 ParlayANN 一样为 0
    float  ef_alpha;               // 候选池倍率
    float  prune_alpha;            // RNG 剪枝松弛 α（Vamana 式：>1 保留更多边、图更密、recall/ef↑）
    size_t explore_range;          // beam 基础宽度
    float  construction_time;
    std::string folder_path, index_design;
    bool edge_resid_configured_ = false;
    bool edge_resid_cover_ = false;
    bool edge_ball_rng_ = false;
    bool edge_ball_pure_ = false;
    bool edge_ball_occlude_all_ = true;
    bool edge_bubble_auto_ = false;
    bool edge_bubble_auto_full_ = false;
    bool edge_lsg_rng_ = false;
    size_t edge_resid_nk_ = 100;
    size_t edge_resid_budget_ = 0;
    size_t edge_auto_candidates_ = 0;
    size_t edge_auto_max_swaps_ = 0;
    double edge_resid_lambda_ = 1.0;
    double edge_resid_scale2_ = 1.0;
    double edge_lsg_alpha_ = 1.0;
    std::string edge_policy_;
    std::vector<int> edge_resid_ks_;
    std::vector<float> edge_resid_;
    std::vector<float> edge_lsg_mu_;
    std::atomic<uint64_t> edge_auto_calls_{0};
    std::atomic<uint64_t> edge_auto_changed_{0};
    std::atomic<uint64_t> edge_auto_swaps_{0};
    std::atomic<uint64_t> edge_auto_extra_dists_{0};
    std::atomic<uint64_t> edge_auto_targets_{0};
    std::atomic<uint64_t> edge_auto_alternatives_{0};
    std::atomic<uint64_t> edge_auto_required_{0};
    std::atomic<uint64_t> edge_auto_route_before_{0};
    std::atomic<uint64_t> edge_auto_route_after_{0};
    std::atomic<uint64_t> edge_auto_verify_violations_{0};

    // 搜索用 visited。两种实现按规模选（编译期 ANQI_VISITED）：
    //   版本戳数组（uint16，默认）：字节级读写、无 touched、无逐查询 reset，最快；内存 2B/点/线程。
    //   dynamic_bitset（-DANQI_VISITED_BITSET）：1B-safe（125MB/点/线程），位操作较慢。
    std::vector<std::vector<uint16_t>> visit_mark_;
    std::vector<uint16_t> visit_ver_;

    // ---- 块内访问器 ----
    inline float*    vec(size_t i)   const { return store_ + i * stride_; }
    inline float&    vnorm(size_t i) const { return store_[i * stride_ + off_norm_]; }
    inline uint32_t& deg(size_t i)   const { return *reinterpret_cast<uint32_t*>(store_ + i * stride_ + off_deg_); }
    inline uint32_t* lnk(size_t i)   const { return reinterpret_cast<uint32_t*>(store_ + i * stride_ + off_links_); }

    nndindex(std::string data_path, Parms & parms) {
        set_parameters(parms);
        // stride 布局：vector(dim) | norm(1) | deg(1) | links(M)，pad 到 16 float(64B)
        off_norm_  = dim;
        off_deg_   = dim + 1;
        off_links_ = dim + 2;
        stride_    = ((dim + 2 + M + 15) / 16) * 16;
        store_ = static_cast<float*>(_mm_malloc(n * stride_ * sizeof(float), 64));
        check_allocation(store_);
        load_vectors(data_path);
    }

    ~nndindex() { if (store_) _mm_free(store_); }

    void check_allocation(void* p) { if (!p) throw std::runtime_error("alloc failed"); }

    void set_parameters(Parms &parms) {
        n = parms.n; dim = parms.dim; metric = parms.meric;
        nn_k = parms.nn_k;                       // 初始 kNN 图的 k（建图用）
        M    = parms.M ? parms.M : 32;           // 最终图最大度数（与 nn_k 解耦；未设时默认 32，不再回退到 nn_k）
        n_trees = parms.n_trees; leaf_size = parms.leaf_size; max_depth = parms.max_depth;
        random_seed = parms.random_seed; n_threads = parms.n_threads;
        ef_alpha = parms.ef_alpha; explore_range = parms.explore_range;
        prune_alpha = parms.prune_alpha > 0 ? parms.prune_alpha : 1.2f;
        folder_path = parms.folder_path; index_design = "ANQI fixed-degree";
        dist = DistCal(dim, metric);
        if (leaf_size == 0) leaf_size = std::max(size_t(10), nn_k);
        if (n_trees == 0) n_trees = 16;
        if (max_depth == 0) max_depth = 100;
        angular = (metric == "ip") || (metric == "cos");
        if (n_threads == 0 || n_threads > std::thread::hardware_concurrency())
            n_threads = std::thread::hardware_concurrency();
        prepare_folder(folder_path.c_str());
    }

    // 把 n×dim 紧凑数据读入 store_ 的 vector 槽（strided），并内联算 norm。
    void load_vectors(const std::string& path) {
        // 临时连续缓冲读入，再铺到 strided 槽（store_ 已分配 n*stride_）
        float* tmp = static_cast<float*>(_mm_malloc(n * dim * sizeof(float), 64));
        check_allocation(tmp);
        read_data(path, tmp, n, dim, false, false);
        #pragma omp parallel for num_threads(n_threads) schedule(static)
        for (size_t i = 0; i < n; i++) {
            std::memcpy(vec(i), tmp + i * dim, dim * sizeof(float));
            vnorm(i) = angular ? 0.0f : dist.norm(vec(i));
            deg(i) = 0;
        }
        _mm_free(tmp);
    }

    // L2: 复用预存 vnorm + 一个内积 → d²=‖a‖²+‖b‖²−2⟨a,b⟩。内积比 Σ(a-b)² 每元素省一个 sub(~20-30% 快)。
    //   仅相对比较用(kNN/建图)，float 精度足够；与 dataset_gt(也用 norm 复用)一致。IP/angular 走原路。
    inline float d2(size_t a, size_t b) const {
        if (!angular)
            return vnorm(a) + vnorm(b) - 2.0f * DistCal::InnerProductSIMD16ExtAVX512_(vec(a), vec(b), (uint32_t)dim);
        return dist.calculate(vec(a), vec(b));
    }

    static int envi(const char* k, int d) {
        const char* v = std::getenv(k);
        return v ? atoi(v) : d;
    }

    static double envd(const char* k, double d) {
        const char* v = std::getenv(k);
        return v ? atof(v) : d;
    }

    // Vamana 原版是单入口图搜索；默认用 ParlayANN 的 start_point=0。
    // 需要时可设 ANQI_VAMANA_ENTRY=<id>，或 ANQI_VAMANA_ENTRY=medoid 做轻量 centroid-medoid 近似。
    uint32_t choose_vamana_entry() const {
        const char* ev = std::getenv("ANQI_VAMANA_ENTRY");
        if (!ev || !*ev) return 0;
        std::string s(ev);
        if (s == "medoid" || s == "centroid") {
            size_t sample = (size_t)envi("ANQI_VAMANA_ENTRY_SAMPLE", (int)std::min<size_t>(n, 100000));
            sample = std::max<size_t>(1, std::min(sample, n));
            std::vector<double> mean(dim, 0.0);
            size_t step = std::max<size_t>(1, n / sample);
            for (size_t t = 0; t < sample; t++) {
                size_t id = (t * step) % n;
                const float* x = vec(id);
                for (size_t j = 0; j < dim; j++) mean[j] += (double)x[j];
            }
            for (size_t j = 0; j < dim; j++) mean[j] /= (double)sample;
            uint32_t best = 0;
            double bestd = std::numeric_limits<double>::infinity();
            for (size_t t = 0; t < sample; t++) {
                size_t id = (t * step) % n;
                const float* x = vec(id);
                double dd = 0.0;
                for (size_t j = 0; j < dim; j++) {
                    double z = (double)x[j] - mean[j];
                    dd += z * z;
                }
                if (dd < bestd) { bestd = dd; best = (uint32_t)id; }
            }
            return best;
        }
        char* endp = nullptr;
        unsigned long long id = strtoull(ev, &endp, 10);
        if (endp != ev && n > 0) return (uint32_t)(id % n);
        return 0;
    }

    void sort_unique_candidates(size_t base, std::vector<std::pair<float,uint32_t>>& cand) const {
        cand.erase(std::remove_if(cand.begin(), cand.end(), [&](const auto& x) {
            return x.second == UINT32_MAX || x.second >= n || x.second == (uint32_t)base;
        }), cand.end());
        std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) {
            return a.second < b.second || (a.second == b.second && a.first < b.first);
        });
        size_t w = 0;
        for (size_t r = 0; r < cand.size();) {
            uint32_t id = cand[r].second;
            float best = cand[r].first;
            size_t e = r + 1;
            while (e < cand.size() && cand[e].second == id) {
                if (cand[e].first < best) best = cand[e].first;
                e++;
            }
            cand[w++] = {best, id};
            r = e;
        }
        cand.resize(w);
        std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) {
            return a.first < b.first || (a.first == b.first && a.second < b.second);
        });
    }

    static std::vector<int> parse_int_list(const std::string& s) {
        std::vector<int> out;
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) continue;
            int v = atoi(tok.c_str());
            if (v > 0) out.push_back(v);
        }
        return out;
    }

    bool selected_has(const std::vector<uint32_t>& ids, uint32_t id) const {
        for (uint32_t x : ids) if (x == id) return true;
        return false;
    }

    void rng_extend(const std::vector<std::pair<float,uint32_t>>& cand, size_t cap,
                    std::vector<uint32_t>& ids) {
        for (const auto& c : cand) {
            if (ids.size() >= cap) break;
            float d2c = c.first;
            uint32_t cid = c.second;
            if (selected_has(ids, cid)) continue;
            bool occluded = false;
            for (uint32_t sid : ids) {
                float dsc = d2(sid, cid);
                if (prune_alpha * dsc < d2c) { occluded = true; break; }
            }
            if (!occluded) ids.push_back(cid);
        }
    }

    bool edge_policy_needs_residual(const std::string& pol) const {
        return pol == "residual_cover" || pol == "ball_rng" ||
               pol == "ball_hybrid" || pol == "ball_multik" ||
               pol == "bubble_rng" || pol == "bubble_hybrid" ||
               pol == "bubble_multik" || pol == "bubble_auto" ||
               pol == "bubble_auto_full" || pol == "lsg_bubble_hybrid";
    }

    double edge_radj(uint32_t id, size_t kidx) const {
        return edge_resid_scale2_ * (double)edge_resid_[(size_t)id * edge_resid_nk_ + kidx];
    }

    double edge_sort_radj(uint32_t id) const {
        if (!edge_ball_rng_ || edge_resid_ks_.empty()) return 0.0;
        double best = -std::numeric_limits<double>::infinity();
        for (int kk : edge_resid_ks_) {
            size_t kidx = (size_t)(kk - 1);
            best = std::max(best, edge_radj(id, kidx));
        }
        return std::isfinite(best) ? best : 0.0;
    }

    void configure_edge_residual_cover() {
        if (edge_resid_configured_) return;
        edge_resid_configured_ = true;
        const char* pol = std::getenv("ANQI_EDGE_POLICY");
        if (!pol || !*pol) return;
        edge_policy_ = pol;
        const bool lsg_policy = edge_policy_ == "lsg_rng" ||
                                edge_policy_ == "lsg_bubble_hybrid";
        if (lsg_policy) {
            edge_lsg_alpha_ = envd("ANQI_LSG_ALPHA", 1.0);
            if (!std::isfinite(edge_lsg_alpha_) || edge_lsg_alpha_ < 0.0) {
                fprintf(stderr, "[edge-policy] invalid ANQI_LSG_ALPHA=%.17g\n", edge_lsg_alpha_);
                exit(2);
            }
            std::string mu_path = std::getenv("ANQI_LSG_MU_FILE") ?
                                  std::getenv("ANQI_LSG_MU_FILE") : "";
            if (mu_path.empty()) {
                fprintf(stderr, "[edge-policy] lsg_rng requires ANQI_LSG_MU_FILE\n");
                exit(2);
            }
            edge_lsg_mu_.resize(n);
            std::ifstream mf(mu_path, std::ios::binary);
            if (!mf) {
                fprintf(stderr, "[edge-policy] lsg_rng cannot open mu table %s\n", mu_path.c_str());
                exit(2);
            }
            const size_t expected_bytes = n * sizeof(float);
            mf.seekg(0, std::ios::end);
            std::streamoff actual_bytes = mf.tellg();
            if (actual_bytes < 0 || (size_t)actual_bytes != expected_bytes) {
                fprintf(stderr,
                        "[edge-policy] lsg mu size mismatch %s: expected %zu bytes, got %lld\n",
                        mu_path.c_str(), expected_bytes, (long long)actual_bytes);
                exit(2);
            }
            mf.seekg(0, std::ios::beg);
            mf.read(reinterpret_cast<char*>(edge_lsg_mu_.data()),
                    (std::streamsize)expected_bytes);
            if ((size_t)mf.gcount() != expected_bytes) {
                fprintf(stderr, "[edge-policy] short lsg mu table %s\n", mu_path.c_str());
                exit(2);
            }
            double mu_min = std::numeric_limits<double>::infinity();
            double mu_max = 0.0;
            long double mu_sum = 0.0;
            for (size_t i = 0; i < n; i++) {
                double mu = edge_lsg_mu_[i];
                if (!std::isfinite(mu) || mu <= 0.0) {
                    fprintf(stderr, "[edge-policy] invalid lsg mu[%zu]=%.9g\n", i, mu);
                    exit(2);
                }
                mu_min = std::min(mu_min, mu);
                mu_max = std::max(mu_max, mu);
                mu_sum += mu;
            }
            edge_lsg_rng_ = true;
            fprintf(stderr,
                    "[edge-policy] %s lsg enabled alpha=%.6g mu=%s min=%.6g mean=%.6g max=%.6g\n",
                    edge_policy_.c_str(), edge_lsg_alpha_, mu_path.c_str(), mu_min,
                    n ? (double)(mu_sum / n) : 0.0, mu_max);
            if (edge_policy_ == "lsg_rng") return;
        }
        if (!edge_policy_needs_residual(edge_policy_)) return;
        edge_resid_nk_ = (size_t)envi("ANQI_EDGE_NK", 100);
        if (edge_resid_nk_ == 0) edge_resid_nk_ = 100;
        edge_resid_budget_ = (size_t)std::max(0, envi("ANQI_EDGE_RESID_BUDGET", 16));
        edge_resid_budget_ = std::min(edge_resid_budget_, M);
        edge_auto_candidates_ = (size_t)std::max(1, envi("ANQI_EDGE_AUTO_CANDIDATES", (int)M));
        edge_auto_max_swaps_ = (size_t)std::max(1, envi("ANQI_EDGE_AUTO_MAX_SWAPS", (int)M));
        edge_resid_lambda_ = envd("ANQI_EDGE_RESID_LAMBDA", 1.0);
        edge_resid_ks_ = parse_int_list(std::getenv("ANQI_EDGE_KS") ? std::getenv("ANQI_EDGE_KS") : "10,50,100");
        std::vector<int> valid_ks;
        for (int k : edge_resid_ks_)
            if (k >= 1 && (size_t)k <= edge_resid_nk_) valid_ks.push_back(k);
        edge_resid_ks_.swap(valid_ks);
        if (edge_resid_ks_.empty()) return;
        edge_bubble_auto_ = edge_policy_ == "bubble_auto" ||
                            edge_policy_ == "bubble_auto_full";
        edge_bubble_auto_full_ = edge_policy_ == "bubble_auto_full";
        if (edge_resid_budget_ == 0 && edge_policy_ != "ball_rng" &&
            !edge_bubble_auto_) return;

        {
            std::ifstream mf(folder_path + "_anqi_meta.txt");
            double mstar = 0.0, scale = 1.0;
            if (mf >> mstar >> scale) edge_resid_scale2_ = scale * scale;
        }
        std::string resid_path = std::getenv("ANQI_EDGE_RESID_FILE") ?
                                 std::getenv("ANQI_EDGE_RESID_FILE") :
                                 (folder_path + "_anqi_rankm_residual.bin");
        edge_resid_.resize(n * edge_resid_nk_);
        std::ifstream rf(resid_path, std::ios::binary);
        if (!rf) {
            fprintf(stderr, "[edge-policy] residual_cover requested but missing %s\n", resid_path.c_str());
            exit(2);
        }
        const size_t expected_bytes = edge_resid_.size() * sizeof(float);
        rf.seekg(0, std::ios::end);
        std::streamoff actual_bytes = rf.tellg();
        if (actual_bytes < 0 || (size_t)actual_bytes != expected_bytes) {
            fprintf(stderr, "[edge-policy] residual table stride/size mismatch %s: expected %zu bytes "
                    "(n=%zu edge_nk=%zu), got %lld bytes; set ANQI_EDGE_NK to the table stride\n",
                    resid_path.c_str(), expected_bytes, n, edge_resid_nk_, (long long)actual_bytes);
            exit(2);
        }
        rf.seekg(0, std::ios::beg);
        rf.read(reinterpret_cast<char*>(edge_resid_.data()),
                (std::streamsize)expected_bytes);
        if ((size_t)rf.gcount() != expected_bytes) {
            fprintf(stderr, "[edge-policy] bad residual table %s expected %.3f MB got %.3f MB\n",
                    resid_path.c_str(), expected_bytes / 1e6,
                    (double)rf.gcount() / 1e6);
            exit(2);
        }
        edge_resid_cover_ = true;
        if (edge_policy_ != "residual_cover") {
            edge_resid_cover_ = false;
            edge_ball_rng_ = true;
            edge_ball_pure_ = (edge_policy_ == "ball_rng" || edge_policy_ == "bubble_rng");
            std::string occ = std::getenv("ANQI_EDGE_BALL_OCCLUSION") ?
                              std::getenv("ANQI_EDGE_BALL_OCCLUSION") :
                              (edge_policy_ == "ball_multik" || edge_policy_ == "bubble_multik" ? "all" : "all");
            edge_ball_occlude_all_ = (occ != "any");
        }
        fprintf(stderr, "[edge-policy] %s enabled ks=%s budget=%zu lambda=%.4g scale2=%.6g nk=%zu%s%s%s\n",
                edge_policy_.c_str(),
                (std::getenv("ANQI_EDGE_KS") ? std::getenv("ANQI_EDGE_KS") : "10,50,100"),
                edge_resid_budget_, edge_resid_lambda_, edge_resid_scale2_, edge_resid_nk_,
                edge_ball_rng_ ? " ball_occlusion=" : "",
                edge_ball_rng_ ? (edge_ball_occlude_all_ ? "all" : "any") : "",
                edge_bubble_auto_ ?
                    (std::string(" auto_candidates=") +
                     (edge_bubble_auto_full_ ? "all" : std::to_string(edge_auto_candidates_)) +
                     " auto_max_swaps=" + std::to_string(edge_auto_max_swaps_)).c_str() : "");
    }

    void add_residual_cover_edges(size_t base,
                                  const std::vector<std::pair<float,uint32_t>>& cand,
                                  std::vector<uint32_t>& ids) {
        if (!edge_resid_cover_ || ids.size() >= M) return;
        size_t total_budget = std::min(edge_resid_budget_, M - ids.size());
        if (total_budget == 0 || edge_resid_ks_.empty()) return;
        size_t per_k = std::max<size_t>(1, (total_budget + edge_resid_ks_.size() - 1) / edge_resid_ks_.size());
        size_t added_total = 0;
        for (int kk : edge_resid_ks_) {
            if (added_total >= total_budget || ids.size() >= M) break;
            size_t kidx = (size_t)(kk - 1);
            std::vector<std::pair<double,uint32_t>> scored;
            scored.reserve(cand.size());
            for (const auto& c : cand) {
                uint32_t cid = c.second;
                if (cid == UINT32_MAX || cid >= n || cid == (uint32_t)base || selected_has(ids, cid)) continue;
                double residual = (double)edge_resid_[(size_t)cid * edge_resid_nk_ + kidx];
                double risk = residual > 0.0 ? residual : 0.0;
                if (risk <= 0.0) continue;
                double score = (double)c.first - edge_resid_lambda_ * edge_resid_scale2_ * risk;
                scored.emplace_back(score, cid);
            }
            std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
                return a.first < b.first || (a.first == b.first && a.second < b.second);
            });
            size_t added_k = 0;
            for (const auto& s : scored) {
                if (ids.size() >= M || added_total >= total_budget || added_k >= per_k) break;
                if (selected_has(ids, s.second)) continue;
                ids.push_back(s.second);
                added_k++;
                added_total++;
            }
        }
    }

    bool ball_occludes_distance(size_t base, uint32_t sid, float dbase_cid,
                                float dsc) const {
        if (!edge_ball_rng_ || edge_resid_ks_.empty()) return false;
        bool any = false;
        bool all = true;
        for (int kk : edge_resid_ks_) {
            size_t kidx = (size_t)(kk - 1);
            double lhs = prune_alpha * (double)dsc - edge_resid_lambda_ * edge_radj(sid, kidx);
            double rhs = (double)dbase_cid - edge_resid_lambda_ * edge_radj((uint32_t)base, kidx);
            bool occ = lhs < rhs;
            any = any || occ;
            all = all && occ;
            if (!edge_ball_occlude_all_ && any) return true;
            if (edge_ball_occlude_all_ && !all) return false;
        }
        return edge_ball_occlude_all_ ? all : any;
    }

    bool ball_occludes(size_t base, uint32_t sid, uint32_t cid, float dbase_cid) const {
        return ball_occludes_distance(base, sid, dbase_cid, d2(sid, cid));
    }

    void ball_sort_candidates(size_t, const std::vector<std::pair<float,uint32_t>>& cand,
                              std::vector<std::pair<float,uint32_t>>& sorted) const {
        sorted = cand;
        if (!edge_ball_rng_) return;
        std::sort(sorted.begin(), sorted.end(), [&](const auto& a, const auto& b) {
            double ka = (double)a.first - edge_resid_lambda_ * edge_sort_radj(a.second);
            double kb = (double)b.first - edge_resid_lambda_ * edge_sort_radj(b.second);
            return ka < kb || (ka == kb && a.second < b.second);
        });
    }

    void rng_extend_ball(size_t base, const std::vector<std::pair<float,uint32_t>>& cand, size_t cap,
                         std::vector<uint32_t>& ids) {
        for (const auto& c : cand) {
            if (ids.size() >= cap) break;
            float d2c = c.first;
            uint32_t cid = c.second;
            if (selected_has(ids, cid)) continue;
            bool occluded = false;
            for (uint32_t sid : ids) {
                if (ball_occludes(base, sid, cid, d2c)) { occluded = true; break; }
            }
            if (!occluded) ids.push_back(cid);
        }
    }

    // Replace center-RNG edges only when every baseline raw-distance descent
    // remains covered.  Among feasible swaps, greedily increase the number of
    // candidate targets reached by at least one Bubble power-distance descent.
    // The lazy mode evaluates only the center-RNG edges plus the top-M Bubble
    // alternatives, both as edge choices and as certification targets.  Full
    // mode uses the complete candidate pool and is intended as an oracle.
    void optimize_verification_constrained_bubble(
        size_t base,
        const std::vector<std::pair<float,uint32_t>>& cand,
        std::vector<uint32_t>& ids) {
        if (!edge_bubble_auto_ || ids.empty() || cand.empty()) return;

        std::vector<std::pair<float,uint32_t>> bubble_order;
        ball_sort_candidates(base, cand, bubble_order);

        std::vector<uint32_t> universe = ids;
        const size_t alternative_limit = edge_bubble_auto_full_
            ? cand.size() : edge_auto_candidates_;
        size_t alternatives = 0;
        for (const auto& c : bubble_order) {
            if (selected_has(universe, c.second)) continue;
            universe.push_back(c.second);
            if (++alternatives >= alternative_limit) break;
        }
        if (alternatives == 0) return;

        std::vector<std::pair<float,uint32_t>> targets;
        if (edge_bubble_auto_full_) {
            targets = cand;
        } else {
            targets.reserve(universe.size());
            for (uint32_t id : universe) {
                for (const auto& c : cand) {
                    if (c.second == id) {
                        targets.push_back(c);
                        break;
                    }
                }
            }
            std::sort(targets.begin(), targets.end(), [](const auto& a, const auto& b) {
                return a.first < b.first || (a.first == b.first && a.second < b.second);
            });
        }
        const size_t target_count = targets.size();
        const size_t words = (target_count + 63) / 64;

        struct Signature {
            uint32_t id = UINT32_MAX;
            double power_key = std::numeric_limits<double>::infinity();
            std::vector<uint64_t> verify;
            std::vector<uint64_t> route;
        };
        std::vector<Signature> sig(universe.size());
        uint64_t local_extra_dists = 0;
        for (size_t ui = 0; ui < universe.size(); ui++) {
            Signature& s = sig[ui];
            s.id = universe[ui];
            s.verify.assign(words, 0);
            s.route.assign(words, 0);
            float dbase_u = 0.0f;
            for (const auto& c : cand) {
                if (c.second == s.id) { dbase_u = c.first; break; }
            }
            s.power_key = (double)dbase_u -
                          edge_resid_lambda_ * edge_sort_radj(s.id);
            for (size_t ti = 0; ti < target_count; ti++) {
                uint32_t target = targets[ti].second;
                float dsc = 0.0f;
                if (s.id != target) {
                    dsc = d2(s.id, target);
                    local_extra_dists++;
                }
                const uint64_t bit = 1ull << (ti & 63u);
                if (s.id == target || prune_alpha * dsc < targets[ti].first)
                    s.verify[ti >> 6] |= bit;
                if (ball_occludes_distance(base, s.id, targets[ti].first, dsc))
                    s.route[ti >> 6] |= bit;
            }
        }

        std::vector<size_t> selected;
        selected.reserve(ids.size());
        std::vector<uint8_t> is_selected(sig.size(), 0);
        for (uint32_t id : ids) {
            for (size_t ui = 0; ui < sig.size(); ui++) {
                if (sig[ui].id == id) {
                    selected.push_back(ui);
                    is_selected[ui] = 1;
                    break;
                }
            }
        }
        if (selected.empty()) return;

        std::vector<uint16_t> verify_count(target_count, 0);
        std::vector<uint16_t> route_count(target_count, 0);
        auto add_signature = [&](size_t ui, int delta) {
            for (size_t ti = 0; ti < target_count; ti++) {
                const uint64_t bit = 1ull << (ti & 63u);
                if (sig[ui].verify[ti >> 6] & bit)
                    verify_count[ti] = (uint16_t)((int)verify_count[ti] + delta);
                if (sig[ui].route[ti >> 6] & bit)
                    route_count[ti] = (uint16_t)((int)route_count[ti] + delta);
            }
        };
        for (size_t ui : selected) add_signature(ui, 1);
        std::vector<uint8_t> required(target_count, 0);
        size_t required_count = 0;
        size_t route_before = 0;
        for (size_t ti = 0; ti < target_count; ti++) {
            if (verify_count[ti] > 0) { required[ti] = 1; required_count++; }
            if (route_count[ti] > 0) route_before++;
        }

        size_t swaps = 0;
        const size_t max_swaps = std::min(edge_auto_max_swaps_, alternatives);
        while (swaps < max_swaps) {
            std::vector<uint64_t> route_uncovered(words, 0);
            for (size_t ti = 0; ti < target_count; ti++)
                if (route_count[ti] == 0)
                    route_uncovered[ti >> 6] |= 1ull << (ti & 63u);

            long best_route_gain = std::numeric_limits<long>::min();
            double best_power_gain = -std::numeric_limits<double>::infinity();
            size_t best_q = SIZE_MAX;
            size_t best_slot = SIZE_MAX;
            for (size_t q = 0; q < sig.size(); q++) {
                if (is_selected[q]) continue;
                long newly_covered = 0;
                for (size_t w = 0; w < words; w++)
                    newly_covered += __builtin_popcountll(sig[q].route[w] & route_uncovered[w]);

                for (size_t slot = 0; slot < selected.size(); slot++) {
                    size_t old = selected[slot];
                    bool feasible = true;
                    long lost = 0;
                    for (size_t ti = 0; ti < target_count; ti++) {
                        const uint64_t bit = 1ull << (ti & 63u);
                        const bool old_verifies = (sig[old].verify[ti >> 6] & bit) != 0;
                        const bool q_verifies = (sig[q].verify[ti >> 6] & bit) != 0;
                        if (required[ti] && old_verifies && verify_count[ti] == 1 &&
                            !q_verifies) {
                            feasible = false;
                            break;
                        }
                        const bool old_routes = (sig[old].route[ti >> 6] & bit) != 0;
                        const bool q_routes = (sig[q].route[ti >> 6] & bit) != 0;
                        if (old_routes && route_count[ti] == 1 && !q_routes) lost++;
                    }
                    if (!feasible) continue;
                    long route_gain = newly_covered - lost;
                    double power_gain = sig[old].power_key - sig[q].power_key;
                    if (route_gain > best_route_gain ||
                        (route_gain == best_route_gain && power_gain > best_power_gain)) {
                        best_route_gain = route_gain;
                        best_power_gain = power_gain;
                        best_q = q;
                        best_slot = slot;
                    }
                }
            }
            if (best_q == SIZE_MAX || best_route_gain < 0 ||
                (best_route_gain == 0 && best_power_gain <= 1e-12)) break;

            size_t old = selected[best_slot];
            add_signature(old, -1);
            add_signature(best_q, 1);
            is_selected[old] = 0;
            is_selected[best_q] = 1;
            selected[best_slot] = best_q;
            swaps++;
        }

        size_t violations = 0;
        size_t route_after = 0;
        for (size_t ti = 0; ti < target_count; ti++) {
            if (required[ti] && verify_count[ti] == 0) violations++;
            if (route_count[ti] > 0) route_after++;
        }
        for (size_t slot = 0; slot < selected.size(); slot++)
            ids[slot] = sig[selected[slot]].id;

        edge_auto_calls_.fetch_add(1, std::memory_order_relaxed);
        edge_auto_changed_.fetch_add(swaps > 0, std::memory_order_relaxed);
        edge_auto_swaps_.fetch_add(swaps, std::memory_order_relaxed);
        edge_auto_extra_dists_.fetch_add(local_extra_dists, std::memory_order_relaxed);
        edge_auto_targets_.fetch_add(target_count, std::memory_order_relaxed);
        edge_auto_alternatives_.fetch_add(alternatives, std::memory_order_relaxed);
        edge_auto_required_.fetch_add(required_count, std::memory_order_relaxed);
        edge_auto_route_before_.fetch_add(route_before, std::memory_order_relaxed);
        edge_auto_route_after_.fetch_add(route_after, std::memory_order_relaxed);
        edge_auto_verify_violations_.fetch_add(violations, std::memory_order_relaxed);
    }

    double lsg_distance2(uint32_t a, uint32_t b, double raw_d2) const {
        if (!edge_lsg_rng_ || edge_lsg_alpha_ == 0.0) return raw_d2;
        double mu_product = (double)edge_lsg_mu_[a] * (double)edge_lsg_mu_[b];
        double denom = std::pow(mu_product, edge_lsg_alpha_);
        if (!std::isfinite(denom) || denom <= 0.0) return raw_d2;
        return raw_d2 / denom;
    }

    void lsg_sort_candidates(size_t base,
                             const std::vector<std::pair<float,uint32_t>>& cand,
                             std::vector<std::pair<double,uint32_t>>& sorted) const {
        sorted.clear();
        sorted.reserve(cand.size());
        for (const auto& c : cand)
            sorted.emplace_back(lsg_distance2((uint32_t)base, c.second, c.first), c.second);
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.first < b.first || (a.first == b.first && a.second < b.second);
        });
    }

    void rng_extend_lsg(const std::vector<std::pair<double,uint32_t>>& cand, size_t cap,
                        std::vector<uint32_t>& ids) {
        for (const auto& c : cand) {
            if (ids.size() >= cap) break;
            double d2c = c.first;
            uint32_t cid = c.second;
            if (selected_has(ids, cid)) continue;
            bool occluded = false;
            for (uint32_t sid : ids) {
                double dsc = lsg_distance2(sid, cid, d2(sid, cid));
                if ((double)prune_alpha * dsc < d2c) {
                    occluded = true;
                    break;
                }
            }
            if (!occluded) ids.push_back(cid);
        }
    }

    void append_current_edges(size_t i, std::vector<std::pair<float,uint32_t>>& pool) {
        uint32_t dg = deg(i);
        uint32_t* L = lnk(i);
        for (uint32_t j = 0; j < dg; j++) {
            uint32_t nb = L[j];
            if (nb != UINT32_MAX && nb < n && nb != (uint32_t)i) pool.emplace_back(d2(i, nb), nb);
        }
    }

    void append_aknn_edges(size_t i, HeapList* g0, std::vector<std::pair<float,uint32_t>>& pool) {
        for (uint32_t j : g0->indices[i])
            if (j != UINT32_MAX && j < n && j != (uint32_t)i) pool.emplace_back(d2(i, j), j);
    }

    void prune_pool_to_ids(size_t i, std::vector<std::pair<float,uint32_t>>& pool,
                           std::vector<uint32_t>& ids) {
        sort_unique_candidates(i, pool);
        ids.clear();
        ids.reserve(M);
        if (edge_bubble_auto_) {
            // Build the ordinary RNG graph first.  The optimizer may exchange
            // edges, but it must preserve every raw-distance descent that this
            // baseline set certifies on the shared candidate pool.
            rng_extend(pool, M, ids);
            optimize_verification_constrained_bubble(i, pool, ids);
            return;
        }
        // LSG changes only the ordering and RNG occlusion metric. Candidate
        // discovery is shared with center-RNG so the edge policies are comparable.
        if (edge_lsg_rng_ && edge_ball_rng_ && edge_lsg_alpha_ != 0.0) {
            std::vector<std::pair<double,uint32_t>> lsg_pool;
            lsg_sort_candidates(i, pool, lsg_pool);
            size_t bubble_budget = std::min(edge_resid_budget_, M);
            size_t lsg_cap = M - bubble_budget;
            if (lsg_cap == 0) lsg_cap = M;
            rng_extend_lsg(lsg_pool, lsg_cap, ids);

            std::vector<std::pair<float,uint32_t>> ball_pool;
            ball_sort_candidates(i, pool, ball_pool);
            rng_extend_ball(i, ball_pool, M, ids);
            rng_extend_lsg(lsg_pool, M, ids);
            return;
        }
        if (edge_lsg_rng_ && edge_lsg_alpha_ != 0.0) {
            std::vector<std::pair<double,uint32_t>> lsg_pool;
            lsg_sort_candidates(i, pool, lsg_pool);
            rng_extend_lsg(lsg_pool, M, ids);
            return;
        }
        if (edge_ball_rng_) {
            std::vector<std::pair<float,uint32_t>> ball_pool;
            ball_sort_candidates(i, pool, ball_pool);
            if (edge_ball_pure_) {
                rng_extend_ball(i, ball_pool, M, ids);
                rng_extend(pool, M, ids);
                return;
            }
            size_t ball_budget = std::min(edge_resid_budget_, M);
            size_t center_cap = M - ball_budget;
            if (center_cap == 0) center_cap = M;
            rng_extend(pool, center_cap, ids);
            rng_extend_ball(i, ball_pool, M, ids);
            rng_extend(pool, M, ids);
            return;
        }
        size_t cover_budget = edge_resid_cover_ ? std::min(edge_resid_budget_, M) : 0;
        size_t normal_cap = M - cover_budget;
        if (normal_cap == 0) normal_cap = M;
        rng_extend(pool, normal_cap, ids);
        add_residual_cover_edges(i, pool, ids);
        rng_extend(pool, M, ids);
    }

    void write_ids(size_t i, const std::vector<uint32_t>& ids) {
        uint32_t sd = (uint32_t)std::min(ids.size(), M);
        uint32_t* L = lnk(i);
        for (uint32_t j = 0; j < sd; j++) L[j] = ids[j];
        deg(i) = sd;
    }

    // RNG prune：候选按距离升序，保留不被已选边遮挡的，直到达 cap。写入 out（id 列表）。
    // cand: (dist_to_base, id) 升序；用 store_ 向量现算遮挡距离。无锁（纯局部）。
    void rng_prune(const std::vector<std::pair<float,uint32_t>>& cand, size_t cap,
                   uint32_t* out, uint32_t& out_deg) {
        out_deg = 0;
        for (auto& c : cand) {
            float d2c = c.first; uint32_t cid = c.second;
            bool occluded = false;
            for (uint32_t k = 0; k < out_deg; k++) {
                float dsc = d2(out[k], cid);
                // Vamana α 遮挡：α·dist(s,c) < dist(base,c) 才算被遮挡。α=1 即标准 RNG；α>1 保留更多边。
                if (prune_alpha * dsc < d2c) { occluded = true; break; }
            }
            if (!occluded) { out[out_deg++] = cid; if (out_deg >= cap) break; }
        }
    }

    // Collect candidates by greedy beam search over the current fixed-degree
    // graph. The graph is read-only and visited state is thread-local.
    void greedy_collect(size_t i, std::vector<std::pair<float,uint32_t>>& pool, size_t Lb) {
        static thread_local std::vector<uint32_t> vmark; static thread_local uint32_t vver = 0;
        if (vmark.size() != n) { vmark.assign(n, 0); vver = 0; }
        if (++vver == 0) { std::fill(vmark.begin(), vmark.end(), 0); vver = 1; }
        NeighborPriorityQueue beam(Lb);
        float* q = vec(i);
        uint32_t ep = search_tree.get_leave_mediod(q, dist, random_seed);
        vmark[ep] = vver; { float d = d2(i, ep); beam.insert(Neighbor(ep, d)); if (ep != (uint32_t)i) pool.emplace_back(d, ep); }
        while (beam.has_unexpanded_node()) {
            uint32_t cur = beam.closest_unexpanded().id();
            uint32_t dg = deg(cur); uint32_t* L = lnk(cur);
            for (uint32_t j = 0; j < dg; j++) {
                uint32_t nb = L[j];
                if (vmark[nb] == vver) continue;
                vmark[nb] = vver;
                float d = d2(i, nb); beam.insert(Neighbor(nb, d)); if (nb != (uint32_t)i) pool.emplace_back(d, nb);
            }
        }
    }

    // ParlayANN/DiskANN Vamana-style:所有 build search 都从同一个 entry 出发，而不是 RPT 多入口。
    void greedy_collect_from_entry(size_t i, std::vector<std::pair<float,uint32_t>>& pool,
                                   size_t Lb, uint32_t entry) {
        static thread_local std::vector<uint32_t> vmark;
        static thread_local uint32_t vver = 0;
        if (vmark.size() != n) { vmark.assign(n, 0); vver = 0; }
        if (++vver == 0) { std::fill(vmark.begin(), vmark.end(), 0); vver = 1; }
        NeighborPriorityQueue beam(Lb);
        uint32_t ep = (entry < n) ? entry : 0;
        vmark[ep] = vver;
        float d0 = d2(i, ep);
        beam.insert(Neighbor(ep, d0));
        if (ep != (uint32_t)i) pool.emplace_back(d0, ep);
        while (beam.has_unexpanded_node()) {
            uint32_t cur = beam.closest_unexpanded().id();
            uint32_t dg = deg(cur);
            uint32_t* L = lnk(cur);
            for (uint32_t j = 0; j < dg; j++) {
                uint32_t nb = L[j];
                if (nb >= n || vmark[nb] == vver) continue;
                vmark[nb] = vver;
                float d = d2(i, nb);
                beam.insert(Neighbor(nb, d));
                if (nb != (uint32_t)i) pool.emplace_back(d, nb);
            }
        }
    }

    // 提交边:反向 scatter(O→R 的 CSR)+ cand=O_i∪R_i RNG-prune 到 M 写 co-located block。
    // 初始建图与每个 Vamana pass 共用。无锁。
    void commit_edges(const std::vector<std::vector<uint32_t>>& O) {
        std::vector<std::atomic<uint32_t>> rcnt(n);
        for (size_t i = 0; i < n; i++) rcnt[i].store(0);
        #pragma omp parallel for num_threads(n_threads) schedule(static)
        for (size_t i = 0; i < n; i++) for (uint32_t j : O[i]) rcnt[j].fetch_add(1, std::memory_order_relaxed);
        std::vector<size_t> roff(n + 1, 0);
        for (size_t i = 0; i < n; i++) roff[i + 1] = roff[i] + rcnt[i].load();
        std::vector<uint32_t> rids(roff[n]);
        std::vector<std::atomic<uint32_t>> rcur(n);
        for (size_t i = 0; i < n; i++) rcur[i].store(0);
        #pragma omp parallel for num_threads(n_threads) schedule(static)
        for (size_t i = 0; i < n; i++)
            for (uint32_t j : O[i]) { uint32_t p = rcur[j].fetch_add(1, std::memory_order_relaxed); rids[roff[j] + p] = (uint32_t)i; }
        #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 64)
        for (size_t i = 0; i < n; i++) {
            std::vector<std::pair<float,uint32_t>> cand;
            cand.reserve(O[i].size() + (roff[i + 1] - roff[i]));
            for (uint32_t j : O[i]) cand.emplace_back(d2(i, j), j);
            for (size_t p = roff[i]; p < roff[i + 1]; p++) { uint32_t j = rids[p]; cand.emplace_back(d2(i, j), j); }
            std::vector<uint32_t> ids;
            prune_pool_to_ids(i, cand, ids);
            write_ids(i, ids);
        }
    }

    // ParlayANN Vamana 对齐版 refine：
    //   - 初始图不随机，从 AKNN/NNDescent 得到的图开始；
    //   - 每点 beam search 从统一 entry 出发；
    //   - 按随机顺序分 batch 插入，batch 出边写回后立即把反向边加到被指向点并 robust-prune。
    void vamana_batch_refine(HeapList* g0, performance_recorder& rec, size_t Lb, int passes) {
        if (n == 0 || passes <= 0) return;
        graph_entry_ = choose_vamana_entry();
        double max_frac = envd("ANQI_VAMANA_BATCH_FRAC", 0.02);
        if (max_frac <= 0.0) max_frac = 0.02;
        size_t max_batch_cap = (size_t)envi("ANQI_VAMANA_MAX_BATCH", 1000000);
        size_t max_batch = std::min<size_t>((size_t)std::max<double>(1.0, max_frac * (double)n), max_batch_cap);
        if (max_batch == 0) max_batch = n;
        double batch_base = envd("ANQI_VAMANA_BATCH_BASE", 2.0);
        if (batch_base <= 1.0) batch_base = 2.0;
        float final_alpha = prune_alpha;
        float early_alpha = std::getenv("ANQI_VAMANA_EARLY_ALPHA") ?
                            (float)atof(std::getenv("ANQI_VAMANA_EARLY_ALPHA")) : 1.0f;
        bool use_early_alpha = envi("ANQI_VAMANA_EARLY_ALPHA_ONE", 1) != 0;
        fprintf(stderr, "[vamana-batch] entry=%u L=%zu pass=%d batch_frac=%.4g max_batch=%zu base=%.3g alpha=%.3g rev=csr\n",
                graph_entry_, Lb, passes, max_frac, max_batch, batch_base, final_alpha);

        std::vector<uint32_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::vector<std::atomic<uint32_t>> rev_count(n), rev_cursor(n);
        std::vector<size_t> rev_offset(n, 0);
        #pragma omp parallel for num_threads(n_threads) schedule(static)
        for (size_t i = 0; i < n; i++) {
            rev_count[i].store(0, std::memory_order_relaxed);
            rev_cursor[i].store(0, std::memory_order_relaxed);
        }
        for (int pass = 0; pass < passes; pass++) {
            prune_alpha = (use_early_alpha && pass + 1 < passes) ? early_alpha : final_alpha;
            std::mt19937_64 rng(random_seed + 0x9e3779b97f4a7c15ULL * (uint64_t)(pass + 1));
            std::shuffle(order.begin(), order.end(), rng);

            size_t count = 0, inc = 0;
            while (count < n) {
                size_t floor = count, ceiling;
                double p0 = std::pow(batch_base, (double)inc);
                if (p0 <= (double)max_batch) {
                    floor = (size_t)p0 - 1;
                    double p1 = std::pow(batch_base, (double)(inc + 1));
                    ceiling = std::min<size_t>((size_t)p1 - 1, n);
                    count = ceiling;
                } else {
                    ceiling = std::min<size_t>(count + max_batch, n);
                    count = ceiling;
                }
                inc++;
                if (ceiling <= floor || floor >= n) continue;
                size_t bsz = ceiling - floor;
                std::vector<std::vector<uint32_t>> batch_out(bsz);

                #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 64)
                for (size_t bi = 0; bi < bsz; bi++) {
                    uint32_t node = order[floor + bi];
                    std::vector<std::pair<float,uint32_t>> pool;
                    pool.reserve(Lb + nn_k + deg(node) + 8);
                    greedy_collect_from_entry(node, pool, Lb, graph_entry_);
                    append_aknn_edges(node, g0, pool);
                    append_current_edges(node, pool);       // robustPrune(add=true)
                    prune_pool_to_ids(node, pool, batch_out[bi]);
                }

                #pragma omp parallel for num_threads(n_threads) schedule(static)
                for (size_t bi = 0; bi < bsz; bi++) {
                    uint32_t node = order[floor + bi];
                    write_ids(node, batch_out[bi]);
                }

                std::vector<std::vector<uint32_t>> touched_tls(n_threads);
                #pragma omp parallel num_threads(n_threads)
                {
                    int tid = omp_get_thread_num();
                    std::vector<uint32_t>& local = touched_tls[(size_t)tid];
                    #pragma omp for schedule(static)
                    for (size_t bi = 0; bi < bsz; bi++) {
                        for (uint32_t dst : batch_out[bi]) {
                            if (dst >= n) continue;
                            uint32_t old = rev_count[dst].fetch_add(1, std::memory_order_relaxed);
                            if (old == 0) local.push_back(dst);
                        }
                    }
                }
                size_t ntouched = 0;
                for (auto& v : touched_tls) ntouched += v.size();
                std::vector<uint32_t> touched;
                touched.reserve(ntouched);
                for (auto& v : touched_tls) touched.insert(touched.end(), v.begin(), v.end());

                std::vector<size_t> off(touched.size() + 1, 0);
                for (size_t ti = 0; ti < touched.size(); ti++) {
                    uint32_t dst = touched[ti];
                    off[ti + 1] = off[ti] + rev_count[dst].load(std::memory_order_relaxed);
                    rev_offset[dst] = off[ti];
                }
                std::vector<uint32_t> rev_src(off.back());
                #pragma omp parallel for num_threads(n_threads) schedule(static)
                for (size_t bi = 0; bi < bsz; bi++) {
                    uint32_t src = order[floor + bi];
                    for (uint32_t dst : batch_out[bi]) {
                        if (dst >= n) continue;
                        uint32_t p = rev_cursor[dst].fetch_add(1, std::memory_order_relaxed);
                        rev_src[rev_offset[dst] + p] = src;
                    }
                }

                #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 64)
                for (size_t ti = 0; ti < touched.size(); ti++) {
                    uint32_t dst = touched[ti];
                    std::vector<std::pair<float,uint32_t>> pool;
                    size_t s = rev_offset[dst];
                    size_t e = s + rev_count[dst].load(std::memory_order_relaxed);
                    pool.reserve(deg(dst) + (e - s));
                    append_current_edges(dst, pool);
                    for (size_t p = s; p < e; p++) {
                        uint32_t src = rev_src[p];
                        if (src != dst && src < n) pool.emplace_back(d2(dst, src), src);
                    }
                    std::vector<uint32_t> ids;
                    prune_pool_to_ids(dst, pool, ids);
                    write_ids(dst, ids);
                }
                #pragma omp parallel for num_threads(n_threads) schedule(static)
                for (size_t ti = 0; ti < touched.size(); ti++) {
                    uint32_t dst = touched[ti];
                    rev_count[dst].store(0, std::memory_order_relaxed);
                    rev_cursor[dst].store(0, std::memory_order_relaxed);
                }
            }
            rec.print_performance(("vamana batch pass " + std::to_string(pass)).c_str());
        }
        prune_alpha = final_alpha;
    }

    // ---- 构建 ----
    // init_nbr != nullptr 时：复用【外部预建的 kNN】(如原始空间的近邻 id)填初始图 g0,
    //   重算本空间(lifted)的距离,跳过 lifted 上的 RP森林共现 init_knn。
    //   —— 让 12h 优化的原始 kNN 信号同时用于半径和图 init,kNN 只算一次。
    std::string build_index(bool rerun, bool show_edge,
                            const std::vector<std::vector<uint32_t>>* init_nbr = nullptr,
                            const uint32_t* init_nbr_flat = nullptr,
                            size_t init_nbr_stride = 0,
                            size_t init_nbr_width = 0,
                            std::vector<uint32_t>* init_nbr_flat_owner = nullptr,
                            const WarmStartStream* init_nbr_stream = nullptr) {
        std::cout << "Index config: " << generate_config() << "\n";
        if (!rerun && load_index()) return index_design;
        performance_recorder rec("Index Construction (fixed-degree)");
        configure_edge_residual_cover();
        const bool has_stream_init =
            init_nbr_stream != nullptr &&
            (!init_nbr_stream->command.empty() ||
             !init_nbr_stream->file_path.empty());
        const bool has_init_nbr =
            init_nbr != nullptr || init_nbr_flat != nullptr || has_stream_init;
        if (init_nbr_flat && (init_nbr_stride == 0 || init_nbr_width == 0 ||
                             init_nbr_width > init_nbr_stride)) {
            throw std::runtime_error("invalid flat warm-start KNN layout");
        }
        if (init_nbr_flat_owner &&
            (init_nbr_flat_owner->empty() ||
             init_nbr_flat_owner->data() != init_nbr_flat)) {
            throw std::runtime_error("flat warm-start owner does not match its data pointer");
        }
        if (has_stream_init &&
            ((!init_nbr_stream->command.empty() &&
              !init_nbr_stream->file_path.empty()) ||
             init_nbr_stream->stride == 0 || init_nbr_stream->width == 0 ||
             init_nbr_stream->width > init_nbr_stream->stride ||
             (init_nbr_stream->unlink_file_after_read &&
              init_nbr_stream->file_path.empty()))) {
            throw std::runtime_error("invalid streamed warm-start KNN layout");
        }

        // Build one RP tree over a temporary contiguous vector array for the
        // query entry structure. AKNN initialization uses the original-space
        // warm start followed by NN-Descent; the fallback path builds the full
        // forest only when no initial neighbors are supplied.
        float* cvec = static_cast<float*>(_mm_malloc(n * dim * sizeof(float), 64));
        #pragma omp parallel for num_threads(n_threads) schedule(static)
        for (size_t i = 0; i < n; i++) std::memcpy(cvec + i * dim, vec(i), dim * sizeof(float));

        size_t nt_build = has_init_nbr ? 1 : n_trees;
        std::vector<rptree> forest = make_forest(cvec, n, dim, nt_build, leaf_size, angular, max_depth, random_seed, n_threads);
        search_tree = forest[0];
        graph_entry_ = choose_vamana_entry();
        _mm_free(cvec);
        rec.print_performance("make_forest (lifted, 1 tree for query entry)");

        // Initialize the temporary AKNN graph from original-space neighbors,
        // recompute their lifted distances, and refine in lifted space with
        // NN-Descent. ANQI_REUSE_ORIG_KNN=1 skips the lifted refinement for
        // the corresponding component experiment.
        HeapList* g0 = new HeapList(n, nn_k, false);
        if (has_stream_init) {
            struct StreamKnnRecord {
                uint32_t id;
                float distance;
            };
            static_assert(
                sizeof(StreamKnnRecord) == 8,
                "unexpected streamed warm-start record layout");
            const bool uses_command = !init_nbr_stream->command.empty();
            FILE* input = uses_command
                ? ::popen(init_nbr_stream->command.c_str(), "r")
                : std::fopen(init_nbr_stream->file_path.c_str(), "rb");
            if (!input) {
                throw std::runtime_error("cannot start streamed warm-start command");
            }
            constexpr size_t target_chunk_bytes = 64ull << 20;
            const size_t record_bytes = init_nbr_stream->ids_only
                ? sizeof(uint32_t) : sizeof(StreamKnnRecord);
            const size_t rows_per_chunk = std::max<size_t>(
                1, target_chunk_bytes /
                       (init_nbr_stream->stride * record_bytes));
            std::vector<StreamKnnRecord> record_chunk;
            std::vector<uint32_t> id_chunk;
            if (init_nbr_stream->ids_only)
                id_chunk.resize(rows_per_chunk * init_nbr_stream->stride);
            else
                record_chunk.resize(rows_per_chunk * init_nbr_stream->stride);
            bool stream_ok = true;
            for (size_t row0 = 0; row0 < n; row0 += rows_per_chunk) {
                const size_t rows = std::min(rows_per_chunk, n - row0);
                const size_t records = rows * init_nbr_stream->stride;
                const size_t got = std::fread(
                    init_nbr_stream->ids_only
                        ? static_cast<void*>(id_chunk.data())
                        : static_cast<void*>(record_chunk.data()),
                    record_bytes, records, input);
                if (got != records) {
                    stream_ok = false;
                    break;
                }
                #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 256)
                for (size_t local = 0; local < rows; local++) {
                    const size_t i = row0 + local;
                    const StreamKnnRecord* record_row =
                        init_nbr_stream->ids_only
                            ? nullptr
                            : record_chunk.data() + local * init_nbr_stream->stride;
                    const uint32_t* id_row =
                        init_nbr_stream->ids_only
                            ? id_chunk.data() + local * init_nbr_stream->stride
                            : nullptr;
                    for (size_t k = 0; k < init_nbr_stream->width; k++) {
                        const uint32_t j = init_nbr_stream->ids_only
                            ? id_row[k] : record_row[k].id;
                        if (j != (uint32_t)i && j < n)
                            g0->push_nolock(i, j, d2(i, j));
                    }
                }
            }
            if (stream_ok && std::fgetc(input) != EOF) stream_ok = false;
            const int stream_status = uses_command
                ? ::pclose(input) : std::fclose(input);
            if (!stream_ok || stream_status != 0) {
                throw std::runtime_error(
                    "streamed warm-start ended early, produced extra data, or failed");
            }
            if (init_nbr_stream->unlink_file_after_read &&
                ::unlink(init_nbr_stream->file_path.c_str()) != 0) {
                throw std::runtime_error(
                    "cannot release consumed streamed warm-start file");
            }
            rec.print_performance(
                init_nbr_stream->ids_only
                    ? "init: streamed compact KNN IDs warm-start (recompute lifted d2)"
                    : "init: streamed original KNN warm-start (recompute lifted d2)");
            if (!std::getenv("ANQI_REUSE_ORIG_KNN"))
                nndescent_refine(g0, rec, 6);
        } else if (init_nbr_flat) {
            // Scale runs keep the same graph algorithm while storing the
            // prebuilt original-space KNN in one contiguous allocation. This
            // avoids millions of tiny vector allocations without changing
            // the lifted-distance warm start or any downstream NN-descent/RNG
            // step.
            #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 256)
            for (size_t i = 0; i < n; i++) {
                const uint32_t* row = init_nbr_flat + i * init_nbr_stride;
                for (size_t k = 0; k < init_nbr_width; k++) {
                    uint32_t j = row[k];
                    if (j != (uint32_t)i && j < n) g0->push_nolock(i, j, d2(i, j));
                }
            }
            rec.print_performance("init: contiguous original KNN warm-start (recompute lifted d2)");
            if (init_nbr_flat_owner) {
                const size_t released_bytes =
                    init_nbr_flat_owner->capacity() * sizeof(uint32_t);
                std::vector<uint32_t>().swap(*init_nbr_flat_owner);
                init_nbr_flat = nullptr;
                std::fprintf(
                    stderr,
                    "[index] released consumed flat warm-start IDs: %.3f GB\n",
                    released_bytes / 1e9);
            }
            if (!std::getenv("ANQI_REUSE_ORIG_KNN"))
                nndescent_refine(g0, rec, 6);
        } else if (init_nbr) {
            // 用原始空间近似 kNN 的邻居 id 填 g0,重算本空间(lifted)距离。每点单线程独占 → push_nolock 无锁。
            #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 256)
            for (size_t i = 0; i < n; i++)
                for (uint32_t j : (*init_nbr)[i])
                    if (j != (uint32_t)i && j < n) g0->push_nolock(i, j, d2(i, j));
            rec.print_performance("init: 原始 kNN warm-start (重算 lifted d2)");
            if (!std::getenv("ANQI_REUSE_ORIG_KNN"))
                nndescent_refine(g0, rec, 6);     // Refine using lifted-space distances.
        } else {
            init_knn(forest, g0);                 // 回退：lifted RP 森林叶内共现
            rec.print_performance("init_knn (lifted RP forest, fallback)");
        }
        g0->push_reverse(n_threads);
        rec.print_performance("push_reverse");

        // B: 每点 beam gather + RNG-prune → O_i（无锁，各写各的）
        std::vector<std::vector<uint32_t>> O(n);
        size_t cand_cap = std::max<size_t>(M, (size_t)(ef_alpha * explore_range));
        #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 64)
        for (size_t i = 0; i < n; i++) {
            std::vector<std::pair<float,uint32_t>> pool; pool.reserve(cand_cap);
            gather_candidates(i, g0, pool, cand_cap);
            std::vector<uint32_t> ids;
            prune_pool_to_ids(i, pool, ids);
            O[i].assign(ids.begin(), ids.end());
        }
        rec.print_performance("gather + prune O");
        commit_edges(O);                       // C+D:反向 scatter + 最终剪枝 → lnk/deg(初始图)
        rec.print_performance("commit initial graph");

        // Optional Vamana/RNG refinement. ANQI_VAMANA_STYLE=batch selects the
        // batched component variant.
        if (std::getenv("ANQI_VAMANA")) {
            size_t Lb = (size_t)envi("ANQI_VAMANA_L", 128); int passes = envi("ANQI_VAMANA_PASS", 2);
            std::string style = std::getenv("ANQI_VAMANA_STYLE") ? std::getenv("ANQI_VAMANA_STYLE") : "global";
            if (style == "global" || style == "old") {
                for (int pass = 0; pass < passes; pass++) {
                    #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 64)
                    for (size_t i = 0; i < n; i++) {
                        std::vector<std::pair<float,uint32_t>> pool; pool.reserve(Lb*2);
                        greedy_collect(i, pool, Lb);                              // RP-tree entry points.
                        append_aknn_edges(i, g0, pool);                           // ∪ lifted AKNN
                        std::vector<uint32_t> ids;
                        prune_pool_to_ids(i, pool, ids);
                        O[i].assign(ids.begin(), ids.end());
                    }
                    commit_edges(O);
                    rec.print_performance(("vamana global pass " + std::to_string(pass)).c_str());
                }
            } else {
                std::vector<std::vector<uint32_t>>().swap(O);
                vamana_batch_refine(g0, rec, Lb, passes);
            }
        }
        delete g0;
        rec.print_performance("graph done");

        if (edge_bubble_auto_) print_edge_auto_stats();

        construction_time = rec.get_time_elapsed();
        if (show_edge) print_degree_stats();
        save_index();
        return index_design;
    }

    // 建【近似 kNN】并返回每点按 d² 升序的 top-K(K=nn_k)。
    //   关键:用 gather_candidates(在 g0 上 beam + 邻居的邻居扩展 = ANQI 实际的 NN-descent 操作),
    //   不是只取 g0 原始(那是扩展前的 RP 森林,质量低)。这才是构图实际拿到的 kNN/半径。
    void build_approx_knn(std::vector<std::vector<uint32_t>>& nbr,
                          std::vector<std::vector<float>>& dist) {
        performance_recorder rec("approx kNN (init + gather 扩展)");
        float* cvec = static_cast<float*>(_mm_malloc(n * dim * sizeof(float), 64));
        #pragma omp parallel for num_threads(n_threads) schedule(static)
        for (size_t i = 0; i < n; i++) std::memcpy(cvec + i * dim, vec(i), dim * sizeof(float));
        std::vector<rptree> forest = make_forest(cvec, n, dim, n_trees, leaf_size, angular, max_depth, random_seed, n_threads);
        _mm_free(cvec);
        rec.print_performance("make_forest");
        HeapList* g0 = new HeapList(n, nn_k, false);
        init_knn(forest, g0);
        rec.print_performance("init_knn");
        // ANQI_NNITERS enables iterative NN-Descent local joins after the
        // leaf-based initialization.
        {
            auto envi=[](const char*k,int d){const char*v=std::getenv(k);return v?atoi(v):d;};
            int nn_iters=envi("ANQI_NNITERS",0), nn_sample=std::min(envi("ANQI_NNSAMPLE",20),120);
            for (int it=0; it<nn_iters; it++) {
                std::atomic<long> upd{0};
                #pragma omp parallel for num_threads(n_threads) schedule(dynamic,256)
                for (size_t u=0; u<n; u++) {
                    uint32_t loc[128]; int cnt=0;
                    { std::unique_lock<std::mutex> lk(g0->mtxs[u]);
                      for (uint32_t v : g0->indices[u]) if (v!=UINT32_MAX){ loc[cnt++]=v; if(cnt>=nn_sample) break; } }
                    for (int i=0;i<cnt;i++) for (int j=i+1;j<cnt;j++) {
                        uint32_t v=loc[i], w=loc[j]; float dd=d2(v,w);
                        if (g0->check_push(v,w,dd,false)) upd.fetch_add(1,std::memory_order_relaxed);
                        if (g0->check_push(w,v,dd,false)) upd.fetch_add(1,std::memory_order_relaxed);
                    }
                }
                rec.print_performance(("nndescent iter"+std::to_string(it)+" upd="+std::to_string(upd.load())).c_str());
                if (upd.load() < (long)(n/200)) break;   // 收敛
            }
        }
        nbr.assign(n, {}); dist.assign(n, {});
        if (std::getenv("ANQI_SKIPGATHER")) {
            // Use the refined NN-Descent graph directly when candidate
            // gathering is disabled for the component experiment.
            #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 256)
            for (size_t i = 0; i < n; i++) {
                std::vector<std::pair<float,uint32_t>> p;
                for (size_t j=0;j<g0->indices[i].size();j++)
                    if (g0->indices[i][j]!=UINT32_MAX) p.push_back({g0->values[i][j], g0->indices[i][j]});
                std::sort(p.begin(),p.end());
                for (auto&pr:p){ if(pr.second==(uint32_t)i)continue; nbr[i].push_back(pr.second); dist[i].push_back(pr.first); if(nbr[i].size()>=nn_k)break; }
            }
            delete g0; rec.print_performance("skip-gather(用 g0)");
        } else {
            g0->push_reverse(n_threads);
            size_t cand_cap = std::max<size_t>(nn_k, (size_t)(ef_alpha * explore_range));
            std::atomic<size_t> _gdone{0}; size_t _gstep = std::max<size_t>(1, n/20);
            #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 64)
            for (size_t i = 0; i < n; i++) {
                std::vector<std::pair<float,uint32_t>> pool; pool.reserve(cand_cap);
                gather_candidates(i, g0, pool, cand_cap);
                std::sort(pool.begin(), pool.end());
                auto& ids = nbr[i]; auto& ds = dist[i];
                for (auto& pr : pool) {
                    if (pr.second == (uint32_t)i) continue;
                    ids.push_back(pr.second); ds.push_back(pr.first);
                    if (ids.size() >= nn_k) break;
                }
                size_t d = ++_gdone;
                if (d % _gstep == 0) fprintf(stderr, "  [gather] %3.0f%%\n", 100.0*d/n);
            }
            delete g0; rec.print_performance("gather 扩展 → top-K");
        }
    }

    // Initialize the AKNN graph from points that share RP-tree leaves. Within
    // one tree, leaves partition the points, so each heap has a single writer.
    // Trees are processed sequentially.
    void init_knn(const std::vector<rptree>& forest, HeapList* g0) {
        size_t nt = forest.size();
        if (std::getenv("ANQI_FIXA")) {
            // The point-major variant computes each unique candidate distance once.
            std::vector<uint32_t> leafmap((size_t)nt * n, UINT32_MAX);   // [i*nt+t] 固定点连续
            for (size_t t = 0; t < nt; t++) {
                const auto& nodes = forest[t].nodes;
                #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 64)
                for (size_t ni = 0; ni < nodes.size(); ni++)
                    for (uint32_t p : nodes[ni].indices) leafmap[(size_t)p*nt + t] = (uint32_t)ni;
            }
            std::atomic<size_t> _ikdone{0}; size_t _ikstep = std::max<size_t>(1, n/20);
            #pragma omp parallel num_threads(n_threads)
            {
                std::vector<uint32_t> vmark(n, 0); uint32_t vver = 0;
                #pragma omp for schedule(dynamic, 256)
                for (size_t i = 0; i < n; i++) {
                    vver++; if (vver==0){ std::fill(vmark.begin(),vmark.end(),0); vver=1; }
                    vmark[i] = vver;
                    for (size_t t = 0; t < nt; t++) {
                        uint32_t ni = leafmap[(size_t)i*nt + t];
                        if (ni == UINT32_MAX) continue;
                        for (uint32_t j : forest[t].nodes[ni].indices)
                            if (vmark[j] != vver) { vmark[j] = vver; g0->push_nolock((uint32_t)i, j, d2(i, (size_t)j)); }
                    }
                    size_t d = ++_ikdone;
                    if (d % _ikstep == 0) fprintf(stderr, "  [init_knn-point-major] %3.0f%%\n", 100.0*d/n);
                }
            }
        } else {
            // The per-leaf path preserves vector locality. ANQI_DEDUP=1 uses
            // the point-major leaf map to skip a pair already compared in an
            // earlier tree. Sequential tree processing makes this check lock-free.
            // Deduplication is opt-in because its benefit depends on dimension
            // and the amount of cross-tree overlap.
            const bool dedup = std::getenv("ANQI_DEDUP") != nullptr;
            std::vector<uint32_t> lmap;
            if (dedup) {
                lmap.assign((size_t)nt * n, UINT32_MAX);            // point-major [x*nt+t]
                for (size_t t = 0; t < nt; t++) {
                    const auto& nodes = forest[t].nodes;
                    #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 64)
                    for (size_t ni = 0; ni < nodes.size(); ni++)
                        for (uint32_t p : nodes[ni].indices) lmap[(size_t)p*nt + t] = (uint32_t)ni;
                }
            }
            const uint32_t* LM = lmap.data();
            // ANQI_DEDUP_MAXD limits how many earlier trees are checked; zero
            // scans all earlier trees.
            const size_t maxd = []{ const char* e = std::getenv("ANQI_DEDUP_MAXD"); return e ? (size_t)atoll(e) : 0; }();
            auto seen_before = [&](uint32_t x, uint32_t y, size_t tcur) -> bool {
                const uint32_t* rx = LM + (size_t)x * nt;
                const uint32_t* ry = LM + (size_t)y * nt;
                size_t lim = (maxd && maxd < tcur) ? maxd : tcur;
                size_t t2 = 0;
                for (; t2 + 16 <= lim; t2 += 16)
                    if (_mm512_cmpeq_epi32_mask(_mm512_loadu_si512(rx + t2),
                                                _mm512_loadu_si512(ry + t2))) return true;
                for (; t2 < lim; t2++) if (rx[t2] == ry[t2]) return true;
                return false;
            };
            for (size_t t = 0; t < nt; t++) {
                const auto& nodes = forest[t].nodes;
                #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1)
                for (size_t ni = 0; ni < nodes.size(); ni++) {
                    const auto& idx = nodes[ni].indices;
                    for (size_t a = 0; a < idx.size(); a++)
                        for (size_t b = a + 1; b < idx.size(); b++) {
                            uint32_t x = idx[a], y = idx[b];
                            if (dedup && seen_before(x, y, t)) continue;   // 跨树重复 → 跳 d2
                            float d = d2(x, y);
                            g0->push_nolock(x, y, d);
                            g0->push_nolock(y, x, d);
                        }
                }
                if ((t+1) % std::max<size_t>(1,nt/20)==0 || t+1==nt)
                    fprintf(stderr, "  [init_knn%s] %3.0f%% (%zu/%zu trees)\n", dedup?"-dedup":"", 100.0*(t+1)/nt, t+1, nt);
            }
        }
        // 补缺：parallel over i，每个 i 单线程独占 → 只 push_nolock(i,r)；反向 r→i 由 push_reverse 重建
        #pragma omp parallel for num_threads(n_threads) schedule(static)
        for (size_t i = 0; i < n; i++) {
            size_t miss = nn_k - g0->heap_size(i);
            for (size_t j = 0; j < miss; j++) {
                uint32_t r = generate_random_integer(0, n - 1, random_seed + i);
                float d = d2(i, r);
                g0->push_nolock(i, r, d);
            }
        }
    }

    // NN-descent local join 精修：在【本空间(lifted)】的 d2 度量上对 g0 做多趟邻居对 join，
    //   把每点 kNN 收敛到本空间真值。用于 lifted 建图——init_knn(RP森林叶内共现)只是粗 init，
    //   高维 lifted 几何下必须 descent 收敛，否则选出的近邻仍偏向原始空间。
    //   迭代轮数 = ANQI_GRAPHNND(默认 def_iters)；采样宽度 = ANQI_NNSAMPLE(默认 20)。
    //   纯无损构建步：只往 g0 里压更近的对，kNN 质量单调不降。
    void nndescent_refine(HeapList* g0, performance_recorder& rec, int def_iters) {
        auto envi=[](const char*k,int d){const char*v=std::getenv(k);return v?atoi(v):d;};
        int nn_iters=envi("ANQI_GRAPHNND",def_iters), nn_sample=std::min(envi("ANQI_NNSAMPLE",20),120);
        for (int it=0; it<nn_iters; it++) {
            std::atomic<long> upd{0};
            #pragma omp parallel for num_threads(n_threads) schedule(dynamic,256)
            for (size_t u=0; u<n; u++) {
                uint32_t loc[128]; int cnt=0;
                { std::unique_lock<std::mutex> lk(g0->mtxs[u]);
                  for (uint32_t v : g0->indices[u]) if (v!=UINT32_MAX){ loc[cnt++]=v; if(cnt>=nn_sample) break; } }
                for (int i=0;i<cnt;i++) for (int j=i+1;j<cnt;j++) {
                    uint32_t v=loc[i], w=loc[j]; float dd=d2(v,w);
                    if (g0->check_push(v,w,dd,false)) upd.fetch_add(1,std::memory_order_relaxed);
                    if (g0->check_push(w,v,dd,false)) upd.fetch_add(1,std::memory_order_relaxed);
                }
            }
            rec.print_performance(("graph nndescent iter"+std::to_string(it)+" upd="+std::to_string(upd.load())).c_str());
            if (upd.load() < (long)(n/200)) break;   // 收敛
        }
    }

    // Collect candidates for point i with read-only beam search over g0.
    // A thread-local version-stamp array resets visited state in constant time.
    void gather_candidates(size_t i, HeapList* g0, std::vector<std::pair<float,uint32_t>>& pool, size_t cap) {
        static thread_local std::vector<uint32_t> vmark;
        static thread_local uint32_t vver = 0;
        if (vmark.size() != n) { vmark.assign(n, 0); vver = 0; }
        if (++vver == 0) { std::fill(vmark.begin(), vmark.end(), 0); vver = 1; }  // 溢出回绕
        NeighborPriorityQueue cand(explore_range);
        auto push = [&](uint32_t id, float d){ pool.emplace_back(d, id); };
        for (size_t j = 0; j < g0->indices[i].size(); j++) {
            uint32_t id = g0->indices[i][j];
            if (id == UINT32_MAX || id >= n) continue;
            if (vmark[id] == vver) continue;
            vmark[id] = vver;
            float d = g0->values[i][j];
            cand.insert(Neighbor(id, d)); push(id, d);
        }
        while (cand.has_unexpanded_node()) {
            uint32_t cur = cand.closest_unexpanded().id();
            for (uint32_t nb : g0->indices[cur]) {
                if (nb == UINT32_MAX || nb >= n) continue;
                if (vmark[nb] == vver) continue;
                vmark[nb] = vver;
                float d = d2(i, nb);
                cand.insert(Neighbor(nb, d)); push(nb, d);
            }
            // 也探一部分反向边，提升候选多样性
            for (uint32_t nb : g0->reverse_indices[cur]) {
                if (nb == UINT32_MAX || nb >= n) continue;
                if (vmark[nb] == vver) continue;
                vmark[nb] = vver;
                float d = d2(i, nb);
                cand.insert(Neighbor(nb, d)); push(nb, d);
            }
            if (pool.size() >= cap) break;
        }
    }

    // Reverse-kNN range traversal. A greedy beam reaches the query
    // neighborhood, then bounded flooding collects points inside the target
    // envelope. eps permits bridge points just outside that envelope.
    // With adaptive flooding, radj supplies each object's target-rank lifted
    // threshold. For targets within the graph horizon radj is non-positive;
    // analytic extrapolation to a larger rank makes it positive. Exploration
    // leaves a beta bridge between the target and graph horizons. A null radj
    // uses the global R2 * (1 + eps)^2 expansion.
    void range_search(float* q, double R2, float eps, size_t L_descent,
                      std::vector<uint32_t>& results, size_t& out_ndist,
                      QueryVisited& visited,
                      const float* radj = nullptr, float beta = 0.5f, bool collect_eps = false) {
        results.clear();
        size_t ndist = 0;
        double e2 = (double)(1.0f + eps) * (double)(1.0f + eps);
        struct SearchScoreCfg {
            enum Mode { CENTER, BALL_GAP, BALL_COLLECT, BALL_EXPAND, BALL_HYBRID } mode = CENTER;
            double lambda = 1.0;
            SearchScoreCfg() {
                const char* m = std::getenv("ANQI_SEARCH_SCORE");
                if (!m || !*m || std::strcmp(m, "center") == 0 || std::strcmp(m, "point") == 0) {
                    mode = CENTER;
                } else if (std::strcmp(m, "ball") == 0 || std::strcmp(m, "ball_gap") == 0 || std::strcmp(m, "gap") == 0) {
                    mode = BALL_GAP;
                } else if (std::strcmp(m, "ball_collect") == 0 || std::strcmp(m, "collect_gap") == 0) {
                    mode = BALL_COLLECT;
                } else if (std::strcmp(m, "ball_expand") == 0 || std::strcmp(m, "expand_gap") == 0) {
                    mode = BALL_EXPAND;
                } else if (std::strcmp(m, "ball_hybrid") == 0 || std::strcmp(m, "hybrid") == 0) {
                    mode = BALL_HYBRID;
                } else {
                    fprintf(stderr, "[range_search] unknown ANQI_SEARCH_SCORE=%s, falling back to center\n", m);
                    mode = CENTER;
                }
                const char* l = std::getenv("ANQI_SEARCH_SCORE_LAMBDA");
                if (l && *l) lambda = atof(l);
            }
        };
        static const SearchScoreCfg score_cfg;
        // collect_eps(recheck 模式):收集阈值放宽到 = 扩展阈值,让 eps 余量内的"略出球"点也进 results,
        //   交给上层精确 recheck 过滤 → 召回靠 eps 拉回、精度由 recheck 保证。默认 false = 原行为(严格 r_k)。
        auto thrStrict = [&](uint32_t id)->double{ return radj ? (R2 + (double)radj[id]) : R2; }; // 严格判定阈值
        auto thrExpanded = [&](uint32_t id)->double{
            if (!radj) return R2 * e2;
            const double strict = R2 + (double)radj[id];
            const double toward_horizon = R2 + (double)radj[id] * (1.0 - (double)beta);
            return std::max(strict, toward_horizon);
        };
        auto thrC = [&](uint32_t id)->double{ return collect_eps ? thrExpanded(id) : thrStrict(id); }; // 收集阈值
        auto thrE = [&](uint32_t id)->double{ return thrExpanded(id); }; // 扩展阈值(余量)
        auto D = [&](uint32_t id)->float{ ndist++; return dist.calculate(vec(id), q); };
        auto beam_key = [&](uint32_t id, double dd)->float{
            double key = dd;
            switch (score_cfg.mode) {
                case SearchScoreCfg::CENTER:
                    key = dd;
                    break;
                case SearchScoreCfg::BALL_GAP:
                    key = dd - thrStrict(id);
                    break;
                case SearchScoreCfg::BALL_COLLECT:
                    key = dd - thrC(id);
                    break;
                case SearchScoreCfg::BALL_EXPAND:
                    key = dd - thrE(id);
                    break;
                case SearchScoreCfg::BALL_HYBRID:
                    // R2 is constant for a fixed query/K, so only the point-wise margin affects ordering.
                    key = radj ? (dd - score_cfg.lambda * (double)radj[id]) : dd;
                    break;
            }
            if (!std::isfinite(key)) key = dd;
            if (key > (double)std::numeric_limits<float>::max()) return std::numeric_limits<float>::max();
            if (key < -(double)std::numeric_limits<float>::max()) return -std::numeric_limits<float>::max();
            return (float)key;
        };

        // ① 贪心下降：beam 走到 q 邻域（导航不受 k 限）
        NeighborPriorityQueue beam(L_descent);
        uint32_t ep = (std::getenv("ANQI_VAMANA_QUERY_ENTRY") || std::getenv("ANQI_QUERY_SINGLE_ENTRY")) ?
                      graph_entry_ : search_tree.get_leave_mediod(q, dist, random_seed);
        double dep = D(ep); visited.first_visit(ep);
        beam.insert(Neighbor(ep, beam_key(ep, dep)));
        std::vector<uint32_t> flood;
        if (dep <= thrE(ep)) flood.push_back(ep);
        if (dep <= thrC(ep)) results.push_back(ep);
        while (beam.has_unexpanded_node()) {
            uint32_t cur = beam.closest_unexpanded().id();
            uint32_t d = deg(cur); uint32_t* L = lnk(cur);
            for (uint32_t j = 0; j < d; j++) {
                uint32_t nb = L[j];
                if (!visited.first_visit(nb)) continue;
                double dd = D(nb);
                beam.insert(Neighbor(nb, beam_key(nb, dd)));
                if (dd <= thrE(nb)) flood.push_back(nb);
                if (dd <= thrC(nb)) results.push_back(nb);
            }
        }

        // ② 有界泛洪：k-自适应下只在 r_k(+余量) 区域扩张 → 小 k 少走节点 → 快
        size_t head = 0;
        while (head < flood.size()) {
            uint32_t cur = flood[head++];
            uint32_t d = deg(cur); uint32_t* L = lnk(cur);
            for (uint32_t j = 0; j < d; j++) {
                uint32_t nb = L[j];
                if (!visited.first_visit(nb)) continue;
                double dd = D(nb);
                if (dd <= thrC(nb)) results.push_back(nb);
                if (dd <= thrE(nb)) flood.push_back(nb);
            }
        }
        out_ndist = ndist;
    }

    void range_search(float* q, double R2, float eps, size_t L_descent,
                      std::vector<uint32_t>& results, size_t& out_ndist,
                      uint16_t* mark, uint16_t ver,
                      const float* radj = nullptr, float beta = 0.5f,
                      bool collect_eps = false) {
        QueryVisited visited;
        visited.stamps = mark;
        visited.version = ver;
        range_search(q, R2, eps, L_descent, results, out_ndist, visited,
                     radj, beta, collect_eps);
    }

    // ---- 查询（定长块 beam-search）----
    float search_index_parallel(float* queries, std::vector<uint32_t>& id_result,
                                std::vector<float>& dist_result, std::vector<size_t>& num_dist,
                                const size_t nq, const size_t k, const size_t ef) {
        if (visit_mark_.empty()) { visit_mark_.assign(n_threads, std::vector<uint16_t>(n, 0)); visit_ver_.assign(n_threads, 0); }
        std::fill(num_dist.begin(), num_dist.end(), 0);

        auto t0 = std::chrono::steady_clock::now();
        #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
        for (size_t i = 0; i < nq; i++) {
            int tid = omp_get_thread_num();
            auto* __restrict mark = visit_mark_[tid].data();
            uint16_t ver = ++visit_ver_[tid];
            if (ver == 0) { std::fill(visit_mark_[tid].begin(), visit_mark_[tid].end(), (uint16_t)0); ver = ++visit_ver_[tid]; } // 回绕重置
            float* q = queries + i * dim;

            NeighborPriorityQueue cands(ef);
            uint32_t ep = (std::getenv("ANQI_VAMANA_QUERY_ENTRY") || std::getenv("ANQI_QUERY_SINGLE_ENTRY")) ?
                          graph_entry_ : search_tree.get_leave_mediod(q, dist, random_seed + i);
            _mm_prefetch((char*)vec(ep), _MM_HINT_T0);
            float d0 = dist.fast_l2(vec(ep), q, vnorm(ep));
            num_dist[i]++;
            cands.insert(Neighbor(ep, d0));
            mark[ep] = ver;

            while (cands.has_unexpanded_node()) {
                uint32_t cur = cands.closest_unexpanded().id();
                uint32_t d = deg(cur); uint32_t* L = lnk(cur);
                if (d > 0) _mm_prefetch((char*)vec(L[0]), _MM_HINT_T0);
                for (uint32_t j = 0; j < d; j++) {
                    uint32_t nb = L[j];
                    if (j + 1 < d) { // 预取下一邻居：整块（向量+norm+边）+ 它的 visited 槽
                        _mm_prefetch((char*)vec(L[j + 1]), _MM_HINT_T0);
                        _mm_prefetch((char*)(mark + L[j + 1]), _MM_HINT_T0);
                    }
                    if (mark[nb] == ver) continue;
                    mark[nb] = ver;
                    float dd = dist.fast_l2(vec(nb), q, vnorm(nb));
                    num_dist[i]++;
                    cands.insert(Neighbor(nb, dd));
                }
            }
            size_t base = i * k;
            for (size_t j = 0; j < k; j++) { id_result[base + j] = cands[j].id(); dist_result[base + j] = cands[j].distance; }
        }
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now() - t0).count() / 1e6f;
    }

    // ---- 杂项 ----
    std::string generate_config() const {
        auto envs=[](const char* k, const char* d) {
            const char* v = std::getenv(k);
            return (v && *v) ? std::string(v) : std::string(d);
        };
        auto hash16=[](const std::string& s) {
            uint64_t h = 1469598103934665603ull;
            for (unsigned char c : s) { h ^= (uint64_t)c; h *= 1099511628211ull; }
            char buf[17];
            std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
            return std::string(buf);
        };
        auto clean=[](std::string s) {
            for (char& c : s) {
                bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                          (c >= 'A' && c <= 'Z') || c == '-' || c == '_';
                if (!ok) c = '_';
            }
            return s;
        };
        std::string cfg = "fd_metric_" + metric + "_M_" + std::to_string(M) + "_nnk_" + std::to_string(nn_k) +
                          "_ntrees_" + std::to_string(n_trees) + "_n_" + std::to_string(n) + "_dim_" + std::to_string(dim) +
                          "_efa_" + std::to_string(ef_alpha) + "_pa_" + std::to_string(prune_alpha) +
                          "_er_" + std::to_string(explore_range) +
                          "_graphnnd_" + envs("ANQI_GRAPHNND", "6");
        const char* gkey = std::getenv("ANQI_GRAPH_CACHE_KEY");
        if (gkey && *gkey) {
            cfg += "_gk_" + hash16(gkey);
        } else if (std::getenv("ANQI_VAMANA")) {
            cfg += "_vam1_style_" + envs("ANQI_VAMANA_STYLE", "global") +
                   "_L_" + envs("ANQI_VAMANA_L", "128") +
                   "_P_" + envs("ANQI_VAMANA_PASS", "2") +
                   "_bf_" + envs("ANQI_VAMANA_BATCH_FRAC", "0.02") +
                   "_mb_" + envs("ANQI_VAMANA_MAX_BATCH", "1000000") +
                   "_bb_" + envs("ANQI_VAMANA_BATCH_BASE", "2") +
                   "_ea_" + envs("ANQI_VAMANA_EARLY_ALPHA", "1.0") +
                   "_eao_" + envs("ANQI_VAMANA_EARLY_ALPHA_ONE", "1") +
                   "_entry_" + envs("ANQI_VAMANA_ENTRY", "0");
        } else {
            cfg += "_vam0";
        }
        if (std::getenv("ANQI_EDGE_POLICY")) {
            std::string edge_policy = envs("ANQI_EDGE_POLICY", "none");
            cfg += "_edge_" + edge_policy;
            const bool lsg_policy = edge_policy == "lsg_rng" ||
                                    edge_policy == "lsg_bubble_hybrid";
            if (lsg_policy) {
                std::string mu_key = envs("ANQI_LSG_CACHE_KEY", "");
                if (mu_key.empty()) mu_key = envs("ANQI_LSG_MU_FILE", "");
                cfg += "_lsa_" + envs("ANQI_LSG_ALPHA", "1.0") +
                       "_lsgmu_" + hash16(mu_key);
            }
            if (edge_policy != "lsg_rng") {
                cfg += "_eks_" + envs("ANQI_EDGE_KS", "10,50,100") +
                       "_enk_" + envs("ANQI_EDGE_NK", "100") +
                       "_erb_" + envs("ANQI_EDGE_RESID_BUDGET", "16") +
                       "_erl_" + envs("ANQI_EDGE_RESID_LAMBDA", "1.0") +
                       "_ebo_" + envs("ANQI_EDGE_BALL_OCCLUSION", "all");
                if (edge_policy == "bubble_auto" || edge_policy == "bubble_auto_full") {
                    cfg += "_eac_" + envs("ANQI_EDGE_AUTO_CANDIDATES", "M") +
                           "_eas_" + envs("ANQI_EDGE_AUTO_MAX_SWAPS", "M");
                }
            }
        }
        if (std::getenv("ANQI_REUSE_ORIG_KNN")) cfg += "_reuseorig1";
        return clean(cfg);
    }

    void print_degree_stats() {
        size_t mn = SIZE_MAX, mx = 0, sum = 0;
        for (size_t i = 0; i < n; i++) { size_t d = deg(i); mn = std::min(mn, d); mx = std::max(mx, d); sum += d; }
        std::cout << "Degree: Max=" << mx << " Min=" << mn << " Avg=" << (double)sum / n << " (fixed cap M=" << M << ")\n";
    }

    void print_edge_auto_stats() const {
        const uint64_t calls = edge_auto_calls_.load(std::memory_order_relaxed);
        const uint64_t targets = edge_auto_targets_.load(std::memory_order_relaxed);
        const uint64_t route_before = edge_auto_route_before_.load(std::memory_order_relaxed);
        const uint64_t route_after = edge_auto_route_after_.load(std::memory_order_relaxed);
        fprintf(stderr,
                "[edge-auto] calls=%llu changed=%llu swaps=%llu extra_dists=%llu "
                "avg_targets=%.2f avg_alternatives=%.2f required=%llu "
                "route_coverage=%.6f->%.6f verify_violations=%llu\n",
                (unsigned long long)calls,
                (unsigned long long)edge_auto_changed_.load(std::memory_order_relaxed),
                (unsigned long long)edge_auto_swaps_.load(std::memory_order_relaxed),
                (unsigned long long)edge_auto_extra_dists_.load(std::memory_order_relaxed),
                calls ? (double)targets / (double)calls : 0.0,
                calls ? (double)edge_auto_alternatives_.load(std::memory_order_relaxed) /
                            (double)calls : 0.0,
                (unsigned long long)edge_auto_required_.load(std::memory_order_relaxed),
                targets ? (double)route_before / (double)targets : 0.0,
                targets ? (double)route_after / (double)targets : 0.0,
                (unsigned long long)edge_auto_verify_violations_.load(std::memory_order_relaxed));
    }

    std::string visualize_parameters() {
        std::ostringstream o;
        o << "Metric: " << metric << " | M(cap): " << M << " | nn_k: " << nn_k
          << " | n_trees: " << n_trees << " | n: " << n << " | dim: " << dim
          << " | stride: " << stride_ << " floats (" << stride_ * 4 << "B/node)"
          << " | build: " << construction_time << "s\n";
        std::cout << o.str(); print_degree_stats(); return o.str();
    }

    // ---- 序列化（只存图：deg+links + tree；向量构造时从数据集重读）----
    bool save_index() {
        std::string p = folder_path + generate_config();
        std::ofstream f(p, std::ios::binary);
        if (!f.is_open()) return false;
        std::cout << "Index saved to: " << p << "\n";
        f.write((char*)&n, sizeof(n)); f.write((char*)&dim, sizeof(dim));
        f.write((char*)&M, sizeof(M)); f.write((char*)&construction_time, sizeof(construction_time));
        // deg + links（每点 1+deg 个 uint32，紧凑）
        for (size_t i = 0; i < n; i++) {
            uint32_t d = deg(i);
            f.write((char*)&d, 4);
            f.write((char*)lnk(i), d * 4);
        }
        save_tree(f);
        return static_cast<bool>(f);
    }

    bool load_index() {
        std::string p = folder_path + generate_config();
        std::ifstream f(p, std::ios::binary);
        if (!f.good()) { std::cerr << "no cached index: " << p << "\n"; return false; }
        std::cout << "Loading index from: " << p << "\n";
        size_t fn = 0, fdim = 0, fM = 0;
        if (!f.read((char*)&fn, sizeof(fn)) || !f.read((char*)&fdim, sizeof(fdim)) ||
            !f.read((char*)&fM, sizeof(fM)) ||
            !f.read((char*)&construction_time, sizeof(construction_time)) ||
            fn != n || fdim != dim || fM != M || !std::isfinite(construction_time)) {
            std::cerr << "incompatible or truncated cached index: " << p << "\n";
            return false;
        }
        for (size_t i = 0; i < n; i++) {
            uint32_t d = 0;
            if (!f.read((char*)&d, 4) || d > M ||
                !f.read((char*)lnk(i), d * 4)) {
                std::cerr << "invalid adjacency data in cached index: " << p << "\n";
                return false;
            }
            for (uint32_t j = 0; j < d; j++) {
                if (lnk(i)[j] >= n) {
                    std::cerr << "invalid neighbor id in cached index: " << p << "\n";
                    return false;
                }
            }
            deg(i) = d;
        }
        if (!load_tree(f)) {
            std::cerr << "invalid RP tree in cached index: " << p << "\n";
            return false;
        }
        graph_entry_ = choose_vamana_entry();
        return true;
    }

    void save_tree(std::ofstream& f) {
        f.write((char*)&search_tree.leaf_size, sizeof(search_tree.leaf_size));
        f.write((char*)&search_tree.n_leaves, sizeof(search_tree.n_leaves));
        size_t ns = search_tree.nodes.size(); f.write((char*)&ns, sizeof(ns));
        for (const auto& nd : search_tree.nodes) {
            f.write((char*)&nd.left, sizeof(nd.left)); f.write((char*)&nd.right, sizeof(nd.right));
            f.write((char*)&nd.offset, sizeof(nd.offset));
            size_t hs = nd.hyperplane.size(); f.write((char*)&hs, sizeof(hs));
            f.write((char*)nd.hyperplane.data(), hs * sizeof(float));
            size_t is = nd.indices.size(); f.write((char*)&is, sizeof(is));
            f.write((char*)nd.indices.data(), is * sizeof(uint32_t));
        }
    }
    bool load_tree(std::ifstream& f) {
        size_t leaf_size = 0, n_leaves = 0, ns = 0;
        if (!f.read((char*)&leaf_size, sizeof(leaf_size)) ||
            !f.read((char*)&n_leaves, sizeof(n_leaves)) ||
            !f.read((char*)&ns, sizeof(ns)) || leaf_size == 0 || leaf_size > 1000000 ||
            n_leaves == 0 || n_leaves > n || ns == 0 || ns > 2 * n) {
            return false;
        }
        search_tree.leaf_size = leaf_size;
        search_tree.n_leaves = n_leaves;
        search_tree.nodes.clear();
        search_tree.nodes.reserve(ns);
        for (size_t i = 0; i < ns; i++) {
            size_t left, right; float offset; std::vector<float> hp; std::vector<uint32_t> idx;
            if (!f.read((char*)&left, sizeof(left)) ||
                !f.read((char*)&right, sizeof(right)) ||
                !f.read((char*)&offset, sizeof(offset)) || !std::isfinite(offset)) {
                return false;
            }
            size_t hs = 0;
            if (!f.read((char*)&hs, sizeof(hs)) || hs > dim) return false;
            hp.resize(hs);
            if (!f.read((char*)hp.data(), hs * sizeof(float))) return false;
            size_t is = 0;
            if (!f.read((char*)&is, sizeof(is)) || is > n) return false;
            idx.resize(is);
            if (!f.read((char*)idx.data(), is * sizeof(uint32_t)) ||
                std::any_of(idx.begin(), idx.end(), [this](uint32_t id) { return id >= n; })) {
                return false;
            }
            const bool leaf = left == UINT32_MAX && right == UINT32_MAX;
            const bool internal = left < i && right < i;
            if ((!leaf && !internal) || (leaf && idx.empty()) ||
                (internal && hs != dim)) {
                return false;
            }
            search_tree.nodes.emplace_back(rptnode(left, right, offset, hp, idx));
        }
        return static_cast<bool>(f);
    }
};

} // namespace nndgraph
