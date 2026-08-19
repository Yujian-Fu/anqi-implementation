#pragma once
// =============================================================================
//  heap.h —— HeapList：Vista 邻近图索引的图存储核心
// -----------------------------------------------------------------------------
//  整张图就存在这里。对每个数据点 i 维护三组并行数组：
//
//      indices[i]          点 i 的出边邻居 id（邻接表）
//      values[i]           对应距离 distance(i, indices[i][j])（与 indices 平行）
//      reverse_indices[i]  点 i 的入边邻居 id（谁指向 i）
//
//  indices[i] 在构建过程中有“双重身份”：
//    1. 初始化阶段 —— 它是容量固定为 nn_k 的【二叉最大堆】（按距离），所以
//       values[i][0] 永远是【当前最远】的邻居。新候选用 check_push 压入，满了就
//       淘汰最远的那个。用最大堆正是为了 O(1) 找到最差邻居、O(log k) 替换。
//    2. 调用 sort_heap_by_dist() 之后 —— 它被重新解释为【按距离升序的邻接表】
//       （values[i][0] = 最近），这是选边 / 搜索代码期望的形态。
//
//  reverse_indices 同步维护，便于构建器低成本地问“谁把 i 当成出边邻居？”。这个
//  入度正是 Vista 分布感知边预算的依据（入度低 == 难被检索到的点）。
//
//  并发：每个点一把锁 mtxs[i]。任何修改 indices[i]/values[i]/reverse_indices[i]
//  的代码都必须持有 mtxs[i]。锁总是一次只持有一个 id（或先释放再取下一个），避免死锁。
// =============================================================================

#include "utils.h"
#include <cstdlib>

namespace nndgraph
{

class HeapList
{
public:
    size_t n_heaps;   // 点数（每点一个堆 / 邻接表）
    size_t n_nodes;   // 初始化阶段每个堆的容量（== nn_k）
    bool error_check; // true 时开启额外（较慢）的一致性断言
    bool skip_dedup = (std::getenv("ANQI_NODEDUP") != nullptr); // 诊断:跳过 O(nn_k) 查重扫描
    bool track_new = false;   // NN-descent new/old 旗标开关（ANQI_NEWOLD）：true 时分配并维护 isnew

    std::vector<std::mutex> mtxs; // 每点一把锁，保护下面三组数组

    std::vector<std::vector<uint32_t>> indices;         // 出边邻居 id
    std::vector<std::vector<float>>    values;          // 距离，与 indices 平行
    std::vector<std::vector<uint32_t>> reverse_indices; // 入边邻居 id
    // NN-descent new/old 旗标（仅 track_new 时分配）：isnew[i][slot]=1 表示该槽邻居为"新"
    //（上一轮才进堆、本轮需 join）；与 indices/values 平行，随堆 sift-down 一并搬移。
    std::vector<std::vector<uint8_t>> isnew;

    HeapList() : n_heaps(0), n_nodes(0), error_check(false) {}

    // 分配 n_heaps 个点，每个点一个容量 n_nodes 的堆。槽位初值：
    // index = UINT32_MAX（哨兵“无邻居”），value = FLT_MAX。
    HeapList(size_t n_heaps, size_t n_nodes, bool error_check, bool track_new = false)
        : n_heaps(n_heaps), n_nodes(n_nodes), error_check(error_check), track_new(track_new), mtxs(n_heaps)
    {
        indices.resize(n_heaps);
        values.resize(n_heaps);
        reverse_indices.resize(n_heaps);
        if (track_new) isnew.resize(n_heaps);
        for (size_t i = 0; i < n_heaps; i++) {
            indices[i].resize(n_nodes, UINT32_MAX);
            values[i].resize(n_nodes, FLT_MAX);
            if (track_new) isnew[i].resize(n_nodes, 0);
        }
    }

    // id0 与 id1 之间是否已存在（无向）边？两个邻接表都查。一次只锁一端。
    bool check_exist(uint32_t id0, uint32_t id1)
    {
        {
            std::unique_lock<std::mutex> lock(mtxs[id0]);
            if (std::find(indices[id0].begin(), indices[id0].end(), id1) != indices[id0].end())
                return true;
        }
        {
            std::unique_lock<std::mutex> lock(mtxs[id1]);
            if (std::find(indices[id1].begin(), indices[id1].end(), id0) != indices[id1].end())
                return true;
        }
        return false;
    }

    // 点 heap_idx 当前的真实邻居数（非哨兵）。
    size_t heap_size(size_t heap_idx)
    {
        std::unique_lock<std::mutex> lock(mtxs[heap_idx]);
        size_t count = 0;
        for (size_t i = 0; i < n_nodes; i++)
            if (indices[heap_idx][i] != UINT32_MAX) count++;
        return count;
    }

    // 尝试把邻居 (node_idx, value) 压入点 heap_idx 的【最大堆】。
    //
    // 返回 false（不插入）的情况：
    //   - value 比当前最远邻居还差（堆已被更近的点占满），或
    //   - node_idx 已存在。
    // 否则把当前最大值下沉挤出、放入 (node_idx, value)；当 update_reverse 为真时
    // 同步维护反向边：
    //   * 新增反向边 node_idx -> heap_idx
    //   * 删除被淘汰者 max_id -> heap_idx 的旧反向边
    //
    // 注意：最大堆意味着 values[heap_idx][0] 是最大（最远）值。
    bool check_push(uint32_t heap_idx, uint32_t node_idx, float value, bool update_reverse, uint8_t mark_new = 1)
    {
        uint32_t max_id;
        {
            std::unique_lock<std::mutex> lock(mtxs[heap_idx]);

            // 比当前最远（堆顶）还差，直接拒绝。
            if (value > values[heap_idx][0]) return false;

            auto bound_size = indices[heap_idx].size();
            for (size_t i = 0; i < bound_size; ++i) // 去重
                if (indices[heap_idx][i] == node_idx) return false;

            max_id = indices[heap_idx][0]; // 即将被淘汰的邻居（当前最远）
            uint8_t* hn = track_new ? isnew[heap_idx].data() : nullptr; // new/old 标志随堆 sift 一并搬移

            // 从堆顶下沉：把较大的子节点上提，直到找到新（较小）值该放的位置。
            size_t current = 0;
            size_t swap;
            while (true) {
                size_t left_child  = 2 * current + 1;
                size_t right_child = left_child + 1;

                if (left_child >= bound_size) {
                    break; // 到叶子
                } else if (right_child >= bound_size) {
                    if (values[heap_idx][left_child] > value) swap = left_child; else break;
                } else if (values[heap_idx][left_child] >= values[heap_idx][right_child]) {
                    if (values[heap_idx][left_child] > value) swap = left_child; else break;
                } else {
                    if (values[heap_idx][right_child] > value) swap = right_child; else break;
                }
                indices[heap_idx][current] = indices[heap_idx][swap];
                values[heap_idx][current]  = values[heap_idx][swap];
                if (hn) hn[current] = hn[swap];
                current = swap;
            }
            indices[heap_idx][current] = node_idx;
            values[heap_idx][current]  = value;
            if (hn) hn[current] = mark_new;
        }

        if (!update_reverse) return true;

        // 新增反向边 node_idx -> heap_idx。
        {
            std::unique_lock<std::mutex> lock(mtxs[node_idx]);
            reverse_indices[node_idx].emplace_back(heap_idx);
        }

        // 删除被淘汰邻居的旧反向边（max_id -> heap_idx）。
        // max_id == UINT32_MAX 表示该槽原来是空的（无反向边可删）。
        if (max_id != UINT32_MAX) {
            if (max_id >= n_heaps) { std::cout << "Invalid max_id " << max_id << "\n"; exit(0); }
            std::unique_lock<std::mutex> lock(mtxs[max_id]);
            bool found_flag = false;
            auto size = reverse_indices[max_id].size();
            for (size_t i = 0; i < size; i++) {
                if (reverse_indices[max_id][i] == heap_idx) {
                    // 与末尾交换再缩容：O(1) 删除，不关心顺序
                    std::swap(reverse_indices[max_id][i], reverse_indices[max_id][size - 1]);
                    found_flag = true;
                    break;
                }
            }
            if (found_flag) {
                reverse_indices[max_id].resize(size - 1);
            } else {
                std::cerr << "Cannot find the reverse edge to be deleted from max_id: "
                          << max_id << " to " << heap_idx << "\n";
                exit(0);
            }
        }
        return true;
    }

    // 无锁版 check_push（update_reverse=false 语义）。
    // ★ 仅当调用方保证【heap_idx 被单线程独占】时安全 —— per-tree 叶内共现即满足：
    //   一棵树内叶子是对点的划分，每个点恰在一个叶子→只有一个线程碰它的堆。
    //   去掉 mutex，省掉 init_knn 里 ~3.2e11 次锁(大 leaf 的主瓶颈)。
    bool push_nolock(uint32_t heap_idx, uint32_t node_idx, float value, uint8_t mark_new = 1)
    {
        if (value > values[heap_idx][0]) return false;
        auto bound_size = indices[heap_idx].size();
        if (!skip_dedup)
            for (size_t i = 0; i < bound_size; ++i)
                if (indices[heap_idx][i] == node_idx) return false; // 去重 O(nn_k) 扫描
        uint8_t* hn = track_new ? isnew[heap_idx].data() : nullptr; // new/old 标志随堆 sift 一并搬移
        size_t current = 0, swap;
        while (true) {
            size_t left_child = 2 * current + 1, right_child = left_child + 1;
            if (left_child >= bound_size) break;
            else if (right_child >= bound_size) { if (values[heap_idx][left_child] > value) swap = left_child; else break; }
            else if (values[heap_idx][left_child] >= values[heap_idx][right_child]) { if (values[heap_idx][left_child] > value) swap = left_child; else break; }
            else { if (values[heap_idx][right_child] > value) swap = right_child; else break; }
            indices[heap_idx][current] = indices[heap_idx][swap];
            values[heap_idx][current]  = values[heap_idx][swap];
            if (hn) hn[current] = hn[swap];
            current = swap;
        }
        indices[heap_idx][current] = node_idx;
        values[heap_idx][current]  = value;
        if (hn) hn[current] = mark_new;
        return true;
    }

    // 从出边整体重建所有 reverse_indices（转置）：对每条 i -> indices[i][j]，
    // 把 i 追加到该邻居的 reverse_indices。
    void push_reverse(size_t n_threads)
    {
        #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
        for (size_t i = 0; i < n_heaps; i++) reverse_indices[i].clear();

        #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
        for (size_t i = 0; i < n_heaps; i++) {
            if (indices[i].size() != n_nodes)
                std::cout << "The number of edges should be: " << n_nodes
                          << " instead of " << indices[i].size() << "\n";
            for (size_t j = 0; j < indices[i].size(); j++) {
                auto nei_id = indices[i][j];
                if (nei_id == UINT32_MAX || nei_id >= n_heaps) continue;
                std::unique_lock<std::mutex> lock(mtxs[nei_id]);
                reverse_indices[nei_id].emplace_back(i);
            }
        }
    }

    // 返回点 id 的邻居 (id, 距离) 列表，按距离【升序】，且【不修改】已存储的堆
    // （在副本上操作）。做法是对最大堆做原地堆排序：反复把堆顶（最大）移到末尾、
    // 对收缩后的前缀重新堆化 => 升序。
    std::vector<std::pair<uint32_t, float>> get_sorted_data(uint32_t id)
    {
        if (id >= indices.size() || id >= values.size())
            throw std::out_of_range("Invalid id: out of bounds.");

        size_t bound_size = indices[id].size();
        std::vector<uint32_t> temp_indices = indices[id]; // 副本，非破坏性
        std::vector<float>    temp_values  = values[id];

        for (size_t sorted_length = 0; sorted_length < bound_size - 1; sorted_length++) {
            size_t last_index = bound_size - sorted_length - 1;
            size_t current = 0;
            size_t swap;

            // 把当前最大值（堆顶）移到未排序区末尾。
            std::swap(temp_values[current],  temp_values[last_index]);
            std::swap(temp_indices[current], temp_indices[last_index]);

            // 对收缩区间 [0, last_index) 重新堆化。
            while (true) {
                size_t left_child  = 2 * current + 1;
                size_t right_child = left_child + 1;
                if (left_child >= last_index) {
                    break;
                } else if (right_child >= last_index) {
                    if (temp_values[left_child] > temp_values[current]) swap = left_child; else break;
                } else if (temp_values[left_child] >= temp_values[right_child]) {
                    if (temp_values[left_child] > temp_values[current]) swap = left_child; else break;
                } else {
                    if (temp_values[right_child] > temp_values[current]) swap = right_child; else break;
                }
                std::swap(temp_values[current],  temp_values[swap]);
                std::swap(temp_indices[current], temp_indices[swap]);
                current = swap;
            }
        }

        std::vector<std::pair<uint32_t, float>> sorted_result;
        for (size_t i = 0; i < bound_size; i++)
            sorted_result.emplace_back(temp_indices[i], temp_values[i]);
        return sorted_result;
    }

    // 与 get_sorted_data 相同的堆排序，但【原地】并行地作用到每个点的存储数组。
    // 调用后 indices[i]/values[i] 从最大堆变为按距离升序的邻接表。
    void sort_heap_by_dist(size_t n_threads)
    {
        #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
        for (size_t heap_idx = 0; heap_idx < n_heaps; heap_idx++) {
            size_t bound_size = indices[heap_idx].size();
            for (size_t sorted_length = 0; sorted_length < bound_size - 1; sorted_length++) {
                size_t last_index = bound_size - sorted_length - 1;
                size_t current = 0;
                size_t swap;

                std::swap(values[heap_idx][current],  values[heap_idx][last_index]);
                std::swap(indices[heap_idx][current], indices[heap_idx][last_index]);

                while (true) {
                    size_t left_child  = 2 * current + 1;
                    size_t right_child = left_child + 1;
                    if (left_child >= last_index) {
                        break;
                    } else if (right_child >= last_index) {
                        if (values[heap_idx][left_child] > values[heap_idx][current]) swap = left_child; else break;
                    } else if (values[heap_idx][left_child] >= values[heap_idx][right_child]) {
                        if (values[heap_idx][left_child] > values[heap_idx][current]) swap = left_child; else break;
                    } else {
                        if (values[heap_idx][right_child] > values[heap_idx][current]) swap = right_child; else break;
                    }
                    std::swap(values[heap_idx][current],  values[heap_idx][swap]);
                    std::swap(indices[heap_idx][current], indices[heap_idx][swap]);
                    current = swap;
                }
            }
        }
    }

    // 对初始 kNN 图做 RNG（相对邻域图）式剪枝。
    // 对每个点 i，按距离升序扫邻居，只有当“没有已保留的更近邻居 id1 比候选 id2 更靠近 id2”
    // （dist.prune_conflict）时才保留 (id2, d2)。为避免把点剪到近乎孤立，设每点保底度数
    // （lower_bound = 平均保留度），度数不足的点恢复部分被剪边。最后从存活出边重建反向边。
    //
    // 注意：默认构建路径并不调用本函数（作为可选的备用剪枝器保留）。
    void pre_prune_sparse(size_t n_threads, DistCal & dist, size_t dim, float * data)
    {
        sort_heap_by_dist(n_threads);

        std::vector<std::vector<bool>> preserve_flags(n_heaps, std::vector<bool>(n_nodes, true));
        #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
        for (size_t i = 0; i < n_heaps; i++) {
            auto size = indices[i].size();
            if (size != n_nodes) {
                #pragma omp critical
                { std::cout << "Error: Initial size should be n_nodes: " << size << "\n"; exit(0); }
            }
            for (size_t j = 0; j < size; j++) {
                if (j > 0 && values[i][j] < values[i][j - 1]) {
                    std::cout << "Sort error: " << values[i][j - 1] << " < " << values[i][j] << "\n"; exit(0);
                }
                float d1 = values[i][j];
                uint32_t id1 = indices[i][j];
                if (!preserve_flags[i][j]) continue; // 已被剪，不能再用来剪别人
                for (size_t k = j + 1; k < size; k++) {
                    float d2 = values[i][k];
                    auto id2 = indices[i][k];
                    if (!preserve_flags[i][k]) continue;
                    float d = dist.calculate(data + size_t(id1) * dim, data + size_t(id2) * dim);
                    if (dist.prune_conflict(d1, d2, d)) preserve_flags[i][k] = false;
                }
            }
        }

        // 求平均保留度 -> 每点保底。
        std::vector<size_t> node_edges(n_heaps, 0);
        #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
        for (size_t i = 0; i < n_heaps; i++)
            node_edges[i] = std::accumulate(preserve_flags[i].begin(), preserve_flags[i].end(), size_t(0));
        size_t sum_preserved_edges = std::accumulate(node_edges.begin(), node_edges.end(), size_t(0));
        size_t lower_bound = sum_preserved_edges / n_heaps + 1;
        std::cout << "Average number of edges after pruning: " << lower_bound << "\n";

        // 度数不足的点恢复被剪边，然后物理压缩数组。
        #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
        for (size_t i = 0; i < n_heaps; i++) {
            size_t total = std::accumulate(preserve_flags[i].begin(), preserve_flags[i].end(), 0);
            if (total < lower_bound) {
                for (size_t j = 0; j < n_nodes; j++) {
                    if (!preserve_flags[i][j]) {
                        preserve_flags[i][j] = true;
                        if (++total == lower_bound) break;
                    }
                }
            }
            for (size_t j = 0; j < indices[i].size();) {
                if (!preserve_flags[i][j]) {
                    preserve_flags[i].erase(preserve_flags[i].begin() + j);
                    indices[i].erase(indices[i].begin() + j);
                    values[i].erase(values[i].begin() + j);
                } else { j++; }
            }
        }

        // 从剪枝后的出边重建反向边。
        for (size_t i = 0; i < n_heaps; i++) reverse_indices[i].clear();
        #pragma omp parallel for num_threads(n_threads) schedule(dynamic)
        for (size_t i = 0; i < n_heaps; i++) {
            auto size = indices[i].size();
            for (size_t j = 0; j < size; j++) {
                auto out_id = indices[i][j];
                std::unique_lock<std::mutex> lock(mtxs[out_id]);
                reverse_indices[out_id].emplace_back(i);
            }
        }
    }

    // 打印出度 / 入度 / 反向度的 max/min/avg 统计。当图维护显式反向边时报告
    // 反向度 + 总度数；否则通过统计入边数推算入度。配合 error_check / mutual_edge
    // 还会断言无重复边、以及出边<->反向边互相一致。values_kept 额外检查 indices/values 对齐。
    void print_edge_info(bool mutual_edge = false, bool values_kept = false, size_t n_threads = 128)
    {
        size_t max_out = 0, min_out = UINT32_MAX;        uint64_t sum_out = 0;
        size_t max_reverse = 0, min_reverse = UINT32_MAX; uint64_t sum_reverse = 0;
        size_t max_total = 0, min_total = UINT32_MAX;    uint64_t sum_total = 0;

        std::vector<size_t> in_edges(n_heaps, 0);
        std::cout << "The information of edges:\n";

        #pragma omp parallel for reduction(max:max_out) reduction(min:min_out) reduction(+:sum_out) \
                                 reduction(max:max_reverse) reduction(min:min_reverse) reduction(+:sum_reverse) \
                                 reduction(max:max_total) reduction(min:min_total) reduction(+:sum_total) num_threads(n_threads)
        for (size_t i = 0; i < n_heaps; i++) {
            for (auto out_edge : indices[i]) {
                if (out_edge >= n_heaps) {
                    #pragma omp critical
                    { std::cout << "Edge error: " << i << " " << out_edge << "\n"; exit(0); }
                }
                std::unique_lock<std::mutex> lock(mtxs[out_edge]);
                in_edges[out_edge]++;
            }

            if (error_check) {
                std::unordered_set<uint32_t> indices_set(indices[i].begin(), indices[i].end());
                if (indices_set.size() < indices[i].size()) {
                    #pragma omp critical
                    { std::cerr << "Duplicate edges in out edges\n"; exit(0); }
                }
                std::unordered_set<uint32_t> reverse_indices_set(reverse_indices[i].begin(), reverse_indices[i].end());
                if (reverse_indices_set.size() < reverse_indices[i].size()) {
                    #pragma omp critical
                    { std::cerr << "Duplicate edges in reverse edges\n"; exit(0); }
                }
            }

            size_t out_size = indices[i].size();
            size_t reverse_size = reverse_indices[i].size();
            size_t total_size = out_size + reverse_size;

            if (out_size > max_out) max_out = out_size;
            if (out_size < min_out) min_out = out_size;
            sum_out += out_size;
            if (reverse_size > max_reverse) max_reverse = reverse_size;
            if (reverse_size < min_reverse) min_reverse = reverse_size;
            sum_reverse += reverse_size;
            if (total_size > max_total) max_total = total_size;
            if (total_size < min_total) min_total = total_size;
            sum_total += total_size;

            if (values_kept && out_size != values[i].size()) {
                #pragma omp critical
                {
                    std::cerr << "The dimension on index and value is not the same for " << i
                              << "th vector: " << out_size << " " << values[i].size() << "\n";
                    exit(0);
                }
            }
        }

        std::cout << std::fixed << std::setprecision(2);
        if (max_reverse > 0) {
            // 图维护显式反向边：报告 出 / 反向 / 总。
            std::cout << "Edge Information:\n-----------------\n";
            std::cout << "Out-edges\n  Max: " << std::setw(8) << max_out
                      << " | Min: " << std::setw(8) << min_out
                      << " | Avg: " << std::setw(8) << static_cast<float>(sum_out) / n_heaps << "\n";
            std::cout << "Reverse-edges\n  Max: " << std::setw(8) << max_reverse
                      << " | Min: " << std::setw(8) << min_reverse
                      << " | Avg: " << std::setw(8) << static_cast<float>(sum_reverse) / n_heaps << "\n";
            std::cout << "Total-edges\n  Max: " << std::setw(8) << max_total
                      << " | Min: " << std::setw(8) << min_total
                      << " | Avg: " << std::setw(8) << static_cast<float>(sum_total) / n_heaps << "\n\n";
        } else {
            // 不存反向边：用入边计数推算入度。
            std::cout << "Edge Information:\n-----------------\n";
            std::cout << "Out-edges\n  Max: " << std::setw(8) << max_out
                      << " | Min: " << std::setw(8) << min_out
                      << " | Avg: " << std::setw(8) << static_cast<float>(sum_out) / n_heaps << "\n";
            std::cout << "In-edges\n";
            size_t sum_in_edges = 0;
            #pragma omp parallel for reduction(+:sum_in_edges)
            for (size_t i = 0; i < in_edges.size(); ++i) sum_in_edges += in_edges[i];
            std::cout << "  Max: " << std::setw(8) << *std::max_element(in_edges.begin(), in_edges.end())
                      << " | Min: " << std::setw(8) << *std::min_element(in_edges.begin(), in_edges.end())
                      << " | Avg: " << std::setw(8) << float(sum_in_edges) / n_heaps << "\n\n";
        }

        if (mutual_edge) {
            if (std::abs(static_cast<int64_t>(sum_out) - static_cast<int64_t>(sum_reverse)) > 1e-9) {
                std::cerr << "Error: Number of out edges and reverse edges are not the same: "
                          << sum_out << " " << sum_reverse << "\n";
                exit(0);
            }
            #pragma omp parallel for num_threads(n_threads)
            for (size_t i = 0; i < n_heaps; i++) {
                for (size_t j = 0; j < indices[i].size(); j++) { // 出边都要有对应反向边
                    auto nei_id = indices[i][j];
                    if (nei_id >= n_heaps) {
                        #pragma omp critical
                        { std::cerr << "Invalid edge exists: " << nei_id << "\n"; exit(0); }
                    }
                    if (std::find(reverse_indices[nei_id].begin(), reverse_indices[nei_id].end(), i)
                        == reverse_indices[nei_id].end()) {
                        #pragma omp critical
                        { std::cerr << "Cannot find the reverse edge for " << nei_id << " to " << i << "\n"; exit(0); }
                    }
                }
                for (size_t j = 0; j < reverse_indices[i].size(); j++) { // 反向边都要有对应出边
                    auto nei_id = reverse_indices[i][j];
                    if (std::find(indices[nei_id].begin(), indices[nei_id].end(), i) == indices[nei_id].end()) {
                        #pragma omp critical
                        { std::cerr << "Cannot find the edge from " << nei_id << " to " << i << "\n"; exit(0); }
                    }
                }
            }
        }
    }
};

} // namespace nndgraph
