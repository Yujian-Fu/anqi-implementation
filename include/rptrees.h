#pragma once
// Random-projection trees for AKNN initialization and query entry points.

#include "utils.h"
#include "distance.h"
#include <atomic>
#include <cstdio>

namespace nndgraph
{

// RP 树节点。内部节点：left/right = 子树下标，hyperplane/offset = 分割超平面；
// 叶子节点：left == right == UINT32_MAX，indices = 落在该叶子的点 id（[0] 为叶心）。
struct rptnode {
    size_t left;
    size_t right;
    float  offset;
    std::vector<float>    hyperplane;
    std::vector<uint32_t> indices;

    rptnode(size_t left, size_t right, float offset,
            const std::vector<float> &hyperplane, const std::vector<uint32_t> &indices)
        : left(left), right(right), offset(offset), hyperplane(hyperplane), indices(indices) {}
};

// 在一组点上做一次随机投影分割，返回 (左子集, 右子集, 超平面向量, 偏置)。
std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<float>, float>
random_projection_split(const float * data, const std::vector<uint32_t> & indices,
                        size_t n, size_t dim, bool angular, uint64_t seed)
{
    // 随机取两个不同的点作为分割参考。
    auto rand0 = generate_random_integer(0, n - 1, seed);
    auto rand1 = generate_random_integer(0, n - 1, seed + 1);
    if (rand0 == rand1) rand0 = (rand1 + 1) % n;
    uint32_t idx0 = indices[rand0];
    uint32_t idx1 = indices[rand1];

    std::vector<float> hyperplane_vector(dim);
    float hyperplane_offset = 0.0f;

    if (angular) {
        // 角度度量：用单位化后的两点之差作为超平面法向，偏置为 0。
        float norm0 = std::sqrt(InnerProductSIMD16ExtAVX512_(data + size_t(idx0) * dim, data + size_t(idx0) * dim, dim));
        float norm1 = std::sqrt(InnerProductSIMD16ExtAVX512_(data + size_t(idx1) * dim, data + size_t(idx1) * dim, dim));
        if (std::abs(norm0) < EPS) norm0 = 1.0f;
        if (std::abs(norm1) < EPS) norm1 = 1.0f;
        for (size_t i = 0; i < dim; i++)
            hyperplane_vector[i] = data[idx0 * dim + i] / norm0 - data[idx1 * dim + i] / norm1;
        float hyperplane_norm = std::sqrt(InnerProductSIMD16ExtAVX512_(hyperplane_vector.data(), hyperplane_vector.data(), dim));
        if (std::abs(hyperplane_norm) < EPS) hyperplane_norm = 1.0f;
        for (size_t i = 0; i < dim; i++) hyperplane_vector[i] /= hyperplane_norm;
    } else {
        // L2 度量：法向 = 两点之差，偏置 = 法向·中点（即过中点的垂直平分面）。
        std::vector<float> midpoint(dim);
        for (size_t i = 0; i < dim; i++) {
            midpoint[i]          = (data[idx0 * dim + i] + data[idx1 * dim + i]) / 2;
            hyperplane_vector[i] =  data[idx0 * dim + i] - data[idx1 * dim + i];
        }
        hyperplane_offset = InnerProductSIMD16ExtAVX512_(hyperplane_vector.data(), midpoint.data(), dim);
    }

    // 按 margin = 法向·点 - 偏置 的符号把点分到左/右；落在面上（|margin|<EPS）随机分。
    std::vector<uint32_t> left_indices, right_indices;
    for (size_t i = 0; i < n; i++) {
        float margin;
        uint32_t indice = indices[i];
        if (angular)
            margin = InnerProductSIMD16ExtAVX512_(hyperplane_vector.data(), data + size_t(indice) * dim, dim);
        else
            margin = InnerProductSIMD16ExtAVX512_(hyperplane_vector.data(), data + size_t(indice) * dim, dim) - hyperplane_offset;

        if (margin < -EPS)      left_indices.push_back(indice);
        else if (margin > EPS)  right_indices.push_back(indice);
        else (generate_random_integer(0, 1, seed + i) == 0 ? left_indices : right_indices).push_back(indice);
    }
    // 万一全落一边：退化为随机均分，保证两边都非空。
    if (left_indices.empty() || right_indices.empty()) {
        left_indices.clear();
        right_indices.clear();
        for (size_t i = 0; i < n; i++)
            (generate_random_integer(0, 1, seed + i) == 0 ? left_indices : right_indices).push_back(indices[i]);
    }
    return std::make_tuple(left_indices, right_indices, hyperplane_vector, hyperplane_offset);
}

class rptree
{
public:
    size_t leaf_size;
    size_t n_leaves;
    std::vector<rptnode> nodes; // 后序追加：子树先于父节点入 nodes，根节点在最后

    rptree() {}
    rptree(size_t leaf_size) : leaf_size(leaf_size), n_leaves(0) {}

    void add_leaf(std::vector<uint32_t> indices) {
        nodes.emplace_back(rptnode(UINT32_MAX, UINT32_MAX, FLT_MAX, {}, indices));
        n_leaves++;
    }

    size_t get_index() const { return nodes.size() - 1; } // 最近加入节点的下标（即子树根）

    void add_node(size_t left_subtree, size_t right_subtree, float offset, const std::vector<float> & hyperplane) {
        nodes.push_back(rptnode(left_subtree, right_subtree, offset, hyperplane, {}));
    }

    // 把查询沿树下行到叶子，返回叶心（叶子 indices[0]，建树后由 index 设为最接近质心的点）。
    uint32_t get_leave_mediod(float * query_data, const DistCal & dist, uint64_t seed) {
        auto index = get_index(); // 从根开始
        while (nodes[index].left != UINT32_MAX) {
            const std::vector<float> &hyperplane_vector = nodes[index].hyperplane;
            float hyperplane_offset = nodes[index].offset;
            float margin = dist.InnerProductSIMD16ExtAVX512_(hyperplane_vector.data(), query_data, hyperplane_vector.size()) - hyperplane_offset;

            if (margin < -EPS)                                   index = nodes[index].left;
            else if (margin > EPS)                               index = nodes[index].right;
            else if (generate_random_integer(0, 1, seed) == 0)   index = nodes[index].left;
            else                                                 index = nodes[index].right;
        }
        // 空叶子兜底(退化分割可能产生 0 点叶子):返回有效入口 0,beam 下降自会修正,避免 indices[0] 越界崩。
        if (nodes[index].indices.empty()) return 0;
        return nodes[index].indices[0];
    }
};

// 递归建一棵 RP 树：点数 <= leaf_size 或到达 max_depth 即成叶子，否则分割后递归左右。
void build_rptree_recursively(rptree & rp_tree, float * data, size_t n, size_t dim,
                              std::vector<uint32_t> & indices, size_t leaf_size,
                              bool angular, size_t max_depth, uint64_t seed)
{
    if (indices.size() <= leaf_size) { rp_tree.add_leaf(indices); return; }
    if (max_depth == 0) { // 深度耗尽：截成一个叶子
        indices.resize(std::min(leaf_size, indices.size()));
        rp_tree.add_leaf(indices);
        return;
    }

    std::vector<uint32_t> left_indices, right_indices;
    std::vector<float> hyperplane;
    float offset;
    std::tie(left_indices, right_indices, hyperplane, offset) =
        random_projection_split(data, indices, n, dim, angular, seed);

    indices.clear();
    std::vector<uint32_t>().swap(indices); // 释放内存（深递归省内存）

    build_rptree_recursively(rp_tree, data, left_indices.size(), dim, left_indices, leaf_size, angular, max_depth - 1, seed + 1);
    auto left_subtree = rp_tree.get_index();
    build_rptree_recursively(rp_tree, data, right_indices.size(), dim, right_indices, leaf_size, angular, max_depth - 1, seed + 2);
    auto right_subtree = rp_tree.get_index();
    rp_tree.add_node(left_subtree, right_subtree, offset, hyperplane);
}

// 建单棵 RP 树（初始 indices = 0..n-1）。
rptree build_rptree(float * data, size_t n, size_t dim, size_t leaf_size,
                    bool angular, size_t max_depth, uint64_t seed)
{
    rptree rp_tree(leaf_size);
    std::vector<uint32_t> point_indices(n);
    std::iota(point_indices.begin(), point_indices.end(), 0);
    build_rptree_recursively(rp_tree, data, n, dim, point_indices, leaf_size, angular, max_depth, seed);
    return rp_tree;
}

// 并行建 n_trees 棵 RP 树组成森林（每棵用不同 seed）。
// n_threads=0 时用 OMP 默认。注：并行度受 n_trees 限（每棵树内部是串行递归），
// n_trees < 核数时上层核会闲（细粒度 per-split 并行属大改，暂未做）。
std::vector<rptree> make_forest(float * data, size_t n, size_t dim, size_t n_trees,
                                size_t leaf_size, bool angular, size_t max_depth = 100,
                                uint64_t seed = 0, size_t n_threads = 0)
{
    std::vector<rptree> forest(n_trees);
    int nt = n_threads ? (int)n_threads : omp_get_max_threads();
    std::atomic<size_t> _done{0};
    size_t _step = std::max<size_t>(1, n_trees / 20);          // 每 ~5% 一报
    #pragma omp parallel for num_threads(nt) schedule(dynamic)
    for (size_t i = 0; i < n_trees; i++) {
        forest[i] = build_rptree(data, n, dim, leaf_size, angular, max_depth, seed + i);
        size_t d = ++_done;
        if (d % _step == 0 || d == n_trees)
            fprintf(stderr, "  [make_forest] %3.0f%% (%zu/%zu trees)\n", 100.0*d/n_trees, d, n_trees);
    }
    return forest;
}

} // namespace nndgraph
