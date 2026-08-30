#pragma once
// AVX-512 distance kernels used by graph construction and search.

#include <vector>
#include <cmath>
#include <functional>
#include <string>
#include <immintrin.h>
#include <unordered_map>
#include <stdexcept>
#include <iostream>

const float EPS = 1e-6;

#ifndef PORTABLE_ALIGN64
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#endif

// 命名空间自由函数版内积（RP 树构造时算超平面/投影 margin 用，见 rptrees.h）。
// 带尾部 mask 处理 dim 余数，使用非对齐 load。
static float InnerProductSIMD16ExtAVX512_(const float* pVect1, const float* pVect2, uint32_t dim) {
    size_t qty16 = dim / 16;
    size_t remainder = dim % 16;
    const float* pEnd1 = pVect1 + (qty16 * 16);

    __m512 v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1); pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2); pVect2 += 16;
        sum = _mm512_fmadd_ps(v1, v2, sum); // sum += v1 * v2
    }
    float res = _mm512_reduce_add_ps(sum);

    if (remainder) { // 处理不足 16 个的尾部
        __m512i mask = _mm512_set_epi32(
            remainder > 15 ? -1 : 0, remainder > 14 ? -1 : 0, remainder > 13 ? -1 : 0, remainder > 12 ? -1 : 0,
            remainder > 11 ? -1 : 0, remainder > 10 ? -1 : 0, remainder > 9 ? -1 : 0,  remainder > 8 ? -1 : 0,
            remainder > 7 ? -1 : 0,  remainder > 6 ? -1 : 0,  remainder > 5 ? -1 : 0,  remainder > 4 ? -1 : 0,
            remainder > 3 ? -1 : 0,  remainder > 2 ? -1 : 0,  remainder > 1 ? -1 : 0,  remainder > 0 ? -1 : 0);
        v1 = _mm512_maskz_loadu_ps(_mm512_test_epi32_mask(mask, mask), pVect1);
        v2 = _mm512_maskz_loadu_ps(_mm512_test_epi32_mask(mask, mask), pVect2);
        res += _mm512_reduce_add_ps(_mm512_mul_ps(v1, v2));
    }
    return res;
}

class DistCal {
public:
    uint32_t dim;
    std::function<float(const float *, const float *, uint32_t)> distance_function_;
    std::unordered_map<std::string, std::function<float(const float *, const float *, uint32_t)>> distance_functions_;

    // L2 平方距离（"l2" 度量），带尾部 mask，非对齐 load。
    static float L2SqrSIMD16ExtAVX512__(const float* pVect1, const float* pVect2, uint32_t dim) {
        size_t qty16 = dim / 16;
        size_t remainder = dim % 16;
        const float* pEnd1 = pVect1 + (qty16 * 16);

        __m512 v1, v2, diff;
        __m512 sum = _mm512_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm512_loadu_ps(pVect1); pVect1 += 16;
            v2 = _mm512_loadu_ps(pVect2); pVect2 += 16;
            diff = _mm512_sub_ps(v1, v2);
            sum = _mm512_fmadd_ps(diff, diff, sum); // sum += diff^2
        }
        float res = _mm512_reduce_add_ps(sum);

        if (remainder) {
            __m512i mask = _mm512_set_epi32(
                remainder > 15 ? -1 : 0, remainder > 14 ? -1 : 0, remainder > 13 ? -1 : 0, remainder > 12 ? -1 : 0,
                remainder > 11 ? -1 : 0, remainder > 10 ? -1 : 0, remainder > 9 ? -1 : 0,  remainder > 8 ? -1 : 0,
                remainder > 7 ? -1 : 0,  remainder > 6 ? -1 : 0,  remainder > 5 ? -1 : 0,  remainder > 4 ? -1 : 0,
                remainder > 3 ? -1 : 0,  remainder > 2 ? -1 : 0,  remainder > 1 ? -1 : 0,  remainder > 0 ? -1 : 0);
            v1 = _mm512_maskz_loadu_ps(_mm512_test_epi32_mask(mask, mask), pVect1);
            v2 = _mm512_maskz_loadu_ps(_mm512_test_epi32_mask(mask, mask), pVect2);
            diff = _mm512_sub_ps(v1, v2);
            res += _mm512_reduce_add_ps(_mm512_mul_ps(diff, diff));
        }
        return res;
    }

    // 类静态版内积（RP 树叶心路由用，见 rptrees.h 的 dist.InnerProductSIMD16ExtAVX512_）。
    static float InnerProductSIMD16ExtAVX512_(const float* pVect1, const float* pVect2, uint32_t dim) {
        size_t qty16 = dim / 16;
        size_t remainder = dim % 16;
        const float* pEnd1 = pVect1 + (qty16 * 16);

        __m512 v1, v2;
        __m512 sum = _mm512_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm512_loadu_ps(pVect1); pVect1 += 16;
            v2 = _mm512_loadu_ps(pVect2); pVect2 += 16;
            sum = _mm512_fmadd_ps(v1, v2, sum);
        }
        float res = _mm512_reduce_add_ps(sum);

        if (remainder) {
            __m512i mask = _mm512_set_epi32(
                remainder > 15 ? -1 : 0, remainder > 14 ? -1 : 0, remainder > 13 ? -1 : 0, remainder > 12 ? -1 : 0,
                remainder > 11 ? -1 : 0, remainder > 10 ? -1 : 0, remainder > 9 ? -1 : 0,  remainder > 8 ? -1 : 0,
                remainder > 7 ? -1 : 0,  remainder > 6 ? -1 : 0,  remainder > 5 ? -1 : 0,  remainder > 4 ? -1 : 0,
                remainder > 3 ? -1 : 0,  remainder > 2 ? -1 : 0,  remainder > 1 ? -1 : 0,  remainder > 0 ? -1 : 0);
            v1 = _mm512_maskz_loadu_ps(_mm512_test_epi32_mask(mask, mask), pVect1);
            v2 = _mm512_maskz_loadu_ps(_mm512_test_epi32_mask(mask, mask), pVect2);
            res += _mm512_reduce_add_ps(_mm512_mul_ps(v1, v2));
        }
        return res;
    }

    // 对齐 load 版内积（fast_l2 用；要求向量地址 64 字节对齐，dim 为 16 倍数时无尾部）。
    static float InnerProductSIMD16ExtAVX512__(const float* pVect1, const float* pVect2, uint32_t dim) {
        size_t qty16 = dim / 16;
        size_t remainder = dim % 16;
        const float* pEnd1 = pVect1 + (qty16 * 16);

        __m512 v1, v2;
        __m512 sum = _mm512_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm512_load_ps(pVect1); pVect1 += 16;
            v2 = _mm512_load_ps(pVect2); pVect2 += 16;
            sum = _mm512_fmadd_ps(v1, v2, sum);
        }
        float res = _mm512_reduce_add_ps(sum);

        if (remainder) {
            __mmask16 mask = (1 << remainder) - 1;
            v1 = _mm512_maskz_load_ps(mask, pVect1);
            v2 = _mm512_maskz_load_ps(mask, pVect2);
            res += _mm512_reduce_add_ps(_mm512_mul_ps(v1, v2));
        }
        return res;
    }

    // 对齐 load 版范数平方 ||x||^2（预计算 data_norm 用）。
    static float NormSIMD16ExtAVX512__(const float* pVect, uint32_t dim) {
        size_t qty16 = dim >> 4;
        const float* pEnd1 = pVect + (qty16 << 4);

        __m512 v;
        __m512 sum = _mm512_set1_ps(0);
        while (pVect < pEnd1) {
            v = _mm512_load_ps(pVect); pVect += 16;
            sum = _mm512_fmadd_ps(v, v, sum); // sum += v^2
        }
        float res = _mm512_reduce_add_ps(sum);
        uint32_t r = dim & 15;
        if (r) {                                        // 尾部 mask 补上(lifted 维度 d+2 等非16倍数必需)
            __mmask16 mask = (__mmask16)((1u << r) - 1);
            v = _mm512_maskz_loadu_ps(mask, pEnd1);
            res += _mm512_reduce_add_ps(_mm512_mul_ps(v, v));
        }
        return res;
    }

    // 标量内积（"ip_origin" 度量）。
    static float inner_product(const float *a, const float *b, uint32_t dim) {
        float product = 0.0;
        for (size_t i = 0; i < dim; ++i) product += a[i] * b[i];
        return product;
    }

    // 标量内积距离 = 1 - 内积（"ip" 度量；越小越近）。
    static float inner_product_dist(const float *a, const float *b, uint32_t dim) {
        float product = 0.0;
        for (size_t i = 0; i < dim; ++i) product += a[i] * b[i];
        return 1 - product;
    }

    DistCal() : dim(0) {}

    // 按度量名绑定距离函数。
    DistCal(uint32_t dim, const std::string& metric) : dim(dim) {
        distance_functions_ = {
            {"l2",        L2SqrSIMD16ExtAVX512__},
            {"ip",        inner_product_dist},
            {"ip_origin", inner_product}
        };
        auto it = distance_functions_.find(metric);
        if (it == distance_functions_.end())
            throw std::invalid_argument("Unsupported metric: " + metric);
        distance_function_ = it->second;
    }

    // 按当前度量计算 a、b 的距离。
    float calculate(const float *a, const float *b) const {
        return distance_function_(a, b, dim);
    }

    // 查询期快速 L2：||x||^2 - 2<q,x>。省掉了 ||q||^2（对同一 query 是常数，不影响排序），
    // 并复用预存的 ||x||^2（x_norm），把每次距离计算从一次 L2 降到一次内积。
    float fast_l2(const float *query, const float *x, float x_norm) {
        return x_norm - 2 * InnerProductSIMD16ExtAVX512__(query, x, dim);
    }

    // 计算 ||x||^2（建索引时预存进 data_norm）。
    float norm(const float * x) {
        return NormSIMD16ExtAVX512__(x, dim);
    }

    // RNG / α 边遮挡判据。已选邻居 (d1) 与候选 (d2) 满足 d1 <= d2；
    // 若已选邻居到候选的距离 d 比 d2 还小（d + EPS < d2），说明候选被遮挡，应剪掉。
    inline bool prune_conflict(float d1, float d2, float d) {
        if (d1 > d2) {
            std::cerr << "Neighbor vector should be in ascending order: " << d1 << " " << d2 << "\n";
            exit(0);
        }
        return (d + EPS) < d2;
    }
};
