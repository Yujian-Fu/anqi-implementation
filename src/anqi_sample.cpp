#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kDim = 8;
constexpr std::size_t kBase = 192;
constexpr std::size_t kQueries = 24;
constexpr std::size_t kMaxK = 10;
constexpr std::size_t kGraphDegree = 48;
constexpr std::size_t kRankM = 8;
constexpr int kQuantBits = 8;
constexpr std::size_t kEfSearch = 128;

using Vector = std::vector<float>;

struct PointSet {
    std::vector<float> values;
    std::size_t size = 0;

    const float* operator[](std::size_t i) const { return values.data() + i * kDim; }
    float* operator[](std::size_t i) { return values.data() + i * kDim; }
};

float l2sqr(const float* a, const float* b) {
    float out = 0.0f;
    for (std::size_t j = 0; j < kDim; ++j) {
        const float delta = a[j] - b[j];
        out += delta * delta;
    }
    return out;
}

PointSet make_base() {
    PointSet base;
    base.size = kBase;
    base.values.resize(kBase * kDim);
    std::mt19937 generator(20260819);
    std::normal_distribution<float> noise(0.0f, 0.8f);

    // A single continuous cloud keeps the fixed-degree toy graph connected,
    // so the sample exercises the algorithm rather than a disconnected-data
    // failure mode.
    for (std::size_t i = 0; i < base.size; ++i) {
        for (std::size_t j = 0; j < kDim; ++j) {
            base[i][j] = noise(generator);
        }
    }
    return base;
}

PointSet make_queries(const PointSet& base) {
    PointSet queries;
    queries.size = kQueries;
    queries.values.resize(kQueries * kDim);
    std::mt19937 generator(20260820);
    std::normal_distribution<float> noise(0.0f, 0.025f);
    for (std::size_t i = 0; i < queries.size; ++i) {
        const float* source = base[(i * 7) % base.size];
        for (std::size_t j = 0; j < kDim; ++j) {
            queries[i][j] = source[j] + noise(generator);
        }
    }
    return queries;
}

void jacobi_eigen_symmetric(
    std::vector<std::vector<double>>& matrix,
    std::vector<double>& eigenvalues,
    std::vector<std::vector<double>>& eigenvectors
) {
    const std::size_t n = matrix.size();
    eigenvectors.assign(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) eigenvectors[i][i] = 1.0;
    for (std::size_t iteration = 0; iteration < 80 * n * n; ++iteration) {
        std::size_t p = 0, q = 1;
        double largest = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                if (std::abs(matrix[i][j]) > largest) {
                    largest = std::abs(matrix[i][j]);
                    p = i;
                    q = j;
                }
            }
        }
        if (largest < 1e-12) break;
        const double angle = 0.5 * std::atan2(2.0 * matrix[p][q], matrix[q][q] - matrix[p][p]);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (std::size_t k = 0; k < n; ++k) {
            const double mkp = matrix[k][p];
            const double mkq = matrix[k][q];
            matrix[k][p] = cosine * mkp - sine * mkq;
            matrix[k][q] = sine * mkp + cosine * mkq;
        }
        for (std::size_t k = 0; k < n; ++k) {
            const double mpk = matrix[p][k];
            const double mqk = matrix[q][k];
            matrix[p][k] = cosine * mpk - sine * mqk;
            matrix[q][k] = sine * mpk + cosine * mqk;
        }
        for (std::size_t k = 0; k < n; ++k) {
            const double vkp = eigenvectors[k][p];
            const double vkq = eigenvectors[k][q];
            eigenvectors[k][p] = cosine * vkp - sine * vkq;
            eigenvectors[k][q] = sine * vkp + cosine * vkq;
        }
    }
    eigenvalues.resize(n);
    for (std::size_t i = 0; i < n; ++i) eigenvalues[i] = matrix[i][i];
}

// Exact base-to-base kNN radii. The diagonal is excluded, matching the
// monochromatic/self-exclusion convention used by the benchmark GT.
std::vector<float> exact_radii(const PointSet& base) {
    std::vector<float> radii(base.size * kMaxK);
    for (std::size_t i = 0; i < base.size; ++i) {
        std::vector<float> distances;
        distances.reserve(base.size - 1);
        for (std::size_t j = 0; j < base.size; ++j) {
            if (i != j) distances.push_back(l2sqr(base[i], base[j]));
        }
        std::nth_element(distances.begin(), distances.begin() + kMaxK, distances.end());
        std::sort(distances.begin(), distances.begin() + kMaxK);
        for (std::size_t k = 0; k < kMaxK; ++k) {
            radii[i * kMaxK + k] = distances[k];
        }
    }
    return radii;
}

struct LiftedIndex {
    std::size_t dim = kDim + 2;
    float scale = 1.0f;
    float mstar2 = 0.0f;
    std::vector<float> points;
    std::vector<std::vector<std::uint32_t>> graph;

    const float* operator[](std::size_t i) const { return points.data() + i * dim; }
};

LiftedIndex build_lifted_index(const PointSet& base, const std::vector<float>& radii) {
    LiftedIndex index;
    double mean_norm2 = 0.0;
    for (std::size_t i = 0; i < base.size; ++i) {
        for (std::size_t j = 0; j < kDim; ++j) {
            mean_norm2 += static_cast<double>(base[i][j]) * base[i][j];
        }
    }
    index.scale = static_cast<float>(1.0 / std::sqrt(mean_norm2 / base.size));

    std::vector<float> hat(base.size * (kDim + 1));
    double max_norm2 = 0.0;
    for (std::size_t i = 0; i < base.size; ++i) {
        double object_norm2 = 0.0;
        for (std::size_t j = 0; j < kDim; ++j) {
            const double value = static_cast<double>(base[i][j]) * index.scale;
            hat[i * (kDim + 1) + j] = static_cast<float>(value);
            object_norm2 += value * value;
        }
        const double radius = radii[i * kMaxK + (kMaxK - 1)] * index.scale * index.scale;
        hat[i * (kDim + 1) + kDim] = static_cast<float>(0.5 * (radius - object_norm2));
        double norm2 = object_norm2 +
            static_cast<double>(hat[i * (kDim + 1) + kDim]) *
            static_cast<double>(hat[i * (kDim + 1) + kDim]);
        max_norm2 = std::max(max_norm2, norm2);
    }
    index.mstar2 = static_cast<float>(max_norm2 * 1.0001);
    index.points.resize(base.size * index.dim);
    for (std::size_t i = 0; i < base.size; ++i) {
        double norm2 = 0.0;
        for (std::size_t j = 0; j < kDim + 1; ++j) {
            index.points[i * index.dim + j] = hat[i * (kDim + 1) + j];
            norm2 += static_cast<double>(index.points[i * index.dim + j]) *
                     static_cast<double>(index.points[i * index.dim + j]);
        }
        index.points[i * index.dim + kDim + 1] =
            static_cast<float>(std::sqrt(std::max(0.0, static_cast<double>(index.mstar2) - norm2)));
    }

    // This tiny sample builds the same shared horizon graph for every query k.
    // The production benchmark uses the parallel fixed-degree builder here.
    index.graph.resize(base.size);
    for (std::size_t i = 0; i < base.size; ++i) {
        std::vector<std::pair<float, std::uint32_t>> nearest;
        nearest.reserve(base.size - 1);
        for (std::size_t j = 0; j < base.size; ++j) {
            if (i != j) nearest.emplace_back(l2sqr(index[i], index[j]), static_cast<std::uint32_t>(j));
        }
        std::sort(nearest.begin(), nearest.end());
        const std::size_t keep = std::min(kGraphDegree, nearest.size());
        for (std::size_t p = 0; p < keep; ++p) index.graph[i].push_back(nearest[p].second);
    }
    return index;
}

struct QuantizedRadii {
    std::vector<float> reconstruction;
    std::vector<float> minimum;
    std::vector<float> step;
    std::vector<std::uint8_t> codes;

    float decode(std::size_t object, std::size_t k) const {
        const auto code = static_cast<float>(codes[object * kMaxK + k]);
        return reconstruction[object * kMaxK + k] + minimum[k] + code * step[k];
    }
};

QuantizedRadii train_u8_lrq(const std::vector<float>& radii) {
    std::vector<float> scale(kMaxK, 1.0f);
    for (std::size_t k = 0; k < kMaxK; ++k) {
        std::vector<float> column;
        column.reserve(kBase);
        for (std::size_t i = 0; i < kBase; ++i) column.push_back(std::abs(radii[i * kMaxK + k]));
        std::nth_element(column.begin(), column.begin() + column.size() / 2, column.end());
        scale[k] = std::max(column[column.size() / 2], 1e-6f);
    }
    std::vector<std::vector<double>> gram(kMaxK, std::vector<double>(kMaxK, 0.0));
    for (std::size_t i = 0; i < kBase; ++i) {
        for (std::size_t a = 0; a < kMaxK; ++a) {
            for (std::size_t b = 0; b < kMaxK; ++b) {
                gram[a][b] += static_cast<double>(radii[i * kMaxK + a] / scale[a]) *
                              static_cast<double>(radii[i * kMaxK + b] / scale[b]);
            }
        }
    }
    std::vector<double> eigenvalues;
    std::vector<std::vector<double>> eigenvectors;
    jacobi_eigen_symmetric(gram, eigenvalues, eigenvectors);
    std::vector<std::size_t> order(kMaxK);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return eigenvalues[a] > eigenvalues[b];
    });
    QuantizedRadii out;
    out.reconstruction.assign(radii.size(), 0.0f);
    out.minimum.assign(kMaxK, std::numeric_limits<float>::infinity());
    out.step.assign(kMaxK, 0.0f);
    out.codes.assign(radii.size(), 0);
    std::vector<float> residuals(radii.size(), 0.0f);

    for (std::size_t i = 0; i < kBase; ++i) {
        for (std::size_t k = 0; k < kMaxK; ++k) {
            float reconstruction = 0.0f;
            for (std::size_t m = 0; m < kRankM; ++m) {
                float coefficient = 0.0f;
                for (std::size_t t = 0; t < kMaxK; ++t) {
                    coefficient += (radii[i * kMaxK + t] / scale[t]) *
                                   static_cast<float>(eigenvectors[t][order[m]]);
                }
                reconstruction += coefficient * static_cast<float>(eigenvectors[k][order[m]]) * scale[k];
            }
            out.reconstruction[i * kMaxK + k] = reconstruction;
            residuals[i * kMaxK + k] = radii[i * kMaxK + k] - reconstruction;
            out.minimum[k] = std::min(out.minimum[k], residuals[i * kMaxK + k]);
        }
    }
    for (std::size_t k = 0; k < kMaxK; ++k) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < kBase; ++i) maximum = std::max(maximum, residuals[i * kMaxK + k]);
        out.step[k] = (maximum - out.minimum[k]) / 255.0f;
        if (!(out.step[k] > 0.0f)) out.step[k] = 1.0f;
    }
    for (std::size_t i = 0; i < kBase; ++i) {
        for (std::size_t k = 0; k < kMaxK; ++k) {
            const float normalized = (residuals[i * kMaxK + k] - out.minimum[k]) / out.step[k];
            const auto code = static_cast<int>(std::floor(std::max(0.0f, std::min(255.0f, normalized))));
            out.codes[i * kMaxK + k] = static_cast<std::uint8_t>(code);
        }
    }
    return out;
}

struct SearchResult {
    std::vector<std::uint32_t> candidates;
    std::size_t graph_distances = 0;
};

SearchResult search_shared_graph(
    const LiftedIndex& index,
    const float* query
) {
    Vector query_lift(index.dim, 0.0f);
    double query_norm2 = 0.0;
    for (std::size_t j = 0; j < kDim; ++j) {
        const float value = query[j] * index.scale;
        query_lift[j] = value;
        query_norm2 += static_cast<double>(value) * value;
    }
    query_lift[kDim] = 1.0f;
    const float threshold = static_cast<float>(query_norm2 + index.mstar2 + 1.0);

    using QueueItem = std::pair<float, std::uint32_t>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> frontier;
    std::vector<char> visited(index.graph.size(), 0);
    frontier.emplace(l2sqr(query_lift.data(), index[0]), 0);
    SearchResult result;
    while (!frontier.empty() && result.graph_distances < kEfSearch) {
        const auto [distance, id] = frontier.top();
        frontier.pop();
        if (visited[id]) continue;
        visited[id] = 1;
        ++result.graph_distances;
        if (distance <= threshold + 1e-6f) result.candidates.push_back(id);
        for (const auto next : index.graph[id]) {
            if (!visited[next]) frontier.emplace(l2sqr(query_lift.data(), index[next]), next);
        }
    }
    return result;
}

std::vector<std::uint32_t> exact_truth(const PointSet& base, const float* query, const std::vector<float>& radii, std::size_t k) {
    std::vector<std::uint32_t> truth;
    for (std::size_t i = 0; i < base.size; ++i) {
        if (l2sqr(query, base[i]) <= radii[i * kMaxK + k - 1] + 1e-6f)
            truth.push_back(static_cast<std::uint32_t>(i));
    }
    return truth;
}

void print_metrics(
    const PointSet& base,
    const PointSet& queries,
    const std::vector<float>& radii,
    const LiftedIndex& index,
    const QuantizedRadii& lrq,
    std::size_t k
) {
    double recall = 0.0;
    double precision = 0.0;
    double candidate_recall = 0.0;
    double graph_distances = 0.0;
    for (std::size_t qi = 0; qi < queries.size; ++qi) {
        const auto truth = exact_truth(base, queries[qi], radii, k);
        const auto search = search_shared_graph(index, queries[qi]);
        std::vector<std::uint32_t> prediction;
        for (const auto id : search.candidates) {
            const float threshold = std::max(0.0f, lrq.decode(id, k - 1));
            if (l2sqr(queries[qi], base[id]) <= threshold + 1e-6f) prediction.push_back(id);
        }
        std::size_t candidate_hits = 0;
        std::size_t hits = 0;
        for (const auto id : search.candidates) {
            if (std::find(truth.begin(), truth.end(), id) != truth.end()) ++candidate_hits;
        }
        for (const auto id : prediction) {
            if (std::find(truth.begin(), truth.end(), id) != truth.end()) ++hits;
        }
        recall += truth.empty() ? 1.0 : static_cast<double>(hits) / truth.size();
        precision += prediction.empty() ? (truth.empty() ? 1.0 : 0.0) : static_cast<double>(hits) / prediction.size();
        candidate_recall += truth.empty() ? 1.0 : static_cast<double>(candidate_hits) / truth.size();
        graph_distances += search.graph_distances;
    }
    const double count = static_cast<double>(queries.size);
    std::cout << "rank-M8+u8-LRQ-floor," << k
              << ',' << std::fixed << std::setprecision(4)
              << recall / count << ',' << precision / count << ',' << candidate_recall / count
              << ',' << std::setprecision(1) << graph_distances / count << '\n';
}

}  // namespace

int main() {
    const auto base = make_base();
    const auto queries = make_queries(base);
    const auto radii = exact_radii(base);
    const auto index = build_lifted_index(base, radii);
    const auto lrq = train_u8_lrq(radii);

    std::cout << "ANQI minimal correctness prototype\n"
              << "base=" << base.size << " query=" << queries.size
              << " dim=" << kDim << " Kmax=" << kMaxK
              << " graph_degree=" << kGraphDegree
              << " rank_M=" << kRankM << " residual_bits=" << kQuantBits << '\n'
              << "shared_graph=1 graph_rank_M=0 slack=0 exact_recheck=0\n"
              << "verifier,k,recall,precision,candidate_recall,avg_graph_dist\n";
    for (const std::size_t k : {1u, 5u, 10u}) {
        print_metrics(base, queries, radii, index, lrq, k);
    }
    std::cout << "done\n";
    return 0;
}
