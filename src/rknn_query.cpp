// ANQI reverse-kNN graph construction, search, and verification driver.
#include "../include/index.h"
#include "../include/packed_code_array.h"
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <unordered_set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
using namespace nndgraph;

static float* load_fbin(const std::string& p, size_t& n, size_t& d) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "[rknn] cannot open fbin: %s\n", p.c_str());
        exit(3);
    }
    const std::streamoff bytes = f.tellg();
    f.seekg(0);
    uint32_t nn = 0, dd = 0;
    if (bytes < 8 || !f.read(reinterpret_cast<char*>(&nn), 4) ||
        !f.read(reinterpret_cast<char*>(&dd), 4) || nn == 0 || dd == 0) {
        fprintf(stderr, "[rknn] invalid fbin header: %s\n", p.c_str());
        exit(3);
    }
    const uint64_t values = static_cast<uint64_t>(nn) * dd;
    const uint64_t expected = 8 + values * sizeof(float);
    if (expected != static_cast<uint64_t>(bytes) ||
        values > std::numeric_limits<size_t>::max() / sizeof(float)) {
        fprintf(stderr, "[rknn] invalid fbin size: %s\n", p.c_str());
        exit(3);
    }
    float* x = static_cast<float*>(_mm_malloc(values * sizeof(float), 64));
    if (!x || !f.read(reinterpret_cast<char*>(x), values * sizeof(float))) {
        fprintf(stderr, "[rknn] cannot read fbin values: %s\n", p.c_str());
        _mm_free(x);
        exit(3);
    }
    n = nn;
    d = dd;
    return x;
}

static float* load_u8bin_as_float(const std::string& path, size_t& n, size_t& d) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[rknn] cannot open uint8 base %s: %s\n",
                path.c_str(), std::strerror(errno));
        exit(3);
    }
    struct stat st {};
    uint32_t shape[2] = {};
    if (::fstat(fd, &st) != 0 ||
        ::pread(fd, shape, sizeof(shape), 0) != (ssize_t)sizeof(shape)) {
        fprintf(stderr, "[rknn] cannot read uint8 base header %s\n", path.c_str());
        ::close(fd);
        exit(3);
    }
    n = shape[0];
    d = shape[1];
    const uint64_t values = (uint64_t)n * (uint64_t)d;
    const uint64_t expected = sizeof(shape) + values;
    if (n == 0 || d == 0 || (uint64_t)st.st_size != expected ||
        values > (uint64_t)std::numeric_limits<size_t>::max() / sizeof(float)) {
        fprintf(stderr,
                "[rknn] bad uint8 base %s: n=%zu d=%zu bytes=%lld expected=%llu\n",
                path.c_str(), n, d, (long long)st.st_size,
                (unsigned long long)expected);
        ::close(fd);
        exit(3);
    }
    void* mapping = ::mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        fprintf(stderr, "[rknn] cannot mmap uint8 base %s: %s\n",
                path.c_str(), std::strerror(errno));
        ::close(fd);
        exit(3);
    }
    float* output = static_cast<float*>(
        _mm_malloc((size_t)values * sizeof(float), 64));
    if (!output) {
        fprintf(stderr, "[rknn] cannot allocate converted uint8 base: %.3fGB\n",
                values * sizeof(float) / 1e9);
        ::munmap(mapping, (size_t)st.st_size);
        ::close(fd);
        exit(3);
    }
    const uint8_t* input = static_cast<const uint8_t*>(mapping) + sizeof(shape);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < (size_t)values; i++) output[i] = (float)input[i];
    ::munmap(mapping, (size_t)st.st_size);
    ::close(fd);
    printf("[rknn] loaded uint8 base directly as float: %s n=%zu d=%zu %.3fGB\n",
           path.c_str(), n, d, values * sizeof(float) / 1e9);
    return output;
}

static std::string base_vector_source(const std::string& fbin_path) {
    const char* u8_path = std::getenv("ANQI_BASE_U8BIN");
    return (u8_path && *u8_path) ? std::string(u8_path) : fbin_path;
}

static float* load_base_vectors(
    const std::string& fbin_path,
    size_t& n,
    size_t& d
) {
    const char* u8_path = std::getenv("ANQI_BASE_U8BIN");
    return (u8_path && *u8_path)
        ? load_u8bin_as_float(u8_path, n, d)
        : load_fbin(fbin_path, n, d);
}

static std::vector<int> parse_knots(const char* s, size_t NK) {
    std::vector<int> knots;
    if (s && *s) {
        char* buf = strdup(s);
        for (char* t = strtok(buf, ", "); t; t = strtok(nullptr, ", ")) {
            int k = atoi(t);
            if (k >= 1 && (size_t)k <= NK) knots.push_back(k);
        }
        free(buf);
    }
    if (knots.empty()) {
        knots.push_back((int)std::min<size_t>(10, NK));
        if (NK != (size_t)knots.front()) knots.push_back((int)NK);
    }
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
    return knots;
}

static bool is_rankm_residual_mode(const std::string& mode) {
    return mode == "rankm_resid_f32" ||
           mode == "rankm_resid_u8_floor" || mode == "rankm_resid_u16_floor" ||
           mode == "rankm_resid_u8_bias" || mode == "rankm_resid_u16_bias" ||
           mode == "rankm_resid_u2_lrq_nearest" || mode == "rankm_resid_u2_lrq_floor" ||
           mode == "rankm_resid_u2_lrq_ceil" ||
           mode == "rankm_resid_u4_lrq_nearest" || mode == "rankm_resid_u4_lrq_floor" ||
           mode == "rankm_resid_u4_lrq_ceil" ||
           mode == "rankm_resid_u8_lrq_nearest" || mode == "rankm_resid_u8_lrq_floor" ||
           mode == "rankm_resid_u8_lrq_ceil" ||
           mode == "rankm_resid_u16_lrq_nearest" || mode == "rankm_resid_u16_lrq_floor" ||
           mode == "rankm_resid_u16_lrq_ceil";
}

static bool is_learned_rq_mode(const std::string& mode) {
    return mode.find("rankm_resid_u2_lrq_") == 0 ||
           mode.find("rankm_resid_u4_lrq_") == 0 ||
           mode.find("rankm_resid_u8_lrq_") == 0 ||
           mode.find("rankm_resid_u16_lrq_") == 0;
}

static std::string learned_rq_policy(const std::string& mode) {
    if (mode.size() >= 6 && mode.compare(mode.size() - 6, 6, "_floor") == 0) return "floor";
    if (mode.size() >= 5 && mode.compare(mode.size() - 5, 5, "_ceil") == 0) return "ceil";
    return "nearest";
}

static int learned_rq_bits(const std::string& mode) {
    if (mode.find("rankm_resid_u2_lrq_") == 0) return 2;
    if (mode.find("rankm_resid_u4_lrq_") == 0) return 4;
    if (mode.find("rankm_resid_u8_lrq_") == 0) return 8;
    if (mode.find("rankm_resid_u16_lrq_") == 0) return 16;
    fprintf(stderr, "[rknn] cannot infer learned RQ bits from mode=%s\n", mode.c_str());
    exit(2);
}

static bool is_rankm_only_mode(const std::string& mode) {
    return mode == "rankm_only";
}

static bool is_analytic_rankm_mode(const std::string& mode) {
    return mode == "analytic_rankm";
}

static std::string rq_objective() {
    const char* s = std::getenv("ANQI_RQ_OBJECTIVE");
    return (s && *s) ? std::string(s) : std::string("radius_reconstruction");
}

static void validate_rq_objective(const std::string& objective) {
    if (objective == "radius_reconstruction") return;
    fprintf(stderr,
            "[rknn] unsupported ANQI_RQ_OBJECTIVE=%s "
            "(implemented: radius_reconstruction; reserved: predicate_boundary)\n",
            objective.c_str());
    exit(2);
}

static std::string hex64(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v);
    return std::string(buf);
}

static std::string file_fingerprint(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return path + ":missing";
    uint64_t h = 1469598103934665603ull;
    uint64_t bytes = 0;
    char buf[8192];
    while (f) {
        f.read(buf, sizeof(buf));
        std::streamsize got = f.gcount();
        for (std::streamsize i = 0; i < got; i++) {
            h ^= (uint64_t)(unsigned char)buf[i];
            h *= 1099511628211ull;
        }
        bytes += (uint64_t)got;
    }
    return path + ":sz" + std::to_string(bytes) + ":fnv" + hex64(h);
}

static std::string file_stat_identity(const std::string& path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return path + ":missing";
    return path + ":dev" + std::to_string((uint64_t)st.st_dev) +
           ":ino" + std::to_string((uint64_t)st.st_ino) +
           ":sz" + std::to_string((uint64_t)st.st_size) +
           ":mt" + std::to_string((int64_t)st.st_mtime) +
           ":ct" + std::to_string((int64_t)st.st_ctime);
}

static std::string hash_string16(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= (uint64_t)c;
        h *= 1099511628211ull;
    }
    return hex64(h);
}

static std::string env_or(const char* k, const char* d) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::string(v) : std::string(d);
}

static std::string cache_dir_prefix(const char* key, const std::string& fallback) {
    std::string path = env_or(key, fallback.c_str());
    if (path.empty() || path.back() != '/') path.push_back('/');
    return path;
}

static bool env_flag(const char* k) {
    const char* v = std::getenv(k);
    if (!v || !*v) return false;
    std::string value(v);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return value != "0" && value != "false" && value != "no" && value != "off";
}

static void release_heap_pages() {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

static std::string sanitize_tag(std::string s) {
    for (char& c : s) {
        bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                  (c >= 'A' && c <= 'Z') || c == '-' || c == '_';
        if (!ok) c = '_';
    }
    return s;
}

static std::string resolve_graph_geometry(int rankm) {
    const char* raw = std::getenv("ANQI_GRAPH_GEOMETRY");
    std::string geometry = (raw && *raw) ? std::string(raw)
                                         : (rankm > 0 ? "rankm" : "anyk_lift");
    if (geometry == "anyk" || geometry == "lift") geometry = "anyk_lift";
    if (geometry == "raw" || geometry == "no_lift" || geometry == "nolift")
        geometry = "original";
    if (geometry != "anyk_lift" && geometry != "rankm" && geometry != "original") {
        fprintf(stderr,
                "[rknn] unsupported ANQI_GRAPH_GEOMETRY=%s "
                "(expected anyk_lift, original, or rankm)\n",
                geometry.c_str());
        exit(2);
    }
    if (geometry == "rankm" && rankm <= 0) {
        fprintf(stderr, "[rknn] ANQI_GRAPH_GEOMETRY=rankm requires ANQI_RANKM>0\n");
        exit(2);
    }
    return geometry;
}

static std::string graph_cache_key(size_t M, size_t leaf, size_t nt, size_t ef,
                                   const std::string& geometry, int graph_rankm) {
    std::string k = "M" + std::to_string(M) +
                    "_leaf" + std::to_string(leaf) +
                    "_nt" + std::to_string(nt) +
                    "_ef" + std::to_string(ef) +
                    "_geom" + geometry +
                    "_rankm" + std::to_string(graph_rankm) +
                    "_specific" + (std::getenv("ANQI_SPECIFIC") ? "1" : "0") +
                    "_alpha" + env_or("ANQI_ALPHA", "1.2") +
                    "_liftnnk" + env_or("ANQI_LIFTNNK", "50") +
                    "_graphnnd" + env_or("ANQI_GRAPHNND", "6");
    if (std::getenv("ANQI_REUSE_ORIG_KNN")) k += "_reuseorig1";
    if (std::getenv("ANQI_EDGE_POLICY")) {
        k += "_edge" + env_or("ANQI_EDGE_POLICY", "none") +
             "_eks" + env_or("ANQI_EDGE_KS", "10,50,100") +
             "_enk" + env_or("ANQI_EDGE_NK", "100") +
             "_erb" + env_or("ANQI_EDGE_RESID_BUDGET", "16") +
             "_erl" + env_or("ANQI_EDGE_RESID_LAMBDA", "1.0") +
             "_ebo" + env_or("ANQI_EDGE_BALL_OCCLUSION", "all");
        if (std::getenv("ANQI_EDGE_RESID_FILE")) {
            k += "_erf" + hash_string16(file_fingerprint(env_or("ANQI_EDGE_RESID_FILE", "")));
        }
    }
    if (std::getenv("ANQI_VAMANA")) {
        std::string style = env_or("ANQI_VAMANA_STYLE", "global");
        k += "_vam1_style" + style +
             "_L" + env_or("ANQI_VAMANA_L", "128") +
             "_P" + env_or("ANQI_VAMANA_PASS", "2") +
             "_bf" + env_or("ANQI_VAMANA_BATCH_FRAC", "0.02") +
             "_mb" + env_or("ANQI_VAMANA_MAX_BATCH", "1000000") +
             "_bb" + env_or("ANQI_VAMANA_BATCH_BASE", "2") +
             "_ea" + env_or("ANQI_VAMANA_EARLY_ALPHA", "1.0") +
             "_eao" + env_or("ANQI_VAMANA_EARLY_ALPHA_ONE", "1") +
             "_entry" + env_or("ANQI_VAMANA_ENTRY", "0");
        if (style == "batch") k += "_implcsr1";
    } else {
        k += "_vam0";
    }
    return sanitize_tag(k);
}

static size_t float_table_bytes(size_t n, size_t k, const char* label) {
    if (k != 0 && n > std::numeric_limits<size_t>::max() / k) {
        fprintf(stderr, "[rknn] %s table size overflow: n=%zu K=%zu\n", label, n, k);
        exit(3);
    }
    size_t count = n * k;
    if (count > std::numeric_limits<size_t>::max() / sizeof(float)) {
        fprintf(stderr, "[rknn] %s table byte size overflow: count=%zu\n", label, count);
        exit(3);
    }
    return count * sizeof(float);
}

static void read_binary_exact(const std::string& path, void* dst, size_t bytes, const char* label) {
    if (bytes > (size_t)std::numeric_limits<std::streamsize>::max()) {
        fprintf(stderr, "[rknn] %s too large to read in one stream: %zu bytes (%s)\n",
                label, bytes, path.c_str());
        exit(3);
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[rknn] missing %s cache file %s\n", label, path.c_str());
        exit(3);
    }
    const std::streamsize need = (std::streamsize)bytes;
    f.read((char*)dst, need);
    if (f.gcount() != need) {
        fprintf(stderr, "[rknn] truncated %s cache %s: expected %zu bytes, got %lld\n",
                label, path.c_str(), bytes, (long long)f.gcount());
        exit(3);
    }
    char extra;
    if (f.read(&extra, 1)) {
        fprintf(stderr, "[rknn] oversized %s cache %s: expected exactly %zu bytes\n",
                label, path.c_str(), bytes);
        exit(3);
    }
}

static std::string rq_codebook_path(const std::string& pfx, int rankm, int bits) {
    const char* s = std::getenv("ANQI_RQ_CODEBOOK");
    if (s && *s) return std::string(s);
    return pfx + "_learned_rq_codebook_M" + std::to_string(rankm) +
           "_u" + std::to_string(bits) + ".bin";
}

static std::string rq_codebook_key(const std::string& pfx, int rankm, int bits) {
    const char* s = std::getenv("ANQI_RQ_CODEBOOK_ID");
    if (s && *s) return std::string(s);
    return file_fingerprint(rq_codebook_path(pfx, rankm, bits));
}

static std::string rq_basis_key(const std::string& pfx, int rankm) {
    const char* s = std::getenv("ANQI_RQ_BASIS_ID");
    if (s && *s) return std::string(s);
    return file_fingerprint(pfx + "_radbasis_M" + std::to_string(rankm) + ".bin");
}

static std::string analytic_rankm_model_path(const std::string& pfx, int rankm) {
    const char* s = std::getenv("ANQI_ANALYTIC_RANKM_MODEL");
    if (s && *s) return std::string(s);
    return pfx + "_analytic_rankm_M" + std::to_string(rankm) + ".bin";
}

static std::vector<float> load_learned_rq_codebook(
    const std::string& path,
    size_t NK,
    int RANKM,
    int expected_bits,
    size_t& code_size
) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[rknn] missing learned RQ codebook %s\n", path.c_str());
        exit(3);
    }
    char magic[8];
    f.read(magic, 8);
    if (f.gcount() != 8 || std::memcmp(magic, "ANQIRQ1\0", 8) != 0) {
        fprintf(stderr, "[rknn] bad learned RQ codebook magic %s\n", path.c_str());
        exit(3);
    }
    int nk = 0, rankm = 0, bits = 0, csize = 0;
    f.read((char*)&nk, 4);
    f.read((char*)&rankm, 4);
    f.read((char*)&bits, 4);
    f.read((char*)&csize, 4);
    if (!f || nk != (int)NK || rankm != RANKM || bits != expected_bits || csize <= 0) {
        fprintf(stderr,
                "[rknn] bad learned RQ codebook header %s: nk=%d rankm=%d bits=%d code_size=%d expected nk=%zu rankm=%d bits=%d\n",
                path.c_str(), nk, rankm, bits, csize, NK, RANKM, expected_bits);
        exit(3);
    }
    code_size = (size_t)csize;
    std::vector<float> codebook((size_t)nk * code_size);
    const size_t bytes = codebook.size() * sizeof(float);
    if (bytes > (size_t)std::numeric_limits<std::streamsize>::max()) {
        fprintf(stderr, "[rknn] learned RQ codebook too large: %s\n", path.c_str());
        exit(3);
    }
    f.read((char*)codebook.data(), (std::streamsize)bytes);
    if (f.gcount() != (std::streamsize)bytes) {
        fprintf(stderr, "[rknn] truncated learned RQ codebook %s\n", path.c_str());
        exit(3);
    }
    char extra;
    if (f.read(&extra, 1)) {
        fprintf(stderr, "[rknn] oversized learned RQ codebook %s\n", path.c_str());
        exit(3);
    }
    return codebook;
}

static double decode_learned_residual(
    double value,
    const float* centers,
    size_t code_size,
    const std::string& policy
) {
    auto begin = centers;
    auto end = centers + code_size;
    if (policy == "floor") {
        auto it = std::upper_bound(begin, end, (float)value);
        if (it == begin) return (double)*begin;
        return (double)*(it - 1);
    }
    if (policy == "ceil") {
        auto it = std::lower_bound(begin, end, (float)value);
        if (it == end) return (double)*(end - 1);
        return (double)*it;
    }
    auto it = std::lower_bound(begin, end, (float)value);
    if (it == begin) return (double)*begin;
    if (it == end) return (double)*(end - 1);
    double hi = (double)*it;
    double lo = (double)*(it - 1);
    return (std::abs(hi - value) < std::abs(value - lo)) ? hi : lo;
}

static uint32_t encode_learned_residual(
    double value,
    const float* centers,
    size_t code_size,
    const std::string& policy
) {
    auto begin = centers;
    auto end = centers + code_size;
    auto selected = begin;
    if (policy == "floor") {
        auto it = std::upper_bound(begin, end, (float)value);
        selected = (it == begin) ? begin : it - 1;
    } else if (policy == "ceil") {
        auto it = std::lower_bound(begin, end, (float)value);
        selected = (it == end) ? end - 1 : it;
    } else {
        auto it = std::lower_bound(begin, end, (float)value);
        if (it == begin) selected = begin;
        else if (it == end) selected = end - 1;
        else {
            const double hi = (double)*it;
            const double lo = (double)*(it - 1);
            selected = (std::abs(hi - value) < std::abs(value - lo)) ? it : it - 1;
        }
    }
    const size_t code = (size_t)(selected - begin);
    if (code_size > 65536 || code >= code_size) {
        fprintf(stderr, "[rknn] learned RQ code index overflow: code=%zu code_size=%zu\n", code, code_size);
        exit(3);
    }
    return (uint32_t)code;
}

struct AnalyticRankMVerifier {
    size_t n = 0;
    size_t train_kmax = 0;
    int rankm = 0;
    size_t fit_kmin = 0;
    uint32_t flags = 0;
    double b0 = 0.0;
    std::vector<float> coeff;

    size_t persistent_bytes() const {
        return 40 + coeff.size() * sizeof(float);
    }

    void load(const std::string& path, size_t expected_n, int expected_rankm) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            fprintf(stderr, "[rknn] missing analytic Rank-M model %s\n", path.c_str());
            exit(3);
        }
        char magic[8] = {};
        uint64_t n64 = 0;
        uint32_t train_kmax32 = 0, rankm32 = 0, fit_kmin32 = 0, flags32 = 0;
        f.read(magic, 8);
        f.read((char*)&n64, sizeof(n64));
        f.read((char*)&train_kmax32, sizeof(train_kmax32));
        f.read((char*)&rankm32, sizeof(rankm32));
        f.read((char*)&fit_kmin32, sizeof(fit_kmin32));
        f.read((char*)&flags32, sizeof(flags32));
        f.read((char*)&b0, sizeof(b0));
        if (!f || std::memcmp(magic, "ANQIPW1\0", 8) != 0 ||
            n64 != expected_n || rankm32 != (uint32_t)expected_rankm ||
            train_kmax32 < 1 || fit_kmin32 < 1 || fit_kmin32 > train_kmax32 ||
            (flags32 != 3u && flags32 != 7u) || !std::isfinite(b0)) {
            fprintf(stderr,
                    "[rknn] analytic Rank-M header mismatch %s: n=%llu train_kmax=%u "
                    "rankm=%u fit_kmin=%u flags=%u b0=%.9g expected n=%zu rankm=%d\n",
                    path.c_str(), (unsigned long long)n64, train_kmax32, rankm32,
                    fit_kmin32, flags32, b0, expected_n, expected_rankm);
            exit(3);
        }
        n = expected_n;
        train_kmax = (size_t)train_kmax32;
        rankm = expected_rankm;
        fit_kmin = (size_t)fit_kmin32;
        flags = flags32;
        coeff.resize(n * (size_t)rankm);
        const size_t bytes = coeff.size() * sizeof(float);
        f.read((char*)coeff.data(), (std::streamsize)bytes);
        if (f.gcount() != (std::streamsize)bytes) {
            fprintf(stderr, "[rknn] truncated analytic Rank-M coefficients %s\n", path.c_str());
            exit(3);
        }
        char extra;
        if (f.read(&extra, 1)) {
            fprintf(stderr, "[rknn] oversized analytic Rank-M model %s\n", path.c_str());
            exit(3);
        }
    }

    void materialize_k(
        size_t k_one_based,
        double radius_scale,
        std::vector<float>& thresholds
    ) const {
        if (k_one_based < 1 || n == 0 || rankm <= 0 ||
            coeff.size() != n * (size_t)rankm) {
            fprintf(stderr, "[rknn] analytic Rank-M verifier is not ready for k=%zu\n",
                    k_one_based);
            exit(3);
        }
        const bool centered_log = (flags & 4u) != 0;
        const double ratio = centered_log
            ? (double)k_one_based / (double)train_kmax
            : (double)k_one_based;
        const double logk = std::log(ratio);
        std::vector<double> basis((size_t)rankm, std::pow(ratio, b0));
        for (int m = 1; m < rankm; m++) basis[(size_t)m] = basis[(size_t)m - 1] * logk;
        thresholds.resize(n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            const float* ci = &coeff[i * (size_t)rankm];
            double value = 0.0;
            for (int m = 0; m < rankm; m++) value += (double)ci[m] * basis[(size_t)m];
            thresholds[i] = (float)std::max(0.0, radius_scale * value);
        }
    }
};

struct CompactLrqVerifier {
    size_t n = 0;
    size_t nk = 0;
    int rankm = 0;
    int bits = 0;
    size_t code_size = 0;
    std::string policy;
    std::vector<float> mu;
    std::vector<float> basis;
    std::vector<float> codebook;
    std::vector<float> coeff;
    std::vector<uint8_t> packed_codes;
    std::vector<float> residual_f32;

    size_t logical_code_count() const { return n * nk; }

    size_t packed_code_bytes() const {
        return anqi::packed_code_bytes(logical_code_count(), bits);
    }

    void set_code(size_t idx, uint32_t code) {
        anqi::set_packed_code(packed_codes.data(), idx, bits, code);
    }

    uint32_t get_code(size_t idx) const {
        return anqi::get_packed_code(packed_codes.data(), idx, bits);
    }

    size_t persistent_bytes() const {
        return coeff.size() * sizeof(float) + packed_codes.size() * sizeof(uint8_t) +
               residual_f32.size() * sizeof(float) +
               mu.size() * sizeof(float) + basis.size() * sizeof(float) +
               codebook.size() * sizeof(float) + 36 + 8 + (bits ? 24 : 0);
    }

    void configure(
        size_t n_in,
        size_t nk_in,
        int rankm_in,
        int bits_in,
        const std::string& policy_in,
        const std::vector<float>& mu_in,
        const std::vector<float>& basis_in,
        const std::vector<float>& codebook_in,
        size_t code_size_in
    ) {
        n = n_in;
        nk = nk_in;
        rankm = rankm_in;
        bits = bits_in;
        policy = policy_in;
        mu = mu_in;
        basis = basis_in;
        codebook = codebook_in;
        code_size = code_size_in;
        const bool bit_width_ok = bits == 0 || bits == 32 ||
                                  anqi::supported_packed_code_bits(bits);
        const size_t expected_code_size = anqi::supported_packed_code_bits(bits)
            ? ((size_t)1 << bits) : 0;
        if (rankm <= 0 || nk == 0 || !bit_width_ok ||
            code_size != expected_code_size || mu.size() != nk ||
            basis.size() != (size_t)rankm * nk || codebook.size() != nk * code_size) {
            fprintf(stderr, "[rknn] invalid CompactLrqVerifier configuration\n");
            exit(3);
        }
        if (anqi::supported_packed_code_bits(bits)) {
            for (size_t k = 0; k < nk; k++) {
                const float* row = &codebook[k * code_size];
                if (!std::is_sorted(row, row + code_size)) {
                    fprintf(stderr, "[rknn] LRQ codebook row is not sorted: k=%zu\n", k + 1);
                    exit(3);
                }
            }
        }
    }

    void build(const std::vector<float>& exact_radii) {
        if (exact_radii.size() != n * nk) {
            fprintf(stderr, "[rknn] LRQ build exact-radii size mismatch: got=%zu expected=%zu\n",
                    exact_radii.size(), n * nk);
            exit(3);
        }
        coeff.assign(n * (size_t)rankm, 0.0f);
        packed_codes.assign(packed_code_bytes(), 0);
        residual_f32.assign(bits == 32 ? logical_code_count() : 0, 0.0f);
        const size_t codes_per_byte = bits == 2 ? 4 : (bits == 4 ? 2 : 1);
        if ((bits == 2 || bits == 4) && nk % codes_per_byte != 0) {
            fprintf(stderr,
                    "[rknn] packed u%d verifier requires NK divisible by %zu, got %zu\n",
                    bits, codes_per_byte, nk);
            exit(3);
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            const float* rk = &exact_radii[i * nk];
            float* ci = &coeff[i * (size_t)rankm];
            for (int m = 0; m < rankm; m++) {
                const float* gm = &basis[(size_t)m * nk];
                double c = 0.0;
                for (size_t k = 0; k < nk; k++)
                    c += ((double)rk[k] - (double)mu[k]) * (double)gm[k];
                ci[m] = (float)c;
            }
            if (bits == 0) continue;
            for (size_t k = 0; k < nk; k++) {
                double recon = (double)mu[k];
                for (int m = 0; m < rankm; m++)
                    recon += (double)ci[m] * (double)basis[(size_t)m * nk + k];
                const double residual = (double)rk[k] - recon;
                if (bits == 32) {
                    residual_f32[i * nk + k] = (float)residual;
                } else {
                    set_code(i * nk + k, encode_learned_residual(
                        residual, &codebook[k * code_size], code_size, policy));
                }
            }
        }
    }

    void materialize_k(
        size_t k_one_based,
        double radius_scale,
        bool is_ip,
        std::vector<float>& thresholds
    ) const {
        if (k_one_based < 1 || k_one_based > nk || coeff.size() != n * (size_t)rankm ||
            packed_codes.size() != packed_code_bytes() ||
            residual_f32.size() != (bits == 32 ? logical_code_count() : 0)) {
            fprintf(stderr, "[rknn] LRQ verifier is not ready for k=%zu\n", k_one_based);
            exit(3);
        }
        const size_t k = k_one_based - 1;
        thresholds.resize(n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            const float* ci = &coeff[i * (size_t)rankm];
            double value = (double)mu[k];
            for (int m = 0; m < rankm; m++)
                value += (double)ci[m] * (double)basis[(size_t)m * nk + k];
            if (bits == 32) {
                value += (double)residual_f32[i * nk + k];
            } else if (bits) {
                const uint32_t code = get_code(i * nk + k);
                value += (double)codebook[k * code_size + (size_t)code];
            }
            const double scaled = radius_scale * value;
            // L2 stores squared radii, while the IP predicate key is -<q,o>
            // and may legitimately be negative.
            thresholds[i] = (float)(is_ip ? scaled : std::max(0.0, scaled));
        }
    }

    void save(const std::string& path) const {
        const std::string tmp_path = path + ".tmp";
        std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
        if (!f) {
            fprintf(stderr, "[rknn] cannot write LRQ verifier artifact %s\n", tmp_path.c_str());
            exit(3);
        }
        const char magic[8] = {'A','N','Q','I','V','F','2','\0'};
        const uint64_t n64 = (uint64_t)n;
        const uint32_t nk32 = (uint32_t)nk;
        const uint32_t rankm32 = (uint32_t)rankm;
        const uint32_t bits32 = (uint32_t)bits;
        const uint32_t code_size32 = (uint32_t)code_size;
        const uint32_t policy32 = policy == "floor" ? 1u : (policy == "ceil" ? 2u : 0u);
        f.write(magic, 8);
        f.write((const char*)&n64, sizeof(n64));
        f.write((const char*)&nk32, sizeof(nk32));
        f.write((const char*)&rankm32, sizeof(rankm32));
        f.write((const char*)&bits32, sizeof(bits32));
        f.write((const char*)&code_size32, sizeof(code_size32));
        f.write((const char*)&policy32, sizeof(policy32));
        f.write((const char*)coeff.data(), (std::streamsize)(coeff.size() * sizeof(float)));
        f.write((const char*)packed_codes.data(), (std::streamsize)packed_codes.size());
        f.write((const char*)residual_f32.data(),
                (std::streamsize)(residual_f32.size() * sizeof(float)));
        if (!f) {
            fprintf(stderr, "[rknn] failed while writing LRQ verifier artifact %s\n", tmp_path.c_str());
            exit(3);
        }
        f.close();
        if (!f || std::rename(tmp_path.c_str(), path.c_str()) != 0) {
            fprintf(stderr, "[rknn] cannot publish LRQ verifier artifact %s\n", path.c_str());
            exit(3);
        }
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            fprintf(stderr, "[rknn] missing LRQ verifier artifact %s\n", path.c_str());
            exit(3);
        }
        char magic[8] = {};
        uint64_t n64 = 0;
        uint32_t nk32 = 0, rankm32 = 0, bits32 = 0, code_size32 = 0, policy32 = 0;
        f.read(magic, 8);
        f.read((char*)&n64, sizeof(n64));
        f.read((char*)&nk32, sizeof(nk32));
        f.read((char*)&rankm32, sizeof(rankm32));
        f.read((char*)&bits32, sizeof(bits32));
        f.read((char*)&code_size32, sizeof(code_size32));
        f.read((char*)&policy32, sizeof(policy32));
        const uint32_t expected_policy = policy == "floor" ? 1u : (policy == "ceil" ? 2u : 0u);
        if (!f || std::memcmp(magic, "ANQIVF2\0", 8) != 0 || n64 != n || nk32 != nk ||
            rankm32 != (uint32_t)rankm || bits32 != (uint32_t)bits ||
            code_size32 != code_size || policy32 != expected_policy) {
            fprintf(stderr, "[rknn] LRQ verifier artifact header mismatch %s\n", path.c_str());
            exit(3);
        }
        coeff.resize(n * (size_t)rankm);
        packed_codes.resize(packed_code_bytes());
        residual_f32.resize(bits == 32 ? logical_code_count() : 0);
        f.read((char*)coeff.data(), (std::streamsize)(coeff.size() * sizeof(float)));
        if (f.gcount() != (std::streamsize)(coeff.size() * sizeof(float))) {
            fprintf(stderr, "[rknn] truncated LRQ verifier coefficients %s\n", path.c_str());
            exit(3);
        }
        f.read((char*)packed_codes.data(), (std::streamsize)packed_codes.size());
        if (f.gcount() != (std::streamsize)packed_codes.size()) {
            fprintf(stderr, "[rknn] truncated LRQ verifier codes %s\n", path.c_str());
            exit(3);
        }
        f.read((char*)residual_f32.data(),
               (std::streamsize)(residual_f32.size() * sizeof(float)));
        if (f.gcount() != (std::streamsize)(residual_f32.size() * sizeof(float))) {
            fprintf(stderr, "[rknn] truncated float32 residuals %s\n", path.c_str());
            exit(3);
        }
        char extra;
        if (f.read(&extra, 1)) {
            fprintf(stderr, "[rknn] oversized LRQ verifier artifact %s\n", path.c_str());
            exit(3);
        }
    }
};

static std::string radius_relax_key() {
    const char* s = std::getenv("ANQI_RADIUS_RELAX");
    if (!s || !*s || atof(s) == 0.0) return "";
    return std::string(":relax") + s;
}

static uint16_t float_to_half_nearest(float fv) {
    uint32_t x;
    std::memcpy(&x, &fv, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half++;
        return (uint16_t)(sign | half);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float half_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03ffu;
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

struct PrebuiltKnnRecord {
    uint32_t id;
    float distance;
};
static_assert(sizeof(PrebuiltKnnRecord) == 8, "unexpected top-k record layout");

static std::string prebuilt_topk_path() {
    const char* path = std::getenv("ANQI_PREBUILT_TOPK_REC");
    return (path && *path) ? std::string(path) : std::string();
}

static size_t prebuilt_topk_width(size_t fallback) {
    const char* value = std::getenv("ANQI_PREBUILT_TOPK_K");
    if (!value || !*value) return fallback;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0' || parsed < 1 ||
        parsed > 10000) {
        fprintf(stderr, "[rknn] invalid ANQI_PREBUILT_TOPK_K=%s\n", value);
        exit(2);
    }
    return (size_t)parsed;
}

static void validate_prebuilt_topk(
    const std::string& path,
    size_t n,
    size_t source_k,
    size_t required_k
) {
    if (source_k < required_k) {
        fprintf(stderr,
                "[rknn] prebuilt top-k width=%zu is smaller than required K=%zu\n",
                source_k, required_k);
        exit(3);
    }
    struct stat st {};
    const uint64_t expected =
        (uint64_t)n * (uint64_t)source_k * sizeof(PrebuiltKnnRecord);
    if (::stat(path.c_str(), &st) != 0 || (uint64_t)st.st_size != expected) {
        fprintf(stderr,
                "[rknn] bad prebuilt top-k table %s: bytes=%lld expected=%llu\n",
                path.c_str(), (long long)(::stat(path.c_str(), &st) == 0 ? st.st_size : -1),
                (unsigned long long)expected);
        exit(3);
    }
}

static void pread_exact(
    int fd,
    void* destination,
    size_t bytes,
    uint64_t offset,
    const std::string& path
) {
    uint8_t* output = static_cast<uint8_t*>(destination);
    while (bytes != 0) {
        const ssize_t got = ::pread(fd, output, bytes, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) {
            fprintf(stderr,
                    "[rknn] cannot read prebuilt top-k during compaction %s "
                    "offset=%llu: %s\n",
                    path.c_str(), (unsigned long long)offset,
                    got == 0 ? "unexpected EOF" : std::strerror(errno));
            exit(3);
        }
        output += (size_t)got;
        bytes -= (size_t)got;
        offset += (uint64_t)got;
    }
}

static void pwrite_exact(
    int fd,
    const void* source,
    size_t bytes,
    uint64_t offset,
    const std::string& path
) {
    const uint8_t* input = static_cast<const uint8_t*>(source);
    while (bytes != 0) {
        const ssize_t put = ::pwrite(fd, input, bytes, (off_t)offset);
        if (put < 0 && errno == EINTR) continue;
        if (put <= 0) {
            fprintf(stderr,
                    "[rknn] cannot write compact warm-start IDs %s offset=%llu: %s\n",
                    path.c_str(), (unsigned long long)offset,
                    put == 0 ? "zero-byte write" : std::strerror(errno));
            exit(3);
        }
        input += (size_t)put;
        bytes -= (size_t)put;
        offset += (uint64_t)put;
    }
}

static void compact_prebuilt_topk_ids_in_place(
    const std::string& path,
    size_t n,
    size_t source_k,
    size_t width
) {
    validate_prebuilt_topk(path, n, source_k, width);
    const int fd = ::open(path.c_str(), O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "[rknn] cannot open prebuilt top-k for compaction %s: %s\n",
                path.c_str(), std::strerror(errno));
        exit(3);
    }
    constexpr size_t target_chunk_bytes = 64ull << 20;
    const size_t rows_per_chunk = std::max<size_t>(
        1, target_chunk_bytes / (source_k * sizeof(PrebuiltKnnRecord)));
    std::vector<PrebuiltKnnRecord> records(rows_per_chunk * source_k);
    std::vector<uint32_t> ids(rows_per_chunk * width);
    std::atomic<uint64_t> first_bad(UINT64_MAX);
    const size_t progress_step = std::max<size_t>(1, n / 20);
    size_t next_progress = progress_step;
    for (size_t row0 = 0; row0 < n; row0 += rows_per_chunk) {
        const size_t rows = std::min(rows_per_chunk, n - row0);
        const size_t input_records = rows * source_k;
        pread_exact(
            fd, records.data(), input_records * sizeof(records[0]),
            (uint64_t)row0 * source_k * sizeof(records[0]), path);
        #pragma omp parallel for schedule(static)
        for (size_t local = 0; local < rows; local++) {
            const PrebuiltKnnRecord* input = records.data() + local * source_k;
            uint32_t* output = ids.data() + local * width;
            for (size_t k = 0; k < width; k++) {
                if (input[k].id >= n) {
                    uint64_t expected = UINT64_MAX;
                    const uint64_t location =
                        ((uint64_t)(row0 + local) << 32) | (uint64_t)k;
                    first_bad.compare_exchange_strong(expected, location);
                }
                output[k] = input[k].id;
            }
        }
        if (first_bad.load() != UINT64_MAX) {
            const uint64_t location = first_bad.load();
            fprintf(stderr,
                    "[rknn] invalid warm-start id at row=%llu k=%u\n",
                    (unsigned long long)(location >> 32),
                    (unsigned)(location & 0xffffffffu));
            ::close(fd);
            exit(3);
        }
        pwrite_exact(
            fd, ids.data(), rows * width * sizeof(uint32_t),
            (uint64_t)row0 * width * sizeof(uint32_t), path);
        if (row0 + rows >= next_progress || row0 + rows == n) {
            printf("[rknn] compact warm-start IDs: %.1f%%\n",
                   100.0 * (double)(row0 + rows) / (double)n);
            next_progress = row0 + rows + progress_step;
        }
    }
    const uint64_t compact_bytes = (uint64_t)n * width * sizeof(uint32_t);
    if (::ftruncate(fd, (off_t)compact_bytes) != 0 || ::fsync(fd) != 0 ||
        ::close(fd) != 0) {
        fprintf(stderr, "[rknn] cannot finalize compact warm-start IDs %s: %s\n",
                path.c_str(), std::strerror(errno));
        exit(3);
    }
    printf("[rknn] compacted regenerable prebuilt top-k to IDs in place: "
           "%s source_k=%zu width=%zu bytes=%.3fGB\n",
           path.c_str(), source_k, width, compact_bytes / 1e9);
}

static std::string exact_radius_source_identity(const std::string& pfx) {
    const std::string topk = prebuilt_topk_path();
    return topk.empty() ? file_stat_identity(pfx + "_baseknn_gt.bin")
                        : file_stat_identity(topk) + ":interleaved_k" +
                              std::to_string(prebuilt_topk_width(100));
}

static std::vector<uint32_t> load_prebuilt_knn_ids(
    const std::string& path,
    size_t n,
    size_t source_k,
    size_t width
) {
    validate_prebuilt_topk(path, n, source_k, width);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fprintf(stderr, "[rknn] cannot open prebuilt top-k table %s\n", path.c_str());
        exit(3);
    }
    std::vector<uint32_t> ids(n * width);
    const size_t rows_per_chunk = std::max<size_t>(1, (1u << 24) / source_k);
    std::vector<PrebuiltKnnRecord> chunk(rows_per_chunk * source_k);
    for (size_t row0 = 0; row0 < n; row0 += rows_per_chunk) {
        const size_t rows = std::min(rows_per_chunk, n - row0);
        const size_t records = rows * source_k;
        input.read((char*)chunk.data(), (std::streamsize)(records * sizeof(chunk[0])));
        if (input.gcount() != (std::streamsize)(records * sizeof(chunk[0]))) {
            fprintf(stderr, "[rknn] truncated prebuilt top-k table at row=%zu\n", row0);
            exit(3);
        }
        #pragma omp parallel for schedule(static)
        for (size_t local = 0; local < rows; local++) {
            uint32_t* output = &ids[(row0 + local) * width];
            const PrebuiltKnnRecord* row = &chunk[local * source_k];
            for (size_t k = 0; k < width; k++) output[k] = row[k].id;
        }
    }
    printf("[rknn] loaded contiguous prebuilt original KNN %s (n=%zu source_k=%zu width=%zu)\n",
           path.c_str(), n, source_k, width);
    return ids;
}

static void load_exact_radii_table(
    const std::string& pfx,
    size_t n,
    size_t NK,
    bool is_ip,
    std::vector<float>& rk2a_all
) {
    (void)is_ip;
    const std::string topk = prebuilt_topk_path();
    if (!topk.empty()) {
        const size_t source_k = prebuilt_topk_width(NK);
        validate_prebuilt_topk(topk, n, source_k, NK);
        std::ifstream input(topk, std::ios::binary);
        const size_t rows_per_chunk = std::max<size_t>(1, (1u << 24) / source_k);
        std::vector<PrebuiltKnnRecord> chunk(rows_per_chunk * source_k);
        for (size_t row0 = 0; row0 < n; row0 += rows_per_chunk) {
            const size_t rows = std::min(rows_per_chunk, n - row0);
            const size_t records = rows * source_k;
            input.read((char*)chunk.data(), (std::streamsize)(records * sizeof(chunk[0])));
            if (input.gcount() != (std::streamsize)(records * sizeof(chunk[0]))) {
                fprintf(stderr, "[rknn] truncated prebuilt radius table at row=%zu\n", row0);
                exit(3);
            }
            #pragma omp parallel for schedule(static)
            for (size_t local = 0; local < rows; local++) {
                float* output = &rk2a_all[(row0 + local) * NK];
                const PrebuiltKnnRecord* row = &chunk[local * source_k];
                for (size_t k = 0; k < NK; k++) output[k] = row[k].distance;
            }
        }
        printf("[rknn] loaded exact radii from interleaved top-k %s (n=%zu K=%zu)\n",
               topk.c_str(), n, NK);
        return;
    }
    std::string path = pfx + "_baseknn_gt.bin";
    std::ifstream gf(path, std::ios::binary);
    if (!gf) { fprintf(stderr, "[rknn] missing exact radius table %s\n", path.c_str()); exit(3); }
    uint32_t gn = 0, gk = 0;
    gf.read((char*)&gn, 4);
    gf.read((char*)&gk, 4);
    if ((size_t)gn != n || (size_t)gk < NK) {
        fprintf(stderr, "[rknn] bad exact radius table %s: n=%u K=%u expected n=%zu K>=%zu\n",
                path.c_str(), gn, gk, n, NK);
        exit(3);
    }
    gf.seekg((std::streamoff)gn * (std::streamoff)gk * 4, std::ios::cur); // skip ids
    std::vector<float> row(gk);
    for (size_t i = 0; i < n; i++) {
        gf.read((char*)row.data(), (std::streamsize)gk * 4);
        float* out = &rk2a_all[i * NK];
        // dataset_gt stores the final predicate key directly: squared L2 for
        // L2, and -<o,p> for IP.  The 1-ip conversion is only needed for the
        // approximate nndindex distances built in this process.
        for (size_t k = 0; k < NK; k++) out[k] = row[k];
    }
    printf("[rknn] loaded exact radius table %s (n=%zu K=%zu)\n", path.c_str(), n, NK);
}

static void load_exact_radius_column(
    const std::string& pfx,
    size_t n,
    size_t k_one_based,
    bool is_ip,
    std::vector<float>& out
) {
    (void)is_ip;
    const std::string topk = prebuilt_topk_path();
    if (!topk.empty()) {
        const size_t source_k = prebuilt_topk_width(k_one_based);
        validate_prebuilt_topk(topk, n, source_k, k_one_based);
        std::ifstream input(topk, std::ios::binary);
        const size_t rows_per_chunk = std::max<size_t>(1, (1u << 24) / source_k);
        std::vector<PrebuiltKnnRecord> chunk(rows_per_chunk * source_k);
        out.resize(n);
        for (size_t row0 = 0; row0 < n; row0 += rows_per_chunk) {
            const size_t rows = std::min(rows_per_chunk, n - row0);
            const size_t records = rows * source_k;
            input.read((char*)chunk.data(), (std::streamsize)(records * sizeof(chunk[0])));
            if (input.gcount() != (std::streamsize)(records * sizeof(chunk[0]))) {
                fprintf(stderr, "[rknn] truncated prebuilt radius table at row=%zu\n", row0);
                exit(3);
            }
            #pragma omp parallel for schedule(static)
            for (size_t local = 0; local < rows; local++)
                out[row0 + local] = chunk[local * source_k + (k_one_based - 1)].distance;
        }
        printf("[rknn] loaded exact radius column from interleaved top-k %s (n=%zu k=%zu)\n",
               topk.c_str(), n, k_one_based);
        return;
    }
    std::string path = pfx + "_baseknn_gt.bin";
    std::ifstream gf(path, std::ios::binary);
    if (!gf) { fprintf(stderr, "[rknn] missing exact radius table %s\n", path.c_str()); exit(3); }
    uint32_t gn = 0, gk = 0;
    gf.read((char*)&gn, 4);
    gf.read((char*)&gk, 4);
    if ((size_t)gn != n || k_one_based < 1 || k_one_based > (size_t)gk) {
        fprintf(stderr, "[rknn] bad exact radius column request %s: n=%u K=%u expected n=%zu k=%zu\n",
                path.c_str(), gn, gk, n, k_one_based);
        exit(3);
    }
    gf.seekg((std::streamoff)gn * (std::streamoff)gk * 4, std::ios::cur);
    std::vector<float> row(gk);
    out.resize(n);
    for (size_t i = 0; i < n; i++) {
        gf.read((char*)row.data(), (std::streamsize)gk * 4);
        if (!gf) {
            fprintf(stderr, "[rknn] truncated exact radius table %s at row=%zu\n", path.c_str(), i);
            exit(3);
        }
        out[i] = row[k_one_based - 1];
    }
    printf("[rknn] loaded exact radius column %s (n=%zu k=%zu)\n",
           path.c_str(), n, k_one_based);
}

static std::string radius_cache_key(const std::string& mode, size_t NK, const std::string& pfx, int rankm) {
    if (is_analytic_rankm_mode(mode)) {
        return mode + ":M" + std::to_string(rankm) + ":model" +
               hash_string16(file_fingerprint(analytic_rankm_model_path(pfx, rankm))) +
               radius_relax_key();
    }
    if (is_rankm_residual_mode(mode) || is_rankm_only_mode(mode)) {
        std::string key = mode + ":M";
        key += std::getenv("ANQI_RANKM") ? std::getenv("ANQI_RANKM") : "0";
        key += ":b";
        key += std::getenv("ANQI_RESID_BIAS") ? std::getenv("ANQI_RESID_BIAS") : "0";
        key += ":obj";
        key += rq_objective();
        key += ":basis";
        key += rq_basis_key(pfx, rankm);
        if (is_learned_rq_mode(mode)) {
            key += ":cb";
            key += rq_codebook_key(pfx, rankm, learned_rq_bits(mode));
        }
        const std::string radius_source_id = std::getenv("ANQI_RADIUS_SOURCE_ID")
            ? env_or("ANQI_RADIUS_SOURCE_ID", "missing_radius_source_id")
            : exact_radius_source_identity(pfx);
        key += ":src" + hash_string16(radius_source_id);
        key += ":rqpred_v3";
        key += radius_relax_key();
        return key;
    }
    if (mode != "knots") return mode + radius_relax_key();
    std::vector<int> knots = parse_knots(std::getenv("ANQI_RADIUS_KNOTS"), NK);
    std::string key = "knots:";
    for (size_t i = 0; i < knots.size(); i++) {
        key += (i ? "," : "");
        key += std::to_string(knots[i]);
    }
    key += radius_relax_key();
    return key;
}

static size_t conceptual_bytes_for_radius_mode(const std::string& mode, size_t n, size_t NK) {
    if (is_analytic_rankm_mode(mode)) {
        const int rankm = std::getenv("ANQI_RANKM") ? atoi(std::getenv("ANQI_RANKM")) : 0;
        return 40 + n * (size_t)std::max(0, rankm) * sizeof(float);
    }
    if (mode == "exact_fp16" || mode == "exact_u16" || mode == "exact_u16_floor")
        return n * NK * sizeof(uint16_t) + NK * 2 * sizeof(float);
    if (mode == "rankm_resid_u8_floor" || mode == "rankm_resid_u8_bias")
        return n * NK * sizeof(uint8_t) + NK * 2 * sizeof(float);
    if (is_learned_rq_mode(mode)) {
        const int bits = learned_rq_bits(mode);
        return (n * NK * (size_t)bits + 7) / 8 +
               NK * ((size_t)1 << bits) * sizeof(float);
    }
    if (mode == "rankm_resid_f32")
        return n * (size_t)std::max(0, std::getenv("ANQI_RANKM")
            ? atoi(std::getenv("ANQI_RANKM")) : 0) * sizeof(float) +
               n * NK * sizeof(float);
    if (mode == "rankm_resid_u16_floor" || mode == "rankm_resid_u16_bias")
        return n * NK * sizeof(uint16_t) + NK * 2 * sizeof(float);
    if (mode == "knots") return n * parse_knots(std::getenv("ANQI_RADIUS_KNOTS"), NK).size() * sizeof(float);
    return n * NK * sizeof(float);
}

static void apply_rankm_residual_mode(
    const std::string& mode,
    const std::string& pfx,
    std::vector<float>& rk2a_all,
    size_t n,
    size_t NK,
    int RANKM,
    const std::vector<float>& Gba,
    const std::vector<float>& muba,
    bool materialize_radii,
    std::vector<float>& rankm_resid_all,
    size_t& conceptual_radius_bytes,
    std::string& mode_desc
) {
    if (RANKM <= 0 || Gba.empty() || muba.empty()) {
        fprintf(stderr, "[rknn] %s requires ANQI_RANKM>0 and a rank-M basis\n", mode.c_str());
        exit(2);
    }
    const std::string objective = rq_objective();
    validate_rq_objective(objective);
    if (is_learned_rq_mode(mode)) {
        const int bits = learned_rq_bits(mode);
        size_t code_size = 0;
        std::string cb_path = rq_codebook_path(pfx, RANKM, bits);
        std::vector<float> codebook = load_learned_rq_codebook(cb_path, NK, RANKM, bits, code_size);
        const std::string policy = learned_rq_policy(mode);
        rankm_resid_all.assign((size_t)n * NK, 0.0f);
        #pragma omp parallel
        {
            std::vector<double> coeff((size_t)RANKM);
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; i++) {
                float* rk = &rk2a_all[i * NK];
                float* resid_row = &rankm_resid_all[i * NK];
                std::fill(coeff.begin(), coeff.end(), 0.0);
                for (int m = 0; m < RANKM; m++) {
                    const float* gm = &Gba[(size_t)m * NK];
                    double c = 0.0;
                    for (size_t k = 0; k < NK; k++) c += ((double)rk[k] - (double)muba[k]) * (double)gm[k];
                    coeff[(size_t)m] = c;
                }
                for (size_t k = 0; k < NK; k++) {
                    double recon = (double)muba[k];
                    for (int m = 0; m < RANKM; m++) recon += coeff[(size_t)m] * (double)Gba[(size_t)m * NK + k];
                    double residual = (double)rk[k] - recon;
                    const float* centers = &codebook[k * code_size];
                    const double decoded = decode_learned_residual(residual, centers, code_size, policy);
                    resid_row[k] = (float)decoded;
                    if (materialize_radii) rk[k] = (float)(recon + decoded);
                }
            }
        }
        conceptual_radius_bytes = (n * NK * (size_t)bits + 7) / 8 +
                                  NK * code_size * sizeof(float);
        mode_desc = mode + ":M" + std::to_string(RANKM) + ":policy" + policy +
                    ":basis" + rq_basis_key(pfx, RANKM) +
                    ":cb" + rq_codebook_key(pfx, RANKM, bits) + ":obj" + objective + ":rqpred_v3";
        printf("[rknn] learned RQ codebook: path=%s policy=%s bits=%d code_size=%zu\n",
               cb_path.c_str(), policy.c_str(), bits, code_size);
        if (materialize_radii)
            printf("[rknn] materialized rank-M+RQ radius table for Any-K Lift during LRQ decode\n");
        return;
    }
    const bool use_u8 = (mode == "rankm_resid_u8_floor" || mode == "rankm_resid_u8_bias");
    const double levels = use_u8 ? 255.0 : 65535.0;
    const double resid_bias = std::getenv("ANQI_RESID_BIAS") ? atof(std::getenv("ANQI_RESID_BIAS")) : 0.0;
    const bool conservative = (resid_bias <= 0.0);
    rankm_resid_all.assign((size_t)n * NK, 0.0f);
    const int nth = std::max(1, omp_get_max_threads());
    std::vector<std::vector<double>> mn_t(nth, std::vector<double>(NK, std::numeric_limits<double>::infinity()));
    std::vector<std::vector<double>> mx_t(nth, std::vector<double>(NK, -std::numeric_limits<double>::infinity()));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        std::vector<double> coeff((size_t)RANKM);
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; i++) {
            float* rk = &rk2a_all[i * NK];
            std::fill(coeff.begin(), coeff.end(), 0.0);
            for (int m = 0; m < RANKM; m++) {
                const float* gm = &Gba[(size_t)m * NK];
                double c = 0.0;
                for (size_t k = 0; k < NK; k++) c += ((double)rk[k] - (double)muba[k]) * (double)gm[k];
                coeff[(size_t)m] = c;
            }
            for (size_t k = 0; k < NK; k++) {
                double recon = (double)muba[k];
                for (int m = 0; m < RANKM; m++) recon += coeff[(size_t)m] * (double)Gba[(size_t)m * NK + k];
                double e = (double)rk[k] - recon;
                if (e < mn_t[tid][k]) mn_t[tid][k] = e;
                if (e > mx_t[tid][k]) mx_t[tid][k] = e;
            }
        }
    }

    std::vector<double> mn(NK, std::numeric_limits<double>::infinity());
    std::vector<double> mx(NK, -std::numeric_limits<double>::infinity());
    for (int t = 0; t < nth; t++) {
        for (size_t k = 0; k < NK; k++) {
            mn[k] = std::min(mn[k], mn_t[t][k]);
            mx[k] = std::max(mx[k], mx_t[t][k]);
        }
    }

    #pragma omp parallel
    {
        std::vector<double> coeff((size_t)RANKM);
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; i++) {
            float* rk = &rk2a_all[i * NK];
            float* resid_row = &rankm_resid_all[i * NK];
            std::fill(coeff.begin(), coeff.end(), 0.0);
            for (int m = 0; m < RANKM; m++) {
                const float* gm = &Gba[(size_t)m * NK];
                double c = 0.0;
                for (size_t k = 0; k < NK; k++) c += ((double)rk[k] - (double)muba[k]) * (double)gm[k];
                coeff[(size_t)m] = c;
            }
            for (size_t k = 0; k < NK; k++) {
                const double exact = (double)rk[k];
                double recon = (double)muba[k];
                for (int m = 0; m < RANKM; m++) recon += coeff[(size_t)m] * (double)Gba[(size_t)m * NK + k];
                double span = mx[k] - mn[k];
                double decoded = exact - recon;
                if (span > 0.0) {
                    double raw_q = (decoded - mn[k]) * levels / span;
                    double q = std::floor(raw_q + resid_bias);
                    if (q < 0.0) q = 0.0;
                    if (q > levels) q = levels;
                    decoded = mn[k] + q * span / levels;
                }
                double corrected = recon + decoded;
                if (conservative && corrected > exact) corrected = exact;
                resid_row[k] = (float)(corrected - recon);
                if (materialize_radii) rk[k] = (float)corrected;
            }
        }
    }

    conceptual_radius_bytes = use_u8
        ? (n * NK * sizeof(uint8_t) + NK * 2 * sizeof(float))
        : (n * NK * sizeof(uint16_t) + NK * 2 * sizeof(float));
    mode_desc = mode + ":M" + std::to_string(RANKM) + ":b" + std::to_string(resid_bias) +
                ":obj" + objective + ":rqpred_v2";
    if (materialize_radii)
        printf("[rknn] materialized rank-M+RQ radius table for Any-K Lift during residual decode\n");
}

static void apply_radius_mode(
    const std::string& mode,
    std::vector<float>& rk2a_all,
    size_t n,
    size_t NK,
    size_t& conceptual_radius_bytes,
    std::string& mode_desc
) {
    conceptual_radius_bytes = n * NK * sizeof(float);
    mode_desc = mode;
    if (mode == "exact_f32" || mode == "approx") return;

    if (mode == "exact_fp16") {
        std::vector<float> mn(NK, std::numeric_limits<float>::infinity()), mx(NK, -std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < n; i++) {
            const float* row = &rk2a_all[i * NK];
            for (size_t k = 0; k < NK; k++) { mn[k] = std::min(mn[k], row[k]); mx[k] = std::max(mx[k], row[k]); }
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            float* row = &rk2a_all[i * NK];
            for (size_t k = 0; k < NK; k++) {
                double span = (double)mx[k] - (double)mn[k];
                if (span <= 0) continue;
                float z = (float)(((double)row[k] - (double)mn[k]) / span);
                row[k] = (float)((double)mn[k] + (double)half_to_float(float_to_half_nearest(z)) * span);
            }
        }
        conceptual_radius_bytes = n * NK * sizeof(uint16_t) + NK * 2 * sizeof(float);
        return;
    }

    if (mode == "exact_u16" || mode == "exact_u16_floor") {
        const bool conservative = (mode == "exact_u16_floor");
        std::vector<float> mn(NK, std::numeric_limits<float>::infinity()), mx(NK, -std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < n; i++) {
            const float* row = &rk2a_all[i * NK];
            for (size_t k = 0; k < NK; k++) { mn[k] = std::min(mn[k], row[k]); mx[k] = std::max(mx[k], row[k]); }
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            float* row = &rk2a_all[i * NK];
            for (size_t k = 0; k < NK; k++) {
                double span = (double)mx[k] - (double)mn[k];
                if (span <= 0) continue;
                double raw_q = ((double)row[k] - (double)mn[k]) * 65535.0 / span;
                double q = conservative ? std::floor(raw_q) : std::round(raw_q);
                if (q < 0) q = 0;
                if (q > 65535) q = 65535;
                row[k] = (float)((double)mn[k] + q * span / 65535.0);
            }
        }
        conceptual_radius_bytes = n * NK * sizeof(uint16_t) + NK * 2 * sizeof(float);
        return;
    }

    if (mode == "knots") {
        const size_t knots_before = conceptual_radius_bytes;
        std::vector<int> knots = parse_knots(std::getenv("ANQI_RADIUS_KNOTS"), NK);
        std::vector<float> original = rk2a_all;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            float* row = &rk2a_all[i * NK];
            const float* src = &original[i * NK];
            for (size_t j = 0; j < NK; j++) {
                int k = (int)j + 1;
                auto hi_it = std::lower_bound(knots.begin(), knots.end(), k);
                if (hi_it != knots.end() && *hi_it == k) { row[j] = src[j]; continue; }
                if (hi_it == knots.begin()) { row[j] = src[(size_t)knots.front() - 1]; continue; }
                if (hi_it == knots.end()) { row[j] = src[(size_t)knots.back() - 1]; continue; }
                int hi = *hi_it, lo = *(hi_it - 1);
                double t = (std::log((double)k) - std::log((double)lo)) / (std::log((double)hi) - std::log((double)lo));
                double v = (1.0 - t) * (double)src[(size_t)lo - 1] + t * (double)src[(size_t)hi - 1];
                row[j] = (float)v;
            }
        }
        conceptual_radius_bytes = n * knots.size() * sizeof(float);
        mode_desc = "knots:";
        for (size_t i = 0; i < knots.size(); i++) {
            mode_desc += (i ? "," : "");
            mode_desc += std::to_string(knots[i]);
        }
        (void)knots_before;
        return;
    }

    fprintf(stderr, "[rknn] unsupported ANQI_RADIUS_MODE=%s\n", mode.c_str());
    exit(2);
}

static void apply_l2_radius_relax(
    std::vector<float>& rk2a_all,
    size_t n,
    size_t NK,
    bool is_ip,
    std::string& mode_desc
) {
    const char* s = std::getenv("ANQI_RADIUS_RELAX");
    if (!s || !*s) return;
    double relax = atof(s);
    if (relax == 0.0) return;
    if (is_ip) {
        fprintf(stderr, "[rknn] ANQI_RADIUS_RELAX is only implemented for L2 squared-distance radii\n");
        exit(2);
    }
    if (relax <= -1.0) {
        fprintf(stderr, "[rknn] ANQI_RADIUS_RELAX=%.6g gives non-positive radius scale\n", relax);
        exit(2);
    }
    const double scale = 1.0 + relax;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n * NK; i++) {
        double v = (double)rk2a_all[i] * scale;
        rk2a_all[i] = (float)(v > 0.0 ? v : 0.0);
    }
    mode_desc += std::string(":relax") + s;
    printf("[rknn] ANQI_RADIUS_RELAX=%s applied to L2 squared radii (factor=%.8g)\n", s, scale);
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 8) {
        fprintf(stderr,
                "usage: %s L_DESCENT_LIST PREFIX [GRAPH_DEGREE] [LEAF_SIZE] "
                "[TREE_COUNT] [BUILD_EF] [QUERY_K]\n",
                argv[0]);
        return 2;
    }

    auto parse_size = [](const char* text, const char* name, size_t& out) {
        if (!text || !*text) {
            fprintf(stderr, "[rknn] invalid %s (expected a positive integer)\n", name);
            return false;
        }
        char* end = nullptr;
        errno = 0;
        const unsigned long long value = std::strtoull(text, &end, 10);
        if (errno == ERANGE || end == text || *end != '\0' ||
            value == 0 || value > std::numeric_limits<size_t>::max()) {
            fprintf(stderr, "[rknn] invalid %s=%s (expected a positive integer)\n",
                    name, text);
            return false;
        }
        out = static_cast<size_t>(value);
        return true;
    };

    std::vector<size_t> Lds;
    std::string list(argv[1]);
    size_t start = 0;
    while (start <= list.size()) {
        const size_t comma = list.find(',', start);
        const std::string item = list.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        size_t value = 0;
        if (!parse_size(item.c_str(), "L_DESCENT", value)) return 2;
        Lds.push_back(value);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }

    size_t Marg = 48;
    size_t leafA = 1000;
    size_t ntA = 64;
    size_t efA = 400;
    if (argc > 3 && !parse_size(argv[3], "GRAPH_DEGREE", Marg)) return 2;
    if (argc > 4 && !parse_size(argv[4], "LEAF_SIZE", leafA)) return 2;
    if (argc > 5 && !parse_size(argv[5], "TREE_COUNT", ntA)) return 2;
    if (argc > 6 && !parse_size(argv[6], "BUILD_EF", efA)) return 2;

    const std::string PFX(argv[2]);
    std::string lifted = PFX + "_lifted_base.bin";
    std::string meta_p = PFX + "_lifted.meta";
    std::string rk_p   = PFX + "_rknn_gt.bin.rk";
    std::string gt_p   = PFX + "_rknn_gt.bin";

    // Metadata fields: Mstar^2, dimension, rank horizon, metric, and scale.
    std::ifstream mf(meta_p);
    double Mstar2 = 0.0;
    size_t d0 = 0;
    size_t kk = 0;
    std::string metric;
    double scale = 0.0;
    if (!(mf >> Mstar2 >> d0 >> kk >> metric >> scale) ||
        !std::isfinite(Mstar2) || Mstar2 < 0.0 || d0 == 0 || kk == 0 ||
        (metric != "l2" && metric != "ip") ||
        !std::isfinite(scale) || scale <= 0.0) {
        fprintf(stderr, "[rknn] invalid or missing lift metadata: %s\n",
                meta_p.c_str());
        return 3;
    }
    bool is_ip = (metric=="ip");
    printf("[rknn] metric=%s M*^2=%.6g d=%zu k=%zu L_descent_points=%zu\n",
           metric.c_str(), Mstar2, d0, kk, Lds.size());

    // The verifier stores r_1..r_NK. By default the graph also lifts with
    // r_NK; ANQI_GRAPH_NK decouples graph geometry from verifier coverage for
    // the horizon generalization experiment.
    size_t NK = 100;
    if (const char* nk_env = std::getenv("ANQI_NK")) {
        char* end = nullptr;
        errno = 0;
        const unsigned long long parsed = std::strtoull(nk_env, &end, 10);
        if (!*nk_env || errno == ERANGE || end == nk_env || *end != '\0' ||
            parsed < 1 || parsed > 10000) {
            fprintf(stderr, "[rknn] invalid ANQI_NK=%s (expected integer in [1,10000])\n", nk_env);
            return 2;
        }
        NK = (size_t)parsed;
    }
    size_t GRAPH_NK = NK;
    if (const char* graph_nk_env = std::getenv("ANQI_GRAPH_NK")) {
        char* end = nullptr;
        errno = 0;
        const unsigned long long parsed = std::strtoull(graph_nk_env, &end, 10);
        if (!*graph_nk_env || errno == ERANGE || end == graph_nk_env || *end != '\0' ||
            parsed < 1 || parsed > 10000) {
            fprintf(stderr,
                    "[rknn] invalid ANQI_GRAPH_NK=%s (expected integer in [1,10000])\n",
                    graph_nk_env);
            return 2;
        }
        GRAPH_NK = (size_t)parsed;
    }
    size_t Kq = kk;
    if (argc > 7 && !parse_size(argv[7], "QUERY_K", Kq)) return 2;
    if (Kq < 1 || Kq > NK) {
        fprintf(stderr, "[rknn] requested Kq=%zu is outside the Any-K horizon [1,%zu]\n", Kq, NK);
        return 2;
    }
    {
        std::string gp = PFX + "_rknn_gt_k" + std::to_string(Kq) + ".bin";
        std::ifstream tf(gp, std::ios::binary);
        if (tf.good()) {
            gt_p = gp;
        } else if (Kq != kk) {
            fprintf(stderr,
                    "[rknn] missing k-specific reverse-kNN GT %s; the generic GT is only valid for metadata k=%zu\n",
                    gp.c_str(), kk);
            exit(3);
        } else {
            std::ifstream generic_gt(gt_p, std::ios::binary);
            if (!generic_gt.good()) {
                fprintf(stderr, "[rknn] missing reverse-kNN GT %s\n", gt_p.c_str());
                exit(3);
            }
        }
    }
    printf("[rknn] any-k: Kq=%zu NK=%zu graph_NK=%zu GT=%s\n",
           Kq, NK, GRAPH_NK, gt_p.c_str());

    // Original vectors are used by lifting and exact candidate verification.
    size_t n, d, nq, dq; float* base = load_base_vectors(PFX+"_base.bin", n, d);
    float* query = load_fbin(PFX+"_query.bin", nq, dq);
    if (d != d0 || dq != d0) {
        fprintf(stderr,
                "[rknn] vector dimension does not match lift metadata: "
                "base=%zu query=%zu metadata=%zu\n",
                d, dq, d0);
        return 3;
    }

    // Load the rank-horizon threshold source.
    std::ifstream rf(rk_p, std::ios::binary | std::ios::ate);
    const std::streamoff rk_bytes = rf
        ? static_cast<std::streamoff>(rf.tellg())
        : static_cast<std::streamoff>(-1);
    rf.seekg(0);
    uint32_t rn = 0, rkk = 0;
    if (!rf || rk_bytes != static_cast<std::streamoff>(8 + n * sizeof(float)) ||
        !rf.read(reinterpret_cast<char*>(&rn), 4) ||
        !rf.read(reinterpret_cast<char*>(&rkk), 4) || rn != n || rkk != kk) {
        fprintf(stderr, "[rknn] invalid rank-threshold source: %s\n", rk_p.c_str());
        return 3;
    }
    std::vector<float> rk2(n);
    if (!rf.read(reinterpret_cast<char*>(rk2.data()), n * sizeof(float))) {
        fprintf(stderr, "[rknn] truncated rank-threshold source: %s\n", rk_p.c_str());
        return 3;
    }

    // Load reverse-kNN ground truth in CSR form. The file also contains one
    // float key per id after the id array.
    std::ifstream gf(gt_p, std::ios::binary | std::ios::ate);
    const std::streamoff gt_bytes = gf
        ? static_cast<std::streamoff>(gf.tellg())
        : static_cast<std::streamoff>(-1);
    gf.seekg(0);
    uint32_t gnq = 0;
    if (!gf || !gf.read(reinterpret_cast<char*>(&gnq), 4) || gnq != nq) {
        fprintf(stderr, "[rknn] invalid reverse-kNN GT header: %s\n", gt_p.c_str());
        return 3;
    }
    std::vector<uint32_t> goff(static_cast<size_t>(gnq) + 1);
    if (!gf.read(reinterpret_cast<char*>(goff.data()), goff.size() * sizeof(uint32_t)) ||
        goff.front() != 0 ||
        !std::is_sorted(goff.begin(), goff.end())) {
        fprintf(stderr, "[rknn] invalid reverse-kNN GT offsets: %s\n", gt_p.c_str());
        return 3;
    }
    const uint64_t expected_gt_bytes = 4 + goff.size() * sizeof(uint32_t) +
        static_cast<uint64_t>(goff.back()) * (sizeof(uint32_t) + sizeof(float));
    if (gt_bytes < 0 || static_cast<uint64_t>(gt_bytes) != expected_gt_bytes) {
        fprintf(stderr, "[rknn] invalid reverse-kNN GT size: %s\n", gt_p.c_str());
        return 3;
    }
    std::vector<uint32_t> gids(goff.back());
    if (!gf.read(reinterpret_cast<char*>(gids.data()), gids.size() * sizeof(uint32_t)) ||
        std::any_of(gids.begin(), gids.end(), [n](uint32_t id) { return id >= n; })) {
        fprintf(stderr, "[rknn] invalid reverse-kNN GT ids: %s\n", gt_p.c_str());
        return 3;
    }

    std::string DSDIR = PFX.substr(0, PFX.rfind('/')+1);
    const std::string graph_cache_dir = cache_dir_prefix("ANQI_GRAPH_CACHE_DIR", DSDIR);
    const std::string verifier_cache_dir = cache_dir_prefix("ANQI_VERIFIER_CACHE_DIR", DSDIR);
    auto tb0 = std::chrono::steady_clock::now();
    std::string approxlift = DSDIR+"_approxlift.bin";
    std::string rk2a_file  = DSDIR+"_anqi_rk2a.bin";
    std::string rankm_resid_file = DSDIR+"_anqi_rankm_residual.bin";
    std::string meta_file  = DSDIR+"_anqi_meta.txt";
    std::string radius_mode_file = DSDIR+"_anqi_radius_mode.txt";
    std::string graph_state_file = DSDIR+"_anqi_graph_state.txt";
    auto fexists=[](const std::string&p){ std::ifstream f(p); return f.good(); };
    std::string radius_mode = std::getenv("ANQI_RADIUS_MODE") ? std::getenv("ANQI_RADIUS_MODE") : "approx";
    int RANKM = 0;
    if (const char* rankm_env = std::getenv("ANQI_RANKM")) {
        char* end = nullptr;
        errno = 0;
        const long value = std::strtol(rankm_env, &end, 10);
        if (!*rankm_env || errno == ERANGE || end == rankm_env || *end != '\0' ||
            value < 0 || value > 1024) {
            fprintf(stderr, "[rknn] invalid ANQI_RANKM=%s (expected integer in [0,1024])\n",
                    rankm_env);
            return 2;
        }
        RANKM = static_cast<int>(value);
    }
    const std::string graph_geometry = resolve_graph_geometry(RANKM);
    const bool graph_uses_rankm = (graph_geometry == "rankm");
    const bool graph_anyk_lift = (graph_geometry == "anyk_lift");
    const bool graph_original = (graph_geometry == "original");
    const bool graph_independent_verifier = graph_anyk_lift || graph_original;
    if (graph_original && is_ip) {
        fprintf(stderr,
                "[rknn] ANQI_GRAPH_GEOMETRY=original currently supports only L2; "
                "the raw-space IP envelope needs a separate predicate mapping\n");
        exit(2);
    }
    const int GRAPH_RANKM = graph_uses_rankm ? RANKM : 0;
    std::string verifier_placement;
    if (std::getenv("ANQI_VERIFIER_PLACEMENT"))
        verifier_placement = env_or("ANQI_VERIFIER_PLACEMENT", "integrated");
    else
        verifier_placement = (graph_independent_verifier &&
                              (is_learned_rq_mode(radius_mode) || radius_mode == "rankm_resid_f32" ||
                               is_rankm_only_mode(radius_mode) ||
                               is_analytic_rankm_mode(radius_mode)))
                                 ? "postfilter" : "integrated";
    if (verifier_placement != "integrated" && verifier_placement != "postfilter") {
        fprintf(stderr, "[rknn] unsupported ANQI_VERIFIER_PLACEMENT=%s (expected integrated or postfilter)\n",
                verifier_placement.c_str());
        exit(2);
    }
    const bool postfilter_lrq = graph_independent_verifier &&
                                verifier_placement == "postfilter" && is_learned_rq_mode(radius_mode);
    const bool postfilter_resid_f32 = graph_independent_verifier &&
                                      verifier_placement == "postfilter" &&
                                      radius_mode == "rankm_resid_f32";
    const bool postfilter_rankm = graph_independent_verifier &&
                                  verifier_placement == "postfilter" && is_rankm_only_mode(radius_mode);
    const bool postfilter_analytic = graph_independent_verifier &&
                                     verifier_placement == "postfilter" &&
                                     is_analytic_rankm_mode(radius_mode);
    const bool postfilter_exact = graph_independent_verifier &&
                                  verifier_placement == "postfilter" && radius_mode == "exact_f32";
    const bool postfilter_compact = postfilter_lrq || postfilter_resid_f32 || postfilter_rankm;
    const bool postfilter_verifier = postfilter_compact || postfilter_analytic || postfilter_exact;
    if (verifier_placement == "postfilter" && !postfilter_verifier) {
        fprintf(stderr,
                "[rknn] postfilter verifier supports Any-K Lift or original geometry "
                "with exact_f32, rankm_only, analytic_rankm, float32 residual, or learned LRQ; "
                "got metric=%s geometry=%s radius_mode=%s\n",
                metric.c_str(), graph_geometry.c_str(), radius_mode.c_str());
        exit(2);
    }
    if (GRAPH_NK != NK && !postfilter_verifier) {
        fprintf(stderr,
                "[rknn] ANQI_GRAPH_NK differs from ANQI_NK only for an independent "
                "postfilter verifier\n");
        exit(2);
    }
    if (graph_original && !postfilter_verifier) {
        fprintf(stderr,
                "[rknn] ANQI_GRAPH_GEOMETRY=original is an ablation-only geometry and requires "
                "ANQI_VERIFIER_PLACEMENT=postfilter with exact_f32, rankm_only, "
                "float32 residual, or learned LRQ\n");
        exit(2);
    }
    if (postfilter_verifier && std::getenv("ANQI_SPECIFIC")) {
        fprintf(stderr,
                "[rknn] independent postfilter verifier requires a shared graph; "
                "unset ANQI_SPECIFIC\n");
        exit(2);
    }
    if (postfilter_analytic && is_ip) {
        fprintf(stderr,
                "[rknn] analytic_rankm uses the L2 LID power law and does not support IP\n");
        exit(2);
    }
    std::string candidate_envelope = std::getenv("ANQI_CANDIDATE_ENVELOPE")
        ? env_or("ANQI_CANDIDATE_ENVELOPE", "graph_horizon")
        : (postfilter_analytic ? "verifier" : "graph_horizon");
    if (candidate_envelope != "graph_horizon" && candidate_envelope != "verifier") {
        fprintf(stderr,
                "[rknn] unsupported ANQI_CANDIDATE_ENVELOPE=%s "
                "(expected graph_horizon or verifier)\n",
                candidate_envelope.c_str());
        exit(2);
    }
    const bool candidate_uses_verifier = candidate_envelope == "verifier";
    if (candidate_uses_verifier &&
        (!postfilter_verifier || !graph_anyk_lift || is_ip)) {
        fprintf(stderr,
                "[rknn] verifier candidate envelope currently requires L2 Any-K Lift "
                "with an independent postfilter verifier\n");
        exit(2);
    }
    bool exact_radius_mode = (radius_mode == "exact_f32" || radius_mode == "exact_fp16" ||
                              radius_mode == "exact_u16" || radius_mode == "exact_u16_floor" ||
                              radius_mode == "knots" || is_rankm_residual_mode(radius_mode) ||
                              is_rankm_only_mode(radius_mode) || is_analytic_rankm_mode(radius_mode));
    if (is_rankm_residual_mode(radius_mode)) validate_rq_objective(rq_objective());
    std::string radius_key = radius_cache_key(radius_mode, NK, PFX, RANKM);
    std::string graph_key = graph_cache_key(Marg, leafA, ntA, efA, graph_geometry, GRAPH_RANKM);
    if (std::getenv("ANQI_SPECIFIC")) graph_key += "_k" + std::to_string(Kq);
    const std::string graph_radius_source =
        env_or("ANQI_GRAPH_RADIUS_SOURCE_PFX", PFX.c_str());
    if (postfilter_verifier) {
        const std::string graph_radius_id = std::getenv("ANQI_GRAPH_RADIUS_ID")
            ? env_or("ANQI_GRAPH_RADIUS_ID", "exact_r100_v1")
            : file_stat_identity(
                  base_vector_source(graph_radius_source + "_base.bin")) + "|" +
              exact_radius_source_identity(graph_radius_source) +
              ":k" + std::to_string(GRAPH_NK) + ":v2";
        graph_key += "_graph_exact_r" + std::to_string(GRAPH_NK) + "_" +
                     hash_string16(graph_radius_id);
    } else graph_key += "_rad" + hash_string16(radius_key);
    graph_key = sanitize_tag(graph_key);
    if (postfilter_verifier) {
        const std::string graph_artifact_tag = hash_string16(graph_key);
        approxlift = graph_cache_dir + "_approxlift_graph_" + graph_artifact_tag + ".bin";
        meta_file = graph_cache_dir + "_anqi_meta_graph_" + graph_artifact_tag + ".txt";
        graph_state_file = graph_cache_dir + "_anqi_graph_state_" + graph_artifact_tag + ".txt";
    }
    const std::string graph_vector_file = graph_original ? (PFX + "_base.bin") : approxlift;
    if (graph_original && std::getenv("ANQI_BASE_U8BIN")) {
        fprintf(stderr,
                "[rknn] direct uint8 base loading is currently supported only for "
                "lifted graph geometry\n");
        exit(2);
    }
    setenv("ANQI_GRAPH_CACHE_KEY", graph_key.c_str(), 1);
    const std::string graph_artifact_dir = postfilter_verifier ? graph_cache_dir : DSDIR;
    std::string marker = graph_artifact_dir + "_anqi_" + graph_key + ".built";
    const std::string cache_state_key = postfilter_verifier
        ? std::string("graph=") + graph_key
        : radius_key + ":graph=" + graph_key;
    printf("[rknn] graph geometry=%s graph_rankm=%d verification_rankm=%d radius_mode=%s placement=%s\n",
           graph_geometry.c_str(), GRAPH_RANKM, RANKM, radius_mode.c_str(), verifier_placement.c_str());
    printf("[rknn] graph cache key=%s\n", graph_key.c_str());
    auto read_text=[](const std::string&p){ std::ifstream f(p); std::string s; std::getline(f,s); return s; };
    const std::string& state_file = postfilter_verifier ? graph_state_file : radius_mode_file;
    bool radius_cache_match = fexists(state_file) && read_text(state_file) == cache_state_key;
    // Reuse the graph only when its state key and all required artifacts match.
    bool can_load = radius_cache_match && fexists(marker) && fexists(graph_vector_file) && fexists(meta_file) &&
                    (postfilter_verifier ||
                     (fexists(rk2a_file) && (!is_rankm_residual_mode(radius_mode) || fexists(rankm_resid_file))));
    if (postfilter_verifier) {
        printf("[rknn] component caches: graph=%s verifier=%s\n",
               graph_cache_dir.c_str(), verifier_cache_dir.c_str());
    }
    if (postfilter_verifier && env_flag("ANQI_REQUIRE_GRAPH_CACHE") && !can_load) {
        fprintf(stderr,
                "[rknn] required graph cache is incomplete or incompatible: dir=%s key=%s\n",
                graph_cache_dir.c_str(), graph_key.c_str());
        exit(3);
    }

    // The rank-M basis is needed during both construction and search.
    std::vector<float> Gba, muba;
    const bool needs_rankm_basis = GRAPH_RANKM > 0 || is_rankm_residual_mode(radius_mode) ||
                                   is_rankm_only_mode(radius_mode);
    if (needs_rankm_basis) {
        std::string rb = PFX + "_radbasis_M" + std::to_string(RANKM) + ".bin";
        std::ifstream bf(rb, std::ios::binary | std::ios::ate);
        const std::streamoff basis_bytes = bf
            ? static_cast<std::streamoff>(bf.tellg())
            : static_cast<std::streamoff>(-1);
        bf.seekg(0);
        int Kb = 0, Mb = 0;
        if (!bf || !bf.read(reinterpret_cast<char*>(&Kb), 4) ||
            !bf.read(reinterpret_cast<char*>(&Mb), 4) || Kb <= 0 || Mb <= 0 ||
            static_cast<size_t>(Kb) != NK || Mb != RANKM) {
            fprintf(stderr,
                    "[rknn] missing or invalid rank-M basis %s "
                    "(K=%d M=%d, expected K=%zu M=%d)\n",
                    rb.c_str(), Kb, Mb, NK, RANKM);
            return 3;
        }
        const uint64_t expected_basis_bytes = 8 +
            static_cast<uint64_t>(Kb) * sizeof(float) +
            static_cast<uint64_t>(Mb) * Kb * sizeof(float);
        if (basis_bytes < 0 ||
            static_cast<uint64_t>(basis_bytes) != expected_basis_bytes) {
            fprintf(stderr, "[rknn] invalid rank-M basis size: %s\n", rb.c_str());
            return 3;
        }
        muba.resize(Kb); Gba.resize((size_t)Mb*Kb);
        if (!bf.read(reinterpret_cast<char*>(muba.data()),
                     static_cast<std::streamsize>(Kb) * sizeof(float)) ||
            !bf.read(reinterpret_cast<char*>(Gba.data()),
                     static_cast<std::streamsize>(Mb) * Kb * sizeof(float))) {
            fprintf(stderr, "[rknn] truncated rank-M basis: %s\n", rb.c_str());
            return 3;
        }
        printf("[rknn] loaded rank-M basis: M=%d path=%s\n", RANKM, rb.c_str());
    }
    // Shared lifted-index parameters.
    Parms par; par.n=n;
    par.dim = graph_original ? (uint32_t)d
                             : (uint32_t)(d + (GRAPH_RANKM > 0 ? GRAPH_RANKM : 0) + 2);
    par.meric="l2"; par.random_seed=10;
    par.nn_k = 50;
    if (const char* lift_nnk = std::getenv("ANQI_LIFTNNK")) {
        if (!parse_size(lift_nnk, "ANQI_LIFTNNK", par.nn_k)) return 2;
    }
    par.M=Marg; par.leaf_size=100; par.max_depth=100; par.n_trees=64;
    par.ef_alpha=8; par.prune_alpha = std::getenv("ANQI_ALPHA") ? (float)atof(std::getenv("ANQI_ALPHA")) : 1.2f;
    par.explore_range=100; par.n_threads=64;
    par.folder_path=postfilter_verifier ? graph_cache_dir : DSDIR;

    std::vector<float> rk2a_all;
    if (!postfilter_verifier) rk2a_all.resize((size_t)n * NK);
    // Exact graph-only r_100 signal. Lift embeds it in d+2; original geometry
    // keeps the same envelope as an external per-object sidecar.
    std::vector<float> graph_horizon_radius;
    std::vector<float> verification_thresholds;  // current-k threshold, independent of graph/search
    AnalyticRankMVerifier analytic_rankm;
    std::string analytic_rankm_file;
    CompactLrqVerifier compact_lrq;
    std::string compact_lrq_file;
    bool compact_lrq_ready = false;
    double verifier_load_ms = 0.0;
    double verifier_encode_save_ms = 0.0;
    if (postfilter_analytic) {
        if (RANKM <= 0) {
            fprintf(stderr, "[rknn] analytic_rankm requires ANQI_RANKM>0\n");
            exit(2);
        }
        analytic_rankm_file = analytic_rankm_model_path(PFX, RANKM);
        const auto load_t0 = std::chrono::steady_clock::now();
        analytic_rankm.load(analytic_rankm_file, n, RANKM);
        verifier_load_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - load_t0).count() / 1000.0;
        if (analytic_rankm.train_kmax != GRAPH_NK) {
            fprintf(stderr,
                    "[rknn] analytic Rank-M training horizon=%zu must match graph horizon=%zu "
                    "for the controlled Any-K experiment\n",
                    analytic_rankm.train_kmax, GRAPH_NK);
            exit(2);
        }
        printf("[rknn] loaded analytic Rank-M verifier %s "
               "(train_kmax=%zu fit_kmin=%zu M=%d b0=%.9g flags=%u persistent=%.3fGB load_ms=%.3f)\n",
               analytic_rankm_file.c_str(), analytic_rankm.train_kmax,
               analytic_rankm.fit_kmin, analytic_rankm.rankm, analytic_rankm.b0,
               analytic_rankm.flags, analytic_rankm.persistent_bytes() / 1e9,
               verifier_load_ms);
    }
    if (postfilter_compact) {
        const int bits = postfilter_lrq ? learned_rq_bits(radius_mode)
                                        : (postfilter_resid_f32 ? 32 : 0);
        size_t code_size = 0;
        std::vector<float> codebook;
        if (postfilter_lrq) {
            const std::string cb_path = rq_codebook_path(PFX, RANKM, bits);
            codebook = load_learned_rq_codebook(cb_path, NK, RANKM, bits, code_size);
        }
        compact_lrq.configure(n, NK, RANKM, bits,
                              postfilter_lrq ? learned_rq_policy(radius_mode)
                                             : (postfilter_resid_f32 ? "float32" : "rankm"),
                              muba, Gba, codebook, code_size);
        compact_lrq_file = verifier_cache_dir + "_anqi_verifier_" + hash_string16(radius_key) + ".bin";
        if (fexists(compact_lrq_file)) {
            const auto load_t0 = std::chrono::steady_clock::now();
            compact_lrq.load(compact_lrq_file);
            compact_lrq_ready = true;
            verifier_load_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - load_t0).count() / 1000.0;
            printf("[rknn] loaded independent compact verifier mode=%s bits=%d %s "
                   "(persistent=%.3fGB load_ms=%.3f)\n",
                   postfilter_lrq ? "rank-M+LRQ" :
                   (postfilter_resid_f32 ? "rank-M+float32-residual" : "rank-M-only"), bits,
                   compact_lrq_file.c_str(), compact_lrq.persistent_bytes() / 1e9,
                   verifier_load_ms);
        }
    }
    std::vector<float> rankm_resid_all;          // decoded residuals for rank-M residual modes
    size_t conceptual_radius_bytes = postfilter_compact
        ? compact_lrq.persistent_bytes()
        : (postfilter_analytic ? analytic_rankm.persistent_bytes()
                               : (size_t)n * NK * sizeof(float));
    std::string radius_mode_desc = radius_mode;
    std::vector<std::vector<uint32_t>> nbr_orig;
    std::vector<uint32_t> nbr_orig_flat;
    size_t nbr_orig_flat_width = 0;
    nndindex::WarmStartStream nbr_orig_stream;
    bool nbr_orig_stream_ready = false;
    bool compact_prebuilt_ids = false;
    size_t prebuilt_source_k = 0;

    auto ensure_compact_lrq = [&](bool keep_graph_horizon_radius) {
        if (!postfilter_compact || compact_lrq_ready) return;
        std::vector<float> exact_radii((size_t)n * NK);
        load_exact_radii_table(PFX, n, NK, is_ip, exact_radii);
        const auto encode_t0 = std::chrono::steady_clock::now();
        compact_lrq.build(exact_radii);
        compact_lrq.save(compact_lrq_file);
        verifier_encode_save_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encode_t0).count() / 1000.0;
        compact_lrq_ready = true;
        if (keep_graph_horizon_radius) {
            load_exact_radius_column(
                graph_radius_source, n, GRAPH_NK, is_ip, graph_horizon_radius);
        }
        printf("[rknn] built independent compact verifier mode=%s bits=%d %s "
               "(persistent=%.3fGB encode_save_ms=%.3f)\n",
               postfilter_lrq ? "rank-M+LRQ" :
               (postfilter_resid_f32 ? "rank-M+float32-residual" : "rank-M-only"),
               compact_lrq.bits,
               compact_lrq_file.c_str(), compact_lrq.persistent_bytes() / 1e9,
               verifier_encode_save_ms);
    };

    double original_knn_s = 0.0;
    double lift_verifier_prepare_s = 0.0;
    double graph_index_s = 0.0;
    if (can_load) {
        // Load compatible cached construction artifacts.
        if (!postfilter_verifier) {
            read_binary_exact(rk2a_file, rk2a_all.data(), float_table_bytes(n, NK, "rk2a"), "rk2a");
            if (is_rankm_residual_mode(radius_mode)) {
                rankm_resid_all.resize((size_t)n * NK);
                read_binary_exact(rankm_resid_file, rankm_resid_all.data(),
                                  float_table_bytes(n, NK, "rankm residual"), "rankm residual");
            }
        }
        {
            std::ifstream f(meta_file);
            if (!(f >> Mstar2 >> scale)) {
                fprintf(stderr, "[rknn] bad meta cache %s\n", meta_file.c_str());
                exit(3);
            }
        }
        ensure_compact_lrq(false);
        if ((graph_original || candidate_uses_verifier) && graph_horizon_radius.empty())
            load_exact_radius_column(
                graph_radius_source, n, GRAPH_NK, is_ip, graph_horizon_radius);
        conceptual_radius_bytes = postfilter_compact
            ? compact_lrq.persistent_bytes()
            : (postfilter_analytic ? analytic_rankm.persistent_bytes()
                                   : conceptual_bytes_for_radius_mode(radius_mode, n, NK));
        radius_mode_desc = postfilter_verifier ? radius_key + ":postfilter" : radius_key;
        printf("[rknn] LOAD 已存索引 (M=%zu): 跳过 kNN+lift+build, M*^2=%.6g scale=%.4g radius=%s\n",
               Marg, Mstar2, scale, radius_mode_desc.c_str());
    } else {
        // Build the original-space AKNN warm start, thresholds, and lift.
        Parms po; po.n=n; po.dim=d; po.meric=metric; po.random_seed=10;
        po.nn_k=100; po.M=32; po.leaf_size=leafA; po.max_depth=100; po.n_trees=ntA;
        po.ef_alpha=8; po.prune_alpha=1.2f; po.explore_range=efA; po.n_threads=64;
        printf("[rknn] original-space AKNN: leaf=%zu n_trees=%zu ef=%zu metric=%s\n", leafA, ntA, efA, metric.c_str());
        po.folder_path=DSDIR;
        std::vector<std::vector<float>> dist_orig;
        const auto original_knn_t0 = std::chrono::steady_clock::now();
        const std::string prebuilt_topk = prebuilt_topk_path();
        if (!prebuilt_topk.empty()) {
            if (!postfilter_verifier) {
                fprintf(stderr,
                        "[rknn] ANQI_PREBUILT_TOPK_REC currently requires an independent "
                        "postfilter verifier\n");
                exit(2);
            }
            const size_t source_k = prebuilt_topk_width(po.nn_k);
            prebuilt_source_k = source_k;
            nbr_orig_flat_width = po.nn_k;
            const char* stream_command =
                std::getenv("ANQI_PREBUILT_WARM_START_COMMAND");
            compact_prebuilt_ids =
                env_flag("ANQI_COMPACT_PREBUILT_TOPK_IDS_INPLACE");
            if (compact_prebuilt_ids && stream_command && *stream_command) {
                fprintf(stderr,
                        "[rknn] compact local warm-start IDs cannot be combined with "
                        "ANQI_PREBUILT_WARM_START_COMMAND\n");
                exit(2);
            }
            if (compact_prebuilt_ids) {
                validate_prebuilt_topk(
                    prebuilt_topk, n, source_k, nbr_orig_flat_width);
                printf("[rknn] deferred original KNN IDs to compact local graph "
                       "warm-start (source_k=%zu width=%zu)\n",
                       source_k, nbr_orig_flat_width);
            } else if (stream_command && *stream_command) {
                nbr_orig_stream.command = stream_command;
                nbr_orig_stream.stride = source_k;
                nbr_orig_stream.width = nbr_orig_flat_width;
                nbr_orig_stream_ready = true;
                validate_prebuilt_topk(
                    prebuilt_topk, n, source_k, nbr_orig_flat_width);
                printf("[rknn] deferred original KNN IDs to streamed graph warm-start "
                       "(source_k=%zu width=%zu)\n",
                       source_k, nbr_orig_flat_width);
            } else {
                nbr_orig_flat = load_prebuilt_knn_ids(
                    prebuilt_topk, n, source_k, nbr_orig_flat_width);
            }
        } else {
            nndindex oindex(PFX+"_base.bin", po);
            oindex.build_approx_knn(nbr_orig, dist_orig);
        }
        original_knn_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - original_knn_t0).count();
        const auto graph_prepare_t0 = std::chrono::steady_clock::now();
        printf("[rknn] 原始 kNN 准备好(复用为图 init), rows=%zu source=%s\n",
               !nbr_orig_stream_ready && !compact_prebuilt_ids
                   ? (nbr_orig_flat.empty() ? nbr_orig.size() : n)
                   : n,
               nbr_orig_stream_ready || compact_prebuilt_ids
                   ? "prebuilt-stream"
                   : (nbr_orig_flat.empty() ? "internal" : "prebuilt-contiguous"));
        if (postfilter_verifier) {
            ensure_compact_lrq(true);
            if (graph_horizon_radius.empty())
                load_exact_radius_column(
                    graph_radius_source, n, GRAPH_NK, is_ip, graph_horizon_radius);
            conceptual_radius_bytes = postfilter_compact
                ? compact_lrq.persistent_bytes()
                : (postfilter_analytic ? analytic_rankm.persistent_bytes()
                                       : (size_t)n * NK * sizeof(float));
            radius_mode_desc = radius_key + ":postfilter";
        } else {
            // rk2a = 第 kk 个的 key。L2: r_k²=dist;IP: −ip_k = (1−⟨⟩)−1 = dist−1。
            for (size_t o=0;o<n;o++){ for(size_t j=0;j<NK;j++){ float v=(dist_orig[o].size()>j)?dist_orig[o][j]:(dist_orig[o].empty()?1e30f:dist_orig[o].back()); rk2a_all[(size_t)o*NK+j]= is_ip ? (v-1.0f) : v; } }
            if (exact_radius_mode) load_exact_radii_table(PFX, n, NK, is_ip, rk2a_all);
            if (is_rankm_residual_mode(radius_mode)) {
                apply_rankm_residual_mode(radius_mode, PFX, rk2a_all, n, NK, RANKM, Gba, muba,
                                          graph_anyk_lift,
                                          rankm_resid_all,
                                          conceptual_radius_bytes, radius_mode_desc);
            } else {
                apply_radius_mode(radius_mode, rk2a_all, n, NK, conceptual_radius_bytes, radius_mode_desc);
            }
            apply_l2_radius_relax(rk2a_all, n, NK, is_ip, radius_mode_desc);
        }
        if (compact_prebuilt_ids) {
            compact_prebuilt_topk_ids_in_place(
                prebuilt_topk, n, prebuilt_source_k, nbr_orig_flat_width);
            nbr_orig_stream.file_path = prebuilt_topk;
            nbr_orig_stream.stride = nbr_orig_flat_width;
            nbr_orig_stream.width = nbr_orig_flat_width;
            nbr_orig_stream.ids_only = true;
            nbr_orig_stream.unlink_file_after_read = true;
            nbr_orig_stream_ready = true;
            release_heap_pages();
        } else if (!prebuilt_topk.empty() &&
            env_flag("ANQI_RELEASE_PREBUILT_TOPK_AFTER_PREP")) {
            if (::unlink(prebuilt_topk.c_str()) != 0) {
                fprintf(stderr,
                        "[rknn] cannot release regenerable prebuilt top-k %s: %s\n",
                        prebuilt_topk.c_str(), std::strerror(errno));
                exit(3);
            }
            printf("[rknn] released regenerable prebuilt top-k after verifier/lift "
                   "preparation: %s\n", prebuilt_topk.c_str());
        }
        printf("[rknn] RADIUS_MODE=%s conceptual_radius_store=%.3fGB\n",
               radius_mode_desc.c_str(), conceptual_radius_bytes / 1e9);
        if (graph_original) {
            if (graph_horizon_radius.empty())
                load_exact_radius_column(
                    graph_radius_source, n, GRAPH_NK, is_ip, graph_horizon_radius);
            Mstar2 = 0.0;
            scale = 1.0;
            printf("[rknn] BUILD original L2 graph (dim=%zu) with external exact-r%zu envelope; "
                   "verifier=%s\n",
                   d, NK, radius_mode_desc.c_str());
        } else {
        double sum_n2=0;
        #pragma omp parallel for reduction(+:sum_n2)
        for(size_t i=0;i<n;i++){const float*o=base+i*d; double s=0; for(size_t j=0;j<d;j++) s+=(double)o[j]*o[j]; sum_n2+=s;}
        double s_scale=1.0/std::sqrt(sum_n2/n+1e-12), sc2=s_scale*s_scale;
        // Graph geometry is independent of verification: Any-K Lift uses only
        // the horizon radius r_NK in d+2 dimensions.
        uint32_t dl = GRAPH_RANKM>0 ? (uint32_t)(d+1+GRAPH_RANKM) : (uint32_t)(d+1);
        std::vector<float> hat((size_t)n*dl); double maxn2=0;
        #pragma omp parallel for reduction(max:maxn2)
        for(size_t i=0;i<n;i++){const float*o=base+i*d; float*h=hat.data()+i*dl; double nr2=0;
            for(size_t j=0;j<d;j++){double v=(double)o[j]*s_scale; h[j]=(float)v; nr2+=v*v;}
            if (GRAPH_RANKM>0) {                                    // rank-M 图：M 个半径曲线坐标
                // L2: ô=[os, −½‖os‖², ½s²c_i];membership ‖q̂−ô‖²≤R²(k) 含 ‖o‖² 项 → 用 h[d] 槽。
                // IP: ô=[os, 0, s²c_i];membership −⟨q,o⟩≤rk2a 线性,无 ‖o‖² 项 → h[d] 槽置 0。
                h[d]= is_ip ? 0.0f : (float)(-0.5*nr2);
                const float* rk=&rk2a_all[(size_t)i*NK];
                for(int m=0;m<GRAPH_RANKM;m++){ double c=0; const float* gm=&Gba[(size_t)m*NK];
                    for(size_t k=0;k<NK;k++) c+=((double)rk[k]-(double)muba[k])*(double)gm[k];   // c_i=(rk2a−μ)·G_i
                    h[d+1+m]=(float)((is_ip?1.0:0.5)*sc2*c); }      // IP: s²c_i;L2: ½s²c_i
                double hn2=0; for(uint32_t j=0;j<dl;j++) hn2+=(double)h[j]*h[j]; if(hn2>maxn2)maxn2=hn2;
            } else {
                size_t lk_=std::getenv("ANQI_SPECIFIC")?Kq:NK;
                float rmax_=postfilter_verifier ? graph_horizon_radius[i]
                                                : rk2a_all[(size_t)i*NK+(lk_-1)];
                h[d]= is_ip ? (float)((double)rmax_*sc2) : (float)(0.5*((double)rmax_*sc2 - nr2));
                double hn2=nr2+(double)h[d]*h[d]; if(hn2>maxn2)maxn2=hn2;
            }}
        Mstar2=maxn2*1.0001; scale=s_scale;
        uint32_t dout = GRAPH_RANKM>0 ? (uint32_t)(d+GRAPH_RANKM+2) : (uint32_t)(d+2), N=n;
        { std::ofstream of(approxlift,std::ios::binary); of.write((char*)&N,4); of.write((char*)&dout,4);
          std::vector<float> ob(dout);
          for(size_t i=0;i<n;i++){const float*h=hat.data()+i*dl; double hn2=0; for(uint32_t j=0;j<dl;j++){ob[j]=h[j];hn2+=(double)h[j]*h[j];}
            double pad=Mstar2-hn2; if(pad<0)pad=0; ob[dl]=(float)std::sqrt(pad); of.write((char*)ob.data(),dout*4);} }
        }
        if (!postfilter_verifier) {
            { std::ofstream f(rk2a_file, std::ios::binary); f.write((char*)rk2a_all.data(), (std::streamsize)n*NK*4); }
            if (is_rankm_residual_mode(radius_mode)) {
                std::ofstream f(rankm_resid_file, std::ios::binary);
                f.write((char*)rankm_resid_all.data(), (std::streamsize)n * NK * 4);
            }
        }
        { std::ofstream f(meta_file); f << std::setprecision(17) << Mstar2 << " " << scale << "\n"; }
        { std::ofstream f(state_file); f << cache_state_key << "\n"; }
        if (!graph_original) {
            const std::string graph_radius_desc = postfilter_verifier
                ? ("exact_r" + std::to_string(GRAPH_NK)) : "radius_mode table";
            printf("[rknn] BUILD lift M*^2=%.6g scale=%.4g (graph radius=%s; verifier=%s)\n",
                   Mstar2, scale, graph_radius_desc.c_str(),
                   postfilter_verifier?radius_mode_desc.c_str():"integrated");
        }
        lift_verifier_prepare_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - graph_prepare_t0).count();
    }

    bool compact_verifier_released_for_index = false;
    if (env_flag("ANQI_RELEASE_VERIFIER_DURING_GRAPH_INDEX") &&
        postfilter_compact && compact_lrq_ready) {
        const size_t released_bytes =
            compact_lrq.coeff.capacity() * sizeof(float) +
            compact_lrq.packed_codes.capacity() * sizeof(uint8_t);
        std::vector<float>().swap(compact_lrq.coeff);
        std::vector<uint8_t>().swap(compact_lrq.packed_codes);
        compact_lrq_ready = false;
        compact_verifier_released_for_index = true;
        release_heap_pages();
        printf("[rknn] released compact verifier payload during graph index "
               "construction: %.3fGB\n", released_bytes / 1e9);
    }

    bool base_released_for_index = false;
    if (env_flag("ANQI_RELEASE_BASE_DURING_GRAPH_INDEX")) {
        _mm_free(base);
        base = nullptr;
        base_released_for_index = true;
        release_heap_pages();
        printf("[rknn] released original base before graph index construction: %.3fGB\n",
               (double)n * d * sizeof(float) / 1e9);
    }

    printf("[rknn] 图度数 M=%zu\n", Marg);
    nndindex index(graph_vector_file, par);
    const auto graph_index_t0 = std::chrono::steady_clock::now();
    if (can_load) index.build_index(false, false);                          // 载已存图
    else {
        if (nbr_orig_stream_ready)
            index.build_index(
                true, false, nullptr, nullptr, 0, 0, nullptr,
                &nbr_orig_stream);
        else if (!nbr_orig_flat.empty())
            index.build_index(true, false, nullptr, nbr_orig_flat.data(),
                              nbr_orig_flat_width, nbr_orig_flat_width,
                              &nbr_orig_flat);
        else
            index.build_index(true, false, &nbr_orig);
        std::ofstream(marker) << "1\n";
    }  // 建+存图+marker
    graph_index_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - graph_index_t0).count();
    std::vector<std::vector<uint32_t>>().swap(nbr_orig);
    std::vector<uint32_t>().swap(nbr_orig_flat);
    if (compact_verifier_released_for_index) {
        compact_lrq.load(compact_lrq_file);
        compact_lrq_ready = true;
        printf("[rknn] reloaded compact verifier after graph index construction: "
               "%.3fGB\n", compact_lrq.persistent_bytes() / 1e9);
    }
    if (base_released_for_index) {
        size_t reload_n = 0, reload_d = 0;
        base = load_base_vectors(PFX + "_base.bin", reload_n, reload_d);
        if (reload_n != n || reload_d != d) {
            fprintf(stderr,
                    "[rknn] reloaded base shape mismatch: got n=%zu d=%zu "
                    "expected n=%zu d=%zu\n",
                    reload_n, reload_d, n, d);
            exit(3);
        }
        printf("[rknn] reloaded original base after graph index construction: %.3fGB\n",
               (double)n * d * sizeof(float) / 1e9);
    }
    index.visualize_parameters();
    double build_s = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-tb0).count()/1000.0;
    // Legacy materialized verifier counts coefficients separately. The compact
    // postfilter artifact already includes coefficients, packed codes, basis, and codebook.
    size_t conceptual_coeff_bytes = (!postfilter_compact && graph_anyk_lift && is_rankm_residual_mode(radius_mode))
        ? (size_t)n * (size_t)RANKM * sizeof(float) : 0;
    size_t conceptual_verification_bytes = conceptual_radius_bytes + conceptual_coeff_bytes;
    const size_t conceptual_envelope_bytes = graph_original ? n * sizeof(float) : 0;
    double idx_gb = ((double)n*par.M*4 + (double)n*(double)par.dim*4 +
                     (double)conceptual_verification_bytes +
                     (double)conceptual_envelope_bytes)/1e9;
    printf("[rknn] BUILD_TIME=%.1fs INDEX_SIZE=%.2fGB "
           "(graph M=%zu + graph dim=%zu + verification=%.3fGB + envelope=%.3fGB)\n",
           build_s, idx_gb, par.M, par.dim,
           conceptual_verification_bytes / 1e9, conceptual_envelope_bytes / 1e9);
    if (env_flag("ANQI_BUILD_BREAKDOWN")) {
        printf("[rknn] BUILD_BREAKDOWN mode=%s original_knn_s=%.6f "
               "lift_verifier_prepare_s=%.6f graph_index_s=%.6f total_s=%.6f "
               "verifier_encode_save_ms=%.3f verifier_load_ms=%.3f\n",
               can_load ? "load" : "build", original_knn_s,
               lift_verifier_prepare_s, graph_index_s, build_s,
               verifier_encode_save_ms, verifier_load_ms);
    }

    // L2 adaptive flooding uses radj[o] = scale^2 * (r_Kq^2 - r_graph^2).
    // IP uses the global flood because its lifted factor differs.
    std::vector<float> radj; const float* radjp = nullptr;
    float beta = std::getenv("ANQI_BETA") ? (float)atof(std::getenv("ANQI_BETA"))
                                           : (graph_original ? 0.0f : 0.5f);
    if (graph_original && beta != 0.0f) {
        fprintf(stderr,
                "[rknn] original-geometry ablation requires ANQI_BETA=0 so its external "
                "r100 envelope exactly matches the lifted arm; got %.6g\n", beta);
        exit(2);
    }
    // Optional exact recheck collects the flood envelope and then evaluates
    // each candidate against its original-space rank-k threshold.
    int   RECHECK = env_flag("ANQI_RECHECK") ? 1 : 0;
    float RCEPS   = std::getenv("ANQI_RCEPS") ? (float)atof(std::getenv("ANQI_RCEPS")) : 0.0f;
    if (graph_original && RCEPS != 0.0f) {
        fprintf(stderr,
                "[rknn] original-geometry ablation requires ANQI_RCEPS=0 so its external "
                "r100 envelope exactly matches the lifted arm; got %.6g\n", RCEPS);
        exit(2);
    }
    if (postfilter_verifier && RECHECK) {
        fprintf(stderr,
                "[rknn] ANQI_RECHECK cannot be mixed with an independent postfilter verifier; "
                "select ANQI_RADIUS_MODE=exact_f32 for the exact component comparison\n");
        exit(2);
    }
    bool  collect_eps = (RECHECK != 0) && !postfilter_verifier;
    double rankm_slack = std::getenv("ANQI_RANKM_SLACK") ? atof(std::getenv("ANQI_RANKM_SLACK")) : 0.0;
    double slack_factor = 1.0 + rankm_slack;
    if (slack_factor <= 0.0) {
        fprintf(stderr, "[rknn] ANQI_RANKM_SLACK=%.6g makes non-positive radius scale\n", rankm_slack);
        exit(2);
    }
    if (postfilter_exact && rankm_slack != 0.0) {
        fprintf(stderr, "[rknn] exact_f32 postfilter does not accept ANQI_RANKM_SLACK\n");
        exit(2);
    }
    if (postfilter_compact && is_ip && rankm_slack != 0.0) {
        fprintf(stderr,
                "[rknn] IP compact postfilter currently requires ANQI_RANKM_SLACK=0; "
                "multiplicative scaling is not monotone for signed -ip thresholds\n");
        exit(2);
    }
    if (postfilter_verifier && std::getenv("ANQI_RADIUS_RELAX") &&
        atof(std::getenv("ANQI_RADIUS_RELAX")) != 0.0) {
        fprintf(stderr, "[rknn] postfilter verifier uses ANQI_RANKM_SLACK; ANQI_RADIUS_RELAX is not accepted\n");
        exit(2);
    }
    double verifier_materialize_ms = 0.0;
    if (postfilter_analytic) {
        const auto materialize_t0 = std::chrono::steady_clock::now();
        analytic_rankm.materialize_k(Kq, slack_factor, verification_thresholds);
        verifier_materialize_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - materialize_t0).count() / 1000.0;
    } else if (postfilter_compact) {
        ensure_compact_lrq(false);
        const auto materialize_t0 = std::chrono::steady_clock::now();
        compact_lrq.materialize_k(Kq, slack_factor, is_ip, verification_thresholds);
        verifier_materialize_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - materialize_t0).count() / 1000.0;
    } else if (postfilter_exact) {
        const auto materialize_t0 = std::chrono::steady_clock::now();
        load_exact_radius_column(PFX, n, Kq, is_ip, verification_thresholds);
        verifier_materialize_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - materialize_t0).count() / 1000.0;
    }
    if (postfilter_verifier) {
        printf("[rknn] independent verifier ready: mode=%s k=%zu slack=%.6g persistent=%.3fGB "
               "temporary_thresholds=%.3fGB materialize_ms=%.3f bits=%d\n",
               postfilter_lrq ? "rank-M+LRQ" :
               (postfilter_resid_f32 ? "rank-M+float32-residual" :
               (postfilter_rankm ? "rank-M-only" :
                (postfilter_analytic ? "analytic-rank-M" : "exact_f32"))),
               Kq, rankm_slack,
               conceptual_verification_bytes / 1e9,
               verification_thresholds.size() * sizeof(float) / 1e9,
               verifier_materialize_ms, postfilter_compact ? compact_lrq.bits : 0);
    }
    const bool anyk_rankm_verification = !postfilter_verifier && graph_anyk_lift && !is_ip &&
                                         is_rankm_residual_mode(radius_mode);
    bool fixed_k_lift = (graph_anyk_lift && std::getenv("ANQI_SPECIFIC"));
    if (graph_original) {
        if (graph_horizon_radius.size() != n) {
            fprintf(stderr, "[rknn] original geometry missing exact-r%zu envelope: got %zu expected %zu\n",
                    GRAPH_NK, graph_horizon_radius.size(), n);
            exit(3);
        }
        radjp = graph_horizon_radius.data();
    } else if (candidate_uses_verifier) {
        if (graph_horizon_radius.size() != n || verification_thresholds.size() != n) {
            fprintf(stderr,
                    "[rknn] verifier candidate envelope is missing per-object radii: "
                    "graph=%zu verifier=%zu expected=%zu\n",
                    graph_horizon_radius.size(), verification_thresholds.size(), n);
            exit(3);
        }
        radj.resize(n);
        const double sc2 = scale * scale;
        #pragma omp parallel for schedule(static)
        for (size_t o = 0; o < n; o++) {
            radj[o] = (float)(sc2 * ((double)verification_thresholds[o] -
                                     (double)graph_horizon_radius[o]));
        }
        radjp = radj.data();
        double min_radj = std::numeric_limits<double>::infinity();
        double max_radj = -std::numeric_limits<double>::infinity();
        double sum_radj = 0.0;
        size_t positive_radj = 0;
        for (float value : radj) {
            min_radj = std::min(min_radj, (double)value);
            max_radj = std::max(max_radj, (double)value);
            sum_radj += (double)value;
            positive_radj += value > 0.0f;
        }
        printf("[rknn] verifier envelope radj: k=%zu min=%.9g mean=%.9g max=%.9g "
               "positive=%zu/%zu\n",
               Kq, min_radj, sum_radj / std::max<size_t>(1, n), max_radj,
               positive_radj, n);
    } else if (!postfilter_verifier && !is_ip && graph_anyk_lift && !fixed_k_lift) { radj.resize(n); double sc2=scale*scale;
        const double target_scale = anyk_rankm_verification ? slack_factor : 1.0;
        for (size_t o=0;o<n;o++)
            radj[o]=(float)(sc2*(target_scale*(double)rk2a_all[o*NK+(Kq-1)]-
                                      (double)rk2a_all[o*NK+(NK-1)]));
        radjp=radj.data(); }
    if (graph_original) {
        printf("[rknn] candidate generator: original L2 graph, external exact-r%zu envelope, "
               "center graph search, beta=0; verifier is post-filter only\n", GRAPH_NK);
    } else {
        printf("[rknn] k-adaptive flood: Kq=%zu radj=%s beta=%.2f%s%s\n",
               Kq, radjp?"on(L2)":"off", beta, fixed_k_lift?" (fixed-k lift)":"",
               anyk_rankm_verification?" (rank-M+RQ verification)":"");
    }
    if (postfilter_verifier && graph_anyk_lift) {
        if (candidate_uses_verifier) {
            printf("[rknn] candidate generator: modeled per-object target-k envelope on "
                   "exact-r%zu Lift graph, center graph search; verifier=%s\n",
                   GRAPH_NK, postfilter_analytic ? "analytic-rank-M" :
                   (postfilter_lrq ? "rank-M+LRQ" :
                    (postfilter_rankm ? "rank-M-only" : "exact_f32")));
        } else {
            printf("[rknn] candidate generator: exact-r%zu envelope, center graph search; "
                   "verifier is post-filter only\n", GRAPH_NK);
        }
    }
    {
        std::string ss = env_or("ANQI_SEARCH_SCORE", "center");
        std::string sl = env_or("ANQI_SEARCH_SCORE_LAMBDA", "1.0");
        printf("[rknn] search score: mode=%s lambda=%s\n", ss.c_str(), sl.c_str());
    }
    auto item = [&](size_t i){ return base + i*d; };
    // rank-M 图才扩展 query 维度；Any-K Lift 始终使用共享的 d+2 query。
    uint32_t LD = graph_original
        ? (uint32_t)d
        : (uint32_t)(d + (GRAPH_RANKM>0?GRAPH_RANKM:0) + 2);
    std::vector<float> g_q(GRAPH_RANKM>0?GRAPH_RANKM:0); double R2_rankm=0;
    if (GRAPH_RANKM>0) { double sc2q=(double)scale*scale, sg=0;
        for(int m=0;m<GRAPH_RANKM;m++){ g_q[m]=(float)(slack_factor*(double)Gba[(size_t)m*NK+(Kq-1)]); sg+=(double)g_q[m]*g_q[m]; }
        // L2: R²(k)=M*²+1+s²μ(k)+Σg_i² (与 q 无关);
        // IP: R²(k)=qn2s+Σg_i²+M*²+2s²μ'(k) (μ'=mean rk2a=−ip_k);qn2s 每 query 加,这里存 q-无关基。
        double mu_eff = slack_factor * (double)muba[Kq-1];
        R2_rankm = is_ip ? (Mstar2 + sg + 2.0*sc2q*mu_eff)
                         : (Mstar2 + 1.0 + sc2q*mu_eff + sg);
        radjp = nullptr;   // rank-M:membership 折进几何,不用 radj
        printf("[rknn] rank-M query: metric=%s Kq=%zu slack=%.6g factor=%.6g R²(k)%s=%.6g (无 radj; recheck=%d)\n",
               metric.c_str(), Kq, rankm_slack, slack_factor, is_ip?"_base(+qn2s)":"", R2_rankm, RECHECK);
    }
    std::vector<float> exact_recheck_radii;
    if (RECHECK && anyk_rankm_verification) {
        exact_recheck_radii.resize((size_t)n * NK);
        load_exact_radii_table(PFX, n, NK, is_ip, exact_recheck_radii);
        printf("[rknn] exact recheck is independent of rank-M+RQ; loaded exact radius table\n");
    }
    const std::vector<float>& recheck_radii = exact_recheck_radii.empty() ? rk2a_all : exact_recheck_radii;
    // recheck key（原始 d 维）：L2 → d²；IP → −⟨q,o⟩。membership 统一: key ≤ rk2[o]。
    auto recheck_key = [&](size_t id, float* q){
        float ip = DistCal::InnerProductSIMD16ExtAVX512_(item(id), q, (uint32_t)d);
        if (is_ip) return -ip;
        float on=DistCal::InnerProductSIMD16ExtAVX512_(item(id),item(id),(uint32_t)d);
        float qn=DistCal::InnerProductSIMD16ExtAVX512_(q,q,(uint32_t)d);
        return on + qn - 2.0f*ip;
    };

    // Keep enough chunks to occupy all query threads without inflating the
    // scheduler overhead on larger batches.
    const size_t query_threads =
        std::max<size_t>(1, static_cast<size_t>(omp_get_max_threads()));
    const size_t target_chunks = 4 * query_threads;
    const size_t adaptive_chunk = std::max<size_t>(1, nq / target_chunks);
    const int query_chunk = static_cast<int>(
        std::max<size_t>(1, std::min<size_t>(16, adaptive_chunk)));
    const std::string query_visited_mode =
        env_or("ANQI_QUERY_VISITED", "stamps");
    const bool compact_query_visited = query_visited_mode == "bitset";
    if (!compact_query_visited && query_visited_mode != "stamps") {
        fprintf(stderr,
                "[rknn] invalid ANQI_QUERY_VISITED=%s "
                "(expected stamps or bitset)\n",
                query_visited_mode.c_str());
        return 2;
    }
    printf("[rknn] query visited mode=%s aggregate_store=%.3fGB\n",
           query_visited_mode.c_str(),
           compact_query_visited
               ? (double)query_threads * ((n + 63) / 64) * sizeof(uint64_t) / 1e9
               : (double)query_threads * n * sizeof(uint16_t) / 1e9);

    // Warm OpenMP workers and page in the index before timed queries.
    {
        size_t Lw = Lds.back();
        #pragma omp parallel
        {
            std::vector<uint16_t> mk(compact_query_visited ? 0 : n, 0);
            std::vector<uint64_t> bits(
                compact_query_visited ? (n + 63) / 64 : 0, 0);
            std::vector<uint32_t> touched;
            uint16_t vr=0; std::vector<float> qb(LD);
            #pragma omp for schedule(dynamic,query_chunk)
            for (size_t i=0;i<nq;i++){
                float* q=query+i*d; double qn2s=0;
                double R2 = 0.0;
                if (graph_original) {
                    std::copy(q, q + d, qb.begin());
                } else {
                    for(size_t j=0;j<d;j++){double v=(double)q[j]*scale; qb[j]=(float)v; qn2s+=v*v;}
                    qb[d]=(GRAPH_RANKM>0 && is_ip)?0.0f:1.0f;   // IP rank-M:‖o‖² 槽不用 → 0
                    if(GRAPH_RANKM>0){ for(int m=0;m<GRAPH_RANKM;m++) qb[d+1+m]=g_q[m]; qb[LD-1]=0.0f; }
                    else qb[d+1]=0.0f;
                    R2 = GRAPH_RANKM>0 ? (is_ip?(qn2s+R2_rankm):R2_rankm)
                                       : (is_ip?(qn2s+1.0+Mstar2):(Mstar2+1.0));
                }
                std::vector<uint32_t> cd; size_t nd;
                nndindex::QueryVisited visited;
                if (compact_query_visited) {
                    for (uint32_t id : touched)
                        bits[id >> 6] &= ~(1ull << (id & 63u));
                    touched.clear();
                    visited.bits = bits.data();
                    visited.touched = &touched;
                } else {
                    vr++;
                    if(vr==0){std::fill(mk.begin(),mk.end(),0);vr=1;}
                    visited.stamps = mk.data();
                    visited.version = vr;
                }
                index.range_search(qb.data(),R2,RCEPS,Lw,cd,nd,visited,
                                   radjp,beta,collect_eps);
            }
        }
    }

    size_t bench_repeats = 1;
    if (const char* repeats_env = std::getenv("ANQI_BENCH_REPEATS")) {
        bool digits_only = *repeats_env != '\0';
        for (const char* p = repeats_env; *p; p++)
            digits_only = digits_only && std::isdigit(static_cast<unsigned char>(*p));
        char* end = nullptr;
        errno = 0;
        const unsigned long long parsed = std::strtoull(repeats_env, &end, 10);
        const unsigned long long max_safe =
            std::numeric_limits<size_t>::max() / std::max<size_t>(1, nq);
        if (!digits_only || errno == ERANGE || end == repeats_env || *end != '\0' ||
            parsed == 0 || parsed > max_safe) {
            fprintf(stderr, "[rknn] invalid ANQI_BENCH_REPEATS=%s\n", repeats_env);
            return 2;
        }
        bench_repeats = static_cast<size_t>(parsed);
    }
    const size_t max_safe_batches =
        std::numeric_limits<size_t>::max() / std::max<size_t>(1, nq);
    const bool calibrated_timing = env_flag("ANQI_BENCH_CALIBRATED_TIMING");
    auto positive_seconds_env = [](const char* name, double fallback) {
        const char* value = std::getenv(name);
        if (!value || !*value) return fallback;
        char* end = nullptr;
        errno = 0;
        const double parsed = std::strtod(value, &end);
        if (errno == ERANGE || end == value || *end != '\0' ||
            !std::isfinite(parsed) || parsed <= 0.0) {
            fprintf(stderr, "[rknn] invalid %s=%s\n", name, value);
            std::exit(2);
        }
        return parsed;
    };
    const double bench_min_total_s = positive_seconds_env(
        "ANQI_BENCH_MIN_TOTAL_S", 30.0);
    const double bench_min_repeat_s = positive_seconds_env(
        "ANQI_BENCH_MIN_REPEAT_S", 10.0);
    const bool time_breakdown = env_flag("ANQI_TIME_BREAKDOWN");
    printf("[rknn] benchmark repeats=%zu calibrated=%s "
           "(timing uses steady_clock sub-millisecond precision)\n",
           bench_repeats, calibrated_timing ? "on" : "off");
    printf("[rknn] query parallelism: nq=%zu threads=%d chunk=%d\n",
           nq, omp_get_max_threads(), query_chunk);
    if (calibrated_timing) {
        printf("[rknn] calibrated timing: min_total_s=%.3f min_repeat_s=%.3f "
               "windows=%zu\n",
               bench_min_total_s, bench_min_repeat_s, bench_repeats);
    } else printf("[rknn] flattened_repeats=1 tasks=%zu\n", nq * bench_repeats);
    if (time_breakdown)
        printf("[rknn] query stage timing enabled (ANQI_TIME_BREAKDOWN=1)\n");

    struct PointStats {
        double sum_recall = 0, sum_prec = 0, sum_f1 = 0;
        double sum_candidate_recall = 0, sum_cand = 0, sum_pred = 0;
        double sum_ndist = 0, sum_graph_ndist = 0, sum_verify_ndist = 0;
        double sum_transform_s = 0, sum_graph_search_s = 0;
        double sum_verifier_s = 0, sum_eval_s = 0;
        size_t nonempty = 0, tasks = 0, batches = 0;
        double secs = 0;
    };

    auto run_point = [&](size_t L_descent, size_t batches) {
        if (batches == 0 || batches > max_safe_batches) {
            fprintf(stderr, "[rknn] unsafe calibrated batch count=%zu\n", batches);
            std::exit(2);
        }
        const size_t benchmark_tasks = nq * batches;
        PointStats stats;
        double sum_recall=0, sum_prec=0, sum_f1=0, sum_candidate_recall=0;
        double sum_cand=0, sum_pred=0, sum_ndist=0;
        double sum_graph_ndist=0, sum_verify_ndist=0;
        double sum_transform_s=0, sum_graph_search_s=0, sum_verifier_s=0, sum_eval_s=0;
        size_t nonempty=0;
        auto t0 = std::chrono::steady_clock::now();
        #pragma omp parallel reduction(+:sum_recall,sum_prec,sum_f1,sum_candidate_recall,sum_cand,sum_pred,sum_ndist,sum_graph_ndist,sum_verify_ndist,sum_transform_s,sum_graph_search_s,sum_verifier_s,sum_eval_s,nonempty)
        {
            std::vector<uint16_t> mark_t(compact_query_visited ? 0 : n, 0);
            std::vector<uint64_t> bits_t(
                compact_query_visited ? (n + 63) / 64 : 0, 0);
            std::vector<uint32_t> touched_t;
            uint16_t ver_t = 0;
            std::vector<float> qbar_t(LD);
            #pragma omp for schedule(dynamic, query_chunk)
            for (size_t task = 0; task < benchmark_tasks; task++) {
                const size_t i = task % nq;
                float* q = query + i*d;
                auto stage_t0 = std::chrono::steady_clock::time_point{};
                if (time_breakdown) stage_t0 = std::chrono::steady_clock::now();
                double qn2s = 0;
                double R2 = 0.0;
                if (graph_original) {
                    std::copy(q, q + d, qbar_t.begin());
                } else {
                    for (size_t j=0;j<d;j++){ double v=(double)q[j]*scale; qbar_t[j]=(float)v; qn2s+=v*v; }
                    qbar_t[d]=(GRAPH_RANKM>0 && is_ip)?0.0f:1.0f;   // IP rank-M:‖o‖² 槽不用 → 0
                    if(GRAPH_RANKM>0){ for(int m=0;m<GRAPH_RANKM;m++) qbar_t[d+1+m]=g_q[m]; qbar_t[LD-1]=0.0f; }
                    else qbar_t[d+1]=0.0f;
                    R2 = GRAPH_RANKM>0 ? (is_ip ? (qn2s + R2_rankm) : R2_rankm)
                                       : (is_ip ? (qn2s + 1.0 + Mstar2) : (Mstar2 + 1.0));
                }
                if (time_breakdown) {
                    const auto stage_t1 = std::chrono::steady_clock::now();
                    sum_transform_s += std::chrono::duration<double>(stage_t1 - stage_t0).count();
                    stage_t0 = stage_t1;
                }
                std::vector<uint32_t> cand; size_t ndist;
                nndindex::QueryVisited visited;
                if (compact_query_visited) {
                    for (uint32_t id : touched_t)
                        bits_t[id >> 6] &= ~(1ull << (id & 63u));
                    touched_t.clear();
                    visited.bits = bits_t.data();
                    visited.touched = &touched_t;
                } else {
                    ver_t++;
                    if (ver_t==0){
                        std::fill(mark_t.begin(),mark_t.end(),0);
                        ver_t=1;
                    }
                    visited.stamps = mark_t.data();
                    visited.version = ver_t;
                }
                index.range_search(qbar_t.data(), R2, RCEPS, L_descent, cand,
                                   ndist, visited, radjp, beta, collect_eps);
                const size_t graph_ndist = ndist;
                if (time_breakdown) {
                    const auto stage_t1 = std::chrono::steady_clock::now();
                    sum_graph_search_s += std::chrono::duration<double>(stage_t1 - stage_t0).count();
                    stage_t0 = stage_t1;
                }
                std::unordered_set<uint32_t> pred;
                if (postfilter_verifier) {
                    for (uint32_t id : cand)
                        if (recheck_key(id, q) <= (double)verification_thresholds[id]) pred.insert(id);
                    ndist += cand.size();
                } else if(GRAPH_RANKM>0){
                    if(RECHECK){ for(uint32_t id:cand) if(recheck_key(id,q) <= recheck_radii[(size_t)id*NK+(Kq-1)]) pred.insert(id); ndist += cand.size(); }
                    else for(uint32_t id:cand) pred.insert(id);
                }
                else if (anyk_rankm_verification && !RECHECK) {
                    for (uint32_t id : cand) pred.insert(id);  // rank-M+RQ predicate already folded into Any-K Lift threshold.
                } else {
                    const double threshold_scale = anyk_rankm_verification ? slack_factor : 1.0;
                    for (uint32_t id : cand)
                        if (recheck_key(id, q) <= threshold_scale * (double)recheck_radii[(size_t)id*NK + (Kq-1)])
                            pred.insert(id);
                    ndist += cand.size();
                }
                if (time_breakdown) {
                    const auto stage_t1 = std::chrono::steady_clock::now();
                    sum_verifier_s += std::chrono::duration<double>(stage_t1 - stage_t0).count();
                    stage_t0 = stage_t1;
                }
                std::unordered_set<uint32_t> truth(gids.begin()+goff[i], gids.begin()+goff[i+1]);
                if (truth.empty()) {
                    if (time_breakdown)
                        sum_eval_s += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - stage_t0).count();
                    continue;
                }
                nonempty++;
                size_t candidate_hit=0; for (uint32_t id : cand) if (truth.count(id)) candidate_hit++;
                size_t hit=0; for (uint32_t id : pred) if (truth.count(id)) hit++;
                double rec = (double)hit/truth.size();
                double prec = pred.empty()?1.0:(double)hit/pred.size();
                sum_recall+=rec; sum_prec+=prec; sum_f1+=(rec+prec>0?2*rec*prec/(rec+prec):0);
                sum_candidate_recall+=(double)candidate_hit/truth.size();
                sum_cand+=cand.size(); sum_pred+=pred.size(); sum_ndist+=ndist;
                sum_graph_ndist+=graph_ndist; sum_verify_ndist+=ndist-graph_ndist;
                if (time_breakdown)
                    sum_eval_s += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - stage_t0).count();
            }
        }
        stats.secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now()-t0).count();
        stats.sum_recall = sum_recall;
        stats.sum_prec = sum_prec;
        stats.sum_f1 = sum_f1;
        stats.sum_candidate_recall = sum_candidate_recall;
        stats.sum_cand = sum_cand;
        stats.sum_pred = sum_pred;
        stats.sum_ndist = sum_ndist;
        stats.sum_graph_ndist = sum_graph_ndist;
        stats.sum_verify_ndist = sum_verify_ndist;
        stats.sum_transform_s = sum_transform_s;
        stats.sum_graph_search_s = sum_graph_search_s;
        stats.sum_verifier_s = sum_verifier_s;
        stats.sum_eval_s = sum_eval_s;
        stats.nonempty = nonempty;
        stats.tasks = benchmark_tasks;
        stats.batches = batches;
        return stats;
    };

    auto add_stats = [](PointStats& dst, const PointStats& src) {
        dst.sum_recall += src.sum_recall;
        dst.sum_prec += src.sum_prec;
        dst.sum_f1 += src.sum_f1;
        dst.sum_candidate_recall += src.sum_candidate_recall;
        dst.sum_cand += src.sum_cand;
        dst.sum_pred += src.sum_pred;
        dst.sum_ndist += src.sum_ndist;
        dst.sum_graph_ndist += src.sum_graph_ndist;
        dst.sum_verify_ndist += src.sum_verify_ndist;
        dst.sum_transform_s += src.sum_transform_s;
        dst.sum_graph_search_s += src.sum_graph_search_s;
        dst.sum_verifier_s += src.sum_verifier_s;
        dst.sum_eval_s += src.sum_eval_s;
        dst.nonempty += src.nonempty;
        dst.tasks += src.tasks;
        dst.batches += src.batches;
        dst.secs += src.secs;
    };

    auto print_point = [&](size_t L_descent, const PointStats& stats, double qps) {
        if (stats.nonempty == 0 || stats.tasks == 0 || stats.secs <= 0.0) {
            fprintf(stderr, "[rknn] empty benchmark statistics for L_descent=%zu\n",
                    L_descent);
            std::exit(3);
        }
        const double denom = (double)stats.nonempty;
        printf("[rknn] L_descent=%-5zu | recall=%.4f precision=%.4f F1=%.4f | avg cand=%.0f pred=%.0f #dist=%.0f | QPS=%.0f (parallel)\n",
               L_descent, stats.sum_recall/denom, stats.sum_prec/denom,
               stats.sum_f1/denom, stats.sum_cand/denom, stats.sum_pred/denom,
               stats.sum_ndist/denom, qps);
        printf("[rknn] DIST_BREAKDOWN L_descent=%zu #graph_dist=%.0f #verify_dist=%.0f #total_dist=%.0f\n",
               L_descent, stats.sum_graph_ndist/denom,
               stats.sum_verify_ndist/denom, stats.sum_ndist/denom);
        printf("[rknn] CANDIDATE_RECALL L_descent=%zu recall=%.6f\n",
               L_descent, stats.sum_candidate_recall/denom);
        if (time_breakdown) {
            const double task_scale_us = 1e6 / (double)stats.tasks;
            const double stage_sum_s = stats.sum_transform_s +
                stats.sum_graph_search_s + stats.sum_verifier_s + stats.sum_eval_s;
            printf("[rknn] TIME_BREAKDOWN L_descent=%zu transform_us=%.3f graph_search_us=%.3f "
                   "verification_us=%.3f benchmark_eval_us=%.3f stage_cpu_us=%.3f "
                   "elapsed_us=%.3f tasks=%zu\n",
                   L_descent, stats.sum_transform_s * task_scale_us,
                   stats.sum_graph_search_s * task_scale_us,
                   stats.sum_verifier_s * task_scale_us,
                   stats.sum_eval_s * task_scale_us, stage_sum_s * task_scale_us,
                   stats.secs * task_scale_us, stats.tasks);
        }
    };

    for (size_t L_descent : Lds) {
        if (!calibrated_timing) {
            const PointStats stats = run_point(L_descent, bench_repeats);
            print_point(
                L_descent, stats, (double)stats.tasks / stats.secs);
            continue;
        }

        const PointStats warm = run_point(L_descent, 1);
        const PointStats calibration = run_point(L_descent, 1);
        const double target_repeat_s = std::max(
            bench_min_repeat_s, bench_min_total_s / (double)bench_repeats);
        size_t planned_batches = std::max<size_t>(
            1, (size_t)std::ceil(
                   target_repeat_s / std::max(calibration.secs, 1e-9)));
        if (planned_batches > max_safe_batches) {
            fprintf(stderr,
                    "[rknn] calibrated batch plan overflow for L_descent=%zu: %zu\n",
                    L_descent, planned_batches);
            return 2;
        }

        PointStats aggregate;
        std::vector<double> qps_samples;
        qps_samples.reserve(bench_repeats);
        double min_repeat_observed_s = std::numeric_limits<double>::infinity();
        for (size_t repeat = 0; repeat < bench_repeats; repeat++) {
            size_t actual_batches = planned_batches;
            PointStats measured;
            for (int attempt = 0; attempt < 4; attempt++) {
                measured = run_point(L_descent, actual_batches);
                if (measured.secs + 1e-9 >= target_repeat_s) break;
                const double scaled =
                    (double)actual_batches * target_repeat_s /
                    std::max(measured.secs, 1e-9) * 1.05;
                const size_t next_batches = std::max(
                    actual_batches + 1, (size_t)std::ceil(scaled));
                printf("[rknn] HOT_RETRY L_descent=%zu repeat=%zu "
                       "seconds=%.6f target_s=%.6f batches=%zu next_batches=%zu\n",
                       L_descent, repeat + 1, measured.secs, target_repeat_s,
                       actual_batches, next_batches);
                actual_batches = next_batches;
            }
            if (measured.secs + 1e-9 < target_repeat_s) {
                fprintf(stderr,
                        "[rknn] calibrated repeat too short for L_descent=%zu "
                        "repeat=%zu seconds=%.6f target_s=%.6f\n",
                        L_descent, repeat + 1, measured.secs, target_repeat_s);
                return 3;
            }
            const double repeat_qps = (double)measured.tasks / measured.secs;
            qps_samples.push_back(repeat_qps);
            min_repeat_observed_s =
                std::min(min_repeat_observed_s, measured.secs);
            add_stats(aggregate, measured);
            printf("[rknn] HOT_REPEAT L_descent=%zu repeat=%zu/%zu "
                   "batches=%zu tasks=%zu seconds=%.6f QPS=%.3f\n",
                   L_descent, repeat + 1, bench_repeats, measured.batches,
                   measured.tasks, measured.secs, repeat_qps);
        }
        std::vector<double> sorted_qps = qps_samples;
        std::sort(sorted_qps.begin(), sorted_qps.end());
        const size_t mid = sorted_qps.size() / 2;
        const double median_qps = sorted_qps.size() % 2
            ? sorted_qps[mid]
            : 0.5 * (sorted_qps[mid - 1] + sorted_qps[mid]);
        print_point(L_descent, aggregate, median_qps);
        printf("[rknn] HOT_TIMING L_descent=%zu warm_s=%.6f "
               "calibration_s=%.6f measured_s=%.6f repeats=%zu "
               "planned_batches_per_repeat=%zu actual_batches_total=%zu "
               "min_repeat_s=%.6f qps_median=%.3f qps_samples=",
               L_descent, warm.secs, calibration.secs, aggregate.secs,
               bench_repeats, planned_batches, aggregate.batches,
               min_repeat_observed_s, median_qps);
        for (size_t i = 0; i < qps_samples.size(); i++)
            printf("%s%.3f", i ? "," : "", qps_samples[i]);
        printf("\n");
    }
    return 0;
}
