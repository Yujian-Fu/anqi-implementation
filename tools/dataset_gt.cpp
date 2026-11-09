// Exact base-to-base top-K and reverse-kNN ground-truth generator.
#include "../include/distance.h"
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include <string>
#include <iostream>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <omp.h>
#include <immintrin.h>

static float* load_fbin(const std::string& p, size_t& n, size_t& d) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "[dgt] cannot open " << p << "\n";
        std::exit(3);
    }
    const std::streamoff bytes = f.tellg();
    f.seekg(0);
    uint32_t nn = 0, dd = 0;
    if (bytes < 8 || !f.read(reinterpret_cast<char*>(&nn), 4) ||
        !f.read(reinterpret_cast<char*>(&dd), 4) || nn == 0 || dd == 0) {
        std::cerr << "[dgt] invalid fbin header: " << p << "\n";
        std::exit(3);
    }
    const uint64_t values = static_cast<uint64_t>(nn) * dd;
    const uint64_t expected = 8 + values * sizeof(float);
    if (expected != static_cast<uint64_t>(bytes) ||
        values > std::numeric_limits<size_t>::max() / sizeof(float)) {
        std::cerr << "[dgt] invalid fbin size: " << p << "\n";
        std::exit(3);
    }
    n = nn;
    d = dd;
    float* x = static_cast<float*>(_mm_malloc(values * sizeof(float), 64));
    if (!x || !f.read(reinterpret_cast<char*>(x), values * sizeof(float))) {
        std::cerr << "[dgt] cannot read fbin values: " << p << "\n";
        _mm_free(x);
        std::exit(3);
    }
    return x;
}

static size_t parse_positive_size(const char* text, const char* name) {
    if (!text || !*text) {
        std::cerr << "[dgt] missing value for " << name << "\n";
        std::exit(2);
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value == 0 ||
        value > std::numeric_limits<size_t>::max()) {
        std::cerr << "[dgt] invalid " << name << "=" << text << "\n";
        std::exit(2);
    }
    return static_cast<size_t>(value);
}

static void require_write(const std::ofstream& stream, const std::string& path) {
    if (!stream) {
        std::cerr << "[dgt] failed to write " << path << "\n";
        std::exit(3);
    }
}

static void load_threshold_column(
        const std::string& path,
        size_t expected_n,
        size_t rank_one_based,
        std::vector<float>& thresholds) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const std::streamoff bytes = input
        ? static_cast<std::streamoff>(input.tellg())
        : static_cast<std::streamoff>(-1);
    input.seekg(0);
    uint32_t n = 0, k = 0;
    if (!input.read(reinterpret_cast<char*>(&n), 4) ||
        !input.read(reinterpret_cast<char*>(&k), 4) ||
        n != expected_n || rank_one_based == 0 || rank_one_based > k) {
        std::cerr << "[dgt] invalid threshold table: " << path << "\n";
        std::exit(3);
    }
    const uint64_t cells = static_cast<uint64_t>(n) * k;
    const uint64_t expected = 8 + 2 * cells * sizeof(uint32_t);
    if (expected != static_cast<uint64_t>(bytes) ||
        cells > std::numeric_limits<size_t>::max() / sizeof(float)) {
        std::cerr << "[dgt] invalid threshold-table size: " << path << "\n";
        std::exit(3);
    }
    input.seekg(static_cast<std::streamoff>(8 + cells * sizeof(uint32_t)));
    thresholds.resize(n);
    constexpr size_t target_bytes = 64ull << 20;
    const size_t rows_per_chunk = std::max<size_t>(1, target_bytes / (k * sizeof(float)));
    std::vector<float> block(rows_per_chunk * k);
    for (size_t first = 0; first < n; first += rows_per_chunk) {
        const size_t rows = std::min(rows_per_chunk, static_cast<size_t>(n) - first);
        const size_t block_values = rows * k;
        if (!input.read(reinterpret_cast<char*>(block.data()),
                        static_cast<std::streamsize>(block_values * sizeof(float)))) {
            std::cerr << "[dgt] truncated threshold table: " << path << "\n";
            std::exit(3);
        }
        for (size_t row = 0; row < rows; row++) {
            thresholds[first + row] = block[row * k + rank_one_based - 1];
        }
    }
}

static void write_timing_metadata(
        const std::string& prefix,
        const std::string& metric,
        size_t n,
        size_t d,
        size_t k,
        int threads,
        double seconds) {
    const std::string path = prefix + "_threshold_build_time.meta";
    std::ofstream output(path);
    output << "ANQI_THRESHOLD_BUILD_V1\n"
           << "seconds " << std::setprecision(17) << seconds << "\n"
           << "metric " << metric << "\n"
           << "n " << n << "\n"
           << "d " << d << "\n"
           << "k " << k << "\n"
           << "threads " << threads << "\n";
    require_write(output, path);
    std::cout << "[dgt] wrote " << path << "\n";
}

int main(int argc, char** argv) {
    std::string base_p, query_p, prefix, metric="l2", mode="both", threshold_table;
    size_t K=100, rk_k=10, TILE=512, nthreads=0;
    auto next_value = [&](int& i, const std::string& option) -> const char* {
        if (i + 1 >= argc) {
            std::cerr << "[dgt] missing value for " << option << "\n";
            std::exit(2);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--base") base_p = next_value(i, a);
        else if (a == "--query") query_p = next_value(i, a);
        else if (a == "--K") K = parse_positive_size(next_value(i, a), "K");
        else if (a == "--rk_k") rk_k = parse_positive_size(next_value(i, a), "rk_k");
        else if (a == "--out_prefix") prefix = next_value(i, a);
        else if (a == "--tile") TILE = parse_positive_size(next_value(i, a), "tile");
        else if (a == "--metric") metric = next_value(i, a);
        else if (a == "--threads") nthreads = parse_positive_size(next_value(i, a), "threads");
        else if (a == "--mode") mode = next_value(i, a);
        else if (a == "--threshold-table") threshold_table = next_value(i, a);
        else {
            std::cerr << "[dgt] unknown option: " << a << "\n";
            return 2;
        }
    }
    if (mode != "both" && mode != "thresholds" && mode != "ground-truth") {
        std::cerr << "[dgt] mode must be both, thresholds, or ground-truth\n";
        return 2;
    }
    const bool build_thresholds = mode != "ground-truth";
    const bool build_ground_truth = mode != "thresholds";
    if (base_p.empty() || prefix.empty() || (build_ground_truth && query_p.empty())) {
        std::cerr << "usage: " << argv[0]
                  << " --base BASE.bin [--query QUERY.bin] --out_prefix PREFIX"
                  << " [--K 100] [--rk_k 10] [--tile 512] [--metric l2|ip]"
                  << " [--threads N] [--mode both|thresholds|ground-truth]"
                  << " [--threshold-table FILE]\n";
        return 2;
    }
    if (metric != "l2" && metric != "ip") {
        std::cerr << "[dgt] metric must be l2 or ip\n";
        return 2;
    }
    if (nthreads > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::cerr << "[dgt] thread count is too large\n";
        return 2;
    }
    bool is_ip = (metric=="ip");
    if(nthreads) omp_set_num_threads((int)nthreads);
    int T=omp_get_max_threads(); auto t_all=std::chrono::steady_clock::now();
    const auto threshold_t0 = t_all;

    size_t n,D; float* base=load_fbin(base_p,n,D);
    if (K >= n || rk_k > K) {
        std::cerr << "[dgt] require rk_k <= K < base vector count\n";
        _mm_free(base);
        return 2;
    }
    std::cout<<"[dgt] mode="<<mode<<" n="<<n<<" D="<<D<<" K="<<K
             <<" rk_k="<<rk_k<<" tile="<<TILE<<" threads="<<T<<"\n";
    auto item=[&](float*p,size_t i){return p+i*D;};
    std::vector<float> bnorm(n);
    #pragma omp parallel for
    for(size_t i=0;i<n;i++) bnorm[i]=DistCal::InnerProductSIMD16ExtAVX512_(item(base,i),item(base,i),D);

    std::vector<float> rk2(n);
    double threshold_build_s = 0.0;
    if (build_thresholds) {
        // Exact tiled base-to-base top-K with self exclusion.
        std::vector<uint32_t> knn_ids((size_t)n*K);
        std::vector<float> knn_d((size_t)n*K);
        auto t1=std::chrono::steady_clock::now();
        size_t ntile=(n+TILE-1)/TILE;
        #pragma omp parallel for schedule(dynamic)
        for(size_t t=0;t<ntile;t++){
            size_t ts=t*TILE, te=std::min(n,ts+TILE);
            size_t tn=te-ts;
            // Each row maintains a size-K max heap. Self matches are skipped
            // explicitly so ties and duplicate vectors do not retain the diagonal.
            std::vector<std::priority_queue<std::pair<float,uint32_t>>> heaps(tn);
            for(size_t j=0;j<n;j++){
                const float* bj=item(base,j); float nj=bnorm[j];
                for(size_t a=0;a<tn;a++){
                    size_t ti=ts+a;
                    if(j==ti) continue;
                    float ip=DistCal::InnerProductSIMD16ExtAVX512_(item(base,ti),bj,D);
                    float key = is_ip ? (-ip) : (bnorm[ti]+nj-2.0f*ip);
                    auto& h=heaps[a];
                    if(h.size()<K) h.emplace(key,(uint32_t)j);
                    else if(key<h.top().first){ h.pop(); h.emplace(key,(uint32_t)j); }
                }
            }
            for(size_t a=0;a<tn;a++){
                size_t ti=ts+a; auto& h=heaps[a];
                std::vector<std::pair<float,uint32_t>> v; v.reserve(h.size());
                while(!h.empty()){ v.push_back(h.top()); h.pop(); }
                std::sort(v.begin(),v.end());
                if(v.size()!=K){
                    #pragma omp critical
                    std::cerr<<"[dgt] insufficient non-self neighbors row="<<ti
                             <<" got="<<v.size()<<" expected="<<K<<"\n";
                    std::abort();
                }
                for(size_t r=0;r<K;r++){ knn_ids[ti*K+r]=v[r].second; knn_d[ti*K+r]=v[r].first; }
                rk2[ti]=knn_d[ti*K+(rk_k-1)];
            }
        }
        std::cout<<"[dgt] base-to-base top-K done in "
                 <<std::chrono::duration<double>(std::chrono::steady_clock::now()-t1).count()<<"s\n";

        { const std::string path = prefix+"_baseknn_gt.bin"; std::ofstream f(path,std::ios::binary); uint32_t un=n,uk=K;
          f.write((char*)&un,4); f.write((char*)&uk,4);
          f.write((char*)knn_ids.data(),(size_t)n*K*4); f.write((char*)knn_d.data(),(size_t)n*K*4);
          require_write(f, path);
          std::cout<<"[dgt] wrote "<<path<<" (n*K = "<<n<<"*"<<K<<")\n"; }
        { const std::string path = prefix+"_rknn_gt.bin.rk"; std::ofstream f(path,std::ios::binary); uint32_t un=n,uk=rk_k;
          f.write((char*)&un,4); f.write((char*)&uk,4); f.write((char*)rk2.data(),(size_t)n*4);
          require_write(f, path);
          std::cout<<"[dgt] wrote "<<path<<" (rank-k threshold array)\n"; }
        threshold_build_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now()-threshold_t0).count();
        write_timing_metadata(prefix, metric, n, D, K, T, threshold_build_s);
        std::cout << "[dgt] THRESHOLD_BUILD_TIME=" << threshold_build_s << "s\n";
    } else {
        if (threshold_table.empty()) threshold_table = prefix + "_baseknn_gt.bin";
        load_threshold_column(threshold_table, n, rk_k, rk2);
        std::cout << "[dgt] loaded rank-" << rk_k << " threshold column from "
                  << threshold_table << "\n";
    }

    double ground_truth_s = 0.0;
    if (build_ground_truth) {
        size_t nq,dq; float* query=load_fbin(query_p,nq,dq);
        if (D != dq) {
            std::cerr << "[dgt] base/query dimension mismatch: " << D << " != " << dq << "\n";
            _mm_free(base);
            _mm_free(query);
            return 3;
        }
        std::cout << "[dgt] nq=" << nq << "\n";
        std::vector<float> qnorm(nq);
        #pragma omp parallel for
        for(size_t q=0;q<nq;q++) qnorm[q]=DistCal::InnerProductSIMD16ExtAVX512_(item(query,q),item(query,q),D);

        // Exact reverse-kNN membership using each object's rank-k threshold.
        std::vector<std::vector<uint32_t>> rids(nq); std::vector<std::vector<float>> rdst(nq);
        auto t2=std::chrono::steady_clock::now();
        #pragma omp parallel for schedule(dynamic,64)
        for(size_t q=0;q<nq;q++){
            const float* qv=item(query,q); float qn=qnorm[q];
            std::vector<std::pair<float,uint32_t>> hits;
            for(size_t j=0;j<n;j++){
                float ip=DistCal::InnerProductSIMD16ExtAVX512_(qv,item(base,j),D);
                float key = is_ip ? (-ip) : (qn+bnorm[j]-2.0f*ip);
                if(key<=rk2[j]) hits.emplace_back(key,(uint32_t)j);
            }
            std::sort(hits.begin(),hits.end());
            for(auto&h:hits){ rids[q].push_back(h.second); rdst[q].push_back(h.first); }
        }
        ground_truth_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now()-t2).count();
        std::cout<<"[dgt] reverse-kNN membership done in "<<ground_truth_s<<"s\n";

        { const std::string path = prefix+"_rknn_gt.bin"; std::ofstream f(path,std::ios::binary); uint32_t unq=nq;
          uint64_t total = 0; for (const auto& ids : rids) total += ids.size();
          if (total > std::numeric_limits<uint32_t>::max()) {
              std::cerr << "[dgt] reverse-kNN output exceeds uint32 CSR capacity\n";
              return 3;
          }
          std::vector<uint32_t> off(nq+1,0); for(size_t q=0;q<nq;q++) off[q+1]=off[q]+(uint32_t)rids[q].size();
          f.write((char*)&unq,4); f.write((char*)off.data(),(nq+1)*4);
          for(size_t q=0;q<nq;q++) f.write((char*)rids[q].data(),rids[q].size()*4);
          for(size_t q=0;q<nq;q++) f.write((char*)rdst[q].data(),rdst[q].size()*4);
          require_write(f, path);
          std::cout<<"[dgt] wrote "<<path<<" (total_pairs="<<off[nq]<<", mean |R|="<<(double)off[nq]/nq<<")\n"; }
        if (!build_thresholds) {
            const std::string path = prefix+"_rknn_gt.bin.rk"; std::ofstream f(path,std::ios::binary); uint32_t un=n,uk=rk_k;
            f.write((char*)&un,4); f.write((char*)&uk,4); f.write((char*)rk2.data(),(size_t)n*4);
            require_write(f, path);
            std::cout<<"[dgt] wrote "<<path<<" (rank-k threshold array)\n";
        }
        _mm_free(query);
    }

    double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-t_all).count();
    std::cout<<"[dgt] summary mode="<<mode<<" threads="<<T
             <<" threshold_build_time="<<threshold_build_s<<"s"
             <<" ground_truth_time="<<ground_truth_s<<"s total_time="<<secs<<"s\n";
    _mm_free(base);
    return 0;
}
