// =============================================================================
//  dataset_gt.cpp —— 每数据集一次算齐：base-to-base kNN GT + reverse-kNN GT（tiled 加速）
// -----------------------------------------------------------------------------
//  ① base-to-base top-K kNN（每 base 点的精确 k 近邻）—— tiled 分块，base 向量留 cache，
//     base 只读 n/TILE 遍而非 n 遍 → 从 O(n²) 内存带宽 bound 的几十分钟降到秒级。
//     用途: (a) 提供 r_k²(o)=第 rk_k 近距离 给 rknn GT/lift；(b) 当 base AKNN 准确性判据。
//  ② reverse-kNN GT: 用 r_k 做 membership（每 query 扫 base，δ²≤r_k(o)²）。
//
//  输出:
//    <name>_baseknn_gt.bin : [n:u32][K:u32][n*K u32 ids][n*K f32 d²]  (base-to-base top-K, 排除自身, 升序)
//    <name>_rknn_gt.bin     : [nq:u32][(nq+1) u32 off][total u32 ids][total f32 d²]  (CSR)
//    <name>_rknn_gt.bin.rk  : [n:u32][k:u32][n f32 r_k²]
//  编译: g++ -O3 -mavx512f -march=native -fopenmp -std=c++17 -I ../include dataset_gt.cpp -o dataset_gt
//  用法: ./dataset_gt --base B.bin --query Q.bin --K 100 --rk_k 10 --out_prefix /path/NAME [--tile 512]
// =============================================================================
#include "../include/distance.h"
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include <string>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <sys/resource.h>
#include <immintrin.h>

static float* load_fbin(const std::string& p, size_t& n, size_t& d) {
    std::ifstream f(p, std::ios::binary); if(!f){std::cerr<<"open "<<p<<"\n";exit(1);}
    uint32_t nn,dd; f.read((char*)&nn,4); f.read((char*)&dd,4); n=nn; d=dd;
    float* x=static_cast<float*>(_mm_malloc((size_t)n*d*4,64)); f.read((char*)x,(size_t)n*d*4); return x;
}
static double peak_gb(){ struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/1e6; }

int main(int argc, char** argv) {
    std::string base_p, query_p, prefix, metric="l2"; size_t K=100, rk_k=10, TILE=512, nthreads=0;
    for (int i=1;i<argc;i++){ std::string a=argv[i];
        if(a=="--base")base_p=argv[++i]; else if(a=="--query")query_p=argv[++i];
        else if(a=="--K")K=atoi(argv[++i]); else if(a=="--rk_k")rk_k=atoi(argv[++i]);
        else if(a=="--out_prefix")prefix=argv[++i]; else if(a=="--tile")TILE=atoi(argv[++i]);
        else if(a=="--metric")metric=argv[++i]; else if(a=="--threads")nthreads=atoi(argv[++i]); }
    bool is_ip = (metric=="ip");
    // 统一 key：小 key=更优。L2: key=d²(越小越近)。IP: key=−⟨a,b⟩(越小内积越大)。
    // 存的 rk2[] = 第 rk_k 个的 key（L2 即 r_k²；IP 即 −ip_k）。membership 统一: key(q,o) ≤ rk2[o]。
    if(nthreads) omp_set_num_threads((int)nthreads);
    int T=omp_get_max_threads(); auto t_all=std::chrono::steady_clock::now();

    size_t n,D,nq,dq; float* base=load_fbin(base_p,n,D); float* query=load_fbin(query_p,nq,dq);
    std::cout<<"[dgt] n="<<n<<" nq="<<nq<<" D="<<D<<" K="<<K<<" rk_k="<<rk_k<<" tile="<<TILE<<" threads="<<T<<"\n";
    auto item=[&](float*p,size_t i){return p+i*D;};
    std::vector<float> bnorm(n),qnorm(nq);
    #pragma omp parallel for
    for(size_t i=0;i<n;i++) bnorm[i]=DistCal::InnerProductSIMD16ExtAVX512_(item(base,i),item(base,i),D);
    #pragma omp parallel for
    for(size_t q=0;q<nq;q++) qnorm[q]=DistCal::InnerProductSIMD16ExtAVX512_(item(query,q),item(query,q),D);

    // ---- ① tiled base-to-base top-K ----
    std::vector<uint32_t> knn_ids((size_t)n*K); std::vector<float> knn_d((size_t)n*K);
    std::vector<float> rk2(n);
    auto t1=std::chrono::steady_clock::now();
    size_t ntile=(n+TILE-1)/TILE;
    #pragma omp parallel for schedule(dynamic)
    for(size_t t=0;t<ntile;t++){
        size_t ts=t*TILE, te=std::min(n,ts+TILE);
        size_t tn=te-ts;
        // 每个 tile 点维护一个容量 K 的大顶堆。显式跳过自身，避免
        // 浮点并列或重复向量使 self-match 不在排序首位时被错误保留。
        std::vector<std::priority_queue<std::pair<float,uint32_t>>> heaps(tn);
        for(size_t j=0;j<n;j++){
            const float* bj=item(base,j); float nj=bnorm[j];
            for(size_t a=0;a<tn;a++){
                size_t ti=ts+a;
                if(j==ti) continue;
                float ip=DistCal::InnerProductSIMD16ExtAVX512_(item(base,ti),bj,D);
                float key = is_ip ? (-ip) : (bnorm[ti]+nj-2.0f*ip);   // 统一 key：小=优
                auto& h=heaps[a];
                if(h.size()<K) h.emplace(key,(uint32_t)j);
                else if(key<h.top().first){ h.pop(); h.emplace(key,(uint32_t)j); }
            }
        }
        for(size_t a=0;a<tn;a++){
            size_t ti=ts+a; auto& h=heaps[a];
            std::vector<std::pair<float,uint32_t>> v; v.reserve(h.size());
            while(!h.empty()){ v.push_back(h.top()); h.pop(); }
            std::sort(v.begin(),v.end());                 // 升序
            if(v.size()!=K){
                #pragma omp critical
                std::cerr<<"[dgt] insufficient non-self neighbors row="<<ti
                         <<" got="<<v.size()<<" expected="<<K<<"\n";
                std::abort();
            }
            for(size_t r=0;r<K;r++){ knn_ids[ti*K+r]=v[r].second; knn_d[ti*K+r]=v[r].first; }
            rk2[ti]=knn_d[ti*K+(rk_k-1)];                 // 第 rk_k 近(排除自身) = r_k²
        }
    }
    std::cout<<"[dgt] ① base-to-base top-K done in "
             <<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t1).count()/1000.0<<"s\n";

    // ---- ② rknn GT: membership ----
    std::vector<std::vector<uint32_t>> rids(nq); std::vector<std::vector<float>> rdst(nq);
    auto t2=std::chrono::steady_clock::now();
    #pragma omp parallel for schedule(dynamic,64)
    for(size_t q=0;q<nq;q++){
        const float* qv=item(query,q); float qn=qnorm[q];
        std::vector<std::pair<float,uint32_t>> hits;
        for(size_t j=0;j<n;j++){
            float ip=DistCal::InnerProductSIMD16ExtAVX512_(qv,item(base,j),D);
            float key = is_ip ? (-ip) : (qn+bnorm[j]-2.0f*ip);
            if(key<=rk2[j]) hits.emplace_back(key,(uint32_t)j);   // key(q,o) ≤ rk2[o] 统一 membership
        }
        std::sort(hits.begin(),hits.end());
        for(auto&h:hits){ rids[q].push_back(h.second); rdst[q].push_back(h.first); }
    }
    std::cout<<"[dgt] ② rknn membership done in "
             <<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t2).count()/1000.0<<"s\n";

    // ---- 写文件 ----
    { std::ofstream f(prefix+"_baseknn_gt.bin",std::ios::binary); uint32_t un=n,uk=K;
      f.write((char*)&un,4); f.write((char*)&uk,4);
      f.write((char*)knn_ids.data(),(size_t)n*K*4); f.write((char*)knn_d.data(),(size_t)n*K*4);
      std::cout<<"[dgt] wrote "<<prefix<<"_baseknn_gt.bin (n×K = "<<n<<"×"<<K<<")\n"; }
    { std::ofstream f(prefix+"_rknn_gt.bin",std::ios::binary); uint32_t unq=nq;
      std::vector<uint32_t> off(nq+1,0); for(size_t q=0;q<nq;q++) off[q+1]=off[q]+(uint32_t)rids[q].size();
      f.write((char*)&unq,4); f.write((char*)off.data(),(nq+1)*4);
      for(size_t q=0;q<nq;q++) f.write((char*)rids[q].data(),rids[q].size()*4);
      for(size_t q=0;q<nq;q++) f.write((char*)rdst[q].data(),rdst[q].size()*4);
      std::cout<<"[dgt] wrote "<<prefix<<"_rknn_gt.bin (total_pairs="<<off[nq]<<", mean |R|="<<(double)off[nq]/nq<<")\n"; }
    { std::ofstream f(prefix+"_rknn_gt.bin.rk",std::ios::binary); uint32_t un=n,uk=rk_k;
      f.write((char*)&un,4); f.write((char*)&uk,4); f.write((char*)rk2.data(),(size_t)n*4);
      std::cout<<"[dgt] wrote "<<prefix<<"_rknn_gt.bin.rk (r_k² array)\n"; }

    double secs=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t_all).count()/1000.0;
    std::cout<<"[dgt] === SUMMARY === threads="<<T<<" | total_time="<<secs<<"s | peak_mem="<<peak_gb()<<" GB\n";
    return 0;
}
