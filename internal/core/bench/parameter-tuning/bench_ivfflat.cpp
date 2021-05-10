// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <benchmark/benchmark.h>

#include "index/knowhere/knowhere/index/vector_index/helpers/IndexParameter.h"
#include "index/knowhere/knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "indexbuilder/IndexWrapper.h"
#include "indexbuilder/index_c.h"
#include "indexbuilder/utils.h"
#include "test_utils/indexbuilder_test_utils.h"
#include "bench_utils/parameter_tuning_utils.h"
#include "bench_utils/sift.h"

namespace knowhere = milvus::knowhere;

//// TODO(jigao): Add more metrics.
//auto metric_type_collections = [] {
//    static std::unordered_map<int64_t, milvus::knowhere::MetricType> collections{
//        {0, milvus::knowhere::Metric::L2},
//    };
//    return collections;
//}();

////    auto is_binary = state.range(2);
//auto is_binary = false;
//auto dataset = GenDataset(NB, metric_type, is_binary);

class IVFFlat_Fixture_SIFT1M : public benchmark::Fixture {
 public:
    const knowhere::IndexType index_type = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;
    knowhere::MetricType metric_type = knowhere::Metric::L2;
    knowhere::VecIndexPtr index = knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type);
    IVFFLAT_Parameter para;
    knowhere::Config conf = knowhere::Config{
            {knowhere::meta::DIM, 128},
            {knowhere::meta::TOPK, 100},
            {knowhere::IndexParams::nlist, -1},
            {knowhere::IndexParams::nprobe, -1},
            {knowhere::Metric::TYPE, metric_type},
            {knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
#ifdef MILVUS_GPU_VERSION
            {knowhere::meta::DEVICEID, DEVICEID},
#endif
    };

    size_t d;
    size_t nt;
    size_t nb;
    size_t nq;
    size_t k;                // nb of results per query in the GT
    int64_t* gt; // nq * k matrix of ground-truth nearest-neighbors
    knowhere::DatasetPtr xq_dataset;

    double train_time_min, train_time_max, train_time_avg;
    double add_points_time_min, add_points_time_max, add_points_time_avg;

    IVFFlat_Fixture_SIFT1M() {
        std::cout << "[Benchmark] IVFFlat_Fixture Benchmark Constructor only called once per fixture testcase hat uses it." << std::endl;
    }

    void SetUp(::benchmark::State& state) {
        static bool needBuildIndex = true;
        static bool needLoadQuery = true;
        if (para.IsInvalid()) {
            para.Set(state.range(0), state.range(1));
            conf[knowhere::IndexParams::nlist] = para.Get_nlist();
            conf[knowhere::IndexParams::nprobe] = para.Get_nprobe();
        }

        // Build Parameters: nlist
        auto nlist = state.range(0);
        state.counters["nlist"] = nlist;
        if (!para.CheckBuildPara(nlist)) {
            std::cout << "[Benchmark] IVFFlat_Fixture Benchmark Constructor only called once per fixture testcase hat uses it." << std::endl;
            conf[knowhere::IndexParams::nlist] = para.Get_nlist();
            needBuildIndex = true;
        }

        // Search Parameters: nprobe
        auto nprobe = state.range(1);
        state.counters["nprobe"] = nprobe;
        conf[knowhere::IndexParams::nprobe] = para.Get_nprobe();
        para.SetSearchPara(nprobe);

        // Build Index if necessary
        if (needBuildIndex) {
            std::tie(train_time_min, train_time_max, train_time_avg) = train(d, nt, index, conf);
            std::tie(add_points_time_min, add_points_time_max, add_points_time_avg) = add_points(index_type, d, nb, index, conf, nt);
            needBuildIndex = false;
        }
        state.counters["train time min"] = train_time_min;
        state.counters["train time max"] = train_time_max;
        state.counters["train time avg"] = train_time_avg;
        state.counters["add points time min"] = add_points_time_min;
        state.counters["add points time max"] = add_points_time_max;
        state.counters["add points time avg"] = add_points_time_avg;
        index->UpdateIndexSize();
        state.counters["Index Size"] = index->Size();

        if (needLoadQuery) {
            xq_dataset = load_queries(d, nq);
            load_ground_truth(nq, k, gt, conf);
        }
        needLoadQuery = false;
    }

    void TearDown(const ::benchmark::State& state) {}
};

BENCHMARK_DEFINE_F(IVFFlat_Fixture_SIFT1M, IVFFlat_SIFT1M)(benchmark::State& state) {
    auto result = milvus::knowhere::DatasetPtr(nullptr);
    conf[knowhere::IndexParams::nprobe] = para.Get_nprobe();
    assert(para.Get_nprobe() == state.range(1));
    {
        std::cout << "[Benchmark] Perform a search on " << nq << " queries" << std::endl;
        std::cout << para;
        // Wash the cache
        result = index->Query(xq_dataset, conf, nullptr);
        for (auto _ : state) {
            result = index->Query(xq_dataset, conf, nullptr);
        }
    }

    // QPS
    state.SetItemsProcessed(state.iterations() * nq);

    auto recall = compute_recall(nq, k, result, gt, k);
    state.counters["Recall"] = recall;
    auto recall_1 = compute_recall(nq, k, result, gt, 1);
    state.counters["Recall@1"] = recall_1;
    if (k >= 10) {
        auto recall_10 = compute_recall(nq, k, result, gt, 10);
        state.counters["Recall@10"] = recall_10;
    }
    if (k >= 100) {
        auto recall_100 = compute_recall(nq, k, result, gt, 100);
        state.counters["Recall@100"] = recall_100;
    }

//    ReleaseQuery(xq_dataset);
    ReleaseQueryResult(result);
}

// nlist \in [1024, 2048, 4096, 8192, ..., 65536]
// nprobe \in [1, 2, 4, 8, ..., nlist]
static void
CustomArguments(benchmark::internal::Benchmark* b) {
    for (int nlist = 1024; nlist <= 65536; nlist *= 2) {
        for (int nprobe = 1; nprobe <= nlist; nprobe *= 2) {
            b->Args({nlist, nprobe});
        }
    }
}

// ->Name("IVFFLAT/L2/VectorFloat")
BENCHMARK_REGISTER_F(IVFFlat_Fixture_SIFT1M, IVFFlat_SIFT1M)->Unit(benchmark::kMillisecond)->Apply(CustomArguments);

