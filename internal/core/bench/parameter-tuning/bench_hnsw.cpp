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

class HNSW_Fixture_SIFT1M : public benchmark::Fixture {
public:
    const knowhere::IndexType index_type = knowhere::IndexEnum::INDEX_HNSW;
    knowhere::MetricType metric_type = knowhere::Metric::L2;
    knowhere::VecIndexPtr index = knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type);
    HNSW_Parameter para;
    knowhere::Config conf = knowhere::Config{
            {knowhere::meta::DIM, 128},
            {knowhere::meta::TOPK, 100},
            {knowhere::IndexParams::M, -1},
            {knowhere::IndexParams::efConstruction, -1},
            {knowhere::IndexParams::ef, -1},
            {knowhere::Metric::TYPE, metric_type},
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

    HNSW_Fixture_SIFT1M() {
        std::cout << "[Benchmark] HNSW_Fixture Benchmark Constructor only called once per fixture testcase hat uses it." << std::endl;
        xq_dataset = load_queries(SIFT_DIM, nq);
        load_ground_truth(nq, k, gt, conf);
    }

    ~HNSW_Fixture_SIFT1M() {
        std::cout << "[Benchmark] HNSW_Fixture Benchmark Destructor." << std::endl;
        ReleaseQuery(xq_dataset);
        delete[] gt;
    }

    void SetUp(::benchmark::State& state) {
        std::cout << "[Benchmark] SetUp." << std::endl;
        static bool needBuildIndex = true;
        if (para.IsInvalid()) {
            para.Set(state.range(0), state.range(1), state.range(2));
            conf[knowhere::IndexParams::m] = para.Get_m();
            conf[knowhere::IndexParams::efConstruction] = para.Get_efConstruction();
            conf[knowhere::IndexParams::ef] = para.Get_ef();
        }

        // Build Parameters: m, Get_efConstruction
        auto m = state.range(0);
        state.counters["m"] = m;
        auto efConstruction = state.range(1);
        state.counters["efConstruction"] = efConstruction;
        if (!para.CheckBuildPara(m, efConstruction)) {
            std::cout << "[Benchmark] HNSW_Fixture Benchmark Constructor only called once per fixture testcase hat uses it." << std::endl;
            conf[knowhere::IndexParams::m] = para.Get_m();
            conf[knowhere::IndexParams::efConstruction] = para.Get_efConstruction();
            needBuildIndex = true;
        }

        // Search Parameters: ef
        auto ef = state.range(2);
        state.counters["ef"] = ef;
        conf[knowhere::IndexParams::ef] = para.Get_ef();
        para.SetSearchPara(ef);

        // Build Index if necessary
        if (needBuildIndex) {
            std::tie(train_time_min, train_time_max, train_time_avg) = train(d, nt, index, conf);
            std::tie(add_points_time_min, add_points_time_max, add_points_time_avg) = add_points(index_type, d, nb, index, conf, nt);
            needBuildIndex = false;
        }
        state.counters["train time min (ms)"] = train_time_min;
        state.counters["train time max (ms)"] = train_time_max;
        state.counters["train time avg (ms)"] = train_time_avg;
        state.counters["add points time min (ms)"] = add_points_time_min;
        state.counters["add points time max (ms)"] = add_points_time_max;
        state.counters["add points time avg (ms)"] = add_points_time_avg;
        index->UpdateIndexSize();
        state.counters["Index Size (B)"] = index->Size();
    }

    void TearDown(const ::benchmark::State& state) {
        std::cout << "[Benchmark] TearDown." << std::endl;
    }
};

BENCHMARK_DEFINE_F(HNSW_Fixture_SIFT1M, HNSW_SIFT1M)(benchmark::State& state) {
    auto result = milvus::knowhere::DatasetPtr(nullptr);
    conf[knowhere::IndexParams::ef] = para.Get_ef();
    assert(para.Get_ef() == state.range(2));
    {
        std::cout << "[Benchmark] Perform a search on " << nq << " queries" << std::endl;
        std::cout << para;
        // Wash the cache
//        result = index->Query(xq_dataset, conf, nullptr);
//        ReleaseQueryResult(result);
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

    ReleaseQueryResult(result);
}

// m \in [4, ..., 64]
// efConstruction \in [8, ..., 512]
// ef \in [k, ..., 32768]
static void
CustomArguments(benchmark::internal::Benchmark* b) {
//    for (int m = 4; m <= 64; m *= 2) {
//        for (int efConstruction = 8; efConstruction <= 512; efConstruction *= 2) {
//            for (int ef = 100 /*TODO: the TOPK*/; ef <= 32768; ef *= 2) {
//                b->Args({m, efConstruction, ef});
//            }
//        }
//    }
    for (int m = 16; m <= 64; m *= 2) {
        for (int efConstruction = 200; efConstruction <= 512; efConstruction *= 2) {
//            for (int ef = 100 /*TODO: the TOPK*/; ef <= 32768; ef *= 2) {
                b->Args({m, efConstruction, 200});
//            }
        }
    }
}

BENCHMARK_REGISTER_F(HNSW_Fixture_SIFT1M, HNSW_SIFT1M)->Unit(benchmark::kMillisecond)->Apply(CustomArguments);
