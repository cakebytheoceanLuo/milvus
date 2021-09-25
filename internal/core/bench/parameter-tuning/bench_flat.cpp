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

class Flat_Fixture_SIFT1M : public benchmark::Fixture {
 public:
    const knowhere::IndexType index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP;
    knowhere::MetricType metric_type = knowhere::Metric::L2;
    knowhere::VecIndexPtr index = knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type);
    IVFFLAT_Parameter para;
    knowhere::Config conf = knowhere::Config{
        {knowhere::meta::DIM, 128},
        {knowhere::meta::TOPK, 100},
        {knowhere::Metric::TYPE, metric_type},
        {knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
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

    Flat_Fixture_SIFT1M() {
        std::cout << "[Benchmark] Flat_Fixture Benchmark Constructor only called once per fixture testcase hat uses it." << std::endl;
        xq_dataset = load_queries(SIFT_DIM, nq);
        load_ground_truth(nq, k, gt, conf);
    }

    ~Flat_Fixture_SIFT1M() {
        std::cout << "[Benchmark] FLAT_Fixture Benchmark Destructor." << std::endl;
        ReleaseQuery(xq_dataset);
        delete[] gt;
    }

    void SetUp(::benchmark::State& state) {
        std::cout << "[Benchmark] SetUp." << std::endl;
        std::tie(train_time_min, train_time_max, train_time_avg) = train(d, nt, index, conf);
        std::tie(add_points_time_min, add_points_time_max, add_points_time_avg) = add_points(index_type, d, nb, index, conf, nt);
        index->UpdateIndexSize();
        state.counters["Index Size"] = index->Size();
    }

    void TearDown(const ::benchmark::State& state) {
        std::cout << "[Benchmark] TearDown." << std::endl;
    }
};

BENCHMARK_DEFINE_F(Flat_Fixture_SIFT1M, Flat_SIFT1M)(benchmark::State& state) {
    auto result = milvus::knowhere::DatasetPtr(nullptr);
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
    ReleaseQueryResult(result);
}

BENCHMARK_REGISTER_F(Flat_Fixture_SIFT1M, Flat_SIFT1M)->Unit(benchmark::kMillisecond);

