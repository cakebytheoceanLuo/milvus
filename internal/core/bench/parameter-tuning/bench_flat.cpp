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
#include <tuple>
#include <unordered_map>
#include <google/protobuf/text_format.h>

#include "pb/index_cgo_msg.pb.h"
#include "index/knowhere/knowhere/index/vector_index/helpers/IndexParameter.h"
#include "index/knowhere/knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "indexbuilder/IndexWrapper.h"
#include "indexbuilder/index_c.h"
#include "indexbuilder/utils.h"
#include "test_utils/indexbuilder_test_utils.h"
#include "bench_utils/parameter_tuning_utils.h"
#include "bench_utils/sift.h"

constexpr int64_t NB = 1000000;

namespace indexcgo = milvus::proto::indexcgo;

// TODO(jigao): Add more metrics.
auto metric_type_collections = [] {
    static std::unordered_map<int64_t, milvus::knowhere::MetricType> collections{
        {0, milvus::knowhere::Metric::L2},
    };
    return collections;
}();

static void
FLAT_build(benchmark::State& state) {
    auto index_type = milvus::knowhere::IndexEnum::INDEX_FAISS_IDMAP;
    //    auto metric_type = metric_type_collections.at(state.range(1));
    auto metric_type = milvus::knowhere::Metric::L2;

    indexcgo::TypeParams type_params;
    indexcgo::IndexParams index_params;

    std::tie(type_params, index_params) = flat_generate_params(index_type, metric_type);

    std::string type_params_str, index_params_str;
    bool ok;
    ok = google::protobuf::TextFormat::PrintToString(type_params, &type_params_str);
    assert(ok);
    ok = google::protobuf::TextFormat::PrintToString(index_params, &index_params_str);
    assert(ok);

    //    auto is_binary = state.range(2);
    auto is_binary = false;
    auto dataset = GenDataset(NB, metric_type, is_binary);
    auto xb_data = dataset.get_col<float>(0);
    auto xb_dataset = milvus::knowhere::GenDataset(NB, DIM, xb_data.data());

    std::unique_ptr<milvus::indexbuilder::IndexWrapper> index(nullptr);
    for (auto _ : state) {
        index = std::make_unique<milvus::indexbuilder::IndexWrapper>(type_params_str.c_str(), index_params_str.c_str());
        index->BuildWithoutIds(xb_dataset);
    }
    index->UpdateIndexSize();
    state.counters["Index Size"] = index->Size();
}

static void
FLAT_search(benchmark::State& state) {
    auto index_type = milvus::knowhere::IndexEnum::INDEX_FAISS_IDMAP;
    auto metric_type = milvus::knowhere::Metric::L2;

    auto index = milvus::knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type);

    double t0 = elapsed();
    size_t d;

    auto conf = milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, 128},
            {milvus::knowhere::meta::TOPK, 100},
            {milvus::knowhere::Metric::TYPE, metric_type},
            {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
    };

    size_t nt;
    size_t nb;
    size_t nq;
    size_t k;                // nb of results per query in the GT
    int64_t* gt; // nq * k matrix of ground-truth nearest-neighbors

    train(t0, index_type, d, nt, index, conf);

    add_points(t0, index_type, d, nb, index, conf);

    auto xq_dataset = load_queries(t0, d, nq);

    load_ground_truth(t0, nq, k, gt, conf);

    auto result = milvus::knowhere::DatasetPtr(nullptr);

    {
        printf("[%.3f s] Perform a search on %ld queries\n",
               elapsed() - t0,
               nq);

        for (auto _ : state) {
            result = index->Query(xq_dataset, conf, nullptr);
        }
    }

    auto recall = compute_recall(t0, nq, k, result, gt, k);
    state.counters["Recall"] = recall;

    auto recall_1 = compute_recall(t0, nq, k, result, gt, 1);
    state.counters["Recall@1"] = recall_1;

    if (k > 10) {
        auto recall_10 = compute_recall(t0, nq, k, result, gt, 10);
        state.counters["Recall@10"] = recall_10;
    }

    if (k > 100) {
        auto recall_100 = compute_recall(t0, nq, k, result, gt, 100);
        state.counters["Recall@100"] = recall_100;
    }

    ReleaseQuery(xq_dataset);
    ReleaseQueryResult(result);
}

// FLAT, L2, VectorFloat
//BENCHMARK(FLAT_build)->Name("Build: FLAT/L2/VectorFloat");

//BENCHMARK(FLAT_search)->Name("Search: FLAT/L2/VectorFloat")->Unit(benchmark::kMillisecond);

class MyFixture : public benchmark::Fixture
{
public:
    // add members as needed

    MyFixture()
    {
        std::cout << "Ctor only called once per fixture testcase hat uses it" << std::endl;
        // call whatever setup functions you need in the fixtures ctor
    }

    void SetUp(const ::benchmark::State& state) {
        static bool callSetup = true;
        static int old_zeroth = -1;
        if (old_zeroth != state.range(0))
        {
            old_zeroth = state.range(0);
            // Call your setup function
            std::cout << "X" << state.range(0) << std::endl;
        }
        callSetup = false;
        // call whatever setup functions you need in the fixtures ctor
    }

    void TearDown(const ::benchmark::State& state) {
    }
};

BENCHMARK_DEFINE_F(MyFixture, TestcaseName)(benchmark::State& state)
{
    std::cout << "Benchmark function called more than once: " << state.range(0) << std::endl;
    for (auto _ : state)
    {
        //run your benchmark
        std::string ll;
    }
}

BENCHMARK_REGISTER_F(MyFixture, TestcaseName)->Name("Search: FLAT/L2/VectorFloat")->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(1024, 65536);;
