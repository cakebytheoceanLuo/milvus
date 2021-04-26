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
#include <array>
#include <google/protobuf/text_format.h>

#include "pb/index_cgo_msg.pb.h"
#include "index/knowhere/knowhere/index/vector_index/helpers/IndexParameter.h"
#include "index/knowhere/knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "indexbuilder/IndexWrapper.h"
#include "indexbuilder/index_c.h"
#include "indexbuilder/utils.h"
#include "test_utils/indexbuilder_test_utils.h"
#include "bench_utils/parameter_tuning_utils.h"

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
IndexBuilder_build(benchmark::State& state) {
    auto nlist = state.range(0);
    state.counters["nlist"] = nlist;
    auto nbits = state.range(1);
    state.counters["nbits"] = nbits;
    auto index_type = milvus::knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;
    //    auto metric_type = metric_type_collections.at(state.range(1));
    auto metric_type = milvus::knowhere::Metric::L2;

    indexcgo::TypeParams type_params;
    indexcgo::IndexParams index_params;

    std::tie(type_params, index_params) = ivfsq8_generate_params(index_type, metric_type, nlist, nbits);

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

// IVF_SQ8, L2, VectorFloat
// nlist \in [1024, 2048, 4096, 8192, ..., 65536]
// nbits \in [4, 6, 8, 16]

static void
CustomArguments(benchmark::internal::Benchmark* b) {
    static std::array<int, 4> nbits_arr{4, 6, 8, 16};
    for (int nlist = 1024; nlist <= 65536; nlist *= 2) {
        for (auto nbits : nbits_arr) {
            b->Args({nlist, nbits});
        }
    }
}
BENCHMARK(IndexBuilder_build)->Name("IVF_SQ8/L2/VectorFloat")->Apply(CustomArguments);
