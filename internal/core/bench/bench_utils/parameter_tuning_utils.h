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

#pragma once

#include <tuple>
#include <limits>
#include <cmath>
#include <google/protobuf/text_format.h>

#include "pb/index_cgo_msg.pb.h"
#include "index/knowhere/knowhere/index/vector_index/helpers/IndexParameter.h"
#include "index/knowhere/knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "indexbuilder/IndexWrapper.h"
#include "indexbuilder/index_c.h"
#include "faiss/MetricType.h"
#include "index/knowhere/knowhere/index/vector_index/VecIndexFactory.h"
#include "indexbuilder/utils.h"
#include "test_utils/indexbuilder_test_utils.h"

namespace {
auto
TODELETE_generate_conf(const milvus::knowhere::IndexType& index_type, const milvus::knowhere::MetricType& metric_type) {
    if (index_type == milvus::knowhere::IndexEnum::INDEX_FAISS_IDMAP) {
        //            return milvus::knowhere::Config{
        //                    {milvus::knowhere::meta::DIM, DIM},
        //                    {milvus::knowhere::meta::TOPK, K},
        //                    {milvus::knowhere::Metric::TYPE, metric_type},
        //                    {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
        //            };
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_FAISS_IVFPQ) {
        //            return milvus::knowhere::Config{
        //                    {milvus::knowhere::meta::DIM, DIM},
        //                    {milvus::knowhere::meta::TOPK, K},
        //                    {milvus::knowhere::IndexParams::nlist, 100},
        //                    {milvus::knowhere::IndexParams::nprobe, 4},
        //                    {milvus::knowhere::IndexParams::m, 4},
        //                    {milvus::knowhere::IndexParams::nbits, 8},
        //                    {milvus::knowhere::Metric::TYPE, metric_type},
        //                    {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
        //            };
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_FAISS_IVFFLAT) {
        //            return milvus::knowhere::Config{
        //                    {milvus::knowhere::meta::DIM, DIM},
        //                    {milvus::knowhere::meta::TOPK, K},
        //                    {milvus::knowhere::IndexParams::nlist, 1024},
        //                    {milvus::knowhere::IndexParams::nprobe, 4},
        //                    {milvus::knowhere::Metric::TYPE, metric_type},
        //                    {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
        //#ifdef MILVUS_GPU_VERSION
        //                    {milvus::knowhere::meta::DEVICEID, DEVICEID},
        //#endif
        //            };
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_FAISS_IVFSQ8) {
        //            return milvus::knowhere::Config{
        //                    {milvus::knowhere::meta::DIM, DIM},
        //                    {milvus::knowhere::meta::TOPK, K},
        //                    {milvus::knowhere::IndexParams::nlist, 100},
        //                    {milvus::knowhere::IndexParams::nprobe, 4},
        //                    {milvus::knowhere::IndexParams::nbits, 8},
        //                    {milvus::knowhere::Metric::TYPE, metric_type},
        //                    {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
        //#ifdef MILVUS_GPU_VERSION
        //                    {milvus::knowhere::meta::DEVICEID, DEVICEID},
        //#endif
        //            };
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT) {
        return milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, DIM},
            {milvus::knowhere::meta::TOPK, K},
            {milvus::knowhere::IndexParams::nlist, 100},
            {milvus::knowhere::IndexParams::nprobe, 4},
            {milvus::knowhere::IndexParams::m, 4},
            {milvus::knowhere::IndexParams::nbits, 8},
            {milvus::knowhere::Metric::TYPE, metric_type},
            {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
        };
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP) {
        return milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, DIM},
            {milvus::knowhere::meta::TOPK, K},
            {milvus::knowhere::Metric::TYPE, metric_type},
        };
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_NSG) {
        return milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, DIM},
            {milvus::knowhere::IndexParams::nlist, 163},
            {milvus::knowhere::meta::TOPK, K},
            {milvus::knowhere::IndexParams::nprobe, 8},
            {milvus::knowhere::IndexParams::knng, 20},
            {milvus::knowhere::IndexParams::search_length, 40},
            {milvus::knowhere::IndexParams::out_degree, 30},
            {milvus::knowhere::IndexParams::candidate, 100},
            {milvus::knowhere::Metric::TYPE, metric_type},
        };
#ifdef MILVUS_SUPPORT_SPTAG
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_SPTAG_KDT_RNT) {
        return milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, DIM},
            // {milvus::knowhere::meta::TOPK, 10},
            {milvus::knowhere::Metric::TYPE, metric_type},
            {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
        };
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_SPTAG_BKT_RNT) {
        return milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, DIM},
            // {milvus::knowhere::meta::TOPK, 10},
            {milvus::knowhere::Metric::TYPE, metric_type},
            {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
        };
#endif
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_HNSW) {
        return milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, DIM},       {milvus::knowhere::meta::TOPK, K},
            {milvus::knowhere::IndexParams::M, 16},   {milvus::knowhere::IndexParams::efConstruction, 200},
            {milvus::knowhere::IndexParams::ef, 200}, {milvus::knowhere::Metric::TYPE, metric_type},
        };
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_ANNOY) {
        return milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, DIM},
            {milvus::knowhere::meta::TOPK, K},
            {milvus::knowhere::IndexParams::n_trees, 4},
            {milvus::knowhere::IndexParams::search_k, 100},
            {milvus::knowhere::Metric::TYPE, metric_type},
            {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
        };
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_RHNSWFlat) {
        return milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, DIM},
            {milvus::knowhere::meta::TOPK, K},
            {milvus::knowhere::IndexParams::M, 16},
            {milvus::knowhere::IndexParams::efConstruction, 200},
            {milvus::knowhere::IndexParams::ef, 200},
            {milvus::knowhere::Metric::TYPE, metric_type},
            {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
        };
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_RHNSWPQ) {
        return milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, DIM},
            {milvus::knowhere::meta::TOPK, K},
            {milvus::knowhere::IndexParams::M, 16},
            {milvus::knowhere::IndexParams::efConstruction, 200},
            {milvus::knowhere::IndexParams::ef, 200},
            {milvus::knowhere::Metric::TYPE, metric_type},
            {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
            {milvus::knowhere::IndexParams::PQM, 8},
        };
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_RHNSWSQ) {
        return milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, DIM},
            {milvus::knowhere::meta::TOPK, K},
            {milvus::knowhere::IndexParams::M, 16},
            {milvus::knowhere::IndexParams::efConstruction, 200},
            {milvus::knowhere::IndexParams::ef, 200},
            {milvus::knowhere::Metric::TYPE, metric_type},
            {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
        };
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_NGTPANNG) {
        return milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, DIM},
            {milvus::knowhere::meta::TOPK, K},
            {milvus::knowhere::Metric::TYPE, metric_type},
            {milvus::knowhere::IndexParams::edge_size, 10},
            {milvus::knowhere::IndexParams::epsilon, 0.1},
            {milvus::knowhere::IndexParams::max_search_edges, 50},
            {milvus::knowhere::IndexParams::forcedly_pruned_edge_size, 60},
            {milvus::knowhere::IndexParams::selectively_pruned_edge_size, 30},
            {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
        };
    } else if (index_type == milvus::knowhere::IndexEnum::INDEX_NGTONNG) {
        return milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, DIM},
            {milvus::knowhere::meta::TOPK, K},
            {milvus::knowhere::Metric::TYPE, metric_type},
            {milvus::knowhere::IndexParams::edge_size, 20},
            {milvus::knowhere::IndexParams::epsilon, 0.1},
            {milvus::knowhere::IndexParams::max_search_edges, 50},
            {milvus::knowhere::IndexParams::outgoing_edge_size, 5},
            {milvus::knowhere::IndexParams::incoming_edge_size, 40},
            {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
        };
    }
    return milvus::knowhere::Config();
}

auto
flat_generate_params(const milvus::knowhere::IndexType& index_type, const milvus::knowhere::MetricType& metric_type) {
    namespace indexcgo = milvus::proto::indexcgo;

    assert(index_type == milvus::knowhere::IndexEnum::INDEX_FAISS_IDMAP);
    indexcgo::TypeParams type_params;
    indexcgo::IndexParams index_params;
    auto configs = milvus::knowhere::Config{
        {milvus::knowhere::meta::DIM, DIM},
        {milvus::knowhere::meta::TOPK, K},
        {milvus::knowhere::Metric::TYPE, metric_type},
        {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
    };
    for (auto& [key, value] : configs.items()) {
        auto param = index_params.add_params();
        auto value_str = value.is_string() ? value.get<std::string>() : value.dump();
        param->set_key(key);
        param->set_value(value_str);
    }

    auto param = index_params.add_params();
    param->set_key("index_type");
    param->set_value(std::string(index_type));

    return std::make_tuple(type_params, index_params);
}

auto
ivfflat_generate_params(const milvus::knowhere::IndexType& index_type,
                        const milvus::knowhere::MetricType& metric_type,
                        int64_t nlist,
                        int64_t nprobe = 4) {
    namespace indexcgo = milvus::proto::indexcgo;

    assert(index_type == milvus::knowhere::IndexEnum::INDEX_FAISS_IVFFLAT);
    indexcgo::TypeParams type_params;
    indexcgo::IndexParams index_params;
    auto configs = milvus::knowhere::Config{
        {milvus::knowhere::meta::DIM, DIM},
        {milvus::knowhere::meta::TOPK, K},
        {milvus::knowhere::IndexParams::nlist, nlist},
        {milvus::knowhere::IndexParams::nprobe, nprobe},
        {milvus::knowhere::Metric::TYPE, metric_type},
        {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
#ifdef MILVUS_GPU_VERSION
        {milvus::knowhere::meta::DEVICEID, DEVICEID},
#endif
    };
    for (auto& [key, value] : configs.items()) {
        auto param = index_params.add_params();
        auto value_str = value.is_string() ? value.get<std::string>() : value.dump();
        param->set_key(key);
        param->set_value(value_str);
    }

    auto param = index_params.add_params();
    param->set_key("index_type");
    param->set_value(std::string(index_type));

    return std::make_tuple(type_params, index_params);
}

auto
ivfsq8_generate_params(const milvus::knowhere::IndexType& index_type,
                       const milvus::knowhere::MetricType& metric_type,
                       int64_t nlist,
                       int64_t nbits,
                       int64_t nprobe = 4) {
    namespace indexcgo = milvus::proto::indexcgo;

    assert(index_type == milvus::knowhere::IndexEnum::INDEX_FAISS_IVFSQ8);
    indexcgo::TypeParams type_params;
    indexcgo::IndexParams index_params;
    auto configs = milvus::knowhere::Config{
        {milvus::knowhere::meta::DIM, DIM},
        {milvus::knowhere::meta::TOPK, K},
        {milvus::knowhere::IndexParams::nlist, nlist},
        {milvus::knowhere::IndexParams::nprobe, nprobe},
        {milvus::knowhere::IndexParams::nbits, nbits},
        {milvus::knowhere::Metric::TYPE, metric_type},
        {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
#ifdef MILVUS_GPU_VERSION
        {milvus::knowhere::meta::DEVICEID, DEVICEID},
#endif
    };

    for (auto& [key, value] : configs.items()) {
        auto param = index_params.add_params();
        auto value_str = value.is_string() ? value.get<std::string>() : value.dump();
        param->set_key(key);
        param->set_value(value_str);
    }

    auto param = index_params.add_params();
    param->set_key("index_type");
    param->set_value(std::string(index_type));

    return std::make_tuple(type_params, index_params);
}
auto
ivfpq_generate_params(const milvus::knowhere::IndexType& index_type,
                      const milvus::knowhere::MetricType& metric_type,
                      int64_t nlist,
                      int64_t nbits,
                      int64_t m,
                      int64_t nprobe = 4) {
    namespace indexcgo = milvus::proto::indexcgo;

    assert(index_type == milvus::knowhere::IndexEnum::INDEX_FAISS_IVFSQ8);
    indexcgo::TypeParams type_params;
    indexcgo::IndexParams index_params;
    auto configs = milvus::knowhere::Config{
        {milvus::knowhere::meta::DIM, DIM},
        {milvus::knowhere::meta::TOPK, K},
        {milvus::knowhere::IndexParams::nlist, nlist},
        {milvus::knowhere::IndexParams::nprobe, nprobe},
        {milvus::knowhere::IndexParams::m, m},
        {milvus::knowhere::IndexParams::nbits, nbits},
        {milvus::knowhere::Metric::TYPE, metric_type},
        {milvus::knowhere::INDEX_FILE_SLICE_SIZE_IN_MEGABYTE, 4},
    };

    for (auto& [key, value] : configs.items()) {
        auto param = index_params.add_params();
        auto value_str = value.is_string() ? value.get<std::string>() : value.dump();
        param->set_key(key);
        param->set_value(value_str);
    }

    auto param = index_params.add_params();
    param->set_key("index_type");
    param->set_value(std::string(index_type));

    return std::make_tuple(type_params, index_params);
}
}  // namespace
