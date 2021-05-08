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

std::vector<std::string>
NM_List() {
    static std::vector<std::string> ret{
            milvus::knowhere::IndexEnum::INDEX_FAISS_IVFFLAT,
            milvus::knowhere::IndexEnum::INDEX_NSG,
            milvus::knowhere::IndexEnum::INDEX_RHNSWFlat,
    };
    return ret;
}

template <typename T>
bool
is_in_list(const T& t, std::function<std::vector<T>()> list_func) {
    auto l = list_func();
    return std::find(l.begin(), l.end(), t) != l.end();
}

bool
is_in_nm_list(const milvus::knowhere::IndexType& index_type) {
    return is_in_list<std::string>(index_type, NM_List);
}

}  // namespace
