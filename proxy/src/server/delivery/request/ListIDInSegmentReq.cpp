// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "server/delivery/request/ListIDInSegmentReq.h"
#include "server/ValidationUtil.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"

#include <memory>
#include <vector>

namespace milvus {
namespace server {

ListIDInSegmentReq::ListIDInSegmentReq(const ContextPtr& context, const std::string& collection_name,
                                       int64_t segment_id, engine::IDNumbers& ids)
    : BaseReq(context, ReqType::kListIDInSegment),
      collection_name_(collection_name),
      segment_id_(segment_id),
      ids_(ids) {
}

BaseReqPtr
ListIDInSegmentReq::Create(const ContextPtr& context, const std::string& collection_name, int64_t segment_id,
                           engine::IDNumbers& ids) {
    return std::shared_ptr<BaseReq>(new ListIDInSegmentReq(context, collection_name, segment_id, ids));
}

Status
ListIDInSegmentReq::OnExecute() {


    return Status::OK();
}

}  // namespace server
}  // namespace milvus