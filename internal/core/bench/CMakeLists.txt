# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License

include_directories(${CMAKE_HOME_DIRECTORY}/src)
include_directories(${CMAKE_HOME_DIRECTORY}/unittest)
include_directories(${CMAKE_HOME_DIRECTORY}/src/index/knowhere)
include_directories(${CMAKE_HOME_DIRECTORY}/bench)

set(bench_srcs 
    bench_naive.cpp
    bench_search.cpp
)

set(indexbuilder_bench_srcs
        bench_indexbuilder.cpp
)

set(bench_parameter_tuning_flat_srcs
        bench_utils/parameter_tuning_utils.h
        bench_utils/sift.h
        bench_utils/utils.h
        parameter-tuning/bench_flat.cpp
)

set(bench_parameter_tuning_ivfflat_srcs
        bench_utils/parameter_tuning_utils.h
        bench_utils/sift.h
        bench_utils/utils.h
        parameter-tuning/bench_ivfflat.cpp
)

set(bench_parameter_tuning_ivfsq8_srcs
        bench_utils/parameter_tuning_utils.h
        bench_utils/sift.h
        bench_utils/utils.h
        parameter-tuning/bench_ivfsq8.cpp
)

set(bench_parameter_tuning_ivfpq_srcs
        bench_utils/parameter_tuning_utils.h
        bench_utils/sift.h
        bench_utils/utils.h
        parameter-tuning/bench_ivfpq.cpp
)

add_executable(all_bench ${bench_srcs})
target_link_libraries(all_bench
        milvus_segcore
        log
        pthread
        )

target_link_libraries(all_bench benchmark::benchmark_main)

add_executable(indexbuilder_bench ${indexbuilder_bench_srcs})
target_link_libraries(indexbuilder_bench
        milvus_segcore
        milvus_indexbuilder
        log
        pthread
        )

target_link_libraries(indexbuilder_bench benchmark::benchmark_main)

add_executable(bench_parameter_tuning_flat ${bench_parameter_tuning_flat_srcs} ${util_srcs})
target_link_libraries(bench_parameter_tuning_flat
        milvus_segcore
        milvus_indexbuilder
        log
        pthread
        )

target_link_libraries(bench_parameter_tuning_flat benchmark::benchmark_main)

add_executable(bench_parameter_tuning_ivfflat ${bench_parameter_tuning_ivfflat_srcs})
target_link_libraries(bench_parameter_tuning_ivfflat
        milvus_segcore
        milvus_indexbuilder
        log
        pthread
        )

target_link_libraries(bench_parameter_tuning_ivfflat benchmark::benchmark_main)

add_executable(bench_parameter_tuning_ivfsq8 ${bench_parameter_tuning_ivfsq8_srcs})
target_link_libraries(bench_parameter_tuning_ivfsq8
        milvus_segcore
        milvus_indexbuilder
        log
        pthread
        )

target_link_libraries(bench_parameter_tuning_ivfsq8 benchmark::benchmark_main)

add_executable(bench_parameter_tuning_ivfpq ${bench_parameter_tuning_ivfpq_srcs})
target_link_libraries(bench_parameter_tuning_ivfpq
        milvus_segcore
        milvus_indexbuilder
        log
        pthread
        )

target_link_libraries(bench_parameter_tuning_ivfpq benchmark::benchmark_main)