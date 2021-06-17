#!/bin/bash

/home/jigao/Desktop/milvus/internal/core/cmake-build-release/bench/bench_parameter_tuning_ivfflat --benchmark_counters_tabular=true --benchmark_out=/home/jigao/Desktop/bench_parameter_tuning_ivfflat_0511.csv --benchmark_out_format=csv

/home/jigao/Desktop/milvus/internal/core/cmake-build-release/bench/bench_parameter_tuning_ivfsq8 --benchmark_counters_tabular=true --benchmark_out=/home/jigao/Desktop/bench_parameter_tuning_ivfsq8_0511.csv --benchmark_out_format=csv

/home/jigao/Desktop/milvus/internal/core/cmake-build-release/bench/bench_parameter_tuning_ivfpq --benchmark_counters_tabular=true --benchmark_out=/home/jigao/Desktop/bench_parameter_tuning_ivfpq_0610.csv --benchmark_out_format=csv

# TODO: a speedup for google benchmark
