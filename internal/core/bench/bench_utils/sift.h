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

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>

#include "bench_utils/utils.h"

// TODO(jigao): Fixme
/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the subdirectory sift.
 **/

const std::string sift_path = "/home/jigao/Desktop/sift/";

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

// TODO(jigao) set this one!
int64_t SIFT_DIM = 128;

float*
fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    size_t d_read = fread(&d, 1, sizeof(int), f);
    assert(d_read == sizeof(int));
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    SIFT_DIM = d;
    if (d != SIFT_DIM) {
        std::cerr << "DIM is wrong, DIM := " << SIFT_DIM << ", d :=" << d << std::endl;
    }
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

float*
fvecs_read(std::string fname, size_t* d_out, size_t* n_out) {
    return fvecs_read(fname.c_str(), d_out, n_out);
}


// not very clean, but works as long as sizeof(int) == sizeof(float)
int*
ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

int*
ivecs_read(std::string fname, size_t* d_out, size_t* n_out) {
    return ivecs_read(fname.c_str(), d_out, n_out);
}

double
elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

std::tuple<double, double, double>
Cal_Min_Max_Avg(const std::vector<double>& vec) {
    double min{std::numeric_limits<double>::max()}, max{0}, sum{0};
    for (const auto ele : vec){
        assert(ele >= 0 && "Time should be larger than 0s.");
        sum += ele;
        if (max < ele){
            max = ele;
        }
        if (min > ele){
            min = ele;
        }
    }
    double avg = sum / vec.size();
    return {min, max, avg};
}

std::tuple<double, double, double>
train(size_t& d, size_t& nt, milvus::knowhere::VecIndexPtr& index, milvus::knowhere::Config& conf) {
    float* xb_data = fvecs_read(sift_path + "sift_learn.fvecs", &d, &nt);
    auto xt_dataset = milvus::knowhere::GenDataset(nt, d, static_cast<const void*>(xb_data));
    assert(d == 128);

    // Fix the conf. with dimension from file.
    conf[milvus::knowhere::meta::DIM] = d;

    std::vector<double> vec;
    {
        auto time_n_train = [&](size_t n) {
            for (size_t i = 0; i < n; i++) {
                Timer t;
//                index->Train(xt_dataset, conf);
                double duration = t.elapsed_time<double, std::chrono::milliseconds>();
                std::cout << "[Benchmark] index->Train: " << duration << " ms (milliseconds) on " << nt << " vectors" << std::endl;
//                index->AddWithoutIds(xt_dataset, conf);
                vec.push_back(duration);
            }
        };

index->Train(xt_dataset, conf);
index->AddWithoutIds(xt_dataset, conf);

        time_n_train(1);
        assert(!vec.empty());

//        if (vec.front() < 1000) {
//            // If less than 1000ms (1s), iterate 100 times (together 100s).
//            std::cout << "[Benchmark] index->Train less than 1000ms (1s), iterate 100 times (together 100s)." << std::endl;
//            time_n_train(100);
//        } else if (vec.front() < 5000) {
//            // If less than 5000ms (5s), iterate 20 times (together 100s).
//            std::cout << "[Benchmark] index->Train less than 5000ms (5s), iterate 20 times (together 100s)." << std::endl;
//            time_n_train(20);
//        } else if (vec.front() < 10000) {
//            // If less than 10000ms (10s), iterate 10 times (together 100s).
//            std::cout << "[Benchmark] index->Train less than 10000ms (10s), iterate 10 times (together 10s)." << std::endl;
//            time_n_train(10);
//        } else if (vec.front() < 10000) {
//            // If less than 100000ms (100s), iterate 2 times.
//            std::cout << "[Benchmark] index->Train less than 100000ms (100s), iterate 2 times." << std::endl;
//            time_n_train(2);
//        } else {
//            // If more than 100000ms (100s), iterate NO times.
//            std::cout << "[Benchmark] index->Train more than 100000ms (100s), iterate NO times." << std::endl;
//        }
    }
    delete[] xb_data;
    return Cal_Min_Max_Avg(vec);
}

std::tuple<double, double, double>
add_points(milvus::knowhere::IndexType index_type,
           size_t d, size_t& nb, milvus::knowhere::VecIndexPtr& index, milvus::knowhere::Config& conf, size_t nt) {
    size_t d2;
float* xb = fvecs_read(sift_path + "sift_learn.fvecs", &d2, &nb);
auto xb_dataset = milvus::knowhere::GenDataset(nt, d, static_cast<const void*>(xb));

//    float* xb = fvecs_read(sift_path + "sift_base.fvecs", &d2, &nb);
    assert(d == d2 || !"dataset does not have same dimension as train set");

//    auto xb_dataset = milvus::knowhere::GenDataset(nb, d, static_cast<const void*>(xb));
    double duration{0};

    std::vector<double> vec;
    Timer t;
    index->AddWithoutIds(xb_dataset, conf);
    if (is_in_nm_list(index_type)) {
        milvus::knowhere::BinarySet bs = index->Serialize(conf);
        int64_t dim = xb_dataset->Get<int64_t>(milvus::knowhere::meta::DIM);
        int64_t rows = xb_dataset->Get<int64_t>(milvus::knowhere::meta::ROWS);
        auto raw_data = xb_dataset->Get<const void*>(milvus::knowhere::meta::TENSOR);

        milvus::knowhere::BinaryPtr bptr = std::make_shared<milvus::knowhere::Binary>();
        bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});  // avoid repeated deconstruction
        bptr->size = dim * rows * sizeof(float);
        bs.Append(RAW_DATA, bptr);
        index->Load(bs);
    }
    duration = t.elapsed_time<unsigned int, std::chrono::milliseconds>();
    std::cout << "index->AddWithoutIds: " << duration << " ms (milliseconds) on " << nb << " vectors" << std::endl;
    vec.push_back(duration);
//    {
//        auto time_1_add_points = [&]() {
//            Timer t;
//            index->AddWithoutIds(xb_dataset, conf);
//            if (is_in_nm_list(index_type)) {
//                milvus::knowhere::BinarySet bs = index->Serialize(conf);
//                int64_t dim = xb_dataset->Get<int64_t>(milvus::knowhere::meta::DIM);
//                int64_t rows = xb_dataset->Get<int64_t>(milvus::knowhere::meta::ROWS);
//                auto raw_data = xb_dataset->Get<const void*>(milvus::knowhere::meta::TENSOR);
//
//                milvus::knowhere::BinaryPtr bptr = std::make_shared<milvus::knowhere::Binary>();
//                bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});  // avoid repeated deconstruction
//                bptr->size = dim * rows * sizeof(float);
//                bs.Append(RAW_DATA, bptr);
//                index->Load(bs);
//            }
//            duration = t.elapsed_time<unsigned int, std::chrono::milliseconds>();
//            std::cout << "index->AddWithoutIds: " << duration << " ms (milliseconds) on " << nb << " vectors" << std::endl;
//            vec.push_back(duration);
//        };

        auto time_n_add_points = [&](size_t n) {
            for (size_t i = 0; i < n; i++) {
                // Need to re-train a index.
                milvus::knowhere::VecIndexPtr index_ = milvus::knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type);
                float* xb_data = fvecs_read(sift_path + "sift_learn.fvecs", &d, &nt);
                auto xt_dataset = milvus::knowhere::GenDataset(nt, d, static_cast<const void*>(xb_data));
                index_->Train(xt_dataset, conf);
                delete[] xb_data;

                Timer t;
                index_->AddWithoutIds(xb_dataset, conf);
                if (is_in_nm_list(index_type)) {
                    milvus::knowhere::BinarySet bs = index->Serialize(conf);
                    int64_t dim = xb_dataset->Get<int64_t>(milvus::knowhere::meta::DIM);
                    int64_t rows = xb_dataset->Get<int64_t>(milvus::knowhere::meta::ROWS);
                    auto raw_data = xb_dataset->Get<const void*>(milvus::knowhere::meta::TENSOR);

                    milvus::knowhere::BinaryPtr bptr = std::make_shared<milvus::knowhere::Binary>();
                    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});  // avoid repeated deconstruction
                    bptr->size = dim * rows * sizeof(float);
                    bs.Append(RAW_DATA, bptr);
                    index_->Load(bs);
                }
                duration = t.elapsed_time<unsigned int, std::chrono::milliseconds>();
                std::cout << "index->AddWithoutIds: " << duration << " ms (milliseconds) on " << nb << " vectors" << std::endl;
                vec.push_back(duration);
            }
        };

//        time_1_add_points();
        assert(!vec.empty());
        assert(!vec.empty());

//        if (vec.front() < 1000) {
//            // If less than 1000ms (1s), iterate 100 times (together 100s).
//            std::cout << "[Benchmark] index->AddWithoutIds less than 1000ms (1s), iterate 100 times (together 100s)." << std::endl;
//            time_n_add_points(100);
//        } else if (vec.front() < 5000) {
//            // If less than 5000ms (5s), iterate 20 times (together 100s).
//            std::cout << "[Benchmark] index->AddWithoutIds less than 5000ms (5s), iterate 20 times (together 100s)." << std::endl;
//            time_n_add_points(20);
//        } else if (vec.front() < 10000) {
//            // If less than 10000ms (10s), iterate 10 times (together 100s).
//            std::cout << "[Benchmark] index->AddWithoutIds less than 10000ms (10s), iterate 10 times (together 10s)." << std::endl;
//            time_n_add_points(10);
//        } else if (vec.front() < 100000) {
//            // If less than 100000ms (100s), iterate 2 times.
//            std::cout << "[Benchmark] index->AddWithoutIds more than 10000ms (10s), iterate 2 times." << std::endl;
//            time_n_add_points(2);
//        } else {
//            // If more than 100000ms (100s), iterate NO times.
//            std::cout << "[Benchmark] index->AddWithoutIds more than 100000ms (100s), iterate NO times." << std::endl;
//        }
//    }
    delete[] xb;
    return Cal_Min_Max_Avg(vec);
}

auto
load_queries(size_t d, size_t& nq) {
    std::cout << "[Benchmark] Loading queries" << std::endl;
    size_t d2;
    float* xq = fvecs_read(sift_path + "sift_query.fvecs", &d2, &nq);
    auto xq_dataset = milvus::knowhere::GenDataset(nq, d, static_cast<const void*>(xq));
    assert(d == d2 || !"query does not have same dimension as train set");
    return xq_dataset;
}

auto
load_ground_truth(size_t nq, size_t& k, int64_t*& gt, milvus::knowhere::Config& conf) {
    std::cout << "[Benchmark] Loading ground truth for " << nq << " queries" << std::endl;
    // load ground-truth and convert int to long
    size_t nq2;
    int* gt_int = ivecs_read(sift_path + "sift_groundtruth.ivecs", &k, &nq2);
    assert(nq2 == nq || !"incorrect nb of ground truth entries");

    // Fix the conf. with k from file.
    conf[milvus::knowhere::meta::TOPK] = k;

    gt = new int64_t[k * nq];
    for (int i = 0; i < k * nq; i++) {
        gt[i] = gt_int[i];
    }
    delete[] gt_int;
}

double
compute_recall (size_t nq, size_t k, milvus::knowhere::DatasetPtr& result, int64_t*& gt, size_t num_to_include) {
    std::cout << "[Benchmark] Compute recalls@" << num_to_include << std::endl;

    // evaluate result by hand.
    auto ids = result->Get<int64_t*>(milvus::knowhere::meta::IDS);
    size_t hits = 0;
    for (int i = 0; i < nq; i++) {
        std::vector<int64_t> gt_vec(gt + i * k, gt + i * k + num_to_include);
        std::vector<int64_t> ids_vec(ids + i * k, ids + i * k + num_to_include);
        std::sort(gt_vec.begin(), gt_vec.end());
        std::sort(ids_vec.begin(), ids_vec.end());
        std::vector<int64_t> v(nq * 2);
        std::vector<int64_t>::iterator it;
        it=std::set_intersection(gt_vec.begin(), gt_vec.end(), ids_vec.begin(), ids_vec.end(), v.begin());
        v.resize(it - v.begin());
        hits += v.size();
    }
    return hits * 1.0 / (nq * num_to_include);


//    int n_1 = 0, n_10 = 0, n_100 = 0;
//    for (int i = 0; i < nq; i++) {
//        // The top 1 NN.
//        int gt_nn = gt[i * k];
//        for (int j = 0; j < k; j++) {
//            if (ids[i * k + j] == gt_nn) {
//                if (j < 1)
//                    n_1++;
//                if (j < 10)
//                    n_10++;
//                if (j < 100)
//                    n_100++;
//            }
//        }
//    }
//    printf("R@1 = %.4f\n", n_1 / float(nq));
//    printf("R@10 = %.4f\n", n_10 / float(nq));
//    printf("R@100 = %.4f\n", n_100 / float(nq));
//    double recall =  n_100 / float(nq);
//    printf("Recall = %.4f\n", recall);
//    return recall;
}

void
ReleaseQuery(const milvus::knowhere::DatasetPtr& xq_dataset) {
    const float* tensor = static_cast<const float*>(xq_dataset->Get<const void*>(milvus::knowhere::meta::TENSOR));
    delete[] tensor;
}


void
ReleaseQueryResult(const milvus::knowhere::DatasetPtr& result) {
    float* res_dist = result->Get<float*>(milvus::knowhere::meta::DISTANCE);
    free(res_dist);

    int64_t* res_ids = result->Get<int64_t*>(milvus::knowhere::meta::IDS);
    free(res_ids);
}
