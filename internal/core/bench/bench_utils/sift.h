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

float*
fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
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

void
train(double start_time, milvus::knowhere::IndexType index_type,
    size_t& d, size_t& nt, milvus::knowhere::VecIndexPtr& index, milvus::knowhere::Config& conf) {
    printf("[%.3f s] Loading train set\n", elapsed() - start_time);
    float* xb_data = fvecs_read(sift_path + "sift_learn.fvecs", &d, &nt);

    printf("[%.3f s] Preparing index \"%s\" d=%ld\n",
           elapsed() - start_time,
           index_type.c_str(),
           d);

    printf("[%.3f s] Training on %ld vectors\n", elapsed() - start_time, nt);
    auto xt_dataset = milvus::knowhere::GenDataset(nt, d, static_cast<const void*>(xb_data));

    // Fix the conf. with dimension from file.
    conf[milvus::knowhere::meta::DIM] = d;

    index->Train(xt_dataset, conf);

    delete[] xb_data;
}

void
add_points(double start_time, milvus::knowhere::IndexType index_type,
           size_t d, size_t& nb, milvus::knowhere::VecIndexPtr& index, milvus::knowhere::Config& conf) {
    printf("[%.3f s] Loading database\n", elapsed() - start_time);

    size_t d2;
    float* xb = fvecs_read(sift_path + "sift_base.fvecs", &d2, &nb);
    assert(d == d2 || !"dataset does not have same dimension as train set");

    printf("[%.3f s] Indexing database, size %ld*%ld\n",
           elapsed() - start_time,
           nb,
           d);

    auto xb_dataset = milvus::knowhere::GenDataset(nb, d, static_cast<const void*>(xb));
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

    delete[] xb;
}

auto
load_queries(double start_time,
             size_t d, size_t& nq) {
    printf("[%.3f s] Loading queries\n", elapsed() - start_time);

    size_t d2;
    float* xq = fvecs_read(sift_path + "sift_query.fvecs", &d2, &nq);
    auto xq_dataset = milvus::knowhere::GenDataset(nq, d, static_cast<const void*>(xq));
    assert(d == d2 || !"query does not have same dimension as train set");
    return xq_dataset;
}

auto
load_ground_truth(double start_time, size_t nq, size_t& k, int64_t*& gt, milvus::knowhere::Config& conf) {
    printf("[%.3f s] Loading ground truth for %ld queries\n",
           elapsed() - start_time,
           nq);

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
compute_recall (double start_time,
                size_t nq, size_t k, milvus::knowhere::DatasetPtr& result, int64_t*& gt, size_t num_to_include) {
    printf("[%.3f s] Compute recalls@%zu\n", elapsed() - start_time, num_to_include);

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
        it=std::set_intersection (gt_vec.begin(), gt_vec.end(), ids_vec.begin(), ids_vec.end(), v.begin());
        v.resize(it - v.begin());
        hits += v.size();
    }
    return hits * 1.0 / (nq * num_to_include);


//    int n_1 = 0, n_10 = 0, n_100 = 0;
//    size_t hits = 0;
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
    free((void *) tensor);
}


void
ReleaseQueryResult(const milvus::knowhere::DatasetPtr& result) {
    float* res_dist = result->Get<float*>(milvus::knowhere::meta::DISTANCE);
    free(res_dist);

    int64_t* res_ids = result->Get<int64_t*>(milvus::knowhere::meta::IDS);
    free(res_ids);
}
