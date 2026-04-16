#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <queue>
#include <utility>
#include <omp.h>
#include <chrono>
#include <stdint.h>
#include <array>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <cub/cub.cuh>
#include <thrust/gather.h>
#include <cublas_v2.h>


#include "brute_force.cuh"
#include "tools.cuh"

using namespace std ;

#define half __half
#define half2 __half2

#define HASHLEN 2048
#define HASHSIZE 11
#define HASH_RESET 4
#define queryK 10
#define MERGE_CELL_SIZE 6

// const unsigned TOPM = 40 , DIM = 128 , MAX_ITER = 1000 , DEGREE = 16 ;
const unsigned MAX_ITER = 10000 ;
__device__ __constant__ const float INF_DIS = 1e9f; 
constexpr unsigned EXTEND_POINTS_NUM = 128 ;
constexpr unsigned EXTEND_POINTS_STEP = EXTEND_POINTS_NUM / 2 ; 






__device__ __forceinline__ void swap_dis_and_id(float &a, float &b, unsigned &c, unsigned &d){
    const float t = a;
    a = b;
    b = t;
    const unsigned s = c;
    c = d;
    d = s;
}

__device__ __forceinline__ void swap_dis_and_id_u(unsigned &a, unsigned &b, unsigned &c, unsigned &d){
    const unsigned t = a;
    a = b;
    b = t;
    const unsigned s = c;
    c = d;
    d = s;
}

__device__ __forceinline__ void swap_dis_and_id_iu(int &a, int &b, unsigned &c, unsigned &d){
    const int t = a;
    a = b;
    b = t;
    const unsigned s = c;
    c = d;
    d = s;
}


__device__ __forceinline__ void bitonic_sort_id_by_dis_no_explore(float* shared_arr, unsigned* ids, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned stride = 1; stride < len; stride <<= 1){
        for(unsigned step = stride; step > 0; step >>= 1){
            for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                const unsigned a = 2 * step * (k / step);
                const unsigned b = k % step;
                const unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                const unsigned d = a + b + step;
                if(d < len && shared_arr[u] > shared_arr[d]){
                    // swap(shared_arr[u],shared_arr[d]);
                    // swap_ids(ids[u],ids[d]); 
                    swap_dis_and_id(shared_arr[u],shared_arr[d], ids[u],ids[d]);
                }
                // float lo = (shared_arr[u] > shared_arr[d] ? shared_arr[d] : shared_arr[u]) ;
                // float hi = (shared_arr[u] > shared_arr[d] ? shared_arr[u] : shared_arr[d]) ;
                // unsigned lo_v = (shared_arr[u] > shared_arr[d] ? ids[d] : ids[u]) ;
                // unsigned hi_v = (shared_arr[u] > shared_arr[d] ? ids[u] : ids[d]) ;

                // shared_arr[u] = lo , shared_arr[d] = hi , ids[u] = lo_v , ids[d] = hi_v ; 
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void bitonic_sort_id_by_num(unsigned* shared_arr, unsigned* ids, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned stride = 1; stride < len; stride <<= 1){
        for(unsigned step = stride; step > 0; step >>= 1){
            for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                const unsigned a = 2 * step * (k / step);
                const unsigned b = k % step;
                const unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                const unsigned d = a + b + step;
                if(d < len && shared_arr[u] < shared_arr[d]){
                    // swap(shared_arr[u],shared_arr[d]);
                    // swap_ids(ids[u],ids[d]);
                    swap_dis_and_id_u(shared_arr[u],shared_arr[d], ids[u],ids[d]);
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void bitonic_sort_id_by_num_int(int* shared_arr, unsigned* ids, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned stride = 1; stride < len; stride <<= 1){
        for(unsigned step = stride; step > 0; step >>= 1){
            for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                const unsigned a = 2 * step * (k / step);
                const unsigned b = k % step;
                const unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                const unsigned d = a + b + step;
                if(d < len && shared_arr[u] < shared_arr[d]){
                    // swap(shared_arr[u],shared_arr[d]);
                    // swap_ids(ids[u],ids[d]);
                    swap_dis_and_id_iu(shared_arr[u],shared_arr[d], ids[u],ids[d]);
                }
            }
            __syncthreads();
        }
    }
}


__device__ __forceinline__ void warpSort(float* key, unsigned* val, unsigned lane_id, unsigned len) {
    #pragma unroll
    for (uint32_t range = 1; range <= len; range <<= 1) {
        const uint32_t b = range;
        for (uint32_t c = b / 2; c >= 1; c >>= 1) {
            // 控制排序的顺序, b 位为0的为前半段, 按正序排, c 位为0的为前半段, 正序排, 当两位全0或全1时正序, 两位不同时逆序
            const auto p = (static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c));
            auto k1 = __shfl_xor_sync(__activemask(), *key, c);
            auto v1 = __shfl_xor_sync(__activemask(), *val, c);
            if ((*key != k1) && ((*key < k1) != p)) {
                *key = k1;
                *val = v1;
            }
        }
    }
}


__device__ __forceinline__ void merge_top(unsigned* arr1, unsigned* arr2, float* arr1_val, float* arr2_val, unsigned tid , unsigned DEGREE , unsigned TOPM){
    // 6 == 
    // unsigned res_id_vec[(TOPM + K+6*32-1)/ (6*32)] = {0};
    const unsigned thread_num_per_block = blockDim.x * blockDim.y ;
    const unsigned task_num_per_thread = (TOPM + DEGREE+ thread_num_per_block -1)/ (thread_num_per_block) ;
    unsigned res_id_vec[10] = {0} ;
    float val_vec[10];
    unsigned id_reg_vec[10];
    for(unsigned i = 0; i < task_num_per_thread; i ++){
        // unsigned res_id_vec[i] = 0;
        // float val;
        // if(i * blockDim.x * blockDim.y + tid >= TOPM + DEGREE)
        //     break ;

        if(i * blockDim.x * blockDim.y + tid < TOPM){
            val_vec[i] = arr1_val[i * blockDim.x * blockDim.y + tid];
            id_reg_vec[i] = arr1[i * blockDim.x * blockDim.y + tid];
            unsigned tmp = DEGREE;
            // 找到当前元素的插入位置, 判断后半段数组比自己大的元素数量
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = arr2_val[res_id_vec[i] + halfsize];
                // 说明此时有至少halfsize个元素比自己大
                res_id_vec[i] += ((cand <= val_vec[i]) ? halfsize : 0);
                tmp -= halfsize;
            }
            // 和卡在位置上的元素比较大小
            res_id_vec[i] += (arr2_val[res_id_vec[i]] <= val_vec[i]);
            // 自己在原列表中的位置为 i * blockDim.x * blockDim.y + tid
            res_id_vec[i] += (i * blockDim.x * blockDim.y + tid);
        }
        else if(i * blockDim.x * blockDim.y + tid < TOPM + DEGREE){
            // 同理, 后半段列表中的元素查找自己在前半段列表中的位置
            val_vec[i] = arr2_val[i * blockDim.x * blockDim.y + tid - TOPM];
            id_reg_vec[i] = arr2[i * blockDim.x * blockDim.y + tid - TOPM];
            unsigned tmp = TOPM;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = arr1_val[res_id_vec[i] + halfsize];
                res_id_vec[i] += ((cand < val_vec[i]) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id_vec[i] += (arr1_val[res_id_vec[i]] < val_vec[i]);
            res_id_vec[i] += (i * blockDim.x * blockDim.y + tid - TOPM);
        }
        else{
            res_id_vec[i] = TOPM + DEGREE + 1;
        }
    }
    __syncthreads();
    for(unsigned i = 0; i < task_num_per_thread; i ++){
        if(res_id_vec[i] < TOPM){
            arr1[res_id_vec[i]] = id_reg_vec[i];
            arr1_val[res_id_vec[i]] = val_vec[i];
        } else if(res_id_vec[i] < TOPM + DEGREE) {
            arr2[res_id_vec[i] - TOPM] = id_reg_vec[i] ;
            arr2_val[res_id_vec[i] - TOPM] = val_vec[i] ;
        }
    }
    __syncthreads();
}

template<typename attrType = float>
__device__ __forceinline__ void merge_top_with_recycle_list(unsigned* arr1, unsigned* arr2, float* arr1_val, float* arr2_val, unsigned tid , unsigned DEGREE , 
    unsigned TOPM ,const attrType* l_bound ,const attrType* r_bound ,const attrType* attrs , unsigned attr_dim){
        // 本身传入的为global id 无需再次映射
    // 6 == 
    // unsigned res_id_vec[(TOPM + K+6*32-1)/ (6*32)] = {0};
    const unsigned thread_num_per_block = blockDim.x * blockDim.y ;
    const unsigned task_num_per_thread = (TOPM + DEGREE+ thread_num_per_block -1)/ (thread_num_per_block) ;
    unsigned res_id_vec[10] = {0} ;
    float val_vec[10];
    unsigned id_reg_vec[10];
    for(unsigned i = 0; i < task_num_per_thread; i ++){
        // unsigned res_id_vec[i] = 0;
        // float val;
        // if(i * blockDim.x * blockDim.y + tid >= TOPM + DEGREE)
        //     break ;

        if(i * blockDim.x * blockDim.y + tid < TOPM){
            val_vec[i] = arr1_val[i * blockDim.x * blockDim.y + tid];
            id_reg_vec[i] = arr1[i * blockDim.x * blockDim.y + tid];
            unsigned tmp = DEGREE;
            // 找到当前元素的插入位置, 判断后半段数组比自己大的元素数量
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = arr2_val[res_id_vec[i] + halfsize];
                // 说明此时有至少halfsize个元素比自己大
                res_id_vec[i] += ((cand <= val_vec[i]) ? halfsize : 0);
                tmp -= halfsize;
            }
            // 和卡在位置上的元素比较大小
            res_id_vec[i] += (arr2_val[res_id_vec[i]] <= val_vec[i]);
            // 自己在原列表中的位置为 i * blockDim.x * blockDim.y + tid
            res_id_vec[i] += (i * blockDim.x * blockDim.y + tid);

            if(res_id_vec[i] >= TOPM && id_reg_vec[i] != 0xFFFFFFFF) {
                // arr1_val[i * blockDim.x * blockDim.y + tid]
                // val_vec[i] = INF ;
                unsigned pure_id = id_reg_vec[i] & 0x3FFFFFFF ;
                // if(pure_id >= 1000000)
                //     printf("error: %d!!!!!!\n" , pure_id) ;
                // unsigned global_id = global_idx[pure_id + graph_offset] ;
                unsigned global_id = pure_id ;
                bool flag = true ;
                for(unsigned j = 0 ; j < attr_dim && flag ; j  ++) {
                    flag = (flag && (attrs[global_id * attr_dim + j] >= l_bound[j])
                     && (attrs[global_id * attr_dim + j] <= r_bound[j]))  ;
                }
                // 如果该点不满足属性过滤条件, 则将值标记为INF, ID 标记为全1
                if(!flag) {
                    val_vec[i] = INF_DIS ; 
                    id_reg_vec[i] = 0xFFFFFFFF ;
                }
            }   
        }
        else if(i * blockDim.x * blockDim.y + tid < TOPM + DEGREE){
            // 同理, 后半段列表中的元素查找自己在前半段列表中的位置
            val_vec[i] = arr2_val[i * blockDim.x * blockDim.y + tid - TOPM];
            id_reg_vec[i] = arr2[i * blockDim.x * blockDim.y + tid - TOPM];
            unsigned tmp = TOPM;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = arr1_val[res_id_vec[i] + halfsize];
                res_id_vec[i] += ((cand < val_vec[i]) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id_vec[i] += (arr1_val[res_id_vec[i]] < val_vec[i]);
            res_id_vec[i] += (i * blockDim.x * blockDim.y + tid - TOPM);
            if(res_id_vec[i] >= TOPM) {
                val_vec[i] = INF_DIS ; 
                id_reg_vec[i] = 0xFFFFFFFF ;
            }
        }
        else{
            res_id_vec[i] = TOPM + DEGREE + 1;
        }
    }
    __syncthreads();
    for(unsigned i = 0; i < task_num_per_thread; i ++){
        if(res_id_vec[i] < TOPM){
            arr1[res_id_vec[i]] = id_reg_vec[i];
            arr1_val[res_id_vec[i]] = val_vec[i];
            // printf("merge : res_id_vec[%d]-%d , id_reg_vec[%d]-%d , val_vec[%d]-%f\n" , i , res_id_vec[i] , i , 
            // (id_reg_vec[i] != 0xFFFFFFFF ? (id_reg_vec[i] & 0x7FFFFFFF) : id_reg_vec[i]) , i , val_vec[i]) ;
        } else if(res_id_vec[i] < TOPM + DEGREE) {
            arr2[res_id_vec[i] - TOPM] = id_reg_vec[i] ;
            arr2_val[res_id_vec[i] - TOPM] = val_vec[i] ;
            // printf("merge : res_id_vec[%d]-%d , id_reg_vec[%d]-%d , val_vec[%d]-%f\n" , i , res_id_vec[i] , i ,
            //  (id_reg_vec[i] != 0xFFFFFFFF ? (id_reg_vec[i] & 0x7FFFFFFF) : id_reg_vec[i]), i , val_vec[i]) ;
        }
    }
    __syncthreads();
}


/**
__device__ __forceinline__ unsigned rank_in_desc(
    const float* __restrict__ arr, unsigned n, float v, bool mode_le)
{
    unsigned cnt = 0;
    for (unsigned k = 0; k < n; ++k) {
        float cand = arr[k];
        cnt += (mode_le ? (cand <= v) : (cand < v));
    }
    return cnt;
}

__device__ __forceinline__ void merge_top(
    unsigned* arr1, unsigned* arr2,
    float* arr1_val, float* arr2_val,
    unsigned tid, unsigned DEGREE, unsigned TOPM)
{
    const unsigned tpb   = blockDim.x * blockDim.y;
    const unsigned total = TOPM + DEGREE;

    // 小批次，降低寄存器；保留“两阶段”防止读写冲突
    constexpr unsigned TILE = 2;

    for (unsigned base = tid; base < total; base += tpb * TILE) {
        unsigned pos[TILE];
        unsigned idv[TILE];
        float    vv[TILE];

        //#pragma unroll
        for (unsigned t = 0; t < TILE; ++t) {
            pos[t] = TOPM + DEGREE + 1u;
            idv[t] = 0xFFFFFFFFu;
            vv[t]  = INF_DIS;

            unsigned idx = base + t * tpb;
            if (idx >= total) continue;

            if (idx < TOPM) {
                float v = arr1_val[idx];
                unsigned id = arr1[idx];
                unsigned rank = rank_in_desc(arr2_val, DEGREE, v, true);
                pos[t] = idx + rank;
                idv[t] = id;
                vv[t]  = v;
            } else {
                unsigned j = idx - TOPM;
                float v = arr2_val[j];
                unsigned id = arr2[j];
                unsigned rank = rank_in_desc(arr1_val, TOPM, v, false);
                pos[t] = j + rank;
                idv[t] = id;
                vv[t]  = v;
            }
        }

        __syncthreads(); // 确保全体线程都完成“读+算位”

        #pragma unroll
        for (unsigned t = 0; t < TILE; ++t) {
            unsigned p = pos[t];
            if (p < TOPM) {
                arr1[p] = idv[t];
                arr1_val[p] = vv[t];
            } else if (p < TOPM + DEGREE) {
                unsigned q = p - TOPM;
                arr2[q] = idv[t];
                arr2_val[q] = vv[t];
            }
        }

        __syncthreads(); // 当前 tile 写完再进入下一 tile
    }
}

template<typename attrType = float>
__device__ __forceinline__ void merge_top_with_recycle_list(
    unsigned* arr1, unsigned* arr2,
    float* arr1_val, float* arr2_val,
    unsigned tid, unsigned DEGREE, unsigned TOPM,
    const attrType* l_bound, const attrType* r_bound,
    const attrType* attrs, unsigned attr_dim)
{
    const unsigned tpb   = blockDim.x * blockDim.y;
    const unsigned total = TOPM + DEGREE;

    constexpr unsigned TILE = 2;

    for (unsigned base = tid; base < total; base += tpb * TILE) {
        unsigned pos[TILE];
        unsigned idv[TILE];
        float    vv[TILE];

        //#pragma unroll
        for (unsigned t = 0; t < TILE; ++t) {
            pos[t] = TOPM + DEGREE + 1u;
            idv[t] = 0xFFFFFFFFu;
            vv[t]  = INF_DIS;

            unsigned idx = base + t * tpb;
            if (idx >= total) continue;

            if (idx < TOPM) {
                float v = arr1_val[idx];
                unsigned id = arr1[idx];
                unsigned rank = rank_in_desc(arr2_val, DEGREE, v, true);
                unsigned out_pos = idx + rank;

                if (out_pos >= TOPM && id != 0xFFFFFFFFu) {
                    unsigned pure_id = id & 0x3FFFFFFFu; // 保持你原掩码
                    unsigned global_id = pure_id;

                    bool pass = true;
                    for (unsigned d = 0; d < attr_dim && pass; ++d) {
                        attrType a = attrs[global_id * attr_dim + d];
                        pass = (a >= l_bound[d]) && (a <= r_bound[d]);
                    }
                    if (!pass) {
                        v = INF_DIS;
                        id = 0xFFFFFFFFu;
                    }
                }

                pos[t] = out_pos;
                idv[t] = id;
                vv[t]  = v;
            } else {
                unsigned j = idx - TOPM;
                float v = arr2_val[j];
                unsigned id = arr2[j];
                unsigned rank = rank_in_desc(arr1_val, TOPM, v, false);
                unsigned out_pos = j + rank;

                if (out_pos >= TOPM) {
                    v = INF_DIS;
                    id = 0xFFFFFFFFu;
                }

                pos[t] = out_pos;
                idv[t] = id;
                vv[t]  = v;
            }
        }

        __syncthreads(); // 先完成本 tile 的全部读取与排名

        #pragma unroll
        for (unsigned t = 0; t < TILE; ++t) {
            unsigned p = pos[t];
            if (p < TOPM) {
                arr1[p] = idv[t];
                arr1_val[p] = vv[t];
            } else if (p < TOPM + DEGREE) {
                unsigned q = p - TOPM;
                arr2[q] = idv[t];
                arr2_val[q] = vv[t];
            }
        }

        __syncthreads();
    }
}
**/

template<typename attrType = float>
__device__ __forceinline__ void merge_top_with_recycle_list(unsigned* arr1, unsigned* arr2, float* arr1_val, float* arr2_val, unsigned tid , unsigned DEGREE , 
    unsigned TOPM ,const attrType* l_bound ,const attrType* r_bound ,const attrType* attrs , unsigned attr_dim , const unsigned OP_TYPE){
        // 本身传入的为global id 无需再次映射
    // 6 == 
    // unsigned res_id_vec[(TOPM + K+6*32-1)/ (6*32)] = {0};
    const unsigned thread_num_per_block = blockDim.x * blockDim.y ;
    const unsigned task_num_per_thread = (TOPM + DEGREE+ thread_num_per_block -1)/ (thread_num_per_block) ;
    unsigned res_id_vec[10] = {0} ;
    float val_vec[10];
    unsigned id_reg_vec[10];
    for(unsigned i = 0; i < task_num_per_thread; i ++){
        // unsigned res_id_vec[i] = 0;
        // float val;
        // if(i * blockDim.x * blockDim.y + tid >= TOPM + DEGREE)
        //     break ;

        if(i * blockDim.x * blockDim.y + tid < TOPM){
            val_vec[i] = arr1_val[i * blockDim.x * blockDim.y + tid];
            id_reg_vec[i] = arr1[i * blockDim.x * blockDim.y + tid];
            unsigned tmp = DEGREE;
            // 找到当前元素的插入位置, 判断后半段数组比自己大的元素数量
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = arr2_val[res_id_vec[i] + halfsize];
                // 说明此时有至少halfsize个元素比自己大
                res_id_vec[i] += ((cand <= val_vec[i]) ? halfsize : 0);
                tmp -= halfsize;
            }
            // 和卡在位置上的元素比较大小
            res_id_vec[i] += (arr2_val[res_id_vec[i]] <= val_vec[i]);
            // 自己在原列表中的位置为 i * blockDim.x * blockDim.y + tid
            res_id_vec[i] += (i * blockDim.x * blockDim.y + tid);

            if(res_id_vec[i] >= TOPM && id_reg_vec[i] != 0xFFFFFFFF) {
                // arr1_val[i * blockDim.x * blockDim.y + tid]
                // val_vec[i] = INF ;
                unsigned pure_id = id_reg_vec[i] & 0x3FFFFFFF ;
                // if(pure_id >= 1000000)
                //     printf("error: %d!!!!!!\n" , pure_id) ;
                // unsigned global_id = global_idx[pure_id + graph_offset] ;
                unsigned global_id = pure_id ;
                bool flag = (OP_TYPE == 0 ? true : false) ;
                if(OP_TYPE == 0) {
                    // 逻辑与
                    for(unsigned j = 0 ; j < attr_dim && flag ; j  ++) {
                        flag = (flag && (attrs[global_id * attr_dim + j] >= l_bound[j])
                        && (attrs[global_id * attr_dim + j] <= r_bound[j]))  ;
                    }
                } else if(OP_TYPE == 1) {
                    // 逻辑或
                    for(unsigned j = 0 ; j < attr_dim && flag ; j  ++) {
                        flag = (flag || ((attrs[global_id * attr_dim + j] >= l_bound[j])
                        && (attrs[global_id * attr_dim + j] <= r_bound[j])))  ;
                    }
                }
                // 如果该点不满足属性过滤条件, 则将值标记为INF, ID 标记为全1
                if(!flag) {
                    val_vec[i] = INF_DIS ; 
                    id_reg_vec[i] = 0xFFFFFFFF ;
                }
            }   
        }
        else if(i * blockDim.x * blockDim.y + tid < TOPM + DEGREE){
            // 同理, 后半段列表中的元素查找自己在前半段列表中的位置
            val_vec[i] = arr2_val[i * blockDim.x * blockDim.y + tid - TOPM];
            id_reg_vec[i] = arr2[i * blockDim.x * blockDim.y + tid - TOPM];
            unsigned tmp = TOPM;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = arr1_val[res_id_vec[i] + halfsize];
                res_id_vec[i] += ((cand < val_vec[i]) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id_vec[i] += (arr1_val[res_id_vec[i]] < val_vec[i]);
            res_id_vec[i] += (i * blockDim.x * blockDim.y + tid - TOPM);
            if(res_id_vec[i] >= TOPM) {
                val_vec[i] = INF_DIS ; 
                id_reg_vec[i] = 0xFFFFFFFF ;
            }
        }
        else{
            res_id_vec[i] = TOPM + DEGREE + 1;
        }
    }
    __syncthreads();
    for(unsigned i = 0; i < task_num_per_thread; i ++){
        if(res_id_vec[i] < TOPM){
            arr1[res_id_vec[i]] = id_reg_vec[i];
            arr1_val[res_id_vec[i]] = val_vec[i];
            // printf("merge : res_id_vec[%d]-%d , id_reg_vec[%d]-%d , val_vec[%d]-%f\n" , i , res_id_vec[i] , i , 
            // (id_reg_vec[i] != 0xFFFFFFFF ? (id_reg_vec[i] & 0x7FFFFFFF) : id_reg_vec[i]) , i , val_vec[i]) ;
        } else if(res_id_vec[i] < TOPM + DEGREE) {
            arr2[res_id_vec[i] - TOPM] = id_reg_vec[i] ;
            arr2_val[res_id_vec[i] - TOPM] = val_vec[i] ;
            // printf("merge : res_id_vec[%d]-%d , id_reg_vec[%d]-%d , val_vec[%d]-%f\n" , i , res_id_vec[i] , i ,
            //  (id_reg_vec[i] != 0xFFFFFFFF ? (id_reg_vec[i] & 0x7FFFFFFF) : id_reg_vec[i]), i , val_vec[i]) ;
        }
    }
    __syncthreads();
}

template<typename attrType = float>
__device__ __forceinline__ void merge_top_with_recycle_list_logicalor(unsigned* arr1, unsigned* arr2, float* arr1_val, float* arr2_val, unsigned tid , unsigned DEGREE , 
    unsigned TOPM ,const attrType* l_bound ,const attrType* r_bound ,const attrType* attrs , unsigned attr_dim){
        // 本身传入的为global id 无需再次映射
    // 6 == 
    // unsigned res_id_vec[(TOPM + K+6*32-1)/ (6*32)] = {0};
    const unsigned thread_num_per_block = blockDim.x * blockDim.y ;
    const unsigned task_num_per_thread = (TOPM + DEGREE+ thread_num_per_block -1)/ (thread_num_per_block) ;
    unsigned res_id_vec[10] = {0} ;
    float val_vec[10];
    unsigned id_reg_vec[10];
    for(unsigned i = 0; i < task_num_per_thread; i ++){
        // unsigned res_id_vec[i] = 0;
        // float val;
        // if(i * blockDim.x * blockDim.y + tid >= TOPM + DEGREE)
        //     break ;

        if(i * blockDim.x * blockDim.y + tid < TOPM){
            val_vec[i] = arr1_val[i * blockDim.x * blockDim.y + tid];
            id_reg_vec[i] = arr1[i * blockDim.x * blockDim.y + tid];
            unsigned tmp = DEGREE;
            // 找到当前元素的插入位置, 判断后半段数组比自己大的元素数量
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = arr2_val[res_id_vec[i] + halfsize];
                // 说明此时有至少halfsize个元素比自己大
                res_id_vec[i] += ((cand <= val_vec[i]) ? halfsize : 0);
                tmp -= halfsize;
            }
            // 和卡在位置上的元素比较大小
            res_id_vec[i] += (arr2_val[res_id_vec[i]] <= val_vec[i]);
            // 自己在原列表中的位置为 i * blockDim.x * blockDim.y + tid
            res_id_vec[i] += (i * blockDim.x * blockDim.y + tid);

            if(res_id_vec[i] >= TOPM && id_reg_vec[i] != 0xFFFFFFFF) {
                // arr1_val[i * blockDim.x * blockDim.y + tid]
                // val_vec[i] = INF ;
                unsigned pure_id = id_reg_vec[i] & 0x3FFFFFFF ;
                // if(pure_id >= 1000000)
                //     printf("error: %d!!!!!!\n" , pure_id) ;
                // unsigned global_id = global_idx[pure_id + graph_offset] ;
                unsigned global_id = pure_id ;
                bool flag = false ;
                for(unsigned j = 0 ; j < attr_dim ; j  ++) {
                    flag = (flag || ((attrs[global_id * attr_dim + j] >= l_bound[j])
                     && (attrs[global_id * attr_dim + j] <= r_bound[j])))  ;
                }
                // 如果该点不满足属性过滤条件, 则将值标记为INF, ID 标记为全1
                if(!flag) {
                    val_vec[i] = INF_DIS ; 
                    id_reg_vec[i] = 0xFFFFFFFF ;
                }
            }   
        }
        else if(i * blockDim.x * blockDim.y + tid < TOPM + DEGREE){
            // 同理, 后半段列表中的元素查找自己在前半段列表中的位置
            val_vec[i] = arr2_val[i * blockDim.x * blockDim.y + tid - TOPM];
            id_reg_vec[i] = arr2[i * blockDim.x * blockDim.y + tid - TOPM];
            unsigned tmp = TOPM;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = arr1_val[res_id_vec[i] + halfsize];
                res_id_vec[i] += ((cand < val_vec[i]) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id_vec[i] += (arr1_val[res_id_vec[i]] < val_vec[i]);
            res_id_vec[i] += (i * blockDim.x * blockDim.y + tid - TOPM);
            if(res_id_vec[i] >= TOPM) {
                val_vec[i] = INF_DIS ; 
                id_reg_vec[i] = 0xFFFFFFFF ;
            }
        }
        else{
            res_id_vec[i] = TOPM + DEGREE + 1;
        }
    }
    __syncthreads();
    for(unsigned i = 0; i < task_num_per_thread; i ++){
        if(res_id_vec[i] < TOPM){
            arr1[res_id_vec[i]] = id_reg_vec[i];
            arr1_val[res_id_vec[i]] = val_vec[i];
            // printf("merge : res_id_vec[%d]-%d , id_reg_vec[%d]-%d , val_vec[%d]-%f\n" , i , res_id_vec[i] , i , 
            // (id_reg_vec[i] != 0xFFFFFFFF ? (id_reg_vec[i] & 0x7FFFFFFF) : id_reg_vec[i]) , i , val_vec[i]) ;
        } else if(res_id_vec[i] < TOPM + DEGREE) {
            arr2[res_id_vec[i] - TOPM] = id_reg_vec[i] ;
            arr2_val[res_id_vec[i] - TOPM] = val_vec[i] ;
            // printf("merge : res_id_vec[%d]-%d , id_reg_vec[%d]-%d , val_vec[%d]-%f\n" , i , res_id_vec[i] , i ,
            //  (id_reg_vec[i] != 0xFFFFFFFF ? (id_reg_vec[i] & 0x7FFFFFFF) : id_reg_vec[i]), i , val_vec[i]) ;
        }
    }
    __syncthreads();
}


__device__ __forceinline__ unsigned hash_insert(unsigned *hash_table, unsigned key){
    // 看起来是采用线性探测法的哈希表
    const unsigned bit_mask = HASHLEN - 1;
    unsigned index = ((key ^ (key >> HASHSIZE)) & bit_mask);
    const unsigned stride = 1;
    for(unsigned i = 0; i < HASHLEN; i++){
        const unsigned old = atomicCAS(&hash_table[index], 0xFFFFFFFF, key);
        if(old == 0xFFFFFFFF){
            return 1;
        }
        else if(old == key){
            return 0;
        }
        index = (index + stride) & bit_mask;
    }
    return 0;
}

__device__ __forceinline__ unsigned hash_peek(unsigned *hash_table, unsigned key){
    // 看起来是采用线性探测法的哈希表
    const unsigned bit_mask = HASHLEN - 1;
    unsigned index = ((key ^ (key >> HASHSIZE)) & bit_mask);
    const unsigned stride = 1;
    for(unsigned i = 0; i < HASHLEN; i++){
        const unsigned old = hash_table[index] ;
        if(old == 0xFFFFFFFF){
            return 0;
        }
        else if(old == key){
            return 1;
        }
        index = (index + stride) & bit_mask;
    }
    return 0;
}

// TOPM是candidate队列长度，K是TOP-K也是图索引的度数，DIM是数据集维度，MAX_ITER是最大迭代次数（1000足够），INF_DIS表示无穷大的浮点数
__global__ void select_path_v3(unsigned* graph, const half* __restrict__ values, const half* __restrict__ query_data, unsigned* results_id, float* results_dis, unsigned* ent_pts , unsigned DEGREE ,
    unsigned TOPM , unsigned DIM){
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
   
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数
    extern __shared__ unsigned char shared_mem[] ;

    // __shared__ unsigned top_M_Cand[TOPM + DEGREE];
    unsigned* top_M_Cand = (unsigned*) shared_mem ;
    // __shared__ float top_M_Cand_dis[TOPM + DEGREE];
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE) ;

    // 这里应该怎么处理8字节的对齐问题？
    // __shared__ half4 tmp_val_sha[DIM / 4];
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + (TOPM + DEGREE) * sizeof(unsigned) + (TOPM + DEGREE) * sizeof(float)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8) );

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    unsigned long long start;
    double duration = 0.0;
    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;  

    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
    for(unsigned i = tid; i < DEGREE; i += blockDim.x * blockDim.y){
        top_M_Cand[TOPM + i] = ent_pts[i];
        hash_insert(hash_table, top_M_Cand[TOPM + tid]);
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}
    __syncthreads();
 
    #pragma unroll
    for(unsigned i = threadIdx.y; i < DEGREE; i += blockDim.y){
        // threadIdx.y 0-5 , blockDim.y 即 6
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        // blockDim.x == 32
        // if(top_M_Cand[TOPM + i] >= 62500)
        //         printf("error at 201 top_M_Cand[TOPM + i] = %d\n" , top_M_Cand[TOPM + i])  ;
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            // warp中的每个子线程负责8个字节 即4个half类型分量的计算
            half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].x, val2.x));
            // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].y, val2.y));
        }
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            // 将warp内所有子线程的计算结果归并到一起
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
        }
        if(laneid == 0){
            top_M_Cand_dis[TOPM + i] = __half2float(__hadd(val_res.x, val_res.y));
        }
    }
    __syncthreads();

    // bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], DEGREE);
    if(tid < DEGREE)
        warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
    // 将K个初始节点作为当前的入口节点
    for(unsigned i = tid; i < min(DEGREE, TOPM); i += blockDim.x * blockDim.y){
        top_M_Cand[i] = top_M_Cand[TOPM + i];
        top_M_Cand_dis[i] = top_M_Cand_dis[TOPM + i];
    }
    __syncthreads();

    // begin search
    for(unsigned i = 0; i < MAX_ITER; i++){
        // 每四轮将hash表重置一次
        if((i + 1) % HASH_RESET == 0){
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            for(unsigned j = tid; j < TOPM + DEGREE; j += blockDim.x * blockDim.y){
                // 将此时的TOPM + K 个候选点插入哈希表中
                hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
            }
        }
        __syncthreads();
       
        if(tid < 32){
            // 从前32个点中找到需要扩展的点, 
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = 0;
                if((top_M_Cand[j] & 0x80000000) == 0){
                    n_p = 1;
                }
            
                // *** 坑点: __ballot_sync会等待掩码影响的线程执行到__ballot_sync处, 若把0xffffffff设置为掩码, 个别线程可能因循环条件影响提前退出, 
                // 无法执行到此处, 造成程序无限空等
                // unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
            
                // if(bid == 0)
                //     printf("tid : %d ," , tid) ;
                if(ballot_res > 0){
                    // 这里的ffs必须考虑返回值等于-1的情况
                    if(laneid == (__ffs(ballot_res) - 1)){
                        to_explore = top_M_Cand[j] ;
                        top_M_Cand[j] |= 0x80000000 ;
                    }

                   
                    break;
                }
                to_explore = 0xFFFFFFFF;
            }
        }
        __syncthreads();
    
        if(to_explore == 0xFFFFFFFF) {
            if(bid == 0 && tid == 0) printf("BREAK: %d\n", i);
            // printf("BREAK: %d\n" , i) ;
            break;
        }
        
        for(unsigned j = tid; j < DEGREE; j += blockDim.x * blockDim.y){
            // printf("to explore * degree + j %d" , to_explore * DEGREE + j) ;
            unsigned to_append = graph[to_explore * DEGREE + j];
            if(hash_insert(hash_table, to_append) == 0){
                top_M_Cand_dis[TOPM + j] = INF_DIS;
                // 这里的距离计算是以INF_DIS为基础往上加, 可能会存在浮点数溢出问题
            }
            else{
                top_M_Cand_dis[TOPM + j] = 0.0;
            }
            top_M_Cand[TOPM + j] = to_append;
        }
        __syncthreads();
   
        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < DEGREE; j += blockDim.y){
            // 6个warp, 每个warp负责一个距离计算
            // if(top_M_Cand_dis[TOPM + j] < 1.0){
                half2 val_res;
                val_res.x = top_M_Cand_dis[TOPM + j]; val_res.y = top_M_Cand_dis[TOPM+j];
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    // if(top_M_Cand[TOPM + j] >= 62500) {
                    //     printf("top_M_cand[TOPM + j] : %d\n" , top_M_Cand[TOPM+j]) ;
                    // }
                    half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                    // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].x, val2.x));
                    // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].y, val2.y));
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                    // printf("val_res.x : %f , val_res.y : %f , tmp_val_sha.x : %f , tmp_val_sha.y : %f , val2.x : %f , val2.y : %f \n" ,
                    // __half2float(val_res.x) , __half2float(val_res.y) , __half2float(tmp_val_sha[k].x.x)
                    //  , __half2float(tmp_val_sha[k].y.x) , __half2float(val2.x.x) , __half2float(val2.y.x)) ;
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    // top_M_Cand_dis[TOPM + j] = 1.0 - __half2float(__hadd(val_res.x, val_res.y));
                    top_M_Cand_dis[TOPM + j] = __half2float(__hadd(val_res.x, val_res.y));
                    // printf("dis : %f ," , top_M_Cand_dis[TOPM + j]) ;
                }
            // }
        }
        __syncthreads();
        // if(tid == 0)
        //     printf("epoch : %d , " , i ) ;
        
        // bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], DEGREE);
        if(tid < DEGREE)
            warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
        __syncthreads() ;
        merge_top(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM);
        __syncthreads() ;
    }
    __syncthreads();

   
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
}

__global__ void select_path_v4(unsigned* graph, const half* __restrict__ values, const half* __restrict__ query_data, unsigned* results_id, float* results_dis, unsigned* ent_pts ,unsigned TOPM , unsigned K , unsigned DIM ){
    __shared__ unsigned top_M_Cand[1000];
    __shared__ float top_M_Cand_dis[1000];
    // __shared__ unsigned top_M_Cand2[TOPM + K];
    // __shared__ float top_M_Cand_dis2[TOPM + K];

    __shared__ half4 tmp_val_sha[100];

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    unsigned long long start;
    double duration = 0.0;

    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;  
    
    if(bid == 1 && tid == 0) {
        for(unsigned i = 0 ; i< DIM ; i ++) {
            printf("%d , " , __half2float(values[DIM + i])) ;
        }
    }

    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        // top_M_Cand[TOPM + tid] = graph[bid * K + i];
        top_M_Cand[TOPM + tid] = ent_pts[i];
        hash_insert(hash_table, top_M_Cand[TOPM + tid]);
    }
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}
    __syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < K; i += blockDim.y){
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].x, val2.x));
            // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].y, val2.y));
        }
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
        }
        if(laneid == 0){
            top_M_Cand_dis[TOPM + i] = __half2float(__hadd(val_res.x, val_res.y));
            // top_M_Cand_dis[TOPM + i] = 1.0 - __half2float(__hadd(val_res.x, val_res.y));
        }
    }
    __syncthreads();

    bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
    for(unsigned i = tid; i < min(K, TOPM); i += blockDim.x * blockDim.y){
        top_M_Cand[i] = top_M_Cand[TOPM + i];
        top_M_Cand_dis[i] = top_M_Cand_dis[TOPM + i];
    }
    __syncthreads();

    // begin search
    for(unsigned i = 0; i < MAX_ITER; i++){
        // if(bid == 0 && tid == 0) printf("%d\n", i);
        
        if((i + 1) % HASH_RESET == 0){
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            for(unsigned j = tid; j < TOPM + K; j += blockDim.x * blockDim.y){
                hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
            }
        }
        __syncthreads();

        if(tid < 32){
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = 0;
                if((top_M_Cand[j] & 0x80000000) == 0){
                    n_p = 1;
                }
                unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                if(ballot_res > 0){
                    if(laneid == (__ffs(ballot_res) - 1)){
                        to_explore = top_M_Cand[j];
                        top_M_Cand[j] |= 0x80000000;
                    }
                    break;
                }
                to_explore = 0xFFFFFFFF;
            }
        }
        __syncthreads();
        // duration += ((double)(clock64() - start) / 1582000);
        if(to_explore == 0xFFFFFFFF) {
            if(bid == 0 && tid == 0) printf("BREAK: %d\n", i);
            break;
        }
        
        for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
            unsigned to_append = graph[to_explore * K + j];
            if(hash_insert(hash_table, to_append) == 0){
                top_M_Cand_dis[TOPM + j] = INF_DIS;
            }
            else{
                top_M_Cand_dis[TOPM + j] = 0.0;
            }
            top_M_Cand[TOPM + j] = to_append;
        }
        __syncthreads();
        
        
        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < K; j += blockDim.y){
            if(top_M_Cand_dis[TOPM + j] < 1.0){
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                    // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].x, val2.x));
                    // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].y, val2.y));
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    top_M_Cand_dis[TOPM + j] = __half2float(__hadd(val_res.x, val_res.y));
                    // top_M_Cand_dis[TOPM + j] = 1.0 - __half2float(__hadd(val_res.x, val_res.y));
                }
            }
        }
        __syncthreads();

        bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
        merge_top(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , K , TOPM) ;
        
    }
    __syncthreads();
    
    if(bid == 0 && tid == 0) printf("%lfms %d\n", duration, bid);
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    __syncthreads();
    if(bid == 0 && tid == 0){
        for(unsigned i = 0; i < 10; i++)
            printf("%d, ", top_M_Cand[i] & 0x7FFFFFFF);
        printf("\n");
        for(unsigned i = 0; i < 10; i++)
            printf("%f, ", top_M_Cand_dis[i]);
        printf("\n");
    }
}


__device__ unsigned jump_num_cnt[10000] = {0} ;
__device__ unsigned dis_cal_cnt[10000] = {0} ;
// TOPM是candidate队列长度，K是TOP-K也是图索引的度数，DIM是数据集维度，MAX_ITER是最大迭代次数（1000足够），INF_DIS表示无穷大的浮点数
__global__ void select_path_contain_attr_filter(const unsigned* __restrict__ all_graphs, const half* __restrict__ values, const half* __restrict__ query_data, 
    unsigned* results_id, float* results_dis, unsigned* ent_pts , unsigned DEGREE , unsigned TOPM , unsigned DIM ,  unsigned* graph_start_index , unsigned* graph_size ,
    unsigned* pids , unsigned* p_size , unsigned* p_start_index , float* l_bound , float* r_bound , float* attrs , unsigned attr_dim , unsigned* __restrict__ global_graph , unsigned edges_num_in_global_graph,
    unsigned recycle_list_size , unsigned* __restrict__ global_idx , unsigned* __restrict__ local_idx , unsigned* seg_global_idx_start_idx , unsigned pnum) {
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数
    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    unsigned* recycle_list_id = (unsigned*) shared_mem ;
    float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    // __shared__ unsigned recycle_list_id[64] ;
    // __shared__ float recycle_list_dis[64] ;
    // __shared__ unsigned top_M_Cand[64 + 16];
    unsigned* top_M_Cand = (unsigned*) (recycle_list_dis + recycle_list_size) ;
    // __shared__ float top_M_Cand_dis[64 + 16];
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE) ;

    // 这里应该怎么处理8字节的对齐问题？
    // __shared__ half4 tmp_val_sha[(unsigned)(128 / 4)];
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) +
     (TOPM + DEGREE) * sizeof(unsigned) + (TOPM + DEGREE) * sizeof(float)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    __shared__ unsigned long long start;
    __shared__ double dis_cal_cost , sort_cost , merge_cost , new_merge_cost  , sort_cost2 ;
    __shared__ unsigned total_jump_num ;
    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;  
    unsigned psize_block = p_size[blockIdx.x] , plist_index = p_start_index[blockIdx.x] ;

    if(tid == 0) {
        dis_cal_cost = sort_cost = merge_cost = new_merge_cost = sort_cost2=  0.0 ;
        total_jump_num = 0 ;
    }
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    // 没必要在每个分区开始时都把recycle_list复原, 每个分区得到的入口点肯定是越近越好
    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        recycle_list_id[i] = 0xFFFFFFFF ;
        recycle_list_dis[i] = INF_DIS ;
    }
    // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
    for(unsigned i = tid; i < DEGREE; i += blockDim.x * blockDim.y){
        top_M_Cand[TOPM + i] = global_idx[seg_global_idx_start_idx[pids[plist_index]] + ent_pts[i]];
        // printf("ent_pts[%d] : %d , top_M_Cand[TOPM + %d] : %d , seg_global_idx_start_idx[%d] : %d \n" , i , ent_pts[i] , i , top_M_Cand[TOPM+i] ,
        // pids[plist_index] , seg_global_idx_start_idx[pids[plist_index]]) ;
        // hash_insert(hash_table, top_M_Cand[TOPM + i]);
        hash_insert(hash_table , ent_pts[i]) ;
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}
    // if(tid == 0) {
    //     printf("can click here\n") ;
    // }
    __syncthreads();
 
  
    /**
        1.先明确对每个分区要做什么, 首先, 将候选点载入top_M_Cand 的末DEGREE个位置, 然后排序
        2.将循环池重置为初始值
        3.从初始点开始扩展, 每次将DEGREE个邻居存放入top_M_Cand的末位, 被淘汰的分区内的点会插入循环池中
        4.当前分区查找结束后,将top_M_Cand中不符合范围约束的点去掉,用循环池中的点替代,然后依据这些点从全局图中扩展到下一个分区的入口点

        step4 的具体实现时, 可以先把不符合范围约束的点的距离改为INF,然后对列表排序, 和recycle_list合并
    **/
    __shared__ double overall_cost , select_point_cost , outer_loop_cost , inner_loop_cost, hash_reset_cost , load_point_cost;
    __shared__ unsigned long long overall_start , select_point_start , outer_loop_start , inner_loop_start , hash_reset_start , load_point_start;


    if(tid == 0) {
        overall_cost = 0.0 ;
        overall_start = clock64() ;
        select_point_cost = 0.0 ;
        outer_loop_cost = 0.0 ;
        inner_loop_cost = 0.0 ;
        hash_reset_cost = 0.0 ;
        load_point_cost = 0.0 ;
    }
    __syncthreads() ;
    for(unsigned p_i = 0 ; p_i < psize_block; p_i ++) {
        // 得到图的入口
        unsigned pid = pids[plist_index + p_i] ;
        const unsigned* graph = all_graphs + graph_start_index[pid] ;
        // printf("can click here 542\n") ;
        // 这段代码对初始的入口节点做距离计算, 然后插入到队列中

        // if(tid == 0) {
        //     start = clock64() ;
        // }
        __syncthreads() ;
        #pragma unroll
        for(unsigned i = threadIdx.y; i < DEGREE; i += blockDim.y){
            // threadIdx.y 0-5 , blockDim.y 即 6
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            // blockDim.x == 32
            // if(top_M_Cand[TOPM + i] >= 62500)
            //         printf("error at 201 top_M_Cand[TOPM + i] = %d\n" , top_M_Cand[TOPM + i])  ;
            // if(threadIdx.x == 0)
                // printf("epoch %d/%d , now top_M_Cnad[TOPM + %d] : %d\n" ,p_i , psize_block ,i ,  top_M_Cand[TOPM + i]) ;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
                // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].x, val2.x));
                // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].y, val2.y));
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                // 将warp内所有子线程的计算结果归并到一起
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }
            if(laneid == 0){
                top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                // printf("epoch %d/%d ,top_M_Cand_dis[TOPM + %d] : %f\n" , p_i , psize_block , i , top_M_Cand_dis[TOPM + i]) ;
            }
        }
        __syncthreads();
        // if(tid == 0) {
        //     dis_cal_cost += ((double)(clock64() - start) / 1582000) ;
        // }
        // __syncthreads() ;

        if(tid == 0) {
            // sort_cost += ((double)(clock64() - start) / 1582000) ;
            start = clock64() ;
        }
        __syncthreads() ;
        // 直接对整个列表做距离计算
        if(tid < DEGREE)
            warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
        // bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis + TOPM, top_M_Cand + TOPM, DEGREE);
        __syncthreads() ;
        // if(tid == 0) {
        //     sort_cost += ((double)(clock64() - start) / 1582000) ;
        //     start = clock64() ;
        // }
        // __syncthreads() ;

        // 此处被挤出去的点也应该合并到recycle_list 上
        // 将两个列表归并到一起
        merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , l_bound , r_bound , attrs ,attr_dim) ;
        __syncthreads() ;
        if(tid < DEGREE)
            warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE) ; 
        __syncthreads() ;
        if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
            merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE , recycle_list_size) ;
            __syncthreads() ;
        }
        // if(tid == 0) {
        //     merge_cost += ((double)(clock64() - start) / 1582000) ;
        // }
       
        // 将K个初始节点作为当前的入口节点, 这里可以同时对两个列表排序, 前段列表的dis事先被设置为了Inf
        // for(unsigned i = tid; i < min(DEGREE, TOPM); i += blockDim.x * blockDim.y){
        //     top_M_Cand[i] = top_M_Cand[TOPM + i];
        //     top_M_Cand_dis[i] = top_M_Cand_dis[TOPM + i];
        // }
        // 初始化 recycle_list
        // for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        //     recycle_list_id[i] = 0xFFFFFFFF ;
        //     recycle_list_dis[i] = INF_DIS ;
        // }

        // __syncthreads();
        if(tid == 0)
            inner_loop_start = clock64() ;
        __syncthreads() ;

        // begin search
        for(unsigned i = 0; i < MAX_ITER; i++){

            if(tid == 0) 
                hash_reset_start = clock64() ;
            __syncthreads() ;
            // 每四轮将hash表重置一次
            if((i + 1) % HASH_RESET == 0){
                for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                    hash_table[j] = 0xFFFFFFFF;
                }
                __syncthreads();

                for(unsigned j = tid; j < TOPM + DEGREE; j += blockDim.x * blockDim.y){
                    // 将此时的TOPM + K 个候选点插入哈希表中
                    // hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
                    if(top_M_Cand[j] != 0xFFFFFFFF)
                        hash_insert(hash_table , local_idx[(top_M_Cand[j] & 0x7FFFFFFF)]) ;
                }
            }
            __syncthreads();

            if(tid == 0) {
                hash_reset_cost += ((double)(clock64() - hash_reset_start) / 1582000);
                select_point_start = clock64() ;
            }
            __syncthreads() ;
            if(tid < 32){
                // 从前32个点中找到需要扩展的点, 
                for(unsigned j = laneid; j < TOPM; j+=32){
                    unsigned n_p = 0;
                    if((top_M_Cand[j] & 0x80000000) == 0){
                        n_p = 1;
                    }
                
                    // *** 坑点: __ballot_sync会等待掩码影响的线程执行到__ballot_sync处, 若把0xffffffff设置为掩码, 个别线程可能因循环条件影响提前退出, 
                    // 无法执行到此处, 造成程序无限空等
                    // unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                    unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                
                    // if(bid == 0)
                    //     printf("tid : %d ," , tid) ;
                    if(ballot_res > 0){
                        // 这里的ffs必须考虑返回值等于-1的情况
                        if(laneid == (__ffs(ballot_res) - 1)){
                            to_explore = local_idx[top_M_Cand[j]];
                            top_M_Cand[j] |= 0x80000000;
                        }

                    
                        break;
                    }
                    to_explore = 0xFFFFFFFF;
                }
            }
            __syncthreads();

            if(tid == 0) {
                select_point_cost += ((double)(clock64() - select_point_start) / 1582000);
                load_point_start = clock64() ;
            }
            __syncthreads() ;
            if(to_explore == 0xFFFFFFFF) {
                if(bid == 0 && tid == 0) printf("BREAK: %d\n", i);
                // printf("BREAK: %d\n" , i) ;
                if(tid == 0)
                    total_jump_num += (i + 1) ;
                break;
            }
            
            for(unsigned j = tid; j < DEGREE; j += blockDim.x * blockDim.y){
                // printf("to explore * degree + j %d" , to_explore * DEGREE + j) ;
                unsigned to_append_local = graph[to_explore * DEGREE + j];
                // printf("to_append_local : %d , seg_global_idx_start_idx : %d\n" , to_append_local , seg_global_idx_start_idx[pid]) ;
                unsigned to_append = global_idx[seg_global_idx_start_idx[pid] + to_append_local] ;
                // if(hash_insert(hash_table, to_append_local) == 0){
                //     top_M_Cand_dis[TOPM + j] = INF_DIS;
                    // 这里的距离计算是以INF_DIS为基础往上加, 可能会存在浮点数溢出问题
                // }
                // else{
                //     top_M_Cand_dis[TOPM + j] = 0.0;
                // }
                top_M_Cand_dis[TOPM + j] = (hash_insert(hash_table , to_append_local) == 0 ? INF_DIS : 0.0) ;
                top_M_Cand[TOPM + j] = to_append;
            }
            __syncthreads();
            // if(tid == 0)
            //     printf("can click here 660 epoch : %d/%d\n" , p_i , psize_block) ;
            if(tid == 0) {
                load_point_cost += ((double)(clock64() - load_point_start) / 1582000);
                start = clock64() ;
            }
            __syncthreads() ;
            // calculate distance
            #pragma unroll
            for(unsigned j = threadIdx.y; j < DEGREE; j += blockDim.y){
                // 6个warp, 每个warp负责一个距离计算
                // if(top_M_Cand_dis[TOPM + j] < 1.0){
                    half2 val_res;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                        // if(top_M_Cand[TOPM + j] >= 62500) {
                        //     printf("top_M_cand[TOPM + j] : %d\n" , top_M_Cand[TOPM+j]) ;
                        // }
                        half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                        // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].x, val2.x));
                        // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].y, val2.y));
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                        // printf("val_res.x : %f , val_res.y : %f , tmp_val_sha.x : %f , tmp_val_sha.y : %f , val2.x : %f , val2.y : %f \n" ,
                        // __half2float(val_res.x) , __half2float(val_res.y) , __half2float(tmp_val_sha[k].x.x)
                        //  , __half2float(tmp_val_sha[k].y.x) , __half2float(val2.x.x) , __half2float(val2.y.x)) ;
                    }
                    #pragma unroll
                    for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                        val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                    }

                    if(laneid == 0){
                        // top_M_Cand_dis[TOPM + j] = 1.0 - __half2float(__hadd(val_res.x, val_res.y));
                        top_M_Cand_dis[TOPM + j] += __half2float(val_res.x) + __half2float(val_res.y) ;
                        // printf("epoch %d/%d , round %d , top_M_Cand_dis[TOPM + %d] : %f\n" , p_i , psize_block , i , j , top_M_Cand_dis[TOPM + j]) ;
                        // printf("dis : %f ," , top_M_Cand_dis[TOPM + j]) ;
                    }
                // }
            }
            __syncthreads();
            if(tid == 0) {
                dis_cal_cost += ((double)(clock64() - start) / 1582000) ;
                start = clock64() ;
            }
            __syncthreads() ;
            // if(tid == 0)
            //     printf("can click here 694 epoch : %d/%d\n" , p_i , psize_block) ;
            // if(tid == 0)
            //     printf("epoch : %d , " , i ) ;
            
            // bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], DEGREE);
            if(tid < DEGREE)
                warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
            __syncthreads() ;
            if(tid == 0) {
                sort_cost += ((double)(clock64() - start) / 1582000) ;
                start = clock64() ;
            }
            __syncthreads() ;
            // for(unsigned j = tid ; j < DEGREE + TOPM ; j += blockDim.x * blockDim.y) {
            //     printf("line 716: epoch %d/%d ,round %d ,top_M_Cand[%d] : %d , top_M_Cand_Dis[%d] : %f \n" ,p_i , psize_block, i , j , top_M_Cand[j] & 0x7FFFFFFF , j , top_M_Cand_dis[j]) ;
            // }
            // if(tid == 0)
            //     printf("can click here 700 epoch : %d/%d\n" , p_i , psize_block) ;
            merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , l_bound , r_bound , attrs , attr_dim) ;
            __syncthreads() ;
            if(tid == 0) {
                new_merge_cost += ((double)(clock64() - start) / 1582000) ;
                start = clock64() ;
            }
            __syncthreads() ;
            // for(unsigned j = tid ; j < DEGREE + TOPM ; j += blockDim.x * blockDim.y) {
            //     printf("line 720: epoch %d/%d ,round %d ,top_M_Cand[%d] : %d , top_M_Cand_Dis[%d] : %f \n" ,p_i , psize_block, i , j , top_M_Cand[j] & 0x7FFFFFFF , j , top_M_Cand_dis[j]) ;
            // }
            // if(tid == 0)
            //     printf("can click here 702 epoch : %d/%d\n" , p_i , psize_block) ;
            // 问题来了, 当图开始切换时, 当前部分结果的id应该采用局部id还是全局id? 在列表中的所有id皆采用全局id
            // 对后半段列表sort一次, 然后和recycle pool进行merge
            // bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM] , &top_M_Cand[TOPM] , DEGREE) ;
            if(tid < DEGREE)
                warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
            __syncthreads() ;
            if(tid == 0) {
                sort_cost2 += ((double)(clock64() - start) / 1582000) ;
                start = clock64() ;
            }
            // for(unsigned j = tid ; j < DEGREE + TOPM ; j += blockDim.x * blockDim.y) {
            //     printf("line 726: epoch %d/%d ,round %d ,top_M_Cand[%d] : %d , top_M_Cand_Dis[%d] : %f \n" ,p_i , psize_block, i , j , top_M_Cand[j] & 0x7FFFFFFF , j , top_M_Cand_dis[j]) ;
            // }
            // if(tid == 0)
            //     printf("can click here 707 epoch : %d/%d\n" , p_i , psize_block) ;
            if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                // 此时top_M_Cand + TOPM中的点一定是满足属性过滤条件的点
                merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE , recycle_list_size) ;
                __syncthreads() ;
                if(tid == 0) {
                    merge_cost += ((double)(clock64() - start) / 1582000) ;
                    // start = clock64() ;
                }
                __syncthreads() ;
            }
            // if(tid == 0)
            //     printf("can click here 708 epoch : %d/%d\n" , p_i , psize_block) ;
            // for(unsigned j = tid ; j < DEGREE + TOPM ; j += blockDim.x * blockDim.y) {
            //     printf("line 735: epoch %d/%d ,round %d ,top_M_Cand[%d] : %d , top_M_Cand_Dis[%d] : %f \n" ,p_i , psize_block, i , j , top_M_Cand[j] & 0x7FFFFFFF , j , top_M_Cand_dis[j]) ;
            // }
            // __syncthreads() ;
        }
        __syncthreads();

        if(tid == 0) {
            inner_loop_cost += ((double)(clock64() - inner_loop_start) / 1582000);
            outer_loop_start = clock64() ;
        }
        __syncthreads() ;

        // bug产生原因: 在扩展完候选点后将hash表清空了
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y)
            top_M_Cand[i] |= 0x80000000 ;
        for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
            hash_table[i] = 0xFFFFFFFF;
        }
        __syncthreads() ;

        // 修改一下顺序, 先扩展候选点, 然后再筛除不符合条件的点
        // 从global_edges 中扩展候选点
        if(p_i < psize_block - 1) {
            unsigned next_pid = pids[plist_index + p_i + 1] ;
            for(unsigned i = tid ; i < DEGREE ; i += blockDim.x * blockDim.y) {
                unsigned pure_id = top_M_Cand[i] & 0x7FFFFFFF ;
                for(unsigned j = 0 ; j < edges_num_in_global_graph ; j ++) {
                    unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + j] ;
                    if(hash_insert(hash_table , next_local_id) != 0) {
                        top_M_Cand[TOPM + i] = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ; 
                    // top_M_Cand_dis[TOPM + i]
                        break ; 
                    }
                }
            }
        }
        __syncthreads() ;

        // 一个分区的搜索完成, 将recycle_list中的内容合并到top_M_Cand中
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
            if(top_M_Cand[i] == 0xFFFFFFFF)
                continue ;
            bool flag = true ;
            unsigned pure_id = top_M_Cand[i] & 0x7FFFFFFF ;
            for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
                // printf("attrs[%d * %d + %d] : %f , l_bound[%d] : %f , r_bound[%d] : %f\n" , pure_id , attr_dim , j , attrs[pure_id * attr_dim + j] ,
                // j , l_bound[blockIdx.x * attr_dim +  j] , j , r_bound[blockIdx.x * attr_dim + j]) ;
                flag = (flag && (attrs[pure_id * attr_dim + j] >= l_bound[blockIdx.x * attr_dim + j]) && (attrs[pure_id * attr_dim + j] <= r_bound[blockIdx.x *attr_dim + j])) ;
            }
            if(!flag) {
                top_M_Cand[i] = 0xFFFFFFFF ;
                top_M_Cand_dis[i] = INF_DIS ;
            }
        }
        __syncthreads() ;
        // if(tid == 0)
            // printf("can click 775\n") ;
        // __syncthreads() ;
        // if(tid == 0) {
        //     start=  clock64() ;
        // }
        // __syncthreads() ;
        bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM);
        __syncthreads() ;
        // if(tid == 0) {
        //     sort_cost += ((double)(clock64() - start) / 1582000) ;
        //     start = clock64() ;
        // }
        // __syncthreads() ;
        // 若recycle_list中确实有点, 则合并
        if(recycle_list_id[0] != 0xFFFFFFFF) {
            merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
            __syncthreads() ;
            // if(tid == 0) {
            //     merge_cost += ((double)(clock64() - start) / 1582000) ;
            //     start = clock64() ;
            // }
        }
        // for(unsigned j = tid ; j < DEGREE + TOPM ; j += blockDim.x * blockDim.y) {
            // printf("after merge : epoch %d/%d ,top_M_Cand[%d] : %d , top_M_Cand_Dis[%d] : %f \n" , p_i , psize_block , j , 
            // (top_M_Cand[j] != 0xFFFFFFFF ? top_M_Cand[j] & 0x7FFFFFFF : top_M_Cand[j]) , j , top_M_Cand_dis[j]) ;
        // }
        // __syncthreads() ;

        // for(unsigned j = tid ; j < DEGREE + TOPM ; j += blockDim.x * blockDim.y) {
            // printf("epoch %d/%d ,top_M_Cand[%d] : %d , top_M_Cand_Dis[%d] : %f \n" , p_i , psize_block , j , 
            // (top_M_Cand[j] != 0xFFFFFFFF ? top_M_Cand[j] & 0x7FFFFFFF : top_M_Cand[j]) , j , top_M_Cand_dis[j]) ;
        // }
        // __syncthreads() ;


        // printf("can click to 798\n") ;
        // __syncthreads() ;
        // for(unsigned j = tid ; j < DEGREE + TOPM ; j += blockDim.x * blockDim.y) {
        //     printf("epoch %d/%d ,top_M_Cand[%d] : %d , top_M_Cand_Dis[%d] : %f \n" , p_i , psize_block , j , 
        //     (top_M_Cand[j] != 0xFFFFFFFF ? top_M_Cand[j] & 0x7FFFFFFF : top_M_Cand[j]) , j , top_M_Cand_dis[j]) ;
        // }
        // __syncthreads() ;
        if(tid == 0) {
            outer_loop_cost += ((double)(clock64() - outer_loop_start) / 1582000);
        }
        __syncthreads() ;
    }
    __syncthreads() ;
    if(tid == 0) {
        overall_cost += ((double)(clock64() - overall_start) / 1582000);
    }
    __syncthreads() ;
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    if(tid == 0 ) {
        printf("block : %d ,总体耗时: %f , 距离计算时间: %f , 排序时间: %f , 排序时间(第二次): %f , 合并时间: %f , 合并时间(函数2): %f, 选点耗时: %f,  跳转次数: %d, 其它时间: %f , 内部循环耗时: %f , 外部循环耗时: %f , 重置哈希表耗时: %f , 载入待计算点耗时: %f\n" ,
         blockIdx.x , overall_cost , dis_cal_cost , sort_cost , sort_cost2 , merge_cost , new_merge_cost ,select_point_cost , 
         total_jump_num , overall_cost - dis_cal_cost - sort_cost - sort_cost2 - merge_cost - new_merge_cost , inner_loop_cost , 
         outer_loop_cost , hash_reset_cost , load_point_cost) ;
         jump_num_cnt[blockIdx.x] = total_jump_num ;
    } 
}



// TOPM是candidate队列长度，K是TOP-K也是图索引的度数，DIM是数据集维度，MAX_ITER是最大迭代次数（1000足够），INF_DIS表示无穷大的浮点数
__global__ void select_path_contain_attr_filter_V2(const unsigned* __restrict__ all_graphs, const half* __restrict__ values, const half* __restrict__ query_data, 
    unsigned* results_id, float* results_dis, unsigned* ent_pts , unsigned DEGREE , unsigned TOPM , unsigned DIM ,  unsigned* graph_start_index , unsigned* graph_size ,
    unsigned* pids , unsigned* p_size , unsigned* p_start_index , float* l_bound , float* r_bound , float* attrs , unsigned attr_dim , unsigned* __restrict__ global_graph , unsigned edges_num_in_global_graph,
    unsigned recycle_list_size , unsigned* __restrict__ global_idx , unsigned* __restrict__ local_idx , unsigned* seg_global_idx_start_idx , unsigned pnum) {
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数
    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    unsigned* recycle_list_id = (unsigned*) shared_mem ;
    float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    // __shared__ unsigned recycle_list_id[64] ;
    // __shared__ float recycle_list_dis[64] ;
    // __shared__ unsigned top_M_Cand[64 + 16];
    unsigned* top_M_Cand = (unsigned*) (recycle_list_dis + recycle_list_size) ;
    // __shared__ float top_M_Cand_dis[64 + 16];
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE) ;

    // 这里应该怎么处理8字节的对齐问题？
    // __shared__ half4 tmp_val_sha[(unsigned)(128 / 4)];
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) +
     (TOPM + DEGREE) * sizeof(unsigned) + (TOPM + DEGREE) * sizeof(float)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[128] ;
    __shared__ float work_mem_float[128] ;

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    __shared__ unsigned long long start;
    __shared__ double dis_cal_cost , sort_cost , merge_cost , new_merge_cost  , sort_cost2 ;
    __shared__ unsigned total_jump_num ;
    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;  
    unsigned psize_block = p_size[blockIdx.x] , plist_index = p_start_index[blockIdx.x] ;

    if(tid == 0) {
        dis_cal_cost = sort_cost = merge_cost = new_merge_cost = sort_cost2=  0.0 ;
        total_jump_num = 0 ;
    }
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    // 没必要在每个分区开始时都把recycle_list复原, 每个分区得到的入口点肯定是越近越好
    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        recycle_list_id[i] = 0xFFFFFFFF ;
        recycle_list_dis[i] = INF_DIS ;
    }
    // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
    for(unsigned i = tid; i < DEGREE; i += blockDim.x * blockDim.y){
        top_M_Cand[TOPM + i] = global_idx[seg_global_idx_start_idx[pids[plist_index]] + ent_pts[i]];
        // printf("ent_pts[%d] : %d , top_M_Cand[TOPM + %d] : %d , seg_global_idx_start_idx[%d] : %d \n" , i , ent_pts[i] , i , top_M_Cand[TOPM+i] ,
        // pids[plist_index] , seg_global_idx_start_idx[pids[plist_index]]) ;
        // hash_insert(hash_table, top_M_Cand[TOPM + i]);
        hash_insert(hash_table , ent_pts[i]) ;
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}
    // if(tid == 0) {
    //     printf("can click here\n") ;
    // }
    __syncthreads();
 
  
    /**
        1.先明确对每个分区要做什么, 首先, 将候选点载入top_M_Cand 的末DEGREE个位置, 然后排序
        2.将循环池重置为初始值
        3.从初始点开始扩展, 每次将DEGREE个邻居存放入top_M_Cand的末位, 被淘汰的分区内的点会插入循环池中
        4.当前分区查找结束后,将top_M_Cand中不符合范围约束的点去掉,用循环池中的点替代,然后依据这些点从全局图中扩展到下一个分区的入口点

        step4 的具体实现时, 可以先把不符合范围约束的点的距离改为INF,然后对列表排序, 和recycle_list合并
    **/
    __shared__ double overall_cost , select_point_cost , outer_loop_cost , inner_loop_cost, hash_reset_cost , load_point_cost;
    __shared__ unsigned long long overall_start , select_point_start , outer_loop_start , inner_loop_start , hash_reset_start , load_point_start;


    if(tid == 0) {
        overall_cost = 0.0 ;
        overall_start = clock64() ;
        select_point_cost = 0.0 ;
        outer_loop_cost = 0.0 ;
        inner_loop_cost = 0.0 ;
        hash_reset_cost = 0.0 ;
        load_point_cost = 0.0 ;
    }
    __syncthreads() ;
    for(unsigned p_i = 0 ; p_i < psize_block; p_i ++) {
        // 得到图的入口
        unsigned pid = pids[plist_index + p_i] ;
        const unsigned* graph = all_graphs + graph_start_index[pid] ;
        // printf("can click here 542\n") ;
        // 这段代码对初始的入口节点做距离计算, 然后插入到队列中

        // if(tid == 0) {
        //     start = clock64() ;
        // }
        __syncthreads() ;
        #pragma unroll
        for(unsigned i = threadIdx.y; i < DEGREE; i += blockDim.y){
            // threadIdx.y 0-5 , blockDim.y 即 6
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            // blockDim.x == 32
            // if(top_M_Cand[TOPM + i] >= 62500)
            //         printf("error at 201 top_M_Cand[TOPM + i] = %d\n" , top_M_Cand[TOPM + i])  ;
            // if(threadIdx.x == 0)
                // printf("epoch %d/%d , now top_M_Cnad[TOPM + %d] : %d\n" ,p_i , psize_block ,i ,  top_M_Cand[TOPM + i]) ;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                if(top_M_Cand[TOPM + i] >= 1000000) {
                    printf("error at 1136 , %d\n" , top_M_Cand[TOPM+ i]) ;
                }
                half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
                // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].x, val2.x));
                // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].y, val2.y));
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                // 将warp内所有子线程的计算结果归并到一起
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }
            if(laneid == 0){
                top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                // printf("epoch %d/%d ,top_M_Cand_dis[TOPM + %d] : %f\n" , p_i , psize_block , i , top_M_Cand_dis[TOPM + i]) ;
            }
        }
        __syncthreads();
        // if(tid == 0) {
        //     dis_cal_cost += ((double)(clock64() - start) / 1582000) ;
        // }
        // __syncthreads() ;

        if(tid == 0) {
            // sort_cost += ((double)(clock64() - start) / 1582000) ;
            start = clock64() ;
        }
        __syncthreads() ;
        // 直接对整个列表做距离计算
        if(tid < DEGREE)
            warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
        // bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis + TOPM, top_M_Cand + TOPM, DEGREE);
        __syncthreads() ;
        // if(tid == 0) {
        //     sort_cost += ((double)(clock64() - start) / 1582000) ;
        //     start = clock64() ;
        // }
        // __syncthreads() ;

        // 此处被挤出去的点也应该合并到recycle_list 上
        // 将两个列表归并到一起
        merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , l_bound , r_bound , attrs ,attr_dim) ;
        __syncthreads() ;
        if(tid < DEGREE)
            warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE) ; 
        __syncthreads() ;
        if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
            merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE , recycle_list_size) ;
            __syncthreads() ;
        }
        // if(tid == 0) {
        //     merge_cost += ((double)(clock64() - start) / 1582000) ;
        // }
       
        // 将K个初始节点作为当前的入口节点, 这里可以同时对两个列表排序, 前段列表的dis事先被设置为了Inf
        // for(unsigned i = tid; i < min(DEGREE, TOPM); i += blockDim.x * blockDim.y){
        //     top_M_Cand[i] = top_M_Cand[TOPM + i];
        //     top_M_Cand_dis[i] = top_M_Cand_dis[TOPM + i];
        // }
        // 初始化 recycle_list
        for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
            recycle_list_id[i] = 0xFFFFFFFF ;
            recycle_list_dis[i] = INF_DIS ;
        }

        __syncthreads();
        if(tid == 0)
            inner_loop_start = clock64() ;
        __syncthreads() ;

        // begin search
        for(unsigned i = 0; i < MAX_ITER; i++){

            if(tid == 0) 
                hash_reset_start = clock64() ;
            __syncthreads() ;
            // 每四轮将hash表重置一次
            if((i + 1) % HASH_RESET == 0){
                for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                    hash_table[j] = 0xFFFFFFFF;
                }
                __syncthreads();

                for(unsigned j = tid; j < TOPM + DEGREE; j += blockDim.x * blockDim.y){
                    // 将此时的TOPM + K 个候选点插入哈希表中
                    // hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
                    if(top_M_Cand[j] != 0xFFFFFFFF) {
                        if((top_M_Cand[j] & 0x7FFFFFFF) >= 1000000) {
                            printf("error at %d , %d\n" ,__LINE__ ,top_M_Cand[j] & 0x7FFFFFFF) ;
                        }
                        hash_insert(hash_table , local_idx[(top_M_Cand[j] & 0x7FFFFFFF)]) ;
                        
                    }
                }
            }
            __syncthreads();

            if(tid == 0) {
                hash_reset_cost += ((double)(clock64() - hash_reset_start) / 1582000);
                select_point_start = clock64() ;
            }
            __syncthreads() ;
            if(tid < 32){
                // 从前32个点中找到需要扩展的点, 
                for(unsigned j = laneid; j < TOPM; j+=32){
                    unsigned n_p = 0;
                    if((top_M_Cand[j] & 0x80000000) == 0){
                        n_p = 1;
                    }
                
                    // *** 坑点: __ballot_sync会等待掩码影响的线程执行到__ballot_sync处, 若把0xffffffff设置为掩码, 个别线程可能因循环条件影响提前退出, 
                    // 无法执行到此处, 造成程序无限空等
                    // unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                    unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                
                    // if(bid == 0)
                    //     printf("tid : %d ," , tid) ;
                    if(ballot_res > 0){
                        // 这里的ffs必须考虑返回值等于-1的情况
                        if(laneid == (__ffs(ballot_res) - 1)){
                            to_explore = local_idx[top_M_Cand[j]];
                            top_M_Cand[j] |= 0x80000000;
                        }

                    
                        break;
                    }
                    to_explore = 0xFFFFFFFF;
                }
            }
            __syncthreads();

            if(tid == 0) {
                select_point_cost += ((double)(clock64() - select_point_start) / 1582000);
                load_point_start = clock64() ;
            }
            __syncthreads() ;
            if(to_explore == 0xFFFFFFFF) {
                if(bid == 0 && tid == 0) printf("BREAK: %d\n", i);
                // printf("BREAK: %d\n" , i) ;
                if(tid == 0)
                    total_jump_num += (i + 1) ;
                break;
            }
            
            for(unsigned j = tid; j < DEGREE; j += blockDim.x * blockDim.y){
                // printf("to explore * degree + j %d" , to_explore * DEGREE + j) ;
                unsigned to_append_local = graph[to_explore * DEGREE + j];
                // printf("to_append_local : %d , seg_global_idx_start_idx : %d\n" , to_append_local , seg_global_idx_start_idx[pid]) ;
                unsigned to_append = global_idx[seg_global_idx_start_idx[pid] + to_append_local] ;
                // if(hash_insert(hash_table, to_append_local) == 0){
                //     top_M_Cand_dis[TOPM + j] = INF_DIS;
                    // 这里的距离计算是以INF_DIS为基础往上加, 可能会存在浮点数溢出问题
                // }
                // else{
                //     top_M_Cand_dis[TOPM + j] = 0.0;
                // }
                top_M_Cand_dis[TOPM + j] = (hash_insert(hash_table , to_append_local) == 0 ? INF_DIS : 0.0) ;
                top_M_Cand[TOPM + j] = to_append;
            }
            __syncthreads();
            // if(tid == 0)
            //     printf("can click here 660 epoch : %d/%d\n" , p_i , psize_block) ;
            if(tid == 0) {
                load_point_cost += ((double)(clock64() - load_point_start) / 1582000);
                start = clock64() ;
            }
            __syncthreads() ;
            // calculate distance
            #pragma unroll
            for(unsigned j = threadIdx.y; j < DEGREE; j += blockDim.y){
                // 6个warp, 每个warp负责一个距离计算
                // if(top_M_Cand_dis[TOPM + j] < 1.0){
                    half2 val_res;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                        // if(top_M_Cand[TOPM + j] >= 62500) {
                        //     printf("top_M_cand[TOPM + j] : %d\n" , top_M_Cand[TOPM+j]) ;
                        // }
                        half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                        // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].x, val2.x));
                        // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].y, val2.y));
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                        // printf("val_res.x : %f , val_res.y : %f , tmp_val_sha.x : %f , tmp_val_sha.y : %f , val2.x : %f , val2.y : %f \n" ,
                        // __half2float(val_res.x) , __half2float(val_res.y) , __half2float(tmp_val_sha[k].x.x)
                        //  , __half2float(tmp_val_sha[k].y.x) , __half2float(val2.x.x) , __half2float(val2.y.x)) ;
                    }
                    #pragma unroll
                    for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                        val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                    }

                    if(laneid == 0){
                        // top_M_Cand_dis[TOPM + j] = 1.0 - __half2float(__hadd(val_res.x, val_res.y));
                        top_M_Cand_dis[TOPM + j] += __half2float(val_res.x) + __half2float(val_res.y) ;
                        // printf("epoch %d/%d , round %d , top_M_Cand_dis[TOPM + %d] : %f\n" , p_i , psize_block , i , j , top_M_Cand_dis[TOPM + j]) ;
                        // printf("dis : %f ," , top_M_Cand_dis[TOPM + j]) ;
                    }
                // }
            }
            __syncthreads();
            if(tid == 0) {
                dis_cal_cost += ((double)(clock64() - start) / 1582000) ;
                start = clock64() ;
            }
            __syncthreads() ;
            // if(tid == 0)
            //     printf("can click here 694 epoch : %d/%d\n" , p_i , psize_block) ;
            // if(tid == 0)
            //     printf("epoch : %d , " , i ) ;
            
            // bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], DEGREE);
            if(tid < DEGREE)
                warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
            __syncthreads() ;
            if(tid == 0) {
                sort_cost += ((double)(clock64() - start) / 1582000) ;
                start = clock64() ;
            }
            __syncthreads() ;
            // for(unsigned j = tid ; j < DEGREE + TOPM ; j += blockDim.x * blockDim.y) {
            //     printf("line 716: epoch %d/%d ,round %d ,top_M_Cand[%d] : %d , top_M_Cand_Dis[%d] : %f \n" ,p_i , psize_block, i , j , top_M_Cand[j] & 0x7FFFFFFF , j , top_M_Cand_dis[j]) ;
            // }
            // if(tid == 0)
            //     printf("can click here 700 epoch : %d/%d\n" , p_i , psize_block) ;
            merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , l_bound , r_bound , attrs , attr_dim) ;
            __syncthreads() ;
            if(tid == 0) {
                new_merge_cost += ((double)(clock64() - start) / 1582000) ;
                start = clock64() ;
            }
            __syncthreads() ;
            // for(unsigned j = tid ; j < DEGREE + TOPM ; j += blockDim.x * blockDim.y) {
            //     printf("line 720: epoch %d/%d ,round %d ,top_M_Cand[%d] : %d , top_M_Cand_Dis[%d] : %f \n" ,p_i , psize_block, i , j , top_M_Cand[j] & 0x7FFFFFFF , j , top_M_Cand_dis[j]) ;
            // }
            // if(tid == 0)
            //     printf("can click here 702 epoch : %d/%d\n" , p_i , psize_block) ;
            // 问题来了, 当图开始切换时, 当前部分结果的id应该采用局部id还是全局id? 在列表中的所有id皆采用全局id
            // 对后半段列表sort一次, 然后和recycle pool进行merge
            // bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM] , &top_M_Cand[TOPM] , DEGREE) ;
            if(tid < DEGREE)
                warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
            __syncthreads() ;
            if(tid == 0) {
                sort_cost2 += ((double)(clock64() - start) / 1582000) ;
                start = clock64() ;
            }
            // for(unsigned j = tid ; j < DEGREE + TOPM ; j += blockDim.x * blockDim.y) {
            //     printf("line 726: epoch %d/%d ,round %d ,top_M_Cand[%d] : %d , top_M_Cand_Dis[%d] : %f \n" ,p_i , psize_block, i , j , top_M_Cand[j] & 0x7FFFFFFF , j , top_M_Cand_dis[j]) ;
            // }
            // if(tid == 0)
            //     printf("can click here 707 epoch : %d/%d\n" , p_i , psize_block) ;
            if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                // 此时top_M_Cand + TOPM中的点一定是满足属性过滤条件的点
                merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE , recycle_list_size) ;
                __syncthreads() ;
                if(tid == 0) {
                    merge_cost += ((double)(clock64() - start) / 1582000) ;
                    // start = clock64() ;
                }
                __syncthreads() ;
            }
            // if(tid == 0)
            //     printf("can click here 708 epoch : %d/%d\n" , p_i , psize_block) ;
            // for(unsigned j = tid ; j < DEGREE + TOPM ; j += blockDim.x * blockDim.y) {
            //     printf("line 735: epoch %d/%d ,round %d ,top_M_Cand[%d] : %d , top_M_Cand_Dis[%d] : %f \n" ,p_i , psize_block, i , j , top_M_Cand[j] & 0x7FFFFFFF , j , top_M_Cand_dis[j]) ;
            // }
            // __syncthreads() ;
        }
        __syncthreads();

        if(tid == 0) {
            inner_loop_cost += ((double)(clock64() - inner_loop_start) / 1582000);
            outer_loop_start = clock64() ;
        }
        __syncthreads() ;

        // bug产生原因: 在扩展完候选点后将hash表清空了
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y)
            top_M_Cand[i] |= 0x80000000 ;
        for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
            hash_table[i] = 0xFFFFFFFF;
        }
        __syncthreads() ;

        // 修改一下顺序, 先扩展候选点, 然后再筛除不符合条件的点
        // 从global_edges 中扩展候选点
        if(p_i < psize_block - 1) {
            // 修改一下扩展候选点的逻辑, 向工作区中输送64个候选点, 计算距离后排序, 然后将前16个点送入top_M_Cand + TOPM 中
            unsigned next_pid = pids[plist_index + p_i + 1] ;
            unsigned j = 0 ;
            for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
                if(top_M_Cand[i % 64] == 0xFFFFFFFF) {
                    printf("error : top_M_Cand[i %% 64]!!!!\n") ;
                }
                unsigned pure_id = top_M_Cand[i % 64] & 0x7FFFFFFF ;
                while(j < edges_num_in_global_graph) {
                    unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + j] ;
                    j ++ ;
                    if(hash_insert(hash_table , next_local_id) != 0) {
                        // top_M_Cand[TOPM + i] = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ; 
                    // top_M_Cand_dis[TOPM + i]
                        work_mem_unsigned[i] = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                        break ; 
                    }
                    if(j == edges_num_in_global_graph) {
                        for(unsigned rand_p = 0 ; rand_p < 62500 ; rand_p ++)
                            if(hash_insert(hash_table , rand_p) != 0) {
                                work_mem_unsigned[i] = global_idx[seg_global_idx_start_idx[next_pid] + rand_p] ;
                                break ;
                            }
                    }
                }
            }
            __syncthreads() ;
            // 计算这64个点的距离
            for(unsigned i = threadIdx.y ; i < 128 ; i += blockDim.y) {
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    // if(top_M_Cand[TOPM + j] >= 62500) {
                    //     printf("top_M_cand[TOPM + j] : %d\n" , top_M_Cand[TOPM+j]) ;
                    // }
                    half4 val2 = Load(&values[work_mem_unsigned[i] * DIM + k * 4]);
                    // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].x, val2.x));
                    // val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].y, val2.y));
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                    // printf("val_res.x : %f , val_res.y : %f , tmp_val_sha.x : %f , tmp_val_sha.y : %f , val2.x : %f , val2.y : %f \n" ,
                    // __half2float(val_res.x) , __half2float(val_res.y) , __half2float(tmp_val_sha[k].x.x)
                    //  , __half2float(tmp_val_sha[k].y.x) , __half2float(val2.x.x) , __half2float(val2.y.x)) ;
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                }
            }
            __syncthreads() ;
            bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
            __syncthreads() ;
            // 重置哈希表
            for(unsigned i = tid ; i < HASHLEN ; i += blockDim.x * blockDim.y)
                hash_table[i] = 0xFFFFFFFF ;
            __syncthreads() ;
            // 将前16个点插入, 并同步插入到哈希表中
            for(unsigned i = tid ; i < DEGREE ; i += blockDim.x * blockDim.y) {
                top_M_Cand[TOPM + i] = work_mem_unsigned[i] ;
                hash_insert(hash_table , local_idx[work_mem_unsigned[i]]) ;
            }
            for(unsigned i = tid ; i < 128 - DEGREE ; i += blockDim.x * blockDim.y)
                hash_insert(hash_table , local_idx[work_mem_unsigned[DEGREE + i]]) ;
            // 将后64 - 16 个点与前表合并
            __syncthreads() ;

        }
        __syncthreads() ;

        // 一个分区的搜索完成, 将recycle_list中的内容合并到top_M_Cand中
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
            if(top_M_Cand[i] == 0xFFFFFFFF)
                continue ;
            bool flag = true ;
            unsigned pure_id = top_M_Cand[i] & 0x7FFFFFFF ;
            if(pure_id >= 1000000)
                printf("error at : %d , %d\n" , __LINE__ , pure_id) ;
            for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
                // printf("attrs[%d * %d + %d] : %f , l_bound[%d] : %f , r_bound[%d] : %f\n" , pure_id , attr_dim , j , attrs[pure_id * attr_dim + j] ,
                // j , l_bound[blockIdx.x * attr_dim +  j] , j , r_bound[blockIdx.x * attr_dim + j]) ;
                flag = (flag && (attrs[pure_id * attr_dim + j] >= l_bound[blockIdx.x * attr_dim + j]) && (attrs[pure_id * attr_dim + j] <= r_bound[blockIdx.x *attr_dim + j])) ;
            }
            if(!flag) {
                top_M_Cand[i] = 0xFFFFFFFF ;
                top_M_Cand_dis[i] = INF_DIS ;
            }
        }
        __syncthreads() ;
        // if(tid == 0)
            // printf("can click 775\n") ;
        // __syncthreads() ;
        // if(tid == 0) {
        //     start=  clock64() ;
        // }
        // __syncthreads() ;
        bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM);
        __syncthreads() ;
        // if(tid == 0) {
        //     sort_cost += ((double)(clock64() - start) / 1582000) ;
        //     start = clock64() ;
        // }
        // __syncthreads() ;
        // 若recycle_list中确实有点, 则合并
        if(recycle_list_id[0] != 0xFFFFFFFF) {
            merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
            __syncthreads() ;
            // if(tid == 0) {
            //     merge_cost += ((double)(clock64() - start) / 1582000) ;
            //     start = clock64() ;
            // }
        }
        if(p_i < psize_block - 1)
            merge_top(top_M_Cand , work_mem_unsigned + DEGREE, top_M_Cand_dis , work_mem_float + DEGREE , tid , 64 , TOPM) ;
        __syncthreads() ;
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
            if(top_M_Cand[i] != 0xFFFFFFFF)
                hash_insert(hash_table , local_idx[top_M_Cand[i] & 0x7FFFFFFF]) ;
            // top_M_Cand[i] |= 0x80000000 ;
        }
        __syncthreads() ;
        // for(unsigned j = tid ; j < DEGREE + TOPM ; j += blockDim.x * blockDim.y) {
            // printf("after merge : epoch %d/%d ,top_M_Cand[%d] : %d , top_M_Cand_Dis[%d] : %f \n" , p_i , psize_block , j , 
            // (top_M_Cand[j] != 0xFFFFFFFF ? top_M_Cand[j] & 0x7FFFFFFF : top_M_Cand[j]) , j , top_M_Cand_dis[j]) ;
        // }
        // __syncthreads() ;

        // for(unsigned j = tid ; j < DEGREE + TOPM ; j += blockDim.x * blockDim.y) {
            // printf("epoch %d/%d ,top_M_Cand[%d] : %d , top_M_Cand_Dis[%d] : %f \n" , p_i , psize_block , j , 
            // (top_M_Cand[j] != 0xFFFFFFFF ? top_M_Cand[j] & 0x7FFFFFFF : top_M_Cand[j]) , j , top_M_Cand_dis[j]) ;
        // }
        // __syncthreads() ;


        // printf("can click to 798\n") ;
        // __syncthreads() ;
        // for(unsigned j = tid ; j < DEGREE + TOPM ; j += blockDim.x * blockDim.y) {
        //     printf("epoch %d/%d ,top_M_Cand[%d] : %d , top_M_Cand_Dis[%d] : %f \n" , p_i , psize_block , j , 
        //     (top_M_Cand[j] != 0xFFFFFFFF ? top_M_Cand[j] & 0x7FFFFFFF : top_M_Cand[j]) , j , top_M_Cand_dis[j]) ;
        // }
        // __syncthreads() ;
        if(tid == 0) {
            outer_loop_cost += ((double)(clock64() - outer_loop_start) / 1582000);
        }
        __syncthreads() ;
    }
    __syncthreads() ;
    if(tid == 0) {
        overall_cost += ((double)(clock64() - overall_start) / 1582000);
    }
    __syncthreads() ;
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    if(tid == 0 ) {
        printf("block : %d ,总体耗时: %f , 距离计算时间: %f , 排序时间: %f , 排序时间(第二次): %f , 合并时间: %f , 合并时间(函数2): %f, 选点耗时: %f,  跳转次数: %d, 其它时间: %f , 内部循环耗时: %f , 外部循环耗时: %f , 重置哈希表耗时: %f , 载入待计算点耗时: %f\n" ,
         blockIdx.x , overall_cost , dis_cal_cost , sort_cost , sort_cost2 , merge_cost , new_merge_cost ,select_point_cost , 
         total_jump_num , overall_cost - dis_cal_cost - sort_cost - sort_cost2 - merge_cost - new_merge_cost , inner_loop_cost , 
         outer_loop_cost , hash_reset_cost , load_point_cost) ;
         jump_num_cnt[blockIdx.x] = total_jump_num ; 
    } 
}

// TOPM是candidate队列长度，K是TOP-K也是图索引的度数，DIM是数据集维度，MAX_ITER是最大迭代次数（1000足够），INF_DIS表示无穷大的浮点数
__global__ void select_path_contain_attr_filter_V2_without_output(const unsigned* __restrict__ all_graphs, const half* __restrict__ values, const half* __restrict__ query_data, 
    unsigned* results_id, float* results_dis, unsigned* ent_pts , unsigned DEGREE , unsigned TOPM , unsigned DIM , const unsigned* graph_start_index ,const unsigned* graph_size ,
    const unsigned* __restrict__ pids , const unsigned* p_size ,const unsigned* p_start_index ,const float* l_bound ,const float* r_bound ,const float* __restrict__ attrs , unsigned attr_dim , const unsigned* __restrict__ global_graph , unsigned edges_num_in_global_graph,
    unsigned recycle_list_size ,const unsigned* __restrict__ global_idx ,const unsigned* __restrict__ local_idx ,const unsigned* seg_global_idx_start_idx , unsigned pnum ,unsigned K, unsigned* result_k_id ,const unsigned* task_id_mapper) {
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数
    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    unsigned* recycle_list_id = (unsigned*) shared_mem ;
    float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    // __shared__ unsigned recycle_list_id[64] ;
    // __shared__ float recycle_list_dis[64] ;
    // __shared__ unsigned top_M_Cand[64 + 16];
    unsigned* top_M_Cand = (unsigned*) (recycle_list_dis + recycle_list_size) ;
    // __shared__ float top_M_Cand_dis[64 + 16];
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE) ;
    float* q_l_bound = (float*)(top_M_Cand_dis + TOPM + DEGREE) ;
    float* q_r_bound = (float*)(q_l_bound + attr_dim) ;

    // 这里应该怎么处理8字节的对齐问题？
    // __shared__ half4 tmp_val_sha[(unsigned)(128 / 4)];
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) +
     (TOPM + DEGREE) * sizeof(unsigned) + (TOPM + DEGREE) * sizeof(float) + 2 * attr_dim * sizeof(float)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[128] ;
    __shared__ float work_mem_float[128] ;

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    // __shared__ unsigned long long start;
    // __shared__ double dis_cal_cost , sort_cost , merge_cost , new_merge_cost  , sort_cost2 ;
    // __shared__ unsigned total_jump_num ;
    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = task_id_mapper[blockIdx.x], laneid = threadIdx.x;  
    // 若线程块发射顺序与blockIdx.x的编号无关, 则可以手动实现按到达顺序取任务的排队器
    unsigned psize_block = p_size[bid] , plist_index = p_start_index[bid] ;

    // if(tid == 0) {
    //     dis_cal_cost = sort_cost = merge_cost = new_merge_cost = sort_cost2=  0.0 ;
    //     total_jump_num = 0 ;
    // }
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
        q_l_bound[i] = l_bound[bid * attr_dim + i] ;
        q_r_bound[i] = r_bound[bid * attr_dim + i] ;
    }
    // 没必要在每个分区开始时都把recycle_list复原, 每个分区得到的入口点肯定是越近越好
    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        recycle_list_id[i] = 0xFFFFFFFF ;
        recycle_list_dis[i] = INF_DIS ;
    }
    // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
    for(unsigned i = tid; i < DEGREE; i += blockDim.x * blockDim.y){
        top_M_Cand[TOPM + i] = global_idx[seg_global_idx_start_idx[pids[plist_index]] + ent_pts[i]];
        hash_insert(hash_table , ent_pts[i]) ;
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}
    __syncthreads();
 
  
    /**
        1.先明确对每个分区要做什么, 首先, 将候选点载入top_M_Cand 的末DEGREE个位置, 然后排序
        2.将循环池重置为初始值
        3.从初始点开始扩展, 每次将DEGREE个邻居存放入top_M_Cand的末位, 被淘汰的分区内的点会插入循环池中
        4.当前分区查找结束后,将top_M_Cand中不符合范围约束的点去掉,用循环池中的点替代,然后依据这些点从全局图中扩展到下一个分区的入口点

        step4 的具体实现时, 可以先把不符合范围约束的点的距离改为INF,然后对列表排序, 和recycle_list合并
    **/
    for(unsigned p_i = 0 ; p_i < psize_block; p_i ++) {
        // 得到图的入口
        unsigned pid = pids[plist_index + p_i] ;
        const unsigned* graph = all_graphs + graph_start_index[pid] ;
        __syncthreads() ;
        #pragma unroll
        for(unsigned i = threadIdx.y; i < DEGREE; i += blockDim.y){
            // threadIdx.y 0-5 , blockDim.y 即 6
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                // 将warp内所有子线程的计算结果归并到一起
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }
            if(laneid == 0){
                top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
            }
        }
        __syncthreads();

        

        

      
        // 直接对整个列表做距离计算
        if(tid < DEGREE)
            warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
        __syncthreads() ;
        // 此处被挤出去的点也应该合并到recycle_list 上
        // 将两个列表归并到一起
        merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
        __syncthreads() ;
        if(tid < DEGREE)
            warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE) ; 
        __syncthreads() ;
        if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
            merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE , recycle_list_size) ;
            __syncthreads() ;
        }
        // 初始化 recycle_list
        // for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        //     recycle_list_id[i] = 0xFFFFFFFF ;
        //     recycle_list_dis[i] = INF_DIS ;
        // }

        __syncthreads();

        // begin search
        for(unsigned i = 0; i < MAX_ITER; i++){

            // 每四轮将hash表重置一次
            if((i + 1) % HASH_RESET == 0){
                for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                    hash_table[j] = 0xFFFFFFFF;
                }
                __syncthreads();

                for(unsigned j = tid; j < TOPM + DEGREE; j += blockDim.x * blockDim.y){
                    // 将此时的TOPM + K 个候选点插入哈希表中
                    // hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
                    if(top_M_Cand[j] != 0xFFFFFFFF) {
                        hash_insert(hash_table , local_idx[(top_M_Cand[j] & 0x7FFFFFFF)]) ; 
                    }
                }
            }
            __syncthreads();

            if(tid < 32){
                // 从前32个点中找到需要扩展的点, 
                for(unsigned j = laneid; j < TOPM; j+=32){
                    unsigned n_p = 0;
                    if((top_M_Cand[j] & 0x80000000) == 0){
                        n_p = 1;
                    }
                
                    // *** 坑点: __ballot_sync会等待掩码影响的线程执行到__ballot_sync处, 若把0xffffffff设置为掩码, 个别线程可能因循环条件影响提前退出, 
                    // 无法执行到此处, 造成程序无限空等
                    // unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                    unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                
                    // if(bid == 0)
                    //     printf("tid : %d ," , tid) ;
                    if(ballot_res > 0){
                        // 这里的ffs必须考虑返回值等于-1的情况
                        if(laneid == (__ffs(ballot_res) - 1)){
                            to_explore = local_idx[top_M_Cand[j]];
                            top_M_Cand[j] |= 0x80000000;
                        }

                    
                        break;
                    }
                    to_explore = 0xFFFFFFFF;
                }
            }
            __syncthreads();

            if(to_explore == 0xFFFFFFFF) {
                if(bid == 0 && tid == 0) printf("BREAK: %d\n", i);
                // printf("BREAK: %d\n" , i) ;
                jump_num_cnt[bid] += i ;
                break;
            }
            
            for(unsigned j = tid; j < DEGREE; j += blockDim.x * blockDim.y){
                unsigned to_append_local = graph[to_explore * DEGREE + j];
                unsigned to_append = global_idx[seg_global_idx_start_idx[pid] + to_append_local] ;
                top_M_Cand_dis[TOPM + j] = (hash_insert(hash_table , to_append_local) == 0 ? INF_DIS : 0.0) ;
                top_M_Cand[TOPM + j] = to_append;
            }
            __syncthreads();
            // calculate distance
            #pragma unroll
            for(unsigned j = threadIdx.y; j < DEGREE; j += blockDim.y){
                // 6个warp, 每个warp负责一个距离计算
                if(top_M_Cand_dis[TOPM + j] == INF_DIS)
                    continue ;

                    half2 val_res;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                        half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                    }
                    #pragma unroll
                    for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                        val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                    }

                    if(laneid == 0){
                        top_M_Cand_dis[TOPM + j] += __half2float(val_res.x) + __half2float(val_res.y) ;
                    }
            }
            __syncthreads();

            // #pragma unroll
            // for(unsigned i = tid ; i < DEGREE  ; i += blockDim.x * blockDim.y) {
            //     if(top_M_Cand_dis[TOPM + i] == INF_DIS)
            //         continue ;
            //     atomicAdd(dis_cal_cnt + bid , 1) ;
            //     unsigned pure_id = top_M_Cand[TOPM + i] & 0x3FFFFFFF ;
            //     half2 val_res;
            //     val_res.x = __float2half(0.0f); val_res.y = __float2half(0.0f);
            //     #pragma unroll
            //     for(unsigned j = 0; j < (DIM / 4); j ++){
            //         // warp中的每个子线程负责8个字节 即4个half类型分量的计算
            //         half4 val2 = Load(&values[pure_id * DIM + j * 4]);
            //         val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
            //         val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            //     }
            //     // if(laneid == 0){
            //         top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
            //     // }
            // }
            // __syncthreads() ;

            if(tid < DEGREE)
                warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
            __syncthreads() ;
            if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , q_l_bound , q_r_bound , attrs , attr_dim) ;
                __syncthreads() ;
                if(tid < DEGREE)
                    warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
                __syncthreads() ;
                if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                // 此时top_M_Cand + TOPM中的点一定是满足属性过滤条件的点
                    merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE , recycle_list_size) ;
                    __syncthreads() ;
                }
            }
            // 问题来了, 当图开始切换时, 当前部分结果的id应该采用局部id还是全局id? 在列表中的所有id皆采用全局id
            // 对后半段列表sort一次, 然后和recycle pool进行merge
            // bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM] , &top_M_Cand[TOPM] , DEGREE) ;

        }
        __syncthreads();

        // bug产生原因: 在扩展完候选点后将hash表清空了
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y)
            top_M_Cand[i] |= 0x80000000 ;
        for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
            hash_table[i] = 0xFFFFFFFF;
        }
        __syncthreads() ;

        // 修改一下顺序, 先扩展候选点, 然后再筛除不符合条件的点
        // 从global_edges 中扩展候选点
        if(p_i < psize_block - 1) {
            // 修改一下扩展候选点的逻辑, 向工作区中输送64个候选点, 计算距离后排序, 然后将前16个点送入top_M_Cand + TOPM 中
            unsigned next_pid = pids[plist_index + p_i + 1] ;
            unsigned j = 0 ;
            for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
                work_mem_unsigned[i] = 0xFFFFFFFFu ;
                work_mem_float[i] = INF_DIS ;
                if(TOPM < (i % 64) || top_M_Cand[i % 64] == 0xFFFFFFFFu)
                    continue ;
                unsigned pure_id = top_M_Cand[i % 64] & 0x7FFFFFFF ;
                while(j < edges_num_in_global_graph) {
                    unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + j] ;
                    j ++ ;
                    if(hash_insert(hash_table , next_local_id) != 0) {
                        work_mem_unsigned[i] = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                        break ; 
                    }
                    if(j == edges_num_in_global_graph) {
                        for(unsigned rand_p = 0 ; rand_p < 62500 ; rand_p ++)
                            if(hash_insert(hash_table , rand_p) != 0) {
                                work_mem_unsigned[i] = global_idx[seg_global_idx_start_idx[next_pid] + rand_p] ;
                                break ;
                            }
                    }
                }
            }
            __syncthreads() ;
            // 计算这64个点的距离
            for(unsigned i = threadIdx.y ; i < 128 ; i += blockDim.y) {
                if(work_mem_unsigned[i] == 0xFFFFFFFFu)
                    continue ;
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[work_mem_unsigned[i] * DIM + k * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                }
            }
            __syncthreads() ;
            bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
            __syncthreads() ;
            // 重置哈希表
            for(unsigned i = tid ; i < HASHLEN ; i += blockDim.x * blockDim.y)
                hash_table[i] = 0xFFFFFFFF ;
            __syncthreads() ;
            // 将前16个点插入, 并同步插入到哈希表中
            for(unsigned i = tid ; i < DEGREE ; i += blockDim.x * blockDim.y) {
                top_M_Cand[TOPM + i] = work_mem_unsigned[i] ;
                hash_insert(hash_table , local_idx[work_mem_unsigned[i]]) ;
            }
            for(unsigned i = tid ; i < 128 - DEGREE ; i += blockDim.x * blockDim.y)
                if(work_mem_unsigned[DEGREE + i] != 0xFFFFFFFFu)
                    hash_insert(hash_table , local_idx[work_mem_unsigned[DEGREE + i]]) ;
            // 将后64 - 16 个点与前表合并
            __syncthreads() ;

        }
        __syncthreads() ;

        // 一个分区的搜索完成, 将recycle_list中的内容合并到top_M_Cand中
        // for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
        //     if(top_M_Cand[i] == 0xFFFFFFFF)
        //         continue ;
        //     bool flag = true ;
        //     unsigned pure_id = top_M_Cand[i] & 0x7FFFFFFF ;
        //     for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
        //         flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
        //     }
        //     if(!flag) {
        //         top_M_Cand[i] = 0xFFFFFFFF ;
        //         top_M_Cand_dis[i] = INF_DIS ;
        //     }
        // }
        // __syncthreads() ;
        // bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM);
        // __syncthreads() ;
        // // 若recycle_list中确实有点, 则合并
        // if(recycle_list_id[0] != 0xFFFFFFFF) {
        //     merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
        //     __syncthreads() ;
        // }
        if(p_i < psize_block - 1)
            merge_top(top_M_Cand , work_mem_unsigned + DEGREE, top_M_Cand_dis , work_mem_float + DEGREE , tid , 64 , TOPM) ;
        __syncthreads() ;
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
            if(top_M_Cand[i] != 0xFFFFFFFF)
                hash_insert(hash_table , local_idx[top_M_Cand[i] & 0x7FFFFFFF]) ;
        }
        __syncthreads() ;
    }


    for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
        if(top_M_Cand[i] == 0xFFFFFFFF)
            continue ;
        bool flag = true ;
        unsigned pure_id = top_M_Cand[i] & 0x7FFFFFFF ;
        for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
            flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
        }
        if(!flag) {
            top_M_Cand[i] = 0xFFFFFFFF ;
            top_M_Cand_dis[i] = INF_DIS ;
        }
    }
    __syncthreads() ;
    bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM);
    __syncthreads() ;
    // 若recycle_list中确实有点, 则合并
    if(recycle_list_id[0] != 0xFFFFFFFF) {
        merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
        __syncthreads() ;
    }

    // __syncthreads() ;
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    for(unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        result_k_id[bid * K + i] = (top_M_Cand[i] & 0x7FFFFFFF) ;
        results_dis[bid * K + i] = top_M_Cand_dis[i] ;
    }
}


// TOPM是candidate队列长度，K是TOP-K也是图索引的度数，DIM是数据集维度，MAX_ITER是最大迭代次数（1000足够），INF_DIS表示无穷大的浮点数
__global__ void select_path_contain_attr_filter_V2_without_output_with_postfilter(const unsigned* __restrict__ all_graphs, const half* __restrict__ values, const half* __restrict__ query_data, 
    unsigned* results_id, float* results_dis, unsigned* ent_pts , unsigned* ent_pts_post , unsigned DEGREE , unsigned TOPM , unsigned DIM , const unsigned* graph_start_index ,const unsigned* graph_size ,
    const unsigned* __restrict__ pids , const unsigned* p_size ,const unsigned* p_start_index ,const float* l_bound ,const float* r_bound ,const float* __restrict__ attrs , unsigned attr_dim , const unsigned* __restrict__ global_graph , unsigned edges_num_in_global_graph,
    unsigned recycle_list_size ,const unsigned* __restrict__ global_idx ,const unsigned* __restrict__ local_idx ,const unsigned* seg_global_idx_start_idx , unsigned pnum ,unsigned K, unsigned* result_k_id , const unsigned* __restrict__ bigGraph, const unsigned bigDegree 
    , const unsigned postfilter_threshold, const unsigned* task_id_mapper) {
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数
    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    unsigned* recycle_list_id = (unsigned*) shared_mem ;
    float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    // __shared__ unsigned recycle_list_id[64] ;
    // __shared__ float recycle_list_dis[64] ;
    // __shared__ unsigned top_M_Cand[64 + 16];
    unsigned* top_M_Cand = (unsigned*) (recycle_list_dis + recycle_list_size) ;
    // __shared__ float top_M_Cand_dis[64 + 16];
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + bigDegree) ;
    float* q_l_bound = (float*)(top_M_Cand_dis + TOPM + bigDegree) ;
    float* q_r_bound = (float*)(q_l_bound + attr_dim) ;

    // 这里应该怎么处理8字节的对齐问题？
    // __shared__ half4 tmp_val_sha[(unsigned)(128 / 4)];
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) +
     (TOPM + bigDegree) * sizeof(unsigned) + (TOPM + bigDegree) * sizeof(float) + 2 * attr_dim * sizeof(float)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[128] ;
    __shared__ float work_mem_float[128] ;

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    // __shared__ unsigned long long start;
    // __shared__ double dis_cal_cost , sort_cost , merge_cost , new_merge_cost  , sort_cost2 ;
    // __shared__ unsigned total_jump_num ;
    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = task_id_mapper[blockIdx.x], laneid = threadIdx.x;  
    // 若线程块发射顺序与blockIdx.x的编号无关, 则可以手动实现按到达顺序取任务的排队器
    unsigned psize_block = p_size[bid] , plist_index = p_start_index[bid] ;

    // if(tid == 0) {
    //     dis_cal_cost = sort_cost = merge_cost = new_merge_cost = sort_cost2=  0.0 ;
    //     total_jump_num = 0 ;
    // }
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
        q_l_bound[i] = l_bound[bid * attr_dim + i] ;
        q_r_bound[i] = r_bound[bid * attr_dim + i] ;
    }
    // 没必要在每个分区开始时都把recycle_list复原, 每个分区得到的入口点肯定是越近越好
    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        recycle_list_id[i] = 0xFFFFFFFF ;
        recycle_list_dis[i] = INF_DIS ;
    }

    // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
    for(unsigned i = tid; i < DEGREE; i += blockDim.x * blockDim.y){
        top_M_Cand[TOPM + i] = global_idx[seg_global_idx_start_idx[pids[plist_index]] + ent_pts[i]];
        hash_insert(hash_table , top_M_Cand[TOPM + i]) ;
    }
  
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}
    __syncthreads();
 
    // if(tid == 0) {
    //     printf("psize_block : %d , postfilter_threshold : %d , bigDegree : %d\n" , psize_block , postfilter_threshold , bigDegree) ;
    //     // show ent pts 
    //     printf("small degree ent pts [") ;
    //     for(unsigned i = 0 ; i < DEGREE ; i ++) {
    //         printf("%d" , ent_pts[i]) ;
    //         if(i < DEGREE - 1)
    //             printf(",") ;
    //     }
    //     printf("]\n") ;
    //     printf("big degree ent pts [") ;
    //     for(unsigned i = 0 ; i < bigDegree ; i ++) {
    //         printf("%d" , ent_pts_post[i]) ;
    //         if(i < bigDegree - 1)
    //             printf(",") ;
    //     }
    //     printf("]\n") ;

    //     printf("bigGraph的前32行 :[") ;
    //     for(unsigned i = 0 ; i < 32 ; i ++) {
    //         printf("[") ;
    //         for(unsigned j = 0 ; j < 32 ; j ++) {
    //             printf("%d" , bigGraph[i * bigDegree + j]) ;
    //             if(j < 31)
    //                 printf(",") ;
    //         }
    //         printf("]") ;
    //         if(i < 31)
    //             printf("\n") ;
    //     }
    //     printf("]\n") ;
    // }
    // __syncthreads() ;
  
    /**
        1.先明确对每个分区要做什么, 首先, 将候选点载入top_M_Cand 的末DEGREE个位置, 然后排序
        2.将循环池重置为初始值
        3.从初始点开始扩展, 每次将DEGREE个邻居存放入top_M_Cand的末位, 被淘汰的分区内的点会插入循环池中
        4.当前分区查找结束后,将top_M_Cand中不符合范围约束的点去掉,用循环池中的点替代,然后依据这些点从全局图中扩展到下一个分区的入口点

        step4 的具体实现时, 可以先把不符合范围约束的点的距离改为INF,然后对列表排序, 和recycle_list合并
    **/
    if(psize_block < postfilter_threshold)
    for(unsigned p_i = 0 ; p_i < psize_block; p_i ++) {
        // 得到图的入口
        unsigned pid = pids[plist_index + p_i] ;
        const unsigned* graph = all_graphs + graph_start_index[pid] ;
        __syncthreads() ;
        #pragma unroll
        for(unsigned i = threadIdx.y; i < DEGREE; i += blockDim.y){
            // threadIdx.y 0-5 , blockDim.y 即 6
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                // 将warp内所有子线程的计算结果归并到一起
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }
            if(laneid == 0){
                top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
            }
        }
        __syncthreads();
     
        // 直接对整个列表做距离计算
        if(tid < DEGREE)
            warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
        __syncthreads() ;
        // 此处被挤出去的点也应该合并到recycle_list 上
        // 将两个列表归并到一起
        merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
        __syncthreads() ;
        if(tid < DEGREE)
            warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE) ; 
        __syncthreads() ;
        if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
            merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE , recycle_list_size) ;
            __syncthreads() ;
        }
        // 初始化 recycle_list
        // for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        //     recycle_list_id[i] = 0xFFFFFFFF ;
        //     recycle_list_dis[i] = INF_DIS ;
        // }

        __syncthreads();

        // begin search
        for(unsigned i = 0; i < MAX_ITER; i++){

            // 每四轮将hash表重置一次
            if((i + 1) % HASH_RESET == 0){
                for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                    hash_table[j] = 0xFFFFFFFF;
                }
                __syncthreads();

                for(unsigned j = tid; j < TOPM + DEGREE; j += blockDim.x * blockDim.y){
                    // 将此时的TOPM + K 个候选点插入哈希表中
                    // hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
                    if(top_M_Cand[j] != 0xFFFFFFFF) {
                        hash_insert(hash_table , local_idx[(top_M_Cand[j] & 0x7FFFFFFF)]) ; 
                    }
                }
            }
            __syncthreads();

            if(tid < 32){
                // 从前32个点中找到需要扩展的点, 
                for(unsigned j = laneid; j < TOPM; j+=32){
                    unsigned n_p = 0;
                    if((top_M_Cand[j] & 0x80000000) == 0){
                        n_p = 1;
                    }
                
                    // *** 坑点: __ballot_sync会等待掩码影响的线程执行到__ballot_sync处, 若把0xffffffff设置为掩码, 个别线程可能因循环条件影响提前退出, 
                    // 无法执行到此处, 造成程序无限空等
                    // unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                    unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                
                    // if(bid == 0)
                    //     printf("tid : %d ," , tid) ;
                    if(ballot_res > 0){
                        // 这里的ffs必须考虑返回值等于-1的情况
                        if(laneid == (__ffs(ballot_res) - 1)){
                            to_explore = local_idx[top_M_Cand[j]];
                            top_M_Cand[j] |= 0x80000000;
                        }

                    
                        break;
                    }
                    to_explore = 0xFFFFFFFF;
                }
            }
            __syncthreads();

            if(to_explore == 0xFFFFFFFF) {
                if(bid == 0 && tid == 0) printf("BREAK: %d\n", i);
                // printf("BREAK: %d\n" , i) ;
                jump_num_cnt[bid] += i ;
                break;
            }
            
            for(unsigned j = tid; j < DEGREE; j += blockDim.x * blockDim.y){
                unsigned to_append_local = graph[to_explore * DEGREE + j];
                unsigned to_append = global_idx[seg_global_idx_start_idx[pid] + to_append_local] ;
                top_M_Cand_dis[TOPM + j] = (hash_insert(hash_table , to_append_local) == 0 ? INF_DIS : 0.0) ;
                top_M_Cand[TOPM + j] = to_append;
            }
            __syncthreads();
            // calculate distance
            #pragma unroll
            for(unsigned j = threadIdx.y; j < DEGREE; j += blockDim.y){
                // 6个warp, 每个warp负责一个距离计算
                if(top_M_Cand_dis[TOPM + j] == INF_DIS)
                    continue ;

                    half2 val_res;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                        half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                    }
                    #pragma unroll
                    for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                        val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                    }

                    if(laneid == 0){
                        top_M_Cand_dis[TOPM + j] += __half2float(val_res.x) + __half2float(val_res.y) ;
                    }
            }
            __syncthreads();

            // #pragma unroll
            // for(unsigned i = tid ; i < DEGREE  ; i += blockDim.x * blockDim.y) {
            //     if(top_M_Cand_dis[TOPM + i] == INF_DIS)
            //         continue ;
            //     atomicAdd(dis_cal_cnt + bid , 1) ;
            //     unsigned pure_id = top_M_Cand[TOPM + i] & 0x3FFFFFFF ;
            //     half2 val_res;
            //     val_res.x = __float2half(0.0f); val_res.y = __float2half(0.0f);
            //     #pragma unroll
            //     for(unsigned j = 0; j < (DIM / 4); j ++){
            //         // warp中的每个子线程负责8个字节 即4个half类型分量的计算
            //         half4 val2 = Load(&values[pure_id * DIM + j * 4]);
            //         val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
            //         val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            //     }
            //     // if(laneid == 0){
            //         top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
            //     // }
            // }
            // __syncthreads() ;

            if(tid < DEGREE)
                warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
            __syncthreads() ;
            if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , q_l_bound , q_r_bound , attrs , attr_dim) ;
                __syncthreads() ;
                if(tid < DEGREE)
                    warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
                __syncthreads() ;
                if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                // 此时top_M_Cand + TOPM中的点一定是满足属性过滤条件的点
                    merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE , recycle_list_size) ;
                    __syncthreads() ;
                }
            }
            // 问题来了, 当图开始切换时, 当前部分结果的id应该采用局部id还是全局id? 在列表中的所有id皆采用全局id
            // 对后半段列表sort一次, 然后和recycle pool进行merge
            // bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM] , &top_M_Cand[TOPM] , DEGREE) ;

        }
        __syncthreads();

        // bug产生原因: 在扩展完候选点后将hash表清空了
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y)
            top_M_Cand[i] |= 0x80000000 ;
        for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
            hash_table[i] = 0xFFFFFFFF ;
        }
        __syncthreads() ;

        // 修改一下顺序, 先扩展候选点, 然后再筛除不符合条件的点
        // 从global_edges 中扩展候选点
        if(p_i < psize_block - 1) {
            // 修改一下扩展候选点的逻辑, 向工作区中输送64个候选点, 计算距离后排序, 然后将前16个点送入top_M_Cand + TOPM 中
            unsigned next_pid = pids[plist_index + p_i + 1] ;
            unsigned j = 0 ;
            for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
                work_mem_unsigned[i] = 0xFFFFFFFFu ;
                work_mem_float[i] = INF_DIS ;
                if(TOPM < (i % 64) || top_M_Cand[i % 64] == 0xFFFFFFFFu)
                    continue ;
                unsigned pure_id = top_M_Cand[i % 64] & 0x7FFFFFFF ;
                while(j < edges_num_in_global_graph) {
                    unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + j] ;
                    j ++ ;
                    if(hash_insert(hash_table , next_local_id) != 0) {
                        work_mem_unsigned[i] = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                        break ; 
                    }
                    if(j == edges_num_in_global_graph) {
                        for(unsigned rand_p = 0 ; rand_p < 62500 ; rand_p ++)
                            if(hash_insert(hash_table , rand_p) != 0) {
                                work_mem_unsigned[i] = global_idx[seg_global_idx_start_idx[next_pid] + rand_p] ;
                                break ;
                            }
                    }
                }
            }
            __syncthreads() ;
            // 计算这64个点的距离
            for(unsigned i = threadIdx.y ; i < 128 ; i += blockDim.y) {
                if(work_mem_unsigned[i] == 0xFFFFFFFFu)
                    continue ;
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[work_mem_unsigned[i] * DIM + k * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                }
            }
            __syncthreads() ;
            bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
            __syncthreads() ;
            // 重置哈希表
            for(unsigned i = tid ; i < HASHLEN ; i += blockDim.x * blockDim.y)
                hash_table[i] = 0xFFFFFFFF ;
            __syncthreads() ;
            // 将前16个点插入, 并同步插入到哈希表中
            for(unsigned i = tid ; i < DEGREE ; i += blockDim.x * blockDim.y) {
                top_M_Cand[TOPM + i] = work_mem_unsigned[i] ;
                hash_insert(hash_table , local_idx[work_mem_unsigned[i]]) ;
            }
            for(unsigned i = tid ; i < 128 - DEGREE ; i += blockDim.x * blockDim.y)
                if(work_mem_unsigned[DEGREE + i] != 0xFFFFFFFFu)
                    hash_insert(hash_table , local_idx[work_mem_unsigned[DEGREE + i]]) ;
            // 将后64 - 16 个点与前表合并
            __syncthreads() ;

        }
        __syncthreads() ;

        // 一个分区的搜索完成, 将recycle_list中的内容合并到top_M_Cand中
        // for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
        //     if(top_M_Cand[i] == 0xFFFFFFFF)
        //         continue ;
        //     bool flag = true ;
        //     unsigned pure_id = top_M_Cand[i] & 0x7FFFFFFF ;
        //     for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
        //         flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
        //     }
        //     if(!flag) {
        //         top_M_Cand[i] = 0xFFFFFFFF ;
        //         top_M_Cand_dis[i] = INF_DIS ;
        //     }
        // }
        // __syncthreads() ;
        // bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM);
        // __syncthreads() ;
        // // 若recycle_list中确实有点, 则合并
        // if(recycle_list_id[0] != 0xFFFFFFFF) {
        //     merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
        //     __syncthreads() ;
        // }
        if(p_i < psize_block - 1)
            merge_top(top_M_Cand , work_mem_unsigned + DEGREE, top_M_Cand_dis , work_mem_float + DEGREE , tid , 64 , TOPM) ;
        __syncthreads() ;
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
            if(top_M_Cand[i] != 0xFFFFFFFF)
                hash_insert(hash_table , local_idx[top_M_Cand[i] & 0x7FFFFFFF]) ;
        }
        __syncthreads() ;
    }
    else {
        
        // 先搞定入口点
        for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
            hash_table[i] = 0xFFFFFFFF;
        }
        for(unsigned i = tid; i < bigDegree; i += blockDim.x * blockDim.y){
            top_M_Cand[TOPM + i] = ent_pts_post[i];
            hash_insert(hash_table , top_M_Cand[TOPM + i]) ;
        }
        __syncthreads() ;

        #pragma unroll
        for(unsigned j = threadIdx.y; j < bigDegree ; j += blockDim.y){
            // 6个warp, 每个warp负责一个距离计算
            // if(top_M_Cand[TOPM + j] == 0xFFFFFFFFu)
            //     continue ;

            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            unsigned global_id = top_M_Cand[TOPM + j] ;
            for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                half4 val2 = Load(&values[global_id * DIM + k * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }

            if(laneid == 0){
                top_M_Cand_dis[TOPM + j] = __half2float(val_res.x) + __half2float(val_res.y) ;
            }
        }
        __syncthreads();
        // if(tid == 0 && bid == 0) {
        //     printf("初始化后的列表 [") ;
        //     for(unsigned i = 0 ; i < TOPM + bigDegree; i ++) {
        //         printf("(%d,%f)" , top_M_Cand[i] & 0x7FFFFFFFu , top_M_Cand_dis[i]) ;
        //         if(i < TOPM - 1)
        //             printf(";") ;
        //     }  
        //     printf("]\n") ;
        // }
        // __syncthreads() ;
        
        // if(tid == 0 && bid == 999) {
        //     printf("topMCand[") ;
        //     for(unsigned j = 0 ;j  < TOPM ; j ++) {
        //         printf("(%d,%f)" , top_M_Cand[j]&0x7FFFFFFFu , top_M_Cand_dis[j]) ;
        //         if(j < TOPM - 1)
        //             printf(";") ;
        //     }
        //     printf("]\n") ;

        //     printf("topMCand + TOPM[") ;
        //     for(unsigned j = 0 ;j  < bigDegree ; j ++) {
        //         printf("(%d,%f)" , top_M_Cand[TOPM + j]&0x7FFFFFFFu , top_M_Cand_dis[TOPM + j]) ;
        //         if(j < bigDegree - 1)
        //             printf(";") ;
        //     }
        //     printf("]\n") ;


        //     printf("ent_pts_post[") ;
        //     for(unsigned j = 0 ;j  < bigDegree ; j ++) {
        //         printf("(%d)" , ent_pts_post[j]) ;
        //         if(j < bigDegree - 1)
        //             printf(";") ;
        //     }
        //     printf("]\n") ;
        // }
        // __syncthreads() ;
        if(tid < bigDegree)
            warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , bigDegree);
            // bitonic_sort_id_by_dis_no_explore(work_mem_float, work_mem_unsigned, extend_points_num) ;
        __syncthreads() ;  
        merge_top(top_M_Cand , top_M_Cand + TOPM , top_M_Cand_dis , top_M_Cand_dis + TOPM , tid , bigDegree , TOPM) ;
        __syncthreads() ;
        


        for(unsigned i = 0; i < MAX_ITER; i++) {

            // 每四轮将hash表重置一次
            if((i + 1) % 4 == 0){
                for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                    hash_table[j] = 0xFFFFFFFF;
                }
                __syncthreads();

                for(unsigned j = tid; j < TOPM + bigDegree ; j += blockDim.x * blockDim.y){
                    // 将此时的TOPM + K 个候选点插入哈希表中
                    // hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
                    if(top_M_Cand[j] != 0xFFFFFFFF) {
                        hash_insert(hash_table , top_M_Cand[j] & 0x7FFFFFFF) ; 
                    }
                }
            }
            __syncthreads();

            if(tid < 32){
                // 从前32个点中找到需要扩展的点, 
                for(unsigned j = laneid; j < TOPM; j+=32){
                    unsigned n_p = 0;
                    if((top_M_Cand[j] & 0x80000000) == 0){
                        n_p = 1;
                    }
                
                    // *** 坑点: __ballot_sync会等待掩码影响的线程执行到__ballot_sync处, 若把0xffffffff设置为掩码, 个别线程可能因循环条件影响提前退出, 
                    // 无法执行到此处, 造成程序无限空等
                    // unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                    unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                
                    // if(bid == 0)
                    //     printf("tid : %d ," , tid) ;
                    if(ballot_res > 0){
                        // 这里的ffs必须考虑返回值等于-1的情况
                        if(laneid == (__ffs(ballot_res) - 1)){
                            // to_explore = local_idx[top_M_Cand[j]];
                            to_explore = top_M_Cand[j] ;
                            top_M_Cand[j] |= 0x80000000;
                        }
                        break;
                    }
                    to_explore = 0xFFFFFFFF;
                }
            }
            __syncthreads();

            // if(tid == 0) {
            //     printf("can click %d\n" , __LINE__) ;
            // }
            // __syncthreads() ;

            if(to_explore == 0xFFFFFFFF) {
                if(bid == 0 && tid == 0) printf("BREAK: %d\n", i);
                // printf("BREAK: %d\n" , i) ;
                // jump_num_cnt[bid] += i ;
                break;
            }

            for(unsigned j = tid; j < bigDegree; j += blockDim.x * blockDim.y){
                unsigned to_append = bigGraph[to_explore * bigDegree + j] ;
                top_M_Cand[TOPM + j] = ((hash_insert(hash_table , to_append) == 0) ? 0xFFFFFFFFu : to_append) ;
                top_M_Cand_dis[TOPM + j] = (top_M_Cand[TOPM + j] == 0xFFFFFFFFu ? INF_DIS : 0.0) ;
            }

            __syncthreads() ;
        
            // calculate distance
            #pragma unroll
            for(unsigned j = threadIdx.y; j < bigDegree ; j += blockDim.y){
                // 6个warp, 每个warp负责一个距离计算
                if(top_M_Cand[TOPM + j] == 0xFFFFFFFFu)
                    continue ;

                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                unsigned global_id = top_M_Cand[TOPM + j] ;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[global_id * DIM + k * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    top_M_Cand_dis[TOPM + j] += __half2float(val_res.x) + __half2float(val_res.y) ;
                }
            }
            __syncthreads();


            if(tid < bigDegree)
                warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , bigDegree);
            // bitonic_sort_id_by_dis_no_explore(work_mem_float, work_mem_unsigned, extend_points_num) ;
            __syncthreads() ;
            // 若当前轮次扩展存在有效点, 则合并
            // if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
            //     merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , bigDegree , TOPM , q_l_bound , q_r_bound , attrs , attr_dim) ;
            //     __syncthreads() ;
            //     if(tid < bigDegree)
            //         warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , bigDegree);
            //     __syncthreads() ;
            //     if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
            //     // 此时top_M_Cand + TOPM中的点一定是满足属性过滤条件的点
            //         merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , bigDegree , recycle_list_size) ;
            //         __syncthreads() ;
            //     }
            // }

            // if(bid == 999 && tid == 0 && i < 10) {
            //     printf("epoch%d topMCand + TOPM[" , i) ;
            //     for(unsigned j = 0 ;j  < bigDegree ; j ++) {
            //         printf("(%d,%f)" , top_M_Cand[j]&0x7FFFFFFFu , top_M_Cand_dis[j]) ;
            //         if(j < bigDegree - 1)
            //             printf(";") ;
            //     }
            //     printf("]\n") ;

            //     printf("**epoch%d recycle_list[" , i) ;
            //     for(unsigned j = 0 ;j  < recycle_list_size ; j ++) {
            //         printf("(%d,%f)" , recycle_list_id[j]&0x7FFFFFFFu , recycle_list_dis[j]) ;
            //         if(j < recycle_list_size - 1)
            //             printf(";") ;
            //     }
            //     printf("]\n") ;
            // }   
            // __syncthreads() ;
            merge_top(top_M_Cand , top_M_Cand + TOPM , top_M_Cand_dis , top_M_Cand_dis + TOPM , tid , bigDegree , TOPM) ;
            __syncthreads() ;
        }
    }
    __syncthreads() ;
    
    // if(tid == 0 && bid == 0) {
    //     printf("后过滤前的列表 [") ;
    //     for(unsigned i = 0 ; i < TOPM ; i ++) {
    //         printf("(%d,%f)" , top_M_Cand[i] & 0x7FFFFFFFu , top_M_Cand_dis[i]) ;
    //         if(i < TOPM - 1)
    //             printf(";") ;
    //     }  
    //     printf("]\n") ;
    //     printf("循环池 [") ;
    //     for(unsigned i = 0 ; i < recycle_list_size ; i ++) {
    //         printf("(%d,%f)" , recycle_list_id[i] & 0x7FFFFFFFu , recycle_list_dis[i]) ;
    //         if(i < recycle_list_size - 1)
    //             printf(";") ;
    //     }  
    //     printf("]\n") ;

        
    // }
    // __syncthreads() ;

    // 以下为对top_M_Cand 列表做后过滤, 并将recycle_list合并, 为公共流程
    for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
        if(top_M_Cand[i] == 0xFFFFFFFF)
            continue ;
        bool flag = true ;
        unsigned pure_id = top_M_Cand[i] & 0x7FFFFFFF ;
        for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
            flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
        }
        if(!flag) {
            top_M_Cand[i] = 0xFFFFFFFF ;
            top_M_Cand_dis[i] = INF_DIS ;
        }
    }
    __syncthreads() ;
    bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM);
    __syncthreads() ;
    // 若recycle_list中确实有点, 则合并
    if(recycle_list_id[0] != 0xFFFFFFFF) {
        merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
        __syncthreads() ;
    }

    // __syncthreads() ;
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    for(unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        result_k_id[bid * K + i] = (top_M_Cand[i] & 0x7FFFFFFF) ;
        results_dis[bid * K + i] = top_M_Cand_dis[i] ;
    }
}

__device__ unsigned post_merge_cnt[10000] ;
__constant__ unsigned base_number = 2 ;
// TOPM是candidate队列长度，K是TOP-K也是图索引的度数，DIM是数据集维度，MAX_ITER是最大迭代次数（1000足够），INF_DIS表示无穷大的浮点数
template<typename T = float>
__global__ void select_path_contain_attr_filter_V2_without_output_with_post_pre_filter(const unsigned* __restrict__ all_graphs, const half* __restrict__ values, const half* __restrict__ query_data, 
    unsigned* results_id, float* results_dis, unsigned* ent_pts , unsigned* ent_pts_post , unsigned DEGREE , unsigned TOPM , unsigned DIM , const unsigned* graph_start_index ,const unsigned* graph_size ,
    const unsigned* __restrict__ pids , const unsigned* p_size ,const unsigned* p_start_index ,const T* l_bound ,const T* r_bound ,const T* __restrict__ attrs , unsigned attr_dim , const unsigned* __restrict__ global_graph , unsigned edges_num_in_global_graph,
    unsigned recycle_list_size ,const unsigned* __restrict__ global_idx ,const unsigned* __restrict__ local_idx ,const unsigned* seg_global_idx_start_idx , unsigned pnum ,unsigned K, unsigned* result_k_id , const unsigned* __restrict__ bigGraph, const unsigned bigDegree 
    , const unsigned postfilter_threshold, const unsigned* task_id_mapper, const unsigned* little_seg_counts) {
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图s
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数
    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    unsigned* recycle_list_id = (unsigned*) shared_mem ;
    float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    // __shared__ unsigned recycle_list_id[64] ;
    // __shared__ float recycle_list_dis[64] ;
    // __shared__ unsigned top_M_Cand[64 + 16];
    unsigned* top_M_Cand = (unsigned*) (recycle_list_dis + recycle_list_size) ;
    // __shared__ float top_M_Cand_dis[64 + 16];
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + bigDegree) ;
    T* q_l_bound = (T*)(top_M_Cand_dis + TOPM + bigDegree) ;
    T* q_r_bound = (T*)(q_l_bound + attr_dim) ;

    // 这里应该怎么处理8字节的对齐问题？
    // __shared__ half4 tmp_val_sha[(unsigned)(128 / 4)];
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) +
     (TOPM + bigDegree) * sizeof(unsigned) + (TOPM + bigDegree) * sizeof(float) + 2 * attr_dim * sizeof(T)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[1024] ;
    __shared__ float work_mem_float[1024] ;

    __shared__ unsigned prefilter_pointer ;

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    // __shared__ unsigned long long start;
    // __shared__ double dis_cal_cost , sort_cost , merge_cost , new_merge_cost  , sort_cost2 ;
    // __shared__ unsigned total_jump_num ;
    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = task_id_mapper[blockIdx.x], laneid = threadIdx.x;  
    // 若线程块发射顺序与blockIdx.x的编号无关, 则可以手动实现按到达顺序取任务的排队器
    unsigned psize_block = p_size[bid] , plist_index = p_start_index[bid] , little_seg_num = little_seg_counts[bid] ;

    // if(tid == 0 && bid == 0) {
    //     printf("bid : %d\n" , bid) ;
    //     for(int i = 0 ; i < psize_block ; i ++)
    //         printf("%d," , pids[plist_index + i]) ;
    //     printf("\n") ;
    // }
    // __syncthreads() ;
    // if(tid == 0) {
    //     printf("psize_block : %d , little_seg_num : %d\n" , psize_block , little_seg_num) ;
    // }
    // if(tid == 0) {
    //     dis_cal_cost = sort_cost = merge_cost = new_merge_cost = sort_cost2=  0.0 ;
    //     total_jump_num = 0 ;
    // }
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
        q_l_bound[i] = l_bound[bid * attr_dim + i] ;
        q_r_bound[i] = r_bound[bid * attr_dim + i] ;
    }
    // 没必要在每个分区开始时都把recycle_list复原, 每个分区得到的入口点肯定是越近越好
    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        recycle_list_id[i] = 0xFFFFFFFF ;
        recycle_list_dis[i] = INF_DIS ;
    }

    // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
    for(unsigned i = tid; i < DEGREE; i += blockDim.x * blockDim.y){
        top_M_Cand[TOPM + i] = global_idx[seg_global_idx_start_idx[pids[plist_index]] + ent_pts[i]];
        hash_insert(hash_table , top_M_Cand[TOPM + i]) ;
    }
  
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}
    __syncthreads();
    
    // if(bid == 0 && tid == 0) {
    //     printf("[") ;
    //     for(unsigned i = 0 ; i < 1000 ; i ++) {
    //         printf("(%d,%d)" , little_seg_counts[i] , p_size[i]) ;
    //         if(i < 999)
    //             printf(",") ;
    //     }
    //     printf("]\n") ;
    // }
  
    /**
        1.先明确对每个分区要做什么, 首先, 将候选点载入top_M_Cand 的末DEGREE个位置, 然后排序
        2.将循环池重置为初始值
        3.从初始点开始扩展, 每次将DEGREE个邻居存放入top_M_Cand的末位, 被淘汰的分区内的点会插入循环池中
        4.当前分区查找结束后,将top_M_Cand中不符合范围约束的点去掉,用循环池中的点替代,然后依据这些点从全局图中扩展到下一个分区的入口点

        step4 的具体实现时, 可以先把不符合范围约束的点的距离改为INF,然后对列表排序, 和recycle_list合并
    **/
    if(psize_block < postfilter_threshold && psize_block != little_seg_num)
    // if(true)
    for(unsigned p_i = 0 ; p_i < psize_block; p_i ++) {
        // 得到图的入口
        unsigned pid = pids[plist_index + p_i] ;
        const unsigned* graph = all_graphs + graph_start_index[pid] ;
        __syncthreads() ;
        #pragma unroll
        for(unsigned i = threadIdx.y; i < DEGREE; i += blockDim.y){
            // threadIdx.y 0-5 , blockDim.y 即 6
            if(top_M_Cand[TOPM + i] == 0xFFFFFFFFu)
                continue ;
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                // 将warp内所有子线程的计算结果归并到一起
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }
            if(laneid == 0){
                top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
            }
        }
        __syncthreads();
        // if(tid == 0)
        //     printf("can click here %d\n" , __LINE__) ;
        // __syncthreads() ;
     
        // 直接对整个列表做距离计算
        if(tid < DEGREE)
            warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
        __syncthreads() ;
        // 此处被挤出去的点也应该合并到recycle_list 上
        // 将两个列表归并到一起
        merge_top_with_recycle_list<T>(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
        __syncthreads() ;
        if(tid < DEGREE)
            warpSort(top_M_Cand_dis + TOPM + tid , top_M_Cand + TOPM + tid , tid , DEGREE) ; 
        __syncthreads() ;
        if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
            merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE , recycle_list_size) ;
            __syncthreads() ;
        }

        // 初始化 recycle_list
        // for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        //     recycle_list_id[i] = 0xFFFFFFFF ;
        //     recycle_list_dis[i] = INF_DIS ;
        // }
        // if(tid == 0)
        //     printf("can click here %d\n" , __LINE__) ;
        // __syncthreads() ;
        // __syncthreads();

        // begin search
        for(unsigned i = 0; i < MAX_ITER; i++){

            // 每四轮将hash表重置一次
            if((i + 1) % HASH_RESET == 0){
                for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                    hash_table[j] = 0xFFFFFFFF;
                }
                __syncthreads();

                for(unsigned j = tid; j < TOPM + DEGREE; j += blockDim.x * blockDim.y){
                    // 将此时的TOPM + K 个候选点插入哈希表中
                    // hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
                    if(top_M_Cand[j] != 0xFFFFFFFF) {
                        hash_insert(hash_table , local_idx[(top_M_Cand[j] & 0x7FFFFFFF)]) ; 
                    }
                }
            }
            __syncthreads();

            if(tid < 32){
                // 从前32个点中找到需要扩展的点, 
                for(unsigned j = laneid; j < TOPM; j+=32){
                    unsigned n_p = 0;
                    if((top_M_Cand[j] & 0x80000000) == 0){
                        n_p = 1;
                    }
                
                    // *** 坑点: __ballot_sync会等待掩码影响的线程执行到__ballot_sync处, 若把0xffffffff设置为掩码, 个别线程可能因循环条件影响提前退出, 
                    // 无法执行到此处, 造成程序无限空等
                    // unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                    unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                
                    // if(bid == 0)
                    //     printf("tid : %d ," , tid) ;
                    if(ballot_res > 0){
                        // 这里的ffs必须考虑返回值等于-1的情况
                        if(laneid == (__ffs(ballot_res) - 1)){
                            to_explore = local_idx[top_M_Cand[j]];
                            top_M_Cand[j] |= 0x80000000;
                        }

                    
                        break;
                    }
                    to_explore = 0xFFFFFFFF;
                }
            }
            __syncthreads();
            // if(tid == 0)
            //     printf("can click here %d , epoch : %d\n" , __LINE__ , i) ;
            // __syncthreads() ;

            if(to_explore == 0xFFFFFFFF) {
                if(bid == 0 && tid == 0) printf("BREAK: %d\n", i);
                // printf("BREAK: %d\n" , i) ;
                jump_num_cnt[bid] += i ;
                break;
            }
            // if(tid == 0)
            //     printf("can click here %d , epoch : %d\n" , __LINE__ , i) ;
            // __syncthreads() ;

            for(unsigned j = tid; j < DEGREE; j += blockDim.x * blockDim.y){
                unsigned to_append_local = graph[to_explore * DEGREE + j];
                unsigned to_append = global_idx[seg_global_idx_start_idx[pid] + to_append_local] ;
                top_M_Cand_dis[TOPM + j] = (hash_insert(hash_table , to_append_local) == 0 ? INF_DIS : 0.0) ;
                top_M_Cand[TOPM + j] = to_append;
            }
            __syncthreads();
            // calculate distance
            #pragma unroll
            for(unsigned j = threadIdx.y; j < DEGREE; j += blockDim.y){
                // 6个warp, 每个warp负责一个距离计算
                if(top_M_Cand_dis[TOPM + j] == INF_DIS)
                    continue ;

                    half2 val_res;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                        half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                    }
                    #pragma unroll
                    for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                        val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                    }

                    if(laneid == 0){
                        top_M_Cand_dis[TOPM + j] += __half2float(val_res.x) + __half2float(val_res.y) ;
                    }
            }
            __syncthreads();

            if(tid < DEGREE)
                warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
            __syncthreads() ;
            if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                merge_top_with_recycle_list<T>(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , q_l_bound , q_r_bound , attrs , attr_dim) ;
                __syncthreads() ;
                if(tid < DEGREE)
                    warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
                __syncthreads() ;
                if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                // 此时top_M_Cand + TOPM中的点一定是满足属性过滤条件的点
                    merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE , recycle_list_size) ;
                    __syncthreads() ;
                }
            }
            // 问题来了, 当图开始切换时, 当前部分结果的id应该采用局部id还是全局id? 在列表中的所有id皆采用全局id
            // 对后半段列表sort一次, 然后和recycle pool进行merge
            // bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM] , &top_M_Cand[TOPM] , DEGREE) ;

        }
        __syncthreads();

        // bug产生原因: 在扩展完候选点后将hash表清空了
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y)
            top_M_Cand[i] |= 0x80000000 ;
        for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
            hash_table[i] = 0xFFFFFFFF ;
        }
        __syncthreads() ;

        // 修改一下顺序, 先扩展候选点, 然后再筛除不符合条件的点
        // 从global_edges 中扩展候选点
        if(p_i < psize_block - 1) {
            // 修改一下扩展候选点的逻辑, 向工作区中输送64个候选点, 计算距离后排序, 然后将前16个点送入top_M_Cand + TOPM 中
            unsigned next_pid = pids[plist_index + p_i + 1] ;
            unsigned j = 0 ;
            unsigned base_number_ = 128 / base_number ;
            for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
                work_mem_unsigned[i] = 0xFFFFFFFFu ;
                work_mem_float[i] = INF_DIS ;
                if(TOPM < (i % base_number_) || top_M_Cand[i % base_number_] == 0xFFFFFFFFu)
                    continue ;
                unsigned pure_id = top_M_Cand[i % base_number_] & 0x7FFFFFFF ;
                while(j < edges_num_in_global_graph) {
                    unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + j] ;
                    j ++ ;
                    if(hash_insert(hash_table , next_local_id) != 0) {
                        work_mem_unsigned[i] = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                        break ; 
                    }
                    if(j == edges_num_in_global_graph) {
                        for(unsigned rand_p = 0 ; rand_p < 62500 ; rand_p ++)
                            if(hash_insert(hash_table , rand_p) != 0) {
                                work_mem_unsigned[i] = global_idx[seg_global_idx_start_idx[next_pid] + rand_p] ;
                                break ;
                            }
                    }
                }
            }
            __syncthreads() ;
            // 计算这64个点的距离
            for(unsigned i = threadIdx.y ; i < 128 ; i += blockDim.y) {
                if(work_mem_unsigned[i] == 0xFFFFFFFFu)
                    continue ;
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[work_mem_unsigned[i] * DIM + k * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                }
            }
            __syncthreads() ;
            bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
            __syncthreads() ;
            // 重置哈希表
            for(unsigned i = tid ; i < HASHLEN ; i += blockDim.x * blockDim.y)
                hash_table[i] = 0xFFFFFFFF ;
            __syncthreads() ;
            // 将前16个点插入, 并同步插入到哈希表中
            for(unsigned i = tid ; i < DEGREE ; i += blockDim.x * blockDim.y) {
                top_M_Cand[TOPM + i] = work_mem_unsigned[i] ;
                if(work_mem_unsigned[i] != 0xFFFFFFFFu)
                    hash_insert(hash_table , local_idx[work_mem_unsigned[i]]) ;
            }
            for(unsigned i = tid ; i < 128 - DEGREE ; i += blockDim.x * blockDim.y)
                if(work_mem_unsigned[DEGREE + i] != 0xFFFFFFFFu)
                    hash_insert(hash_table , local_idx[work_mem_unsigned[DEGREE + i]]) ;
            // 将后64 - 16 个点与前表合并
            __syncthreads() ;

        }
        __syncthreads() ;

        if(p_i < psize_block - 1)
            merge_top(top_M_Cand , work_mem_unsigned + DEGREE, top_M_Cand_dis , work_mem_float + DEGREE , tid , 64 , TOPM) ;
            // merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float  , tid , 64 , TOPM) ;
        __syncthreads() ;
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
            if(top_M_Cand[i] != 0xFFFFFFFF)
                hash_insert(hash_table , local_idx[top_M_Cand[i] & 0x7FFFFFFF]) ;
        }
        __syncthreads() ;
    }
    else if(psize_block != little_seg_num && false){
        
        // 先搞定入口点
        for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
            hash_table[i] = 0xFFFFFFFF;
        }
        for(unsigned i = tid; i < bigDegree; i += blockDim.x * blockDim.y){
            top_M_Cand[TOPM + i] = ent_pts_post[i];
            hash_insert(hash_table , top_M_Cand[TOPM + i]) ;
        }
        __syncthreads() ;

        #pragma unroll
        for(unsigned j = threadIdx.y; j < bigDegree ; j += blockDim.y){
            // 6个warp, 每个warp负责一个距离计算
            // if(top_M_Cand[TOPM + j] == 0xFFFFFFFFu)
            //     continue ;

            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            unsigned global_id = top_M_Cand[TOPM + j] ;
            for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                half4 val2 = Load(&values[global_id * DIM + k * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }

            if(laneid == 0){
                top_M_Cand_dis[TOPM + j] = __half2float(val_res.x) + __half2float(val_res.y) ;
            }
        }
        __syncthreads();
  
        if(tid < bigDegree)
            // warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , bigDegree);
            bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , bigDegree) ;
        __syncthreads() ;  
        merge_top(top_M_Cand , top_M_Cand + TOPM , top_M_Cand_dis , top_M_Cand_dis + TOPM , tid , bigDegree , TOPM) ;
        __syncthreads() ;

        for(unsigned i = 0; i < MAX_ITER; i++) {

            // 每四轮将hash表重置一次
            if((i + 1) % HASH_RESET == 0){
                for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                    hash_table[j] = 0xFFFFFFFF;
                }
                __syncthreads();

                for(unsigned j = tid; j < TOPM + bigDegree ; j += blockDim.x * blockDim.y){
                    // 将此时的TOPM + K 个候选点插入哈希表中
                    // hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
                    if(top_M_Cand[j] != 0xFFFFFFFF) {
                        hash_insert(hash_table , top_M_Cand[j] & 0x7FFFFFFF) ; 
                    }
                }
            }
            __syncthreads();

            if(tid < 32){
                // 从前32个点中找到需要扩展的点, 
                for(unsigned j = laneid; j < TOPM; j+=32){
                    unsigned n_p = 0;
                    if((top_M_Cand[j] & 0x80000000) == 0){
                        n_p = 1;
                    }
                
                    // *** 坑点: __ballot_sync会等待掩码影响的线程执行到__ballot_sync处, 若把0xffffffff设置为掩码, 个别线程可能因循环条件影响提前退出, 
                    // 无法执行到此处, 造成程序无限空等
                    // unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                    unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                
                    // if(bid == 0)
                    //     printf("tid : %d ," , tid) ;
                    if(ballot_res > 0){
                        // 这里的ffs必须考虑返回值等于-1的情况
                        if(laneid == (__ffs(ballot_res) - 1)){
                            // to_explore = local_idx[top_M_Cand[j]];
                            to_explore = top_M_Cand[j] ;
                            top_M_Cand[j] |= 0x80000000;
                        }
                        break;
                    }
                    to_explore = 0xFFFFFFFF;
                }
            }
            __syncthreads();

            if(to_explore == 0xFFFFFFFF) {
                if(bid == 0 && tid == 0) printf("BREAK: %d\n", i);
                // printf("BREAK: %d\n" , i) ;
                // jump_num_cnt[bid] += i ;
                break;
            }

            for(unsigned j = tid; j < bigDegree; j += blockDim.x * blockDim.y){
                unsigned to_append = bigGraph[to_explore * bigDegree + j] ;
                top_M_Cand[TOPM + j] = ((hash_insert(hash_table , to_append) == 0) ? 0xFFFFFFFFu : to_append) ;
                top_M_Cand_dis[TOPM + j] = (top_M_Cand[TOPM + j] == 0xFFFFFFFFu ? INF_DIS : 0.0) ;
            }

            __syncthreads() ;
        
            // calculate distance
            #pragma unroll
            for(unsigned j = threadIdx.y; j < bigDegree ; j += blockDim.y){
                // 6个warp, 每个warp负责一个距离计算
                if(top_M_Cand[TOPM + j] == 0xFFFFFFFFu)
                    continue ;

                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                unsigned global_id = top_M_Cand[TOPM + j] ;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[global_id * DIM + k * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    top_M_Cand_dis[TOPM + j] += __half2float(val_res.x) + __half2float(val_res.y) ;
                }
            }
            __syncthreads();


            if(tid < bigDegree)
                // warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , bigDegree);
                bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , bigDegree) ;
            // bitonic_sort_id_by_dis_no_explore(work_mem_float, work_mem_unsigned, extend_points_num) ;
            __syncthreads() ;
         
            // __syncthreads() ;
            merge_top(top_M_Cand , top_M_Cand + TOPM , top_M_Cand_dis , top_M_Cand_dis + TOPM , tid , bigDegree , TOPM) ;
            __syncthreads() ;
        }
    } else {
        // 走预过滤路线
        __shared__ unsigned prefix_sum[1024] ;
        __shared__ unsigned top_prefix[32] ;
        __shared__ unsigned indices_arr[1024] ;
        __shared__ unsigned task_num ; 
        __shared__ unsigned bit_map[32] ;
        // __shared__ unsigned prefix_sum[1] ;
        // __shared__ unsigned top_prefix[1] ;
        // __shared__ unsigned indices_arr[1] ;
        // __shared__ unsigned task_num ; 
        // __shared__ unsigned bit_map[1] ;
        
        // 处理剩余的小分区
        for(unsigned g = 0 ; g < little_seg_num ; g ++) {
          
            // printf("bid : %d , l_bound : %f , r_bound : %d\n" , bid , q_l_bound[0] , q_r_bound[0]) ;
            
            unsigned gpid = pids[plist_index + g] ;
            unsigned global_idx_offset = seg_global_idx_start_idx[gpid] ;
            unsigned num_in_seg = graph_size[gpid] ;
            // unsigned num_in_seg = 62464 ;
    
 
            #pragma unroll
            for(unsigned i = tid ; i < 1024 ; i += blockDim.x * blockDim.y) {
                work_mem_unsigned[i] = 0xFFFFFFFF ;
                work_mem_float[i] = INF_DIS ;
            }
            if(tid == 0)
                prefilter_pointer = 0 ; 
            __syncthreads() ;
    
            // 先把 total_psize_block个点分片, 分片大小可以考虑bit_map中1的个数? 
            // selectivity <= 0.1, 意味着, 每十个点里只有一个点满足条件, 找到32个点平均要遍历320个点
            //  有两种方案 :  1. 动态chunk, 尽可能的设置一组中包含32个点
            //               2. 静态chunk 设置 chunk 大小为10 , 但要维护的候选列表长度至少为320
            //  合并次数 : 1. 6250 / 32 次 ; 2. 625 / 32 次
            //  随着合并次数的增多, 方案2的待合并点数量、以及合并次数会显著减少，故采用方案2. 方案1的前置条件太复杂
            // 表示某个点是否满足属性约束条件
            // for(unsigned i = tid ; i < num_in_seg ; i += blockDim.x * blockDim.y) {
            //     bool flag = true ;
            //     unsigned g_global_id = global_idx[global_idx_offset + i] ;
    
            //     for(unsigned j = 0 ; j < attr_dim ; j ++)
            //         flag = (flag && (attrs[g_global_id * attr_dim + j] >= q_l_bound[j]) && (attrs[g_global_id * attr_dim + j] <= q_r_bound[j])) ;
            //     unsigned ballot_res = __ballot_sync(__activemask() , flag) ;
            //     if(laneid == 0)
            //         bit_map[(i >> 5)] = ballot_res ; 
            // }
            // __syncthreads() ;
    
            unsigned chunk_num = (num_in_seg + 31) >> 5 ;
            unsigned chunk_batch_size = 32 ; 
            for(unsigned chunk_id = 0 ; chunk_id < chunk_num ; chunk_id += chunk_batch_size) {
                unsigned point_batch_size = (chunk_id + chunk_batch_size < chunk_num ? (chunk_batch_size << 5) : num_in_seg - (chunk_id << 5)) ;
                unsigned point_batch_start_id = (chunk_id << 5) ;
                
                #pragma unroll
                for(unsigned i = tid ;i < 32 ; i += blockDim.x * blockDim.y) 
                    bit_map[i] = 0 ;

                for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                    bool flag = true ;
                    unsigned g_global_id = global_idx[global_idx_offset + point_batch_start_id + i] ;
        
                    for(unsigned j = 0 ; j < attr_dim ; j ++)
                        flag = (flag && (attrs[g_global_id * attr_dim + j] >= q_l_bound[j]) && (attrs[g_global_id * attr_dim + j] <= q_r_bound[j])) ;
                    unsigned ballot_res = __ballot_sync(__activemask() , flag) ;
                    if(laneid == 0)
                        bit_map[(i >> 5)] = ballot_res ; 
                }
                __syncthreads() ;

    
                // 扫描出该chunk所需要的前缀和数组
                // 先利用__popc() 得到长度为32的前缀和数组, 再利用0号warp, 把32个局部前缀和数组合并为一个整体
                for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                    // 在bitmap中的单元格位置
                    unsigned cell_id = (i >> 5) ;
                    unsigned bit_pos = i % 32 + 1;
                    unsigned mask = (bit_pos == 32 ? 0xFFFFFFFF : (1 << bit_pos) - 1) ;
                    unsigned index = (bit_map[cell_id] & mask) ;
                    prefix_sum[i] = __popc(index) ; 
                }
                __syncthreads() ;
                if(tid < 32) {
                    unsigned v = (laneid + chunk_id < chunk_num ? __popc(bit_map[laneid]) : 0) ;
                    // 将长度大小为32的各个子prefix合并为最终的prefix_sum 
                    #pragma unroll
                    for(unsigned offset = 1 ; offset < 32 ; offset <<= 1) {
                        unsigned y = __shfl_up_sync(0xFFFFFFFF , v , offset) ;
                        if(laneid >= offset) v += y ;
                    }
                    top_prefix[laneid] = v ;
                }
                if(tid == 0) {
                    task_num = top_prefix[31] ;
                    // if(task_num > 0 && false) {
                    //     printf("task num : %d\n" , task_num) ;
    
                    //     printf("[") ;
                    //     for(unsigned j = 0 ; j < 1024 ; j ++) {
                    //         printf("%d" , prefix_sum[j]) ;
                    //         if(j < 1023)
                    //             printf(", ") ;
                    //     }
                    //     printf("]\n") ;
    
                    //     printf("top_prefix :[") ;
                    //     for(unsigned j = 0 ; j < 32 ; j ++) {
                    //         printf("%d" , top_prefix[j]) ;
                    //         if(j < 31)
                    //             printf(", ") ;
                    //     }
                    //     printf("]\n") ;
                    // }
                }
                __syncthreads() ;
                for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                    unsigned cell_id = (i >> 5) ; 
                    
                    unsigned id_in_cell = i % 32 ;
                    unsigned bit_mask = (1 << id_in_cell) ;
                    if((bit_map[cell_id] & bit_mask) != 0) {
                        // 获取插入位置
                        // unsigned prefix_sum_fetch_pos = 
                        unsigned local_cell_id = (i >> 5) ;
                        unsigned put_pos = (local_cell_id == 0 ? 0 : top_prefix[local_cell_id - 1]) + (id_in_cell == 0 ? 0 : prefix_sum[i - 1]) ;
                        // if(i == 2) {
                        //     printf("i == 2 : %d , cell_id : %d , id_in_cell : %d\n" , put_pos , cell_id , id_in_cell) ;
                        // }
                        // if(put_pos >= 1024) {
    
                        //     // printf("bid : %d ,point_batch_size : %d\n" , bid , point_batch_size) ;
                        //     printf("bid : %d ,put_pos : %d , top_prefix[%d - 1] = %d , prefix_sum[%d - 1] = %d\n" , bid, put_pos , cell_id , top_prefix[(i >> 5) - 1] ,
                        //     i , prefix_sum[i - 1]) ;
                        // }
                        indices_arr[put_pos] = i ;
                    }
                }
                __syncthreads() ;
    
                // if(tid == 0) {
                //     // if(task_num > 0 && false) {
                //         printf("task num : %d\n" , task_num) ;
    
                //         printf("[") ;
                //         for(unsigned j = 0 ; j < 1024 ; j ++) {
                //             printf("%d" , prefix_sum[j]) ;
                //             if(j < 1023)
                //                 printf(", ") ;
                //         }
                //         printf("]\n") ;
    
                //         printf("top_prefix :[") ;
                //         for(unsigned j = 0 ; j < 32 ; j ++) {
                //             printf("%d" , top_prefix[j]) ;
                //             if(j < 31)
                //                 printf(", ") ;
                //         }
                //         printf("]\n") ;
                //     // }
                //         printf("indices_arr: [") ;
                //         for(unsigned j = 0 ; j < 32 ; j ++) {
                //             printf("%d" , indices_arr[j]) ;
                //             if(j < 31)
                //                 printf(", ") ;
                //         }
                //         printf("]\n") ;
                // }
     
                // 计算距离, 然后用原子操作插入队列中
                for(unsigned i = threadIdx.y ; i < task_num ; i += blockDim.y) {
                    unsigned cand_id = indices_arr[i] + point_batch_start_id ;
                    // if(laneid == 0 && cand_id >= 62500) {
                    //     printf("bid : %d , cand_id : %d , indices_arr[i] : %d\n" , bid , cand_id , indices_arr[i]) ;
                    // }
                    // __syncwarp() ;
                    // 计算距离
                    unsigned g_global_id = global_idx[global_idx_offset + cand_id] ;
    
                    float dis = 0.0f ;
    
                    half2 val_res;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                        half4 val2 = Load(&values[g_global_id * DIM + k * 4]);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                    }
                    #pragma unroll
                    for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                        val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                    }
    
                    if(laneid == 0){
                        dis = __half2float(val_res.x) + __half2float(val_res.y) ;
                        atomicAdd(dis_cal_cnt + bid , 1) ;
                    }
                    __syncwarp() ;
                    // 存入work_mem
                    if(laneid == 0 && dis < top_M_Cand_dis[K - 1]) {
                        int cur_pointer = atomicAdd(&prefilter_pointer , 1) ;
                        work_mem_unsigned[cur_pointer] = g_global_id ; 
                        work_mem_float[cur_pointer] = dis ; // 距离变量
                    }
                    __syncwarp() ;
                }
    
                __syncthreads() ;
                if(prefilter_pointer >= 32 || (chunk_id + chunk_batch_size >= chunk_num && prefilter_pointer > 0)) {
                    // 先对前32个点排序, 然后再合并到topK上
                    // if(tid < 32)
                    //     warpSort(work_mem_float , work_mem_unsigned , tid , 32) ; 
                    bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned, prefilter_pointer) ;
                    __syncthreads() ;
                    merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float , tid , prefilter_pointer , K) ;
                    __syncthreads() ;
                    unsigned limit = prefilter_pointer ;
                    #pragma unroll
                    for(unsigned i = tid ; i < limit ; i += blockDim.x * blockDim.y) {
                        work_mem_unsigned[i] = 0xFFFFFFFF ;
                        work_mem_float[i] = INF_DIS ;
                    }
                    if(tid == 0) {
                        prefilter_pointer = 0 ; 
                        atomicAdd(post_merge_cnt + bid , 1) ;
                    }
    
                    __syncthreads() ;
                }
            }
        }
    }
    __syncthreads() ;

    if(psize_block != little_seg_num) {
        // 以下为对top_M_Cand 列表做后过滤, 并将recycle_list合并, 为公共流程
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
            if(top_M_Cand[i] == 0xFFFFFFFF)
                continue ;
            bool flag = true ;
            unsigned pure_id = top_M_Cand[i] & 0x7FFFFFFF ;
            for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
                flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
            }
            if(!flag) {
                top_M_Cand[i] = 0xFFFFFFFF ;
                top_M_Cand_dis[i] = INF_DIS ;
            }
        }
        __syncthreads() ;
        bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM);
        __syncthreads() ;
        // 若recycle_list中确实有点, 则合并
        if(recycle_list_id[0] != 0xFFFFFFFF) {
            merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
            __syncthreads() ;
        }
    }

    // if(tid == 0 && bid == 0) {
    //     printf("dis [") ;
    //     for(unsigned i = 0 ; i < TOPM ; i ++) {
    //         printf("(%d-%f)" , top_M_Cand[i] & 0x7FFFFFFFu , top_M_Cand_dis[i]) ;
    //         if(i < TOPM - 1)
    //             printf(",") ;
    //     }
    //     printf("]\n") ;

    //     printf("pids [") ;
    //     for(int i = 0 ; i < psize_block ; i ++) {
    //         printf("%d" , pids[plist_index + i]) ;
    //         if(i < psize_block - 1)
    //             printf(",") ;
    //     }
    //     printf("]\n") ;
    //     int B[10] = {94267 ,247083 ,356320 ,436821 ,466927 ,544630 ,638190 ,836301 ,888777 ,953104} ;
    //     for(int i = 0 ; i < 10 ; i ++)
    //         top_M_Cand[i] = B[i] , top_M_Cand_dis[i] = INF_DIS ;
    // }
    // __syncthreads() ;


    // #pragma unroll
    // for(unsigned j = threadIdx.y; j < 10 ; j += blockDim.y){
    //     // 6个warp, 每个warp负责一个距离计算
    //     if(top_M_Cand[j] == 0xFFFFFFFFu)
    //         continue ;

    //     half2 val_res;
    //     val_res.x = 0.0; val_res.y = 0.0;
    //     unsigned global_id = top_M_Cand[j] ;
    //     for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
    //         half4 val2 = Load(&values[global_id * DIM + k * 4]);
    //         val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
    //         val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
    //     }
    //     #pragma unroll
    //     for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
    //         val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
    //     }

    //     if(laneid == 0){
    //         top_M_Cand_dis[j] = __half2float(val_res.x) + __half2float(val_res.y) ;
    //     }
    // }
    // __syncthreads();

    // if(tid == 0 && bid == 12) {
    //     printf("dis [") ;
    //     for(unsigned i = 0 ; i < TOPM ; i ++) {
    //         printf("(%d-%f)" , top_M_Cand[i] & 0x7FFFFFFFu , top_M_Cand_dis[i]) ;
    //         if(i < TOPM - 1)
    //             printf(",") ;
    //     }
    //     printf("]\n") ;

    //     printf("pids [") ;
    //     for(int i = 0 ; i < psize_block ; i ++) {
    //         printf("%d" , pids[plist_index + i]) ;
    //         if(i < psize_block - 1)
    //             printf(",") ;
    //     }
    //     printf("]\n") ;

    //     printf("range: [") ;
    //     for(int i = 0 ; i < attr_dim ; i ++) {
    //         printf("(%f , %f)," , q_l_bound[i] , q_r_bound[i]) ;
    //     }
    //     printf("]\n") ;
    // }
    // __syncthreads() ;

    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    for(unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        result_k_id[bid * K + i] = (top_M_Cand[i] & 0x7FFFFFFF) ;
        results_dis[bid * K + i] = top_M_Cand_dis[i] ;
    }
}


// TOPM是candidate队列长度，K是TOP-K也是图索引的度数，DIM是数据集维度，MAX_ITER是最大迭代次数（1000足够），INF_DIS表示无穷大的浮点数
__global__ void select_path_contain_attr_filter_V2_without_output_dual_channel(const unsigned* __restrict__ all_graphs, const half* __restrict__ values, const half* __restrict__ query_data, 
    unsigned* results_id, float* results_dis, unsigned* ent_pts , unsigned DEGREE , unsigned TOPM , unsigned DIM , const unsigned* graph_start_index ,const unsigned* graph_size ,
    const unsigned* __restrict__ pids , const unsigned* p_size ,const unsigned* p_start_index ,const float* l_bound ,const float* r_bound ,const float* __restrict__ attrs , unsigned attr_dim , const unsigned* __restrict__ global_graph , unsigned edges_num_in_global_graph,
    unsigned recycle_list_size ,const unsigned* __restrict__ global_idx ,const unsigned* __restrict__ local_idx ,const unsigned* seg_global_idx_start_idx , unsigned pnum ,unsigned K, unsigned* result_k_id ,const unsigned* task_id_mapper) {
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数
    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    unsigned* recycle_list_id = (unsigned*) shared_mem ;
    float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    unsigned* top_M_Cand = (unsigned*) (recycle_list_dis + recycle_list_size) ;
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE * 2) ;
    float* q_l_bound = (float*)(top_M_Cand_dis + TOPM + DEGREE * 2) ;
    float* q_r_bound = (float*)(q_l_bound + attr_dim) ;

    // 这里应该怎么处理8字节的对齐问题？
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) +
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[128] ;
    __shared__ float work_mem_float[128] ;

    __shared__ unsigned cur_visit_p , main_channel ;
    __shared__ unsigned to_explore[2];

    __shared__ unsigned hash_table[HASHLEN];

    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = task_id_mapper[blockIdx.x], laneid = threadIdx.x , warpId = threadIdx.y ;
    // 若线程块发射顺序与blockIdx.x的编号无关, 则可以手动实现按到达顺序取任务的排队器
    unsigned psize_block = p_size[bid] , plist_index = p_start_index[bid] ;
    unsigned pid[2] = {pids[plist_index] , (psize_block >= 2 ? pids[plist_index + 1] : pids[plist_index])} ;
    const unsigned* graph[2] = {all_graphs + graph_start_index[pid[0]] , all_graphs + graph_start_index[pid[1]]};
    const unsigned DEGREE2 = DEGREE * 2 ;
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
        q_l_bound[i] = l_bound[bid * attr_dim + i] ;
        q_r_bound[i] = r_bound[bid * attr_dim + i] ;
    }
    // 没必要在每个分区开始时都把recycle_list复原, 每个分区得到的入口点肯定是越近越好
    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        recycle_list_id[i] = 0xFFFFFFFF ;
        recycle_list_dis[i] = INF_DIS ;
    }
    // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
    for(unsigned i = tid; i < DEGREE2; i += blockDim.x * blockDim.y){
        unsigned shift_bits = 30 + i / DEGREE ;
        unsigned mask = (pid[0] == pid[1] ? 0 : 1 << shift_bits) ; 
        // 用反向掩码, 将另一位置1
        top_M_Cand[TOPM + i] = global_idx[seg_global_idx_start_idx[pid[i / DEGREE]] + ent_pts[i]] | mask;
        // top_M_Cand[TOPM + DEGREE + i]

        // 哈希表存放的是局部id, 如何统一?  是否可以直接存放全局id? 
        hash_insert(hash_table , top_M_Cand[TOPM + i] & 0x3FFFFFFF) ;
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}

    if(tid == 0) {
        cur_visit_p = (psize_block >= 2 ? 1 : 0) ;
        main_channel = 0 ; 
        // printf("can click 2023\n") ;
    }
    __syncthreads();
    
    // 若只有一个分区, 令两个通道处理同一个分区, 若有超过两个分区, 奇数号warp处理1号分区, 偶数号warp处理0号分区

    // unsigned pid = (psize_block >= 2 ? pids[plist_index + (warpId & 1)] : pids[plist_index]) ;

    // 无限次循环, 直到退出
    for(unsigned epoch = 0 ; epoch < MAX_ITER * 2 ; epoch ++) {

        // 每隔4轮将哈希表重置一次
        if((epoch + 1) % (HASH_RESET) == 0){
            #pragma unroll
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            #pragma unroll
            for(unsigned j = tid; j < TOPM + DEGREE2; j += blockDim.x * blockDim.y){
                // 将此时的TOPM + K 个候选点插入哈希表中
                if(top_M_Cand[j] != 0xFFFFFFFF) {
                    hash_insert(hash_table , top_M_Cand[j] & 0x3FFFFFFF) ; 
                }
            }
            // if(tid == 0 && bid == 0)
            //     printf("epoch : %d\n" , epoch) ;
        }
        __syncthreads();
        // if(tid == 0)
        //     printf("line : %d , epoch : %d\n" , __LINE__ , epoch) ;
        // __syncthreads() ;

        // 此时无论是通道1还是通道2, 所有节点已在TOPM数组的后半段就位, 因此距离计算可以一视同仁
        #pragma unroll
        for(unsigned i = threadIdx.y; i < DEGREE2 ; i += blockDim.y){
            if(top_M_Cand[TOPM + i] == 0xFFFFFFFF)
                continue ;
            atomicAdd(dis_cal_cnt + bid , 1) ;
            // threadIdx.y 0-5 , blockDim.y 即 6
            unsigned pure_id = top_M_Cand[TOPM + i] & 0x3FFFFFFF ;
            // if(pure_id > 1000000)
            //     printf("block : %d , error id - %d\n" , bid ,  pure_id) ;
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                half4 val2 = Load(&values[pure_id * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                // 将warp内所有子线程的计算结果归并到一起
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }
            if(laneid == 0){
                top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
            }
        }
        __syncthreads();
        // if(tid == 0)
        //     printf("line : %d , epoch : %d\n" , __LINE__ , epoch) ;
        // __syncthreads() ;
        // 排序
        if(tid < DEGREE2)
            warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE2);
        __syncthreads() ;
        // 将两个列表归并到一起
        merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE2 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
        __syncthreads() ;
        if(tid < DEGREE2)
            warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
        __syncthreads() ;
        if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
            merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE2 , recycle_list_size) ;
            __syncthreads() ;
        }
        // if(tid == 0)
        //     printf("block : %d , line : %d , epoch : %d , start\n" ,bid ,  __LINE__ , epoch) ;
        // __syncthreads() ;

        // 下面进行点扩展
        if(tid < 64){
            // 需完成的逻辑: 若两个通道处理同一个分区, 令该步寻找同一个分区的前两个点扩展
            if(laneid == 0)
                to_explore[warpId & 1] = 0xFFFFFFFF ;
            __syncwarp() ;
            // 从前32个点中找到需要扩展的点, warpId为偶数的点需要第一位为0 , warpId为奇数的点需要第二位为0
            unsigned shift_bits = (pid[0] == pid[1] ? 31 - (main_channel & 1)  : 31 - (warpId & 1));
            unsigned mask = (1 << shift_bits) ;
            // 若两个通道处理一个图, 则1号warp取第二个候选数, 0号warp取第一个候选点, 否则各自取自己的第一位
            unsigned fetch_pos = (pid[0] == pid[1] ? (warpId & 1) : 0)  + 1;
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = ((top_M_Cand[j] & mask) == 0 ? 1 : 0) ;
            
                // *** 坑点: __ballot_sync会等待掩码影响的线程执行到__ballot_sync处, 若把0xffffffff设置为掩码, 个别线程可能因循环条件影响提前退出, 
                // 无法执行到此处, 造成程序无限空等
                // unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                unsigned one_num = __popc(ballot_res) ;
                if(one_num >= fetch_pos) {
                    // printf("line: %d , onenum : %d , fetch_pos : %d\n" , __LINE__ , one_num , fetch_pos) ;
                    // 这里是单线程执行, 不考虑线程分歧
                    if(laneid == 0) {
                        unsigned idx ;
                        while(fetch_pos > 0) {
                            idx = (__ffs(ballot_res) - 1) ;
                            ballot_res &= (ballot_res - 1) ;
                            fetch_pos -- ;
                        }
                        idx = j - j % 32 + idx ;
                        // if((top_M_Cand[idx] & 0x3FFFFFFF) >= 1000000) {
                        //     printf("block : %d , topmcandidx : %d\n" , bid , (top_M_Cand[idx] & 0x3FFFFFFF)) ;
                        // }
                        to_explore[warpId & 1] = local_idx[top_M_Cand[idx] & 0x3FFFFFFF] ;
                        top_M_Cand[idx] |= mask ; 
                        // printf("line: %d , idx : %d , fetch_pos : %d\n" , __LINE__ , idx , fetch_pos) ;
                        // break ; 
                    }
                    break ;
                }
                // __syncwarp() ;
                fetch_pos -= one_num ; 
            }
        }
        __syncthreads();
        // if(tid == 0)
        //     printf("block : %d ,line : %d , epoch : %d end\n" ,bid ,  __LINE__ , epoch) ;
        // __syncthreads() ;
        /**
            此时有三种情况:  1. 通道1和通道2皆可找到扩展点, 各自扩展即可
                            2. 仅有一个通道能找到扩展点, 找不到扩展点的通道进行栈切换, 用下一个图的入口点填充
                            3. 两个通道都找不到扩展点, 此时算法收敛, 退出
        **/
        // 这里有bug, 需要改一下
        if(to_explore[0] == 0xFFFFFFFF || to_explore[1] == 0xFFFFFFFF) {
            // 将recycle_list合并至 topm 然后返回
            /*
                此时2种情况: 1. 有一个通道切换图栈
                                1.1 判断cur_visit_p == psize - 1, 是则指向另一个图栈, 否则切图
                            2. 有两个通道切换图栈
                                2.1 判断剩余可切换图栈数量, 0则退出, 1则指向同一个
                共有操作: 将recycle_list合并至 topm
            */
            if(cur_visit_p == psize_block - 1 && to_explore[0] == 0xFFFFFFFF && to_explore[1] == 0xFFFFFFFF) {
                // 两个图都没有扩展点, 且没有图可以切换, 退出
                if(tid == 0) {
                    if(bid == 0)
                        printf("block %d: break in %d\n" ,bid ,  epoch) ; 
                    jump_num_cnt[bid] = epoch ;
                }
                break ;
            } else if(cur_visit_p == psize_block - 1) {
                    // 此时已经是最后一个图, 令两个通道处理同一个图即可
                    unsigned idle_channel = (to_explore[0] == 0xFFFFFFFF ? 0 : 1) ;
                    if(pid[0] != pid[1] && tid == 0) {
                        graph[idle_channel] = graph[idle_channel ^ 1] ;
                        pid[idle_channel] = pid[idle_channel ^ 1] ;
                        main_channel = idle_channel ^ 1 ;
                    }

                    // 非主channel用全1填充
                    #pragma unroll
                    for(unsigned j = tid ; j < DEGREE ; j += blockDim.x * blockDim.y) {
                        top_M_Cand[TOPM + idle_channel * DEGREE + j] = 0xFFFFFFFF ;
                        top_M_Cand_dis[TOPM + idle_channel * DEGREE + j] = INF_DIS ; 
                    }
                
            } else {
            // 此时一定有其中一个图要切栈
            // 先获取要切栈的第一个图
                
                // #pragma unroll
                // for(unsigned i = tid ; i < HASHLEN ; i += blockDim.x * blockDim.y) {
                //     hash_table[i] = 0xFFFFFFFF ;
                // }
                
                // if(tid == 0 ) {
                //     printf("block %d , line : %d , cur_visit_p : %d , start\n" , bid , __LINE__ , cur_visit_p) ;
                // }
                // __syncthreads() ;
                /**
                for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
                    if(top_M_Cand[i] == 0xFFFFFFFF)
                        continue ;
                    bool flag = true ;
                    unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
                    for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
                        flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
                    }
                    if(!flag) {
                        top_M_Cand[i] = 0xFFFFFFFF ;
                        top_M_Cand_dis[i] = INF_DIS ;
                    }
                }
                __syncthreads() ;
                bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM);
                __syncthreads() ;
                // 若recycle_list中确实有点, 则合并
                if(recycle_list_id[0] != 0xFFFFFFFF) {
                    #pragma unroll
                    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
                        recycle_list_id[i] |= 0xC0000000 ;
                    }
                    __syncthreads() ;

                    merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
                    __syncthreads() ;
                    #pragma unroll
                    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
                        recycle_list_id[i] = 0xFFFFFFFF ;
                        recycle_list_dis[i] = INF_DIS ;
                    }
                    __syncthreads() ;
                }
                // 重置hash表
                #pragma unroll
                for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
                    if(top_M_Cand[i] != 0xFFFFFFFF)
                        hash_insert(hash_table , top_M_Cand[i] & 0x3FFFFFFF) ;
                    }
                // if(tid == 0) {
                //     printf("block %d , line : %d , cur_visit_p : %d end\n" , bid , __LINE__ , cur_visit_p) ;
                // }
                __syncthreads() ;
                **/ 

            // 此处有一种情况处理不了, 即两个通道都需要切换上下文, 但剩余要处理的图只剩下一个
                unsigned channel_id = (to_explore[0] == 0xFFFFFFFF ? 0 : 1) ;  
                unsigned channel_num = (to_explore[0] == 0xFFFFFFFF) + (to_explore[1] == 0xFFFFFFFF) ;
                // if(tid == 0 && bid == 67) {
                //     printf("block %d , line %d , to_explore[0] = %d , to_explore[1] = %d , psize = %d , cur_visit_p = %d, channel_num = %d\n" , bid , __LINE__ , to_explore[0] , to_explore[1], psize_block , cur_visit_p , channel_num) ;
                    
                // }
                // __syncthreads() ;
                // #pragma unroll
                // for(unsigned i = tid ; i < HASHLEN ; i += blockDim.x * blockDim.y)
                //     hash_table[i] = 0xFFFFFFFF ;
                // __syncthreads() ;
                for(unsigned lp = 0 ; lp < channel_num ; lp ++) {
                        
                        if(cur_visit_p == psize_block - 1) {
                            if(tid == 0) {
                                graph[channel_id] = graph[channel_id ^ 1] ;
                                pid[channel_id] = pid[channel_id ^ 1] ;
                                main_channel = channel_id ^ 1 ;
                            }
        
                            // 非主channel用全1填充
                            #pragma unroll
                            for(unsigned j = tid ; j < DEGREE ; j += blockDim.x * blockDim.y) {
                                top_M_Cand[TOPM + channel_id * DEGREE + j] = 0xFFFFFFFF ;
                                top_M_Cand_dis[TOPM + channel_id * DEGREE + j] = INF_DIS ; 
                            }
                            break ;
                        }

                    //  else {
                        // channel_id 切换图地址, 从全局图中扩展邻居
                        unsigned next_pid = pids[plist_index + cur_visit_p + 1] ;
                        // unsigned j = 0 ;
                        unsigned shift_bits = 30 + (channel_id & 1) ;
                        unsigned mask = 1 << shift_bits ;
                        
                        // 重置哈希表
                        for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
                    
                            if(top_M_Cand[i % 64] == 0xFFFFFFFF) {
                                // 填充一个随机数
                                for(unsigned j = i * 3 ; j < 62500 ; j ++) {
                                    unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + j] ;
                                    if(hash_insert(hash_table , next_global_id) != 0) {
                                        work_mem_unsigned[i] = next_global_id | mask ;
                                        break ;
                                    }
                                }
                                
                                continue ;
                            }
                            // 用前两个填充, 有重复的部分皆替换为随机点
                            unsigned pure_id = top_M_Cand[i % 64] & 0x3FFFFFFF ;
                            unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / 64] ;
                            unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                            if(hash_insert(hash_table , next_global_id) != 0) {
                                    // work_mem_unsigned[i] = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                                work_mem_unsigned[i] = next_global_id | mask ;
                            } else {
                                for(unsigned j = i * 3 ; j < 62500 ; j ++) {
                                    unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + j] ;
                                    if(hash_insert(hash_table , next_global_id) != 0) {
                                        work_mem_unsigned[i] = next_global_id | mask ;
                                        break ;
                                    }
                                }
                            }
                        }
                        
                        __syncthreads() ;
                        // 距离计算
                        #pragma unroll
                        for(unsigned i = threadIdx.y ; i < 128 ; i += blockDim.y) {
                            half2 val_res;
                            unsigned pure_id = work_mem_unsigned[i] & 0x3FFFFFFF ;
                            val_res.x = 0.0; val_res.y = 0.0;
                            for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                                half4 val2 = Load(&values[pure_id * DIM + k * 4]);
                                val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                                val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                            }
                            #pragma unroll
                            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                            }
            
                            if(laneid == 0){
                                work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                            }
                        }
                        __syncthreads() ;
                        bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                        __syncthreads() ;
                        // 此处需调整, 这里topm中被淘汰的点没有进入top_M_Cand + TOPM 中
                        merge_top_with_recycle_list(top_M_Cand , work_mem_unsigned + DEGREE, top_M_Cand_dis , work_mem_float + DEGREE , tid , 64 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
                        // merge_top(top_M_Cand , work_mem_unsigned + DEGREE, top_M_Cand_dis , work_mem_float + DEGREE , tid , 64 , TOPM) ;
                        __syncthreads() ;
                        // if(tid < DEGREE)
                        //     warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
                        bitonic_sort_id_by_dis_no_explore(work_mem_float + DEGREE , work_mem_unsigned + DEGREE , 128 - DEGREE) ;
                        __syncthreads() ;
                        if(work_mem_unsigned[DEGREE] != 0xFFFFFFFF) {
                            merge_top(recycle_list_id , &work_mem_unsigned[DEGREE] , recycle_list_dis , &work_mem_float[DEGREE], tid , 128 - DEGREE , recycle_list_size) ;
                            __syncthreads() ;
                        }

                        
                        // 将前16个点插入, 并同步插入到哈希表中
                        #pragma unroll
                        for(unsigned i = tid ; i < DEGREE ; i += blockDim.x * blockDim.y) {
                            top_M_Cand[TOPM + channel_id * DEGREE + i] = work_mem_unsigned[i] | mask ;
                            // hash_insert(hash_table , work_mem_unsigned[i]) ;
                        }
                        // #pragma unroll
                        // for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
                        //     if(top_M_Cand[i] != 0xFFFFFFFF)
                        //         hash_insert(hash_table , top_M_Cand[i] & 0x3FFFFFFF) ;
                        // }
                        // __syncthreads() ;
                        
                        

                        if(tid == 0) {
                            cur_visit_p ++ ; 
                            graph[channel_id] = all_graphs + graph_start_index[next_pid] ;
                            pid[channel_id] = next_pid ; 
                        }
                        __syncthreads() ;
                        // 处理另一个channel
                        channel_id ^= 1 ; 
                }

                // if(tid == 0 && bid == 67) {
                //     printf("block %d , line %d, to_explore[0] = %d , to_explore[1] = %d\n" , bid , __LINE__ , to_explore[0] , to_explore[1]) ;
                // }
                // __syncthreads() ;
            }
            // }
            
        }
        __syncthreads() ;
        // if(tid == 0)
        //     printf("line : %d , epoch : %d\n" , __LINE__ , epoch) ;
        // __syncthreads() ;
        // 对两个通道的扩展写进两个warp里, 防止warp分歧
        if(threadIdx.y < 2 && to_explore[threadIdx.y] != 0xFFFFFFFF) {
            unsigned channel_id = threadIdx.y ;
            unsigned shift_bits = 30 + (channel_id & 1) ;
            // printf("line : %d , shift_bits : %d\n" , __LINE__ , shift_bits) ;
            unsigned mask = (pid[0] == pid[1] ? 0 : (1 << shift_bits)) ; 
            #pragma unroll
            for(unsigned j = laneid; j < DEGREE; j += blockDim.x){

                const unsigned *ex_graph = graph[channel_id] ;
                unsigned to_append_local = ex_graph[to_explore[channel_id] * DEGREE + j];
                unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_append_local] ;
                top_M_Cand[TOPM + j + channel_id * DEGREE] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                top_M_Cand_dis[TOPM + j + channel_id * DEGREE] = (top_M_Cand[TOPM + j + channel_id * DEGREE] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
            }
        }
        __syncthreads();
        // if(tid == 0)
        //     printf("line : %d , epoch : %d\n" , __LINE__ , epoch) ;
        // __syncthreads() ;

    }    
    // 搜索完成, 将recycle_list中的内容合并到top_M_Cand中
    for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
        if(top_M_Cand[i] == 0xFFFFFFFF)
            continue ;
        bool flag = true ;
        unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
        for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
            flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
        }
        if(!flag) {
            top_M_Cand[i] = 0xFFFFFFFF ;
            top_M_Cand_dis[i] = INF_DIS ;
        }
    }
    __syncthreads() ;
    bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM);
    __syncthreads() ;
    // 若recycle_list中确实有点, 则合并
    if(recycle_list_id[0] != 0xFFFFFFFF) {
        merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
        __syncthreads() ;
    }


    /**
        1.先明确对每个分区要做什么, 首先, 将候选点载入top_M_Cand 的末DEGREE个位置, 然后排序
        2.将循环池重置为初始值
        3.从初始点开始扩展, 每次将DEGREE个邻居存放入top_M_Cand的末位, 被淘汰的分区内的点会插入循环池中
        4.当前分区查找结束后,将top_M_Cand中不符合范围约束的点去掉,用循环池中的点替代,然后依据这些点从全局图中扩展到下一个分区的入口点

        step4 的具体实现时, 可以先把不符合范围约束的点的距离改为INF,然后对列表排序, 和recycle_list合并
    **/
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x3FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    for(unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        result_k_id[bid * K + i] = (top_M_Cand[i] & 0x3FFFFFFF) ;
        results_dis[bid * K + i] = top_M_Cand_dis[i] ;
    }
}


// TOPM是candidate队列长度，K是TOP-K也是图索引的度数，DIM是数据集维度，MAX_ITER是最大迭代次数（1000足够），INF_DIS表示无穷大的浮点数
__global__ void select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn(const unsigned* __restrict__ all_graphs, const half* __restrict__ values, const half* __restrict__ query_data, 
    unsigned* results_id, float* results_dis, unsigned* ent_pts , unsigned DEGREE , unsigned TOPM , unsigned DIM , const unsigned* graph_start_index ,const unsigned* graph_size ,
    const unsigned* __restrict__ pids , const unsigned* p_size ,const unsigned* p_start_index ,const float* l_bound ,const float* r_bound ,const float* __restrict__ attrs , unsigned attr_dim , const unsigned* __restrict__ global_graph , unsigned edges_num_in_global_graph,
    unsigned recycle_list_size ,const unsigned* __restrict__ global_idx ,const unsigned* __restrict__ local_idx ,const unsigned* seg_global_idx_start_idx , unsigned pnum ,unsigned K, unsigned* result_k_id ,const unsigned* task_id_mapper) {
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数
    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    unsigned* recycle_list_id = (unsigned*) shared_mem ;
    float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    unsigned* top_M_Cand = (unsigned*) (recycle_list_dis + recycle_list_size) ;
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE * 2) ;
    float* q_l_bound = (float*)(top_M_Cand_dis + TOPM + DEGREE * 2) ;
    float* q_r_bound = (float*)(q_l_bound + attr_dim) ;

    // 这里应该怎么处理8字节的对齐问题？
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) +
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[128] ;
    __shared__ float work_mem_float[128] ;

    __shared__ unsigned cur_visit_p , main_channel , turn ;
    __shared__ unsigned to_explore[2];

    __shared__ unsigned hash_table[HASHLEN];

    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = task_id_mapper[blockIdx.x], laneid = threadIdx.x , warpId = threadIdx.y ;
    // 若线程块发射顺序与blockIdx.x的编号无关, 则可以手动实现按到达顺序取任务的排队器
    unsigned psize_block = p_size[bid] , plist_index = p_start_index[bid] ;
    unsigned pid[2] = {pids[plist_index] , (psize_block >= 2 ? pids[plist_index + 1] : pids[plist_index])} ;
    // pid[0] = 9 ; 
    const unsigned* graph[2] = {all_graphs + graph_start_index[pid[0]] , all_graphs + graph_start_index[pid[1]]};
    const unsigned DEGREE2 = DEGREE * 2 ;

    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
        q_l_bound[i] = l_bound[bid * attr_dim + i] ;
        q_r_bound[i] = r_bound[bid * attr_dim + i] ;
    }
    // 没必要在每个分区开始时都把recycle_list复原, 每个分区得到的入口点肯定是越近越好
    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        recycle_list_id[i] = 0xFFFFFFFF ;
        recycle_list_dis[i] = INF_DIS ;
    }
    
    // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
    for(unsigned i = tid , L2 = TOPM / 2 ; i < TOPM; i += blockDim.x * blockDim.y){
        unsigned shift_bits = 30 +  i / L2;
        unsigned mask = (pid[0] == pid[1] ? 0 : 1 << shift_bits) ; 
        // 用反向掩码, 将另一位置1
        top_M_Cand[i] = global_idx[seg_global_idx_start_idx[pid[i / L2]] + ent_pts[i]] | mask;
        // top_M_Cand[TOPM + DEGREE + i]

        // 哈希表存放的是局部id, 如何统一?  是否可以直接存放全局id? 
        hash_insert(hash_table , top_M_Cand[i] & 0x3FFFFFFF) ;
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}

    if(tid == 0) {
        // printf("psize block : %d\n" , psize_block) ;
        // printf("pid[0] %d , pid[1] %d\n" , pid[0] , pid[1]) ;
        // printf("[") ;
        // for(int i = 0; i < psize_block ; i ++) {
        //     printf("%d" , pids[plist_index + i]) ;
        //     if(i < psize_block - 1)
        //         printf(", ") ;
        // }
        // printf("]\n") ;

        cur_visit_p = (psize_block >= 2 ? 2 : 1) ;
        main_channel = 0 ; 
        turn = 0 ; 
        // printf("can click 2023\n") ;
    }
    __syncthreads();
    
    // 初始化入口节点
    #pragma unroll
    for(unsigned i = threadIdx.y; i < TOPM ; i += blockDim.y){
        if(top_M_Cand[i] == 0xFFFFFFFF)
            continue ;
        // threadIdx.y 0-5 , blockDim.y 即 6
        unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
        // if(pure_id > 1000000)
        //     printf("block : %d , error id - %d\n" , bid ,  pure_id) ;
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            // warp中的每个子线程负责8个字节 即4个half类型分量的计算
            half4 val2 = Load(&values[pure_id * DIM + j * 4]);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
        }
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            // 将warp内所有子线程的计算结果归并到一起
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
        }
        if(laneid == 0){
            top_M_Cand_dis[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
            atomicAdd(dis_cal_cnt + bid , 1) ;
        }
    }
    __syncthreads();
    // if(tid == 0)
    //     printf("line : %d , epoch : %d\n" , __LINE__ , epoch) ;
    // __syncthreads() ;
    // 排序
    // if(tid < DEGREE2)
    //     warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE2);
    // __syncthreads() ;
    bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis , top_M_Cand , TOPM) ;
    __syncthreads() ;
    // 将两个列表归并到一起
    // merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE2 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
    // __syncthreads() ;
    // if(tid < DEGREE2)
    //     warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
    // __syncthreads() ;
    // if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
    //     merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE2 , recycle_list_size) ;
    //     __syncthreads() ;
    // }
    

    // 无限次循环, 直到退出
    for(unsigned epoch = 0 ; epoch < MAX_ITER ; epoch ++) {

        // 每隔4轮将哈希表重置一次
        if((epoch + 1) % (HASH_RESET) == 0) {
            #pragma unroll
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            #pragma unroll
            for(unsigned j = tid; j < TOPM + DEGREE2; j += blockDim.x * blockDim.y){
                // 将此时的TOPM + K 个候选点插入哈希表中
                if(top_M_Cand[j] != 0xFFFFFFFF) {
                    hash_insert(hash_table , top_M_Cand[j] & 0x3FFFFFFF) ; 
                }
            }
        }
        __syncthreads();

        // 下面进行点扩展
        if(tid < 32){
            /*
                扩展规则: 找到当前turn通道的最近点, 然后从另一个通道找到距离该点最近的点, 进行扩展
            */
            if(laneid < 2)
                to_explore[laneid] = 0xFFFFFFFF ;
            __syncwarp() ;

            unsigned shift_bits = (pid[0] == pid[1] ? 31 - (main_channel & 1)  : 31 - (turn & 1));
            unsigned mask = (1 << shift_bits) ;

            // if(laneid == 0) {
            //     printf("turn : %d\n" , (turn & 1)) ;
            // }
            // __syncwarp() ;
            // 若两个通道处理一个图, 则1号warp取第二个候选数, 0号warp取第一个候选点, 否则各自取自己的第一位
            // unsigned fetch_pos = (pid[0] == pid[1] ? (warpId & 1) : 0)  + 1;
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = ((top_M_Cand[j] & mask) == 0 ? 1 : 0) ;
            
                // *** 坑点: __ballot_sync会等待掩码影响的线程执行到__ballot_sync处, 若把0xffffffff设置为掩码, 个别线程可能因循环条件影响提前退出, 
                // 无法执行到此处, 造成程序无限空等
                // unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                // unsigned one_num = __popc(ballot_res) ;
                // if(one_num >= fetch_pos) {
                if(ballot_res > 0) {

                    if(laneid == (__ffs(ballot_res) - 1)) {
                        to_explore[turn & 1] = local_idx[top_M_Cand[j] & 0x3FFFFFFF] ;
                        top_M_Cand[j] |= mask ;
                        // 扩展下一个explore点
                        // for(unsigned next_expand = 0 ; next_expand < edges_num_in_global_graph ; next_expand ++) {
                        //     unsigned pure_id = top_M_Cand[j] & 0x3FFFFFFF ;
                        //     unsigned to_expand = global_graph[pure_id * pnum * edges_num_in_global_graph + pid[turn ^ 1] * edges_num_in_global_graph + next_expand] ;
                        //     unsigned to_expand_global_id = global_idx[seg_global_idx_start_idx[pid[turn ^ 1]] + to_expand] ;
                        //     if(hash_insert(hash_table , to_expand_global_id) != 0) {
                        //         to_explore[turn ^ 1] = to_expand ; 
                        //         break ;
                        //     }
                        // }
                        // turn ^= 1 ;
                    }
                    break ;
                }
            }
            __syncwarp() ;
            
            if(to_explore[turn] != 0xFFFFFFFF) {
                unsigned global_to_explore = global_idx[seg_global_idx_start_idx[pid[turn]] + to_explore[turn]] ;
                unsigned offset = global_to_explore * pnum * edges_num_in_global_graph + pid[turn ^ 1] * edges_num_in_global_graph ;
                for(unsigned j = laneid ; j < 2; j += 32) {
                    unsigned n_p = !hash_peek(hash_table , global_graph[offset + j]) ;
                    unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;

                    if(ballot_res > 0) {
                        if(laneid == (__ffs(ballot_res) - 1)) {
                            to_explore[turn ^ 1] = global_graph[offset + j] ;
                        }
                        break ; 
                    }
                }
            }
        }
        __syncthreads();


        // 若扩展失败, 切换图栈
        if(to_explore[turn & 1] == 0xFFFFFFFF) {
            
            if(cur_visit_p == psize_block && pid[0] == pid[1]) {
                // 两个图都没有扩展点, 且没有图可以切换, 退出
                if(tid == 0) {
                    if(bid == 0)
                        printf("block %d: break in %d\n" ,bid ,  epoch) ; 
                    jump_num_cnt[bid] = epoch ; 
                }
                break ;
            } else if(cur_visit_p == psize_block) {
                // 此时已经是最后一个图, 令两个通道处理同一个图即可
                // 由turn担任闲置通道
                // unsigned idle_channel = (to_explore[0] == 0xFFFFFFFF ? 0 : 1) ;
                // if(tid == 0) {
                    // printf("converge in epoch : %d\n"  , epoch) ;
                    graph[turn] = graph[turn ^ 1] ;
                    pid[turn] = pid[turn ^ 1] ;
                    if(tid == 0) {
                        main_channel = turn ^ 1 ; 
                        // printf("converge in epoch : %d\n"  , epoch) ;
                    }
                // }
                // __syncthreads() ;
                continue ;
            } else {
                // 此时必然只有一个图需要切换栈, 就是turn所指向的通道
                
                // if(tid == 0) {
                //     printf("change in epoch : %d\n" , epoch) ;
                // }
                // __syncthreads() ;
            // 此处有一种情况处理不了, 即两个通道都需要切换上下文, 但剩余要处理的图只剩下一个

            //  else {
                // channel_id 切换图地址, 从全局图中扩展邻居
                unsigned next_pid = pids[plist_index + cur_visit_p] ;
                // unsigned j = 0 ;
                unsigned shift_bits = 30 + (turn & 1) ;
                unsigned mask = 1 << shift_bits ;
                
                // 重置哈希表
                #pragma unroll
                for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
            
                    if(top_M_Cand[i % 64] == 0xFFFFFFFF) {
                        // 填充一个随机数
                        work_mem_unsigned[i] = 0xFFFFFFFF ;
                        work_mem_float[i] = 0.0f ;
                        continue ;
                    }
                    // 用前两个填充, 有重复的部分皆替换为随机点
                    unsigned pure_id = top_M_Cand[i % 64] & 0x3FFFFFFF ;
                    unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / 64] ;
                    unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                    work_mem_unsigned[i] = (hash_insert(hash_table , next_global_id) != 0 ? next_global_id | mask : 0xFFFFFFFF) ;
                    work_mem_float[i] = (work_mem_unsigned[i] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    // if(hash_insert(hash_table , next_global_id) != 0) {
                            // work_mem_unsigned[i] = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                        // work_mem_unsigned[i] = next_global_id | mask ;
                    // } else {
                        
                    // }
                }
                
                __syncthreads() ;
                // 距离计算
                #pragma unroll
                for(unsigned i = threadIdx.y ; i < 128 ; i += blockDim.y) {
                    if(work_mem_unsigned[i] == 0xFFFFFFFF) {
                        // work_mem_float[i] = INF_DIS ;
                        continue ;  
                    }
                    half2 val_res;
                    unsigned pure_id = work_mem_unsigned[i] & 0x3FFFFFFF ;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                        half4 val2 = Load(&values[pure_id * DIM + k * 4]);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                    }
                    #pragma unroll
                    for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                        val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                    }
    
                    if(laneid == 0){
                        work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                        atomicAdd(dis_cal_cnt + bid , 1) ;
                    }
                }
                __syncthreads() ;
                bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                __syncthreads() ;
                // 此处需调整, 这里topm中被淘汰的点没有进入top_M_Cand + TOPM 中
                merge_top_with_recycle_list(top_M_Cand , work_mem_unsigned, top_M_Cand_dis , work_mem_float , tid , 128 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
                // merge_top(top_M_Cand , work_mem_unsigned + DEGREE, top_M_Cand_dis , work_mem_float + DEGREE , tid , 64 , TOPM) ;
                __syncthreads() ;
                // if(tid < DEGREE)
                //     warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
                bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                __syncthreads() ;
                if(work_mem_unsigned[0] != 0xFFFFFFFF) {
                    merge_top(recycle_list_id , work_mem_unsigned , recycle_list_dis , work_mem_float , tid , 128 , recycle_list_size) ;
                    __syncthreads() ;
                }
                graph[turn] = all_graphs + graph_start_index[next_pid] ;
                pid[turn] = next_pid ; 
                if(tid == 0) {
                    cur_visit_p ++ ; 
                }
                __syncthreads() ;
                // 处理另一个channel
                // channel_id ^= 1 ; 
                continue ;
            }
            
        }
        __syncthreads() ;

        // 对两个通道的扩展写进两个warp里, 防止warp分歧
        if(threadIdx.y < 2) {
            if(to_explore[threadIdx.y & 1] != 0xFFFFFFFF) {
                unsigned channel_id = threadIdx.y ;
                unsigned shift_bits = 30 + (channel_id & 1) ;
                // printf("line : %d , shift_bits : %d\n" , __LINE__ , shift_bits) ;
                unsigned mask = (pid[0] == pid[1] ? 0 : (1 << shift_bits)) ; 
                #pragma unroll
                for(unsigned j = laneid; j < DEGREE - (channel_id == turn ? 0 : 1); j += blockDim.x) {

                    const unsigned *ex_graph = graph[channel_id] ;
                    unsigned to_append_local = ex_graph[to_explore[channel_id] * DEGREE + j];
                    unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_append_local] ;
                    top_M_Cand[TOPM + j + channel_id * DEGREE] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                    top_M_Cand_dis[TOPM + j + channel_id * DEGREE] = (top_M_Cand[TOPM + j + channel_id * DEGREE] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                }
                if(channel_id != turn && laneid == 31) {
                    unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_explore[channel_id]] ;
                    top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                    top_M_Cand_dis[TOPM + (channel_id + 1) * DEGREE - 1] = (top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                }
            } else {
                unsigned channel_id = threadIdx.y ;
                #pragma unroll
                for(unsigned i = laneid ; i < DEGREE ; i += blockDim.x) {
                    top_M_Cand[TOPM + i + channel_id * DEGREE] = 0xFFFFFFFF ;
                    top_M_Cand_dis[TOPM + i + channel_id * DEGREE] = INF_DIS ;
                }
            }
            if(tid == 0)
                turn ^= 1 ; 
        } 
        __syncthreads();
        // if(tid == 0 && epoch >= 135) {
        //     printf("epoch-%d : " , epoch) ;
        //     printf("[") ;
        //     for(unsigned i = 0 ; i < TOPM ; i ++) {
        //         if((top_M_Cand[i] >> 30) == 3)
        //             printf("u") ;
        //         printf("%d" , top_M_Cand[i] & 0x3FFFFFFF) ;
        //         if(i < TOPM - 1)
        //         printf(", ") ;
        //     }
        //     printf("];   ") ;
        //     printf("explore-%d :(" , to_explore[0]) ;
        //     for(unsigned i = 0 ; i < DEGREE ; i ++) {
        //         printf("%d" , top_M_Cand[TOPM + i] & 0x3FFFFFFF) ;
        //         if(i <  DEGREE - 1)
        //             printf(", ") ;
        //     }
        //     printf(");  explore-%d :(" , to_explore[1]) ;
        //     for(unsigned i = 0 ; i < DEGREE ; i ++) {
        //         printf("%d" , top_M_Cand[TOPM + DEGREE + i] & 0x3FFFFFFF) ;
        //         if(i < DEGREE - 1)
        //             printf(", ") ;
        //     }
        //     printf(")") ;
        //     printf("\n") ;
        // }
        // __syncthreads() ;

        // 此时无论是通道1还是通道2, 所有节点已在TOPM数组的后半段就位, 因此距离计算可以一视同仁
        #pragma unroll
        for(unsigned i = threadIdx.y; i < DEGREE2 ; i += blockDim.y){
            if(top_M_Cand[TOPM + i] == 0xFFFFFFFF)
                continue ;
            // threadIdx.y 0-5 , blockDim.y 即 6
            unsigned pure_id = top_M_Cand[TOPM + i] & 0x3FFFFFFF ;
            // if(pure_id > 1000000)
            //     printf("block : %d , error id - %d\n" , bid ,  pure_id) ;
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                half4 val2 = Load(&values[pure_id * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                // 将warp内所有子线程的计算结果归并到一起
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }
            if(laneid == 0){
                top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                atomicAdd(dis_cal_cnt + bid , 1) ;
            }
        }
        __syncthreads();

        // #pragma unroll
        // for(unsigned i = tid ; i < DEGREE2  ; i += blockDim.x * blockDim.y) {
        //     if(top_M_Cand[TOPM + i] == 0xFFFFFFFF)
        //         continue ;
        //     atomicAdd(dis_cal_cnt + bid , 1) ;
        //     unsigned pure_id = top_M_Cand[TOPM + i] & 0x3FFFFFFF ;
        //     half2 val_res;
        //     val_res.x = __float2half(0.0f); val_res.y = __float2half(0.0f);
        //     #pragma unroll
        //     for(unsigned j = 0; j < (DIM / 4); j ++){
        //         // warp中的每个子线程负责8个字节 即4个half类型分量的计算
        //         half4 val2 = Load(&values[pure_id * DIM + j * 4]);
        //         val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
        //         val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
        //     }
        //     // if(laneid == 0){
        //         top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
        //     // }
        // }
        // __syncthreads() ;

        // if(tid == 0 && epoch < 10) {
        //     printf("[") ;
        //     for(unsigned i = 0 ; i < DEGREE2 ; i ++) {
        //         printf("%f" , top_M_Cand_dis[TOPM + i]) ;
        //         if(i < DEGREE2 - 1)
        //             printf(", ") ;
        //     }
        //     printf("]\n") ;
        // }
        // __syncthreads() ;
        // if(tid == 0)
        //     printf("line : %d , epoch : %d\n" , __LINE__ , epoch) ;
        // __syncthreads() ;
        // 排序
        if(tid < DEGREE2)
            warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE2);
        __syncthreads() ;
        if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
            // 将两个列表归并到一起
            merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE2 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
            __syncthreads() ;
            if(tid < DEGREE2)
                warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
            __syncthreads() ;
            if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE2 , recycle_list_size) ;
                __syncthreads() ;
            }
        }
    }    
    // printf("tid : %d ;" , tid) ;
    // __syncthreads() ;
    
    // 搜索完成, 将recycle_list中的内容合并到top_M_Cand中
    for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
        if(top_M_Cand[i] == 0xFFFFFFFF)
            continue ;
        bool flag = true ;
        unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
        for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
            flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
        }
        if(!flag) {
            top_M_Cand[i] = 0xFFFFFFFF ;
            top_M_Cand_dis[i] = INF_DIS ;
        }
    }
    __syncthreads() ;
    bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM);
    __syncthreads() ;
    // 若recycle_list中确实有点, 则合并
    if(recycle_list_id[0] != 0xFFFFFFFF) {
        merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
        __syncthreads() ;
    }


    /**
        1.先明确对每个分区要做什么, 首先, 将候选点载入top_M_Cand 的末DEGREE个位置, 然后排序
        2.将循环池重置为初始值
        3.从初始点开始扩展, 每次将DEGREE个邻居存放入top_M_Cand的末位, 被淘汰的分区内的点会插入循环池中
        4.当前分区查找结束后,将top_M_Cand中不符合范围约束的点去掉,用循环池中的点替代,然后依据这些点从全局图中扩展到下一个分区的入口点

        step4 的具体实现时, 可以先把不符合范围约束的点的距离改为INF,然后对列表排序, 和recycle_list合并
    **/
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x3FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    for(unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        result_k_id[bid * K + i] = (top_M_Cand[i] & 0x3FFFFFFF) ;
        results_dis[bid * K + i] = top_M_Cand_dis[i] ;
    }
}


// TOPM是candidate队列长度，K是TOP-K也是图索引的度数，DIM是数据集维度，MAX_ITER是最大迭代次数（1000足够），INF_DIS表示无穷大的浮点数
__global__ void select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_opt(const unsigned* __restrict__ all_graphs, const half* __restrict__ values, const half* __restrict__ query_data, 
    unsigned* results_id, float* results_dis, unsigned* ent_pts , unsigned DEGREE , unsigned TOPM , unsigned DIM , const unsigned* graph_start_index ,const unsigned* graph_size ,
    const unsigned* __restrict__ pids , const unsigned* p_size ,const unsigned* p_start_index ,const float* l_bound ,const float* r_bound ,const float* __restrict__ attrs , unsigned attr_dim , const unsigned* __restrict__ global_graph , unsigned edges_num_in_global_graph,
    unsigned recycle_list_size ,const unsigned* __restrict__ global_idx ,const unsigned* __restrict__ local_idx ,const unsigned* seg_global_idx_start_idx , unsigned pnum ,unsigned K, unsigned* result_k_id ,const unsigned* task_id_mapper) {
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数
    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    unsigned* recycle_list_id = (unsigned*) shared_mem ;
    float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    unsigned* top_M_Cand = (unsigned*) (recycle_list_dis + recycle_list_size) ;
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE * 2) ;
    float* q_l_bound = (float*)(top_M_Cand_dis + TOPM + DEGREE * 2) ;
    float* q_r_bound = (float*)(q_l_bound + attr_dim) ;

    // 这里应该怎么处理8字节的对齐问题？
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) +
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[128] ;
    __shared__ float work_mem_float[128] ;

    __shared__ unsigned prefix_sum[1024] ;
    __shared__ unsigned top_prefix[32] ;
    __shared__ unsigned indices_arr[1024] ;
    __shared__ unsigned task_num ; 

    __shared__ unsigned cur_visit_p , main_channel , turn ;
    __shared__ unsigned to_explore[2];

    __shared__ unsigned hash_table[HASHLEN];

    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = task_id_mapper[blockIdx.x], laneid = threadIdx.x , warpId = threadIdx.y ;
    // 若线程块发射顺序与blockIdx.x的编号无关, 则可以手动实现按到达顺序取任务的排队器
    unsigned psize_block = p_size[bid] , plist_index = p_start_index[bid] ;
    unsigned pid[2] = {pids[plist_index] , (psize_block >= 2 ? pids[plist_index + 1] : pids[plist_index])} ;
    // pid[0] = 9 ; 
    const unsigned* graph[2] = {all_graphs + graph_start_index[pid[0]] , all_graphs + graph_start_index[pid[1]]};
    const unsigned DEGREE2 = DEGREE * 2 ;

    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
        q_l_bound[i] = l_bound[bid * attr_dim + i] ;
        q_r_bound[i] = r_bound[bid * attr_dim + i] ;
    }
    // 没必要在每个分区开始时都把recycle_list复原, 每个分区得到的入口点肯定是越近越好
    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        recycle_list_id[i] = 0xFFFFFFFF ;
        recycle_list_dis[i] = INF_DIS ;
    }
    
    // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
    for(unsigned i = tid , L2 = TOPM / 2 ; i < TOPM; i += blockDim.x * blockDim.y){
        unsigned shift_bits = 30 +  i / L2;
        unsigned mask = (pid[0] == pid[1] ? 0 : 1 << shift_bits) ; 
        // 用反向掩码, 将另一位置1
        top_M_Cand[i] = global_idx[seg_global_idx_start_idx[pid[i / L2]] + ent_pts[i]] | mask;
        // top_M_Cand[TOPM + DEGREE + i]

        // 哈希表存放的是局部id, 如何统一?  是否可以直接存放全局id? 
        hash_insert(hash_table , top_M_Cand[i] & 0x3FFFFFFF) ;
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}

    // if(bid != 2487)
    //     return  ;
    if(tid == 0) {
        // printf("psize block : %d\n" , psize_block) ;
        // printf("pid[0] %d , pid[1] %d\n" , pid[0] , pid[1]) ;
        // printf("[") ;
        // for(int i = 0; i < psize_block ; i ++) {
        //     printf("%d" , pids[plist_index + i]) ;
        //     if(i < psize_block - 1)
        //         printf(", ") ;
        // }
        // printf("]\n") ;

        cur_visit_p = (psize_block >= 2 ? 2 : 1) ;
        main_channel = 0 ; 
        turn = 0 ; 
        // printf("can click 2023\n") ;
    }
    __syncthreads();
    
    // 初始化入口节点
    #pragma unroll
    for(unsigned i = threadIdx.y; i < TOPM ; i += blockDim.y){
        // if(top_M_Cand[i] == 0xFFFFFFFF)
        //     continue ;
        // threadIdx.y 0-5 , blockDim.y 即 6
        unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
        // if(pure_id > 1000000)
        //     printf("block : %d , error id - %d\n" , bid ,  pure_id) ;
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            // warp中的每个子线程负责8个字节 即4个half类型分量的计算
            half4 val2 = Load(&values[pure_id * DIM + j * 4]);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
        }
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            // 将warp内所有子线程的计算结果归并到一起
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
        }
        if(laneid == 0){
            top_M_Cand_dis[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
            atomicAdd(dis_cal_cnt + bid , 1) ;
        }
    }
    __syncthreads();
    // if(tid == 0)
    //     printf("line : %d , epoch : %d\n" , __LINE__ , epoch) ;
    // __syncthreads() ;
    // 排序
    // if(tid < DEGREE2)
    //     warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE2);
    // __syncthreads() ;
    bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis , top_M_Cand , TOPM) ;
    __syncthreads() ;
    // 将两个列表归并到一起
    // merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE2 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
    // __syncthreads() ;
    // if(tid < DEGREE2)
    //     warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
    // __syncthreads() ;
    // if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
    //     merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE2 , recycle_list_size) ;
    //     __syncthreads() ;
    // }
    

    // 无限次循环, 直到退出
    for(unsigned epoch = 0 ; epoch < MAX_ITER ; epoch ++) {

        // 每隔4轮将哈希表重置一次
        if((epoch + 1) % (HASH_RESET) == 0) {
            #pragma unroll
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            #pragma unroll
            for(unsigned j = tid; j < TOPM + DEGREE2; j += blockDim.x * blockDim.y){
                // 将此时的TOPM + K 个候选点插入哈希表中
                if(top_M_Cand[j] != 0xFFFFFFFF) {
                    hash_insert(hash_table , top_M_Cand[j] & 0x3FFFFFFF) ; 
                }
            }
        }
        __syncthreads();

        // 下面进行点扩展
        if(tid < 32){
            /*
                扩展规则: 找到当前turn通道的最近点, 然后从另一个通道找到距离该点最近的点, 进行扩展
            */
            if(laneid < 2)
                to_explore[laneid] = 0xFFFFFFFF ;
            __syncwarp() ;

            unsigned shift_bits = (pid[0] == pid[1] ? 31 - (main_channel & 1)  : 31 - (turn & 1));
            unsigned mask = (1 << shift_bits) ;

            // if(laneid == 0) {
            //     printf("turn : %d\n" , (turn & 1)) ;
            // }
            // __syncwarp() ;
            // 若两个通道处理一个图, 则1号warp取第二个候选数, 0号warp取第一个候选点, 否则各自取自己的第一位
            // unsigned fetch_pos = (pid[0] == pid[1] ? (warpId & 1) : 0)  + 1;
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = ((top_M_Cand[j] & mask) == 0 ? 1 : 0) ;
            
                // *** 坑点: __ballot_sync会等待掩码影响的线程执行到__ballot_sync处, 若把0xffffffff设置为掩码, 个别线程可能因循环条件影响提前退出, 
                // 无法执行到此处, 造成程序无限空等
                // unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                // unsigned one_num = __popc(ballot_res) ;
                // if(one_num >= fetch_pos) {
                if(ballot_res > 0) {

                    if(laneid == (__ffs(ballot_res) - 1)) {
                        to_explore[turn & 1] = local_idx[top_M_Cand[j] & 0x3FFFFFFF] ;
                        top_M_Cand[j] |= mask ;
                        // 扩展下一个explore点
                        // for(unsigned next_expand = 0 ; next_expand < edges_num_in_global_graph ; next_expand ++) {
                        //     unsigned pure_id = top_M_Cand[j] & 0x3FFFFFFF ;
                        //     unsigned to_expand = global_graph[pure_id * pnum * edges_num_in_global_graph + pid[turn ^ 1] * edges_num_in_global_graph + next_expand] ;
                        //     unsigned to_expand_global_id = global_idx[seg_global_idx_start_idx[pid[turn ^ 1]] + to_expand] ;
                        //     if(hash_insert(hash_table , to_expand_global_id) != 0) {
                        //         to_explore[turn ^ 1] = to_expand ; 
                        //         break ;
                        //     }
                        // }
                        // turn ^= 1 ;
                    }
                    break ;
                }
            }
            __syncwarp() ;
            
            if(to_explore[turn] != 0xFFFFFFFF) {
                unsigned global_to_explore = global_idx[seg_global_idx_start_idx[pid[turn]] + to_explore[turn]] ;
                unsigned offset = global_to_explore * pnum * edges_num_in_global_graph + pid[turn ^ 1] * edges_num_in_global_graph ;
                for(unsigned j = laneid ; j < 2; j += 32) {
                    unsigned n_p = !hash_peek(hash_table , global_graph[offset + j]) ;
                    unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;

                    if(ballot_res > 0) {
                        if(laneid == (__ffs(ballot_res) - 1)) {
                            to_explore[turn ^ 1] = global_graph[offset + j] ;
                        }
                        break ; 
                    }
                }
            }
        }
        __syncthreads();


        // 若扩展失败, 切换图栈
        if(to_explore[turn & 1] == 0xFFFFFFFF) {
            
            if(cur_visit_p == psize_block && pid[0] == pid[1]) {
                // 两个图都没有扩展点, 且没有图可以切换, 退出
                if(tid == 0) {
                    if(bid == 0)
                        printf("block %d: break in %d\n" ,bid ,  epoch) ; 
                    jump_num_cnt[bid] = epoch ; 
                }
                break ;
            } else if(cur_visit_p == psize_block) {
                    graph[turn] = graph[turn ^ 1] ;
                    pid[turn] = pid[turn ^ 1] ;
                    if(tid == 0) {
                        main_channel = turn ^ 1 ; 
                        // printf("converge in epoch : %d\n"  , epoch) ;
                    }
                continue ;
            } else {
               
                unsigned next_pid = pids[plist_index + cur_visit_p] ;
                // unsigned j = 0 ;
                unsigned shift_bits = 30 + (turn & 1) ;
                unsigned mask = 1 << shift_bits ;
                
                // 重置哈希表
                #pragma unroll
                for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
            
                    if(top_M_Cand[i % 64] == 0xFFFFFFFF) {
                        // 填充一个随机数
                        work_mem_unsigned[i] = 0xFFFFFFFF ;
                        work_mem_float[i] = 0.0f ;
                        continue ;
                    }
                    // 用前两个填充, 有重复的部分皆替换为随机点
                    unsigned pure_id = top_M_Cand[i % 64] & 0x3FFFFFFF ;
                    unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / 64] ;
                    unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                    work_mem_unsigned[i] = (hash_insert(hash_table , next_global_id) != 0 ? next_global_id | mask : 0xFFFFFFFF) ;
                    work_mem_float[i] = (work_mem_unsigned[i] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                }
                
                // 先将任务压缩到前面 , 扫描出warp内的前缀和
                for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) {
                    // 在bitmap中的单元格位置
                    bool flag = (work_mem_unsigned[i] != 0xFFFFFFFFu) ;
                    unsigned ballot_res = __ballot_sync(__activemask() , flag) ;


                    // unsigned cell_id = chunk_id + (i >> 5) ;
                    // unsigned bit_pos = i % 32 + 1;
                    unsigned mask = ((1 << laneid) - 1) ;
                    unsigned index = (ballot_res & mask) ;
                    prefix_sum[i] = __popc(index) ;
                    if(laneid == 0)
                        top_prefix[(i >> 5)] = __popc(ballot_res) ;
                }
                __syncthreads() ;
                // 分出一个warp扫描更高层次的前缀和
                if(tid < 32) {
                    unsigned v = (laneid < 4 ? top_prefix[laneid] : 0) ;
                    unsigned x = v ; 
                    // 将长度大小为32的各个子prefix合并为最终的prefix_sum 
                    #pragma unroll
                    for(unsigned offset = 1 ; offset < 32 ; offset <<= 1) {
                        unsigned y = __shfl_up_sync(0xFFFFFFFF , v , offset) ;
                        if(laneid >= offset) v += y ;
                    }
                    top_prefix[laneid] = v - x ;
                }
                if(tid == 0) {
                    task_num = top_prefix[31] ;
                    // for(unsigned i = 0 ; i  <task_num ; i ++)
                    //     indices_arr[i] = 0xFFFFFFFFu ; 
                }
                __syncthreads() ;
        
                for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) {
                    if(work_mem_unsigned[i] != 0xFFFFFFFFu) {
                        unsigned local_cell_id = (i >> 5) ;
                        // unsigned put_pos = (local_cell_id == 0 ? 0 : top_prefix[local_cell_id - 1]) + (id_in_cell == 0 ? 0 : prefix_sum[i - 1]) ;
                        unsigned put_pos = top_prefix[local_cell_id] + prefix_sum[i] ;
                        indices_arr[put_pos] = i ;
                    }
                }
                __syncthreads() ;
                // if(tid == 0 && bid == 2487) {
                //     printf("task_num : %d\n" , task_num) ;

                //     printf("prefix[") ;
                //     for(unsigned i = 0 ; i < 128 ;i ++) {
                //         printf("i%d-%d-%d" , i , prefix_sum[i] , (work_mem_unsigned[i] != 0xFFFFFFFFu ? 1 : 0)) ;
                //         if(i < 127)
                //             printf(" ,") ;
                //     }
                //     printf("]\n") ;
                //     printf("top_prefix[") ;
                //     for(unsigned i = 0 ; i < 32 ;i ++) {
                //         printf("%d" , top_prefix[i]) ;
                //         if(i < 31)
                //             printf(" ,") ;
                //     }
                //     printf("]\n") ;

                //     printf("indices_arr[") ;
                //     for(unsigned i = 0 ; i < task_num ;i ++) {
                //         printf("%d-%d" , indices_arr[i] , work_mem_unsigned[indices_arr[i]] & 0x3FFFFFFFu) ;
                //         if(i < task_num - 1)
                //             printf(" ,") ;
                //     }
                //     printf("]\n") ;
                // }
                // __syncthreads() ;
                // 距离计算
                #pragma unroll
                for(unsigned i = threadIdx.y ; i < task_num ; i += blockDim.y) {
                    unsigned task_id = indices_arr[i] ;
                    // if(work_mem_unsigned[task_id] == 0xFFFFFFFFu && laneid == 0) {
                    //     printf("bid : %d , task_id : %d ,  task_num : %d , line : %d\n" , bid , task_id , task_num , __LINE__) ;
                    // }
                    half2 val_res;
                    unsigned pure_id = work_mem_unsigned[task_id] & 0x3FFFFFFF ;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                        half4 val2 = Load(&values[pure_id * DIM + k * 4]);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                    }
                    #pragma unroll
                    for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                        val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                    }
    
                    if(laneid == 0){
                        work_mem_float[task_id] = __half2float(val_res.x) + __half2float(val_res.y) ;
                        atomicAdd(dis_cal_cnt + bid , 1) ;
                    }
                }
                __syncthreads() ;
                bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                __syncthreads() ;
                // 此处需调整, 这里topm中被淘汰的点没有进入top_M_Cand + TOPM 中
                merge_top_with_recycle_list(top_M_Cand , work_mem_unsigned, top_M_Cand_dis , work_mem_float , tid , 128 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
                // merge_top(top_M_Cand , work_mem_unsigned + DEGREE, top_M_Cand_dis , work_mem_float + DEGREE , tid , 64 , TOPM) ;
                __syncthreads() ;
                // if(tid < DEGREE)
                //     warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
                bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                __syncthreads() ;
                if(work_mem_unsigned[0] != 0xFFFFFFFF) {
                    merge_top(recycle_list_id , work_mem_unsigned , recycle_list_dis , work_mem_float , tid , 128 , recycle_list_size) ;
                    __syncthreads() ;
                }
                graph[turn] = all_graphs + graph_start_index[next_pid] ;
                pid[turn] = next_pid ; 
                if(tid == 0) {
                    cur_visit_p ++ ; 
                }
                __syncthreads() ;
                // 处理另一个channel
                // channel_id ^= 1 ; 
                continue ;
            }
            
        }
        __syncthreads() ;

        // 对两个通道的扩展写进两个warp里, 防止warp分歧
        if(threadIdx.y < 2) {
            if(to_explore[threadIdx.y & 1] != 0xFFFFFFFF) {
                unsigned channel_id = threadIdx.y ;
                unsigned shift_bits = 30 + (channel_id & 1) ;
                // printf("line : %d , shift_bits : %d\n" , __LINE__ , shift_bits) ;
                unsigned mask = (pid[0] == pid[1] ? 0 : (1 << shift_bits)) ; 
                #pragma unroll
                for(unsigned j = laneid; j < DEGREE - (channel_id == turn ? 0 : 1); j += blockDim.x) {

                    const unsigned *ex_graph = graph[channel_id] ;
                    unsigned to_append_local = ex_graph[to_explore[channel_id] * DEGREE + j];
                    unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_append_local] ;
                    top_M_Cand[TOPM + j + channel_id * DEGREE] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                    top_M_Cand_dis[TOPM + j + channel_id * DEGREE] = (top_M_Cand[TOPM + j + channel_id * DEGREE] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                }
                if(channel_id != turn && laneid == 31) {
                    unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_explore[channel_id]] ;
                    top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                    top_M_Cand_dis[TOPM + (channel_id + 1) * DEGREE - 1] = (top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                }
            } else {
                unsigned channel_id = threadIdx.y ;
                #pragma unroll
                for(unsigned i = laneid ; i < DEGREE ; i += blockDim.x) {
                    top_M_Cand[TOPM + i + channel_id * DEGREE] = 0xFFFFFFFF ;
                    top_M_Cand_dis[TOPM + i + channel_id * DEGREE] = INF_DIS ;
                }
            }
            if(tid == 0)
                turn ^= 1 ; 
        } 
        __syncthreads();

        // 先将任务压缩到前面 , 扫描出warp内的前缀和
        for(unsigned i = tid ; i < DEGREE2 ; i += blockDim.x * blockDim.y) {
            // 在bitmap中的单元格位置
            bool flag = (top_M_Cand[i + TOPM] != 0xFFFFFFFFu) ;
            unsigned ballot_res = __ballot_sync(__activemask() , flag) ;


            // unsigned cell_id = chunk_id + (i >> 5) ;
            // unsigned bit_pos = i % 32 + 1;
            unsigned mask = ((1 << laneid) - 1) ;
            unsigned index = (ballot_res & mask) ;
            prefix_sum[i] = __popc(index) ;
            if(laneid == 0)
                top_prefix[(i >> 5)] = __popc(ballot_res) ;
        }
        __syncthreads() ;
        // 分出一个warp扫描更高层次的前缀和
        if(tid < 32) {
            unsigned v = (laneid < (DEGREE2 >> 5) ? top_prefix[laneid] : 0) ;
            unsigned x = v ; 
            // 将长度大小为32的各个子prefix合并为最终的prefix_sum 
            #pragma unroll
            for(unsigned offset = 1 ; offset < 32 ; offset <<= 1) {
                unsigned y = __shfl_up_sync(0xFFFFFFFF , v , offset) ;
                if(laneid >= offset) v += y ;
            }
            top_prefix[laneid] = v - x ;
        }
        if(tid == 0) {
            task_num = top_prefix[31] ;
        }
        __syncthreads() ;
        for(unsigned i = tid ; i < DEGREE2 ; i += blockDim.x * blockDim.y) {
            if(top_M_Cand[i + TOPM] != 0xFFFFFFFFu) {
                unsigned local_cell_id = (i >> 5) ;
                // unsigned put_pos = (local_cell_id == 0 ? 0 : top_prefix[local_cell_id - 1]) + (id_in_cell == 0 ? 0 : prefix_sum[i - 1]) ;
                unsigned put_pos = top_prefix[local_cell_id] + prefix_sum[i] ;
                indices_arr[put_pos] = i ;
            }
        }
        __syncthreads() ;


        // 此时无论是通道1还是通道2, 所有节点已在TOPM数组的后半段就位, 因此距离计算可以一视同仁
        #pragma unroll
        for(unsigned i = threadIdx.y; i < task_num ; i += blockDim.y){
            // if(top_M_Cand[TOPM + i] == 0xFFFFFFFF)
            //     continue ;
            unsigned task_id = indices_arr[i] ;
            // if(top_M_Cand[TOPM + task_id] == 0xFFFFFFFFu && laneid == 0) {
            //     printf("bid : %d , task_id : %d , task_num : %d ,  line : %d\n" , bid , task_id , task_num , __LINE__) ;
            // }
            // threadIdx.y 0-5 , blockDim.y 即 6
            unsigned pure_id = top_M_Cand[TOPM + task_id] & 0x3FFFFFFF ;
            // if(pure_id > 1000000)
            //     printf("block : %d , error id - %d\n" , bid ,  pure_id) ;
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                half4 val2 = Load(&values[pure_id * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                // 将warp内所有子线程的计算结果归并到一起
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }
            if(laneid == 0){
                top_M_Cand_dis[TOPM + task_id] = __half2float(val_res.x) + __half2float(val_res.y) ;
                atomicAdd(dis_cal_cnt + bid , 1) ;
            }
        }
        __syncthreads();

        // 排序
        if(tid < DEGREE2)
            warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE2);
        __syncthreads() ;
        if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
            // 将两个列表归并到一起
            merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE2 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
            __syncthreads() ;
            if(tid < DEGREE2)
                warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
            __syncthreads() ;
            if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE2 , recycle_list_size) ;
                __syncthreads() ;
            }
        }
    }    
    // printf("tid : %d ;" , tid) ;
    // __syncthreads() ;
    
    // 搜索完成, 将recycle_list中的内容合并到top_M_Cand中
    for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
        if(top_M_Cand[i] == 0xFFFFFFFF)
            continue ;
        bool flag = true ;
        unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
        for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
            flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
        }
        if(!flag) {
            top_M_Cand[i] = 0xFFFFFFFF ;
            top_M_Cand_dis[i] = INF_DIS ;
        }
    }
    __syncthreads() ;
    bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM);
    __syncthreads() ;
    // 若recycle_list中确实有点, 则合并
    if(recycle_list_id[0] != 0xFFFFFFFF) {
        merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
        __syncthreads() ;
    }


    /**
        1.先明确对每个分区要做什么, 首先, 将候选点载入top_M_Cand 的末DEGREE个位置, 然后排序
        2.将循环池重置为初始值
        3.从初始点开始扩展, 每次将DEGREE个邻居存放入top_M_Cand的末位, 被淘汰的分区内的点会插入循环池中
        4.当前分区查找结束后,将top_M_Cand中不符合范围约束的点去掉,用循环池中的点替代,然后依据这些点从全局图中扩展到下一个分区的入口点

        step4 的具体实现时, 可以先把不符合范围约束的点的距离改为INF,然后对列表排序, 和recycle_list合并
    **/
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x3FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    for(unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        result_k_id[bid * K + i] = (top_M_Cand[i] & 0x3FFFFFFF) ;
        results_dis[bid * K + i] = top_M_Cand_dis[i] ;
    }
}


// TOPM是candidate队列长度，K是TOP-K也是图索引的度数，DIM是数据集维度，MAX_ITER是最大迭代次数（1000足够），INF_DIS表示无穷大的浮点数
__global__ void select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_V2(const unsigned* __restrict__ all_graphs, const half* __restrict__ values, const half* __restrict__ query_data, 
    unsigned* results_id, float* results_dis, unsigned* ent_pts , unsigned DEGREE , unsigned TOPM , unsigned DIM , const unsigned* graph_start_index ,const unsigned* graph_size ,
    const unsigned* __restrict__ pids , const unsigned* p_size ,const unsigned* p_start_index ,const float* l_bound ,const float* r_bound ,const float* __restrict__ attrs , unsigned attr_dim , const unsigned* __restrict__ global_graph , unsigned edges_num_in_global_graph,
    unsigned recycle_list_size ,const unsigned* __restrict__ global_idx ,const unsigned* __restrict__ local_idx ,const unsigned* seg_global_idx_start_idx , unsigned pnum ,unsigned K, unsigned* result_k_id ,const unsigned* task_id_mapper) {
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数
    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    unsigned* recycle_list_id = (unsigned*) shared_mem ;
    float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    unsigned* top_M_Cand = (unsigned*) (recycle_list_dis + recycle_list_size) ;
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE * 2) ;
    float* q_l_bound = (float*)(top_M_Cand_dis + TOPM + DEGREE * 2) ;
    float* q_r_bound = (float*)(q_l_bound + attr_dim) ;

    // 这里应该怎么处理8字节的对齐问题？
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) +
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[128] ;
    __shared__ float work_mem_float[128] ;

    __shared__ unsigned cur_visit_p , main_channel , turn ;
    __shared__ unsigned to_explore[2];

    __shared__ unsigned hash_table[HASHLEN];

    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = task_id_mapper[blockIdx.x], laneid = threadIdx.x , warpId = threadIdx.y ;
    // 若线程块发射顺序与blockIdx.x的编号无关, 则可以手动实现按到达顺序取任务的排队器
    unsigned psize_block = p_size[bid] , plist_index = p_start_index[bid] ;
    unsigned pid[2] = {pids[plist_index] , (psize_block >= 2 ? pids[plist_index + 1] : pids[plist_index])} ;
    // pid[0] = 9 ; 
    const unsigned* graph[2] = {all_graphs + graph_start_index[pid[0]] , all_graphs + graph_start_index[pid[1]]};
    const unsigned DEGREE2 = DEGREE * 2 ;

    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
        q_l_bound[i] = l_bound[bid * attr_dim + i] ;
        q_r_bound[i] = r_bound[bid * attr_dim + i] ;
    }
    // 没必要在每个分区开始时都把recycle_list复原, 每个分区得到的入口点肯定是越近越好
    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        recycle_list_id[i] = 0xFFFFFFFF ;
        recycle_list_dis[i] = INF_DIS ;
    }
    
    // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
    for(unsigned i = tid , L2 = TOPM / 2 ; i < TOPM; i += blockDim.x * blockDim.y){
        unsigned shift_bits = 30 +  i / L2;
        unsigned mask = (pid[0] == pid[1] ? 0 : 1 << shift_bits) ; 
        // 用反向掩码, 将另一位置1
        top_M_Cand[i] = global_idx[seg_global_idx_start_idx[pid[i / L2]] + ent_pts[i]] | mask;
        // top_M_Cand[TOPM + DEGREE + i]

        // 哈希表存放的是局部id, 如何统一?  是否可以直接存放全局id? 
        hash_insert(hash_table , top_M_Cand[i] & 0x3FFFFFFF) ;
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}

    if(tid == 0) {
        // printf("psize block : %d\n" , psize_block) ;
        // printf("pid[0] %d , pid[1] %d\n" , pid[0] , pid[1]) ;
        // printf("[") ;
        // for(int i = 0; i < psize_block ; i ++) {
        //     printf("%d" , pids[plist_index + i]) ;
        //     if(i < psize_block - 1)
        //         printf(", ") ;
        // }
        // printf("]\n") ;

        cur_visit_p = (psize_block >= 2 ? 2 : 1) ;
        main_channel = 0 ; 
        turn = 0 ; 
        // printf("can click 2023\n") ;
    }
    __syncthreads();
    
    // 初始化入口节点
    #pragma unroll
    for(unsigned i = threadIdx.y; i < TOPM ; i += blockDim.y){
        if(top_M_Cand[i] == 0xFFFFFFFF)
            continue ;
        // threadIdx.y 0-5 , blockDim.y 即 6
        unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
        // if(pure_id > 1000000)
        //     printf("block : %d , error id - %d\n" , bid ,  pure_id) ;
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            // warp中的每个子线程负责8个字节 即4个half类型分量的计算
            half4 val2 = Load(&values[pure_id * DIM + j * 4]);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
        }
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            // 将warp内所有子线程的计算结果归并到一起
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
        }
        if(laneid == 0){
            top_M_Cand_dis[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
        }
    }
    __syncthreads();
    // if(tid == 0)
    //     printf("line : %d , epoch : %d\n" , __LINE__ , epoch) ;
    // __syncthreads() ;
    // 排序
    // if(tid < DEGREE2)
    //     warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE2);
    // __syncthreads() ;
    bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis , top_M_Cand , TOPM) ;
    __syncthreads() ;
    // 将两个列表归并到一起
    // merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE2 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
    // __syncthreads() ;
    // if(tid < DEGREE2)
    //     warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
    // __syncthreads() ;
    // if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
    //     merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE2 , recycle_list_size) ;
    //     __syncthreads() ;
    // }
    

    // 无限次循环, 直到退出
    for(unsigned epoch = 0 ; epoch < MAX_ITER ; epoch ++) {

        // 每隔4轮将哈希表重置一次
        if((epoch + 1) % (HASH_RESET) == 0) {
            #pragma unroll
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            #pragma unroll
            for(unsigned j = tid; j < TOPM + DEGREE2; j += blockDim.x * blockDim.y){
                // 将此时的TOPM + K 个候选点插入哈希表中
                if(top_M_Cand[j] != 0xFFFFFFFF) {
                    hash_insert(hash_table , top_M_Cand[j] & 0x3FFFFFFF) ; 
                }
            }
        }
        __syncthreads();

        // 下面进行点扩展
        if(tid < 32){
            /*
                扩展规则: 找到当前turn通道的最近点, 然后从另一个通道找到距离该点最近的点, 进行扩展
            */
            if(laneid < 2)
                to_explore[laneid] = 0xFFFFFFFF ;
            __syncwarp() ;

            unsigned shift_bits = (pid[0] == pid[1] ? 31 - (main_channel & 1)  : 31 - (turn & 1));
            unsigned mask = (1 << shift_bits) ;

            // if(laneid == 0) {
            //     printf("turn : %d\n" , (turn & 1)) ;
            // }
            // __syncwarp() ;
            // 若两个通道处理一个图, 则1号warp取第二个候选数, 0号warp取第一个候选点, 否则各自取自己的第一位
            // unsigned fetch_pos = (pid[0] == pid[1] ? (warpId & 1) : 0)  + 1;
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = ((top_M_Cand[j] & mask) == 0 ? 1 : 0) ;
            
                // *** 坑点: __ballot_sync会等待掩码影响的线程执行到__ballot_sync处, 若把0xffffffff设置为掩码, 个别线程可能因循环条件影响提前退出, 
                // 无法执行到此处, 造成程序无限空等
                // unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                // unsigned one_num = __popc(ballot_res) ;
                // if(one_num >= fetch_pos) {
                if(ballot_res > 0) {

                    if(laneid == (__ffs(ballot_res) - 1)) {
                        to_explore[turn & 1] = local_idx[top_M_Cand[j] & 0x3FFFFFFF] ;
                        top_M_Cand[j] |= mask ;
                        // 扩展下一个explore点
                        // for(unsigned next_expand = 0 ; next_expand < edges_num_in_global_graph ; next_expand ++) {
                        //     unsigned pure_id = top_M_Cand[j] & 0x3FFFFFFF ;
                        //     unsigned to_expand = global_graph[pure_id * pnum * edges_num_in_global_graph + pid[turn ^ 1] * edges_num_in_global_graph + next_expand] ;
                        //     unsigned to_expand_global_id = global_idx[seg_global_idx_start_idx[pid[turn ^ 1]] + to_expand] ;
                        //     if(hash_insert(hash_table , to_expand_global_id) != 0) {
                        //         to_explore[turn ^ 1] = to_expand ; 
                        //         break ;
                        //     }
                        // }
                        // turn ^= 1 ;
                    }
                    break ;
                }
            }
        }
        __syncthreads();


        // 若扩展失败, 切换图栈
        if(to_explore[turn & 1] == 0xFFFFFFFF) {
            
            if(cur_visit_p == psize_block && pid[0] == pid[1]) {
                // 两个图都没有扩展点, 且没有图可以切换, 退出
                if(tid == 0) {
                    if(bid == 0)
                        printf("block %d: break in %d\n" ,bid ,  epoch) ; 
                    jump_num_cnt[bid] = epoch ; 
                }
                break ;
            } else if(cur_visit_p == psize_block) {
                // 此时已经是最后一个图, 令两个通道处理同一个图即可
                // 由turn担任闲置通道
                // unsigned idle_channel = (to_explore[0] == 0xFFFFFFFF ? 0 : 1) ;
                // if(tid == 0) {
                    // printf("converge in epoch : %d\n"  , epoch) ;
                    graph[turn] = graph[turn ^ 1] ;
                    pid[turn] = pid[turn ^ 1] ;
                    if(tid == 0) {
                        main_channel = turn ^ 1 ; 
                        // printf("converge in epoch : %d\n"  , epoch) ;
                    }
                // }
                // __syncthreads() ;
                continue ;
            } else {
                // 此时必然只有一个图需要切换栈, 就是turn所指向的通道
                
                // if(tid == 0) {
                //     printf("change in epoch : %d\n" , epoch) ;
                // }
                // __syncthreads() ;
            // 此处有一种情况处理不了, 即两个通道都需要切换上下文, 但剩余要处理的图只剩下一个

            //  else {
                // channel_id 切换图地址, 从全局图中扩展邻居
                unsigned next_pid = pids[plist_index + cur_visit_p] ;
                // unsigned j = 0 ;
                unsigned shift_bits = 30 + (turn & 1) ;
                unsigned mask = 1 << shift_bits ;
                
                // 重置哈希表
                #pragma unroll
                for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
            
                    if(top_M_Cand[i % 64] == 0xFFFFFFFF) {
                        // 填充一个随机数
                        work_mem_unsigned[i] = 0xFFFFFFFF ;
                        work_mem_float[i] = 0.0f ;
                        continue ;
                    }
                    // 用前两个填充, 有重复的部分皆替换为随机点
                    unsigned pure_id = top_M_Cand[i % 64] & 0x3FFFFFFF ;
                    unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / 64] ;
                    unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                    work_mem_unsigned[i] = (hash_insert(hash_table , next_global_id) != 0 ? next_global_id | mask : 0xFFFFFFFF) ;
                    work_mem_float[i] = (work_mem_unsigned[i] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    // if(hash_insert(hash_table , next_global_id) != 0) {
                            // work_mem_unsigned[i] = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                        // work_mem_unsigned[i] = next_global_id | mask ;
                    // } else {
                        
                    // }
                }
                
                __syncthreads() ;
                // 距离计算
                #pragma unroll
                for(unsigned i = threadIdx.y ; i < 128 ; i += blockDim.y) {
                    if(work_mem_unsigned[i] == 0xFFFFFFFF) {
                        // work_mem_float[i] = INF_DIS ;
                        continue ;  
                    }
                    half2 val_res;
                    unsigned pure_id = work_mem_unsigned[i] & 0x3FFFFFFF ;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                        half4 val2 = Load(&values[pure_id * DIM + k * 4]);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                        val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                    }
                    #pragma unroll
                    for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                        val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                    }
    
                    if(laneid == 0){
                        work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                    }
                }
                __syncthreads() ;
                bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                __syncthreads() ;
                // 此处需调整, 这里topm中被淘汰的点没有进入top_M_Cand + TOPM 中
                merge_top_with_recycle_list(top_M_Cand , work_mem_unsigned, top_M_Cand_dis , work_mem_float , tid , 128 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
                // merge_top(top_M_Cand , work_mem_unsigned + DEGREE, top_M_Cand_dis , work_mem_float + DEGREE , tid , 64 , TOPM) ;
                __syncthreads() ;
                // if(tid < DEGREE)
                //     warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
                bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                __syncthreads() ;
                if(work_mem_unsigned[0] != 0xFFFFFFFF) {
                    merge_top(recycle_list_id , work_mem_unsigned , recycle_list_dis , work_mem_float , tid , 128 , recycle_list_size) ;
                    __syncthreads() ;
                }
                graph[turn] = all_graphs + graph_start_index[next_pid] ;
                pid[turn] = next_pid ; 
                if(tid == 0) {
                    cur_visit_p ++ ; 
                }
                __syncthreads() ;
                // 处理另一个channel
                // channel_id ^= 1 ; 
                continue ;
            }
            
        }
        __syncthreads() ;

        // 对两个通道的扩展写进两个warp里, 防止warp分歧
        if(threadIdx.y < 2) {
            unsigned channel_id = threadIdx.y ;
            unsigned shift_bits = 30 + (channel_id & 1) ;
            // printf("line : %d , shift_bits : %d\n" , __LINE__ , shift_bits) ;
            unsigned mask = (pid[0] == pid[1] ? 0 : (1 << shift_bits)) ; 
            if(threadIdx.y & 1 == turn) {
                
                #pragma unroll
                for(unsigned j = laneid; j < DEGREE; j += blockDim.x) {

                    const unsigned *ex_graph = graph[channel_id] ;
                    unsigned to_append_local = ex_graph[to_explore[channel_id] * DEGREE + j];
                    unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_append_local] ;
                    top_M_Cand[TOPM + j + channel_id * DEGREE] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                    top_M_Cand_dis[TOPM + j + channel_id * DEGREE] = (top_M_Cand[TOPM + j + channel_id * DEGREE] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                }
            } else {
                unsigned source_point_global = global_idx[seg_global_idx_start_idx[pid[turn]] + to_explore[turn]] ;
                #pragma unroll
                for(unsigned i = laneid ; i < 2 ; i += blockDim.x) {
                    unsigned to_append_local = global_graph[source_point_global * pnum * edges_num_in_global_graph + pid[channel_id] * edges_num_in_global_graph + i] ;
                    unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_append_local] ;
                    top_M_Cand[TOPM + i + channel_id * DEGREE] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                    top_M_Cand_dis[TOPM + i + channel_id * DEGREE] = (top_M_Cand[TOPM + i + channel_id * DEGREE] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                }

                #pragma unroll
                for(unsigned i = laneid ; i < DEGREE - 2 ; i += blockDim.x) {
                    top_M_Cand[2 + TOPM + i + channel_id * DEGREE] = 0xFFFFFFFF ;
                    top_M_Cand_dis[2 + TOPM + i + channel_id * DEGREE] = INF_DIS ; 
                }
            }
            if(tid == 0)
                turn ^= 1 ; 
        } 
        __syncthreads();
        // if(tid == 0 && epoch >= 135) {
        //     printf("epoch-%d : " , epoch) ;
        //     printf("[") ;
        //     for(unsigned i = 0 ; i < TOPM ; i ++) {
        //         if((top_M_Cand[i] >> 30) == 3)
        //             printf("u") ;
        //         printf("%d" , top_M_Cand[i] & 0x3FFFFFFF) ;
        //         if(i < TOPM - 1)
        //         printf(", ") ;
        //     }
        //     printf("];   ") ;
        //     printf("explore-%d :(" , to_explore[0]) ;
        //     for(unsigned i = 0 ; i < DEGREE ; i ++) {
        //         printf("%d" , top_M_Cand[TOPM + i] & 0x3FFFFFFF) ;
        //         if(i <  DEGREE - 1)
        //             printf(", ") ;
        //     }
        //     printf(");  explore-%d :(" , to_explore[1]) ;
        //     for(unsigned i = 0 ; i < DEGREE ; i ++) {
        //         printf("%d" , top_M_Cand[TOPM + DEGREE + i] & 0x3FFFFFFF) ;
        //         if(i < DEGREE - 1)
        //             printf(", ") ;
        //     }
        //     printf(")") ;
        //     printf("\n") ;
        // }
        // __syncthreads() ;

        // 此时无论是通道1还是通道2, 所有节点已在TOPM数组的后半段就位, 因此距离计算可以一视同仁
        #pragma unroll
        for(unsigned i = threadIdx.y; i < DEGREE2 ; i += blockDim.y){
            if(top_M_Cand[TOPM + i] == 0xFFFFFFFF)
                continue ;
            // threadIdx.y 0-5 , blockDim.y 即 6
            unsigned pure_id = top_M_Cand[TOPM + i] & 0x3FFFFFFF ;
            // if(pure_id > 1000000)
            //     printf("block : %d , error id - %d\n" , bid ,  pure_id) ;
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                half4 val2 = Load(&values[pure_id * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                // 将warp内所有子线程的计算结果归并到一起
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }
            if(laneid == 0){
                top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
            }
        }
        __syncthreads();
        // if(tid == 0)
        //     printf("line : %d , epoch : %d\n" , __LINE__ , epoch) ;
        // __syncthreads() ;
        // 排序
        if(tid < DEGREE2)
            warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE2);
        __syncthreads() ;
        // 将两个列表归并到一起
        merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE2 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
        __syncthreads() ;
        if(tid < DEGREE2)
            warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
        __syncthreads() ;
        if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
            merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE2 , recycle_list_size) ;
            __syncthreads() ;
        }
    }    
    // printf("tid : %d ;" , tid) ;
    // __syncthreads() ;
    
    // 搜索完成, 将recycle_list中的内容合并到top_M_Cand中
    for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
        if(top_M_Cand[i] == 0xFFFFFFFF)
            continue ;
        bool flag = true ;
        unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
        for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
            flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
        }
        if(!flag) {
            top_M_Cand[i] = 0xFFFFFFFF ;
            top_M_Cand_dis[i] = INF_DIS ;
        }
    }
    __syncthreads() ;
    bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM);
    __syncthreads() ;
    // 若recycle_list中确实有点, 则合并
    if(recycle_list_id[0] != 0xFFFFFFFF) {
        merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
        __syncthreads() ;
    }


    /**
        1.先明确对每个分区要做什么, 首先, 将候选点载入top_M_Cand 的末DEGREE个位置, 然后排序
        2.将循环池重置为初始值
        3.从初始点开始扩展, 每次将DEGREE个邻居存放入top_M_Cand的末位, 被淘汰的分区内的点会插入循环池中
        4.当前分区查找结束后,将top_M_Cand中不符合范围约束的点去掉,用循环池中的点替代,然后依据这些点从全局图中扩展到下一个分区的入口点

        step4 的具体实现时, 可以先把不符合范围约束的点的距离改为INF,然后对列表排序, 和recycle_list合并
    **/
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x3FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    for(unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        result_k_id[bid * K + i] = (top_M_Cand[i] & 0x3FFFFFFF) ;
        results_dis[bid * K + i] = top_M_Cand_dis[i] ;
    }
}


// 双通道搜索 + 小分区预过滤
__global__ void select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter(const unsigned* __restrict__ all_graphs, const half* __restrict__ values, const half* __restrict__ query_data, 
    unsigned* results_id, float* results_dis, unsigned* ent_pts , unsigned DEGREE , unsigned TOPM , unsigned DIM , const unsigned* graph_start_index ,const unsigned* graph_size ,
    const unsigned* __restrict__ pids , const unsigned* p_size ,const unsigned* p_start_index ,const float* l_bound ,const float* r_bound ,const float* __restrict__ attrs , unsigned attr_dim , const unsigned* __restrict__ global_graph , unsigned edges_num_in_global_graph,
    unsigned recycle_list_size ,const unsigned* __restrict__ global_idx ,const unsigned* __restrict__ local_idx ,const unsigned* seg_global_idx_start_idx , unsigned pnum ,unsigned K, unsigned* result_k_id ,const unsigned* task_id_mapper , const unsigned* little_seg_counts) {
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数
    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    unsigned* recycle_list_id = (unsigned*) shared_mem ;
    float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    unsigned* top_M_Cand = (unsigned*) (recycle_list_dis + recycle_list_size) ;
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE * 2) ;
    float* q_l_bound = (float*)(top_M_Cand_dis + TOPM + DEGREE * 2) ;
    float* q_r_bound = (float*)(q_l_bound + attr_dim) ;

    // 这里应该怎么处理8字节的对齐问题？
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) +
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[1024] ;
    __shared__ float work_mem_float[1024] ;
    // 可用投票一次插入32位
    __shared__ unsigned bit_map[2000] ;
    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = task_id_mapper[blockIdx.x], laneid = threadIdx.x , warpId = threadIdx.y ;
    // 若线程块发射顺序与blockIdx.x的编号无关, 则可以手动实现按到达顺序取任务的排队器
    unsigned psize_block = p_size[bid] - little_seg_counts[bid] , plist_index = p_start_index[bid] , total_psize_block = p_size[bid] , little_seg_num_block = little_seg_counts[bid] ;

    __shared__ unsigned cur_visit_p , main_channel , turn ;
    __shared__ unsigned to_explore[2];
    __shared__ unsigned prefilter_pointer ;

    __shared__ unsigned hash_table[HASHLEN];
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
        q_l_bound[i] = l_bound[bid * attr_dim + i] ;
        q_r_bound[i] = r_bound[bid * attr_dim + i] ;
    }
    // 没必要在每个分区开始时都把recycle_list复原, 每个分区得到的入口点肯定是越近越好
    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        recycle_list_id[i] = 0xFFFFFFFF ;
        recycle_list_dis[i] = INF_DIS ;
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
        tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
    }
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    __syncthreads() ;

    
    if(psize_block > 0) {
        psize_block = total_psize_block ;
        unsigned pid[2] = {pids[plist_index] , (psize_block >= 2 ? pids[plist_index + 1] : pids[plist_index])} ;
        // pid[0] = 9 ; 
        const unsigned* graph[2] = {all_graphs + graph_start_index[pid[0]] , all_graphs + graph_start_index[pid[1]]};
        const unsigned DEGREE2 = DEGREE * 2 ;



        
        // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
        for(unsigned i = tid , L2 = (TOPM + 1)/ 2 ; i < TOPM; i += blockDim.x * blockDim.y){
            unsigned shift_bits = 30 +  i / L2;
            unsigned mask = (pid[0] == pid[1] ? 0 : 1 << shift_bits) ; 
            // 用反向掩码, 将另一位置1
            top_M_Cand[i] = global_idx[seg_global_idx_start_idx[pid[i / L2]] + ent_pts[i]] | mask;
            // top_M_Cand[TOPM + DEGREE + i]

            // 哈希表存放的是局部id, 如何统一?  是否可以直接存放全局id? 
            hash_insert(hash_table , top_M_Cand[i] & 0x3FFFFFFF) ;
        }


        if(tid == 0) {
            // printf("psize block : %d\n" , psize_block) ;
            // printf("pid[0] %d , pid[1] %d\n" , pid[0] , pid[1]) ;
            // printf("[") ;
            // for(int i = 0; i < psize_block ; i ++) {
            //     printf("%d" , pids[plist_index + i]) ;
            //     if(i < psize_block - 1)
            //         printf(", ") ;
            // }
            // printf("]\n") ;

            cur_visit_p = (psize_block >= 2 ? 2 : 1) ;
            main_channel = 0 ; 
            turn = 0 ; 
            // psize_block = total_psize_block ;
            // printf("can click 2023\n") ;
        }
        __syncthreads();
        
        // 初始化入口节点
        #pragma unroll
        for(unsigned i = threadIdx.y; i < TOPM ; i += blockDim.y){
            if(top_M_Cand[i] == 0xFFFFFFFF)
                continue ;
            // threadIdx.y 0-5 , blockDim.y 即 6
            
            unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
            // if(pure_id > 1000000)
            //     printf("block : %d , error id - %d\n" , bid ,  pure_id) ;
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                half4 val2 = Load(&values[pure_id * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                // 将warp内所有子线程的计算结果归并到一起
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }
            if(laneid == 0){
                top_M_Cand_dis[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                atomicAdd(dis_cal_cnt + bid , 1) ;
            }
        }
        __syncthreads();
    
        bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis , top_M_Cand , TOPM) ;
        __syncthreads() ;
        
        

        // 无限次循环, 直到退出
        for(unsigned epoch = 0 ; epoch < MAX_ITER ; epoch ++) {

            // 每隔4轮将哈希表重置一次
            if((epoch + 1) % (HASH_RESET) == 0) {
                #pragma unroll
                for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                    hash_table[j] = 0xFFFFFFFF;
                }
                __syncthreads();
                #pragma unroll
                for(unsigned j = tid; j < TOPM + DEGREE2; j += blockDim.x * blockDim.y){
                    // 将此时的TOPM + K 个候选点插入哈希表中
                    if(top_M_Cand[j] != 0xFFFFFFFF) {
                        hash_insert(hash_table , top_M_Cand[j] & 0x3FFFFFFF) ; 
                    }
                }
            }
            __syncthreads();

            // 下面进行点扩展
            if(tid < 32){
                /*
                    扩展规则: 找到当前turn通道的最近点, 然后从另一个通道找到距离该点最近的点, 进行扩展
                */
                if(laneid < 2)
                    to_explore[laneid] = 0xFFFFFFFF ;
                __syncwarp() ;

                unsigned shift_bits = (pid[0] == pid[1] ? 31 - (main_channel & 1)  : 31 - (turn & 1));
                unsigned mask = (1 << shift_bits) ;

            
                for(unsigned j = laneid; j < TOPM; j+=32){
                    unsigned n_p = ((top_M_Cand[j] & mask) == 0 ? 1 : 0) ;
                
                
                    unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                    if(ballot_res > 0) {

                        if(laneid == (__ffs(ballot_res) - 1)) {
                            to_explore[turn & 1] = local_idx[top_M_Cand[j] & 0x3FFFFFFF] ;
                            top_M_Cand[j] |= mask ;
                        }
                        break ;
                    }
                }
                __syncwarp() ;
                
                if(to_explore[turn] != 0xFFFFFFFF) {
                    unsigned global_to_explore = global_idx[seg_global_idx_start_idx[pid[turn]] + to_explore[turn]] ;
                    unsigned offset = global_to_explore * pnum * edges_num_in_global_graph + pid[turn ^ 1] * edges_num_in_global_graph ;
                    for(unsigned j = laneid ; j < 2; j += 32) {
                        unsigned n_p = !hash_peek(hash_table , global_graph[offset + j]) ;
                        unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;

                        if(ballot_res > 0) {
                            if(laneid == (__ffs(ballot_res) - 1)) {
                                to_explore[turn ^ 1] = global_graph[offset + j] ;
                            }
                            break ; 
                        }
                    }
                }
            }
            __syncthreads();


            // 若扩展失败, 切换图栈
            if(to_explore[turn & 1] == 0xFFFFFFFF) {
                
                if(cur_visit_p == psize_block && pid[0] == pid[1]) {
                    // 两个图都没有扩展点, 且没有图可以切换, 退出
                    if(tid == 0) {
                        if(bid == 0)
                            printf("block %d: break in %d\n" ,bid ,  epoch) ; 
                        jump_num_cnt[bid] = epoch ; 
                    }
                    break ;
                } else if(cur_visit_p == psize_block) {
                        graph[turn] = graph[turn ^ 1] ;
                        pid[turn] = pid[turn ^ 1] ;
                        if(tid == 0) {
                            main_channel = turn ^ 1 ; 
                            // 重置 epoch
                            // epoch = 0 ; 
                        }
                    continue ;
                } else {
                    
                    unsigned next_pid = pids[plist_index + cur_visit_p] ;
                    unsigned shift_bits = 30 + (turn & 1) ;
                    unsigned mask = 1 << shift_bits ;
                    
                    // 重置哈希表
                    #pragma unroll
                    for(unsigned i = tid ; i < EXTEND_POINTS_NUM ; i += blockDim.x * blockDim.y) { 
                        work_mem_unsigned[i] = 0xFFFFFFFFu , work_mem_float[i] = INF_DIS ;
                        //
                        // if(i % 64 >= TOPM) {

                        // }
                
                        if(i % EXTEND_POINTS_STEP >= TOPM || top_M_Cand[i % EXTEND_POINTS_STEP] == 0xFFFFFFFF) {
                            // 填充一个随机数
                            // work_mem_unsigned[i] = 0xFFFFFFFF ;
                            // work_mem_float[i] = 0.0f ;
                            continue ;
                        }
                        // 用前两个填充, 有重复的部分皆替换为随机点
                        unsigned pure_id = top_M_Cand[i % EXTEND_POINTS_STEP] & 0x3FFFFFFF ;
                        unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / EXTEND_POINTS_STEP] ;
                        unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                        work_mem_unsigned[i] = (hash_insert(hash_table , next_global_id) != 0 ? next_global_id | mask : 0xFFFFFFFF) ;
                        work_mem_float[i] = (work_mem_unsigned[i] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    }
                    
                    __syncthreads() ;
                    // 距离计算
                    #pragma unroll
                    for(unsigned i = threadIdx.y ; i < EXTEND_POINTS_NUM ; i += blockDim.y) {
                        if(work_mem_unsigned[i] == 0xFFFFFFFF) {
                            continue ;  
                        }
                        
                        half2 val_res;
                        unsigned pure_id = work_mem_unsigned[i] & 0x3FFFFFFF ;
                        val_res.x = 0.0; val_res.y = 0.0;
                        for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                            half4 val2 = Load(&values[pure_id * DIM + k * 4]);
                            val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                            val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                        }
                        #pragma unroll
                        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                        }
        
                        if(laneid == 0){
                            work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                            atomicAdd(dis_cal_cnt + bid , 1) ;
                        }
                    }
                    __syncthreads() ;

                    // if(tid == 0 && bid == 2) {
                    //     int cnt0 = 0  ;
                    //     for(unsigned c = 0 ; c < EXTEND_POINTS_NUM ; c ++) {
                    //         if(work_mem_float[c] < top_M_Cand_dis[TOPM - 1])
                    //             cnt0 ++ ;
                    //     }
                    //     printf("bid : %d , partition : (%d / %d), next_pid : %d , 可以插入列表的元素个数: %d\n" , bid , cur_visit_p , psize_block , next_pid , cnt0) ;
                    // }
                    // __syncthreads() ;

                    bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , EXTEND_POINTS_NUM) ;
                    __syncthreads() ;
                    // 此处需调整, 这里topm中被淘汰的点没有进入top_M_Cand + TOPM 中
                    merge_top_with_recycle_list(top_M_Cand , work_mem_unsigned, top_M_Cand_dis , work_mem_float , tid , EXTEND_POINTS_NUM , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
                
                    __syncthreads() ;
                
                    bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , EXTEND_POINTS_NUM) ;
                    __syncthreads() ;
                    if(work_mem_unsigned[0] != 0xFFFFFFFF) {
                        merge_top(recycle_list_id , work_mem_unsigned , recycle_list_dis , work_mem_float , tid , EXTEND_POINTS_NUM , recycle_list_size) ;
                        __syncthreads() ;
                    }


                    graph[turn] = all_graphs + graph_start_index[next_pid] ;
                    pid[turn] = next_pid ; 
                    if(tid == 0) {
                        cur_visit_p ++ ; 
                        // 重置 epoch
                        // epoch = 0 ; 
                    }
                    __syncthreads() ;
                    // 处理另一个channel
                    // channel_id ^= 1 ; 
                    continue ;
                }
                
            }
            __syncthreads() ;

            // 对两个通道的扩展写进两个warp里, 防止warp分歧
            if(threadIdx.y < 2) {
                if(to_explore[threadIdx.y & 1] != 0xFFFFFFFF) {
                    unsigned channel_id = threadIdx.y ;
                    unsigned shift_bits = 30 + (channel_id & 1) ;
                    // printf("line : %d , shift_bits : %d\n" , __LINE__ , shift_bits) ;
                    unsigned mask = (pid[0] == pid[1] ? 0 : (1 << shift_bits)) ; 
                    #pragma unroll
                    for(unsigned j = laneid; j < DEGREE - (channel_id == turn ? 0 : 1); j += blockDim.x) {

                        const unsigned *ex_graph = graph[channel_id] ;
                        unsigned to_append_local = ex_graph[to_explore[channel_id] * DEGREE + j];
                        unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_append_local] ;
                        top_M_Cand[TOPM + j + channel_id * DEGREE] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                        top_M_Cand_dis[TOPM + j + channel_id * DEGREE] = (top_M_Cand[TOPM + j + channel_id * DEGREE] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    }
                    if(channel_id != turn && laneid == 31) {
                        unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_explore[channel_id]] ;
                        top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                        top_M_Cand_dis[TOPM + (channel_id + 1) * DEGREE - 1] = (top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    }
                } else {
                    unsigned channel_id = threadIdx.y ;
                    #pragma unroll
                    for(unsigned i = laneid ; i < DEGREE ; i += blockDim.x) {
                        top_M_Cand[TOPM + i + channel_id * DEGREE] = 0xFFFFFFFF ;
                        top_M_Cand_dis[TOPM + i + channel_id * DEGREE] = INF_DIS ;
                    }
                }
                if(tid == 0)
                    turn ^= 1 ; 
            } 
            __syncthreads();

            // 此时无论是通道1还是通道2, 所有节点已在TOPM数组的后半段就位, 因此距离计算可以一视同仁
            #pragma unroll
            for(unsigned i = threadIdx.y; i < DEGREE2 ; i += blockDim.y){
                if(top_M_Cand[TOPM + i] == 0xFFFFFFFF)
                    continue ;
                
                unsigned pure_id = top_M_Cand[TOPM + i] & 0x3FFFFFFF ;
            
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                    // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                    half4 val2 = Load(&values[pure_id * DIM + j * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    // 将warp内所有子线程的计算结果归并到一起
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }
                if(laneid == 0){
                    top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                    atomicAdd(dis_cal_cnt + bid , 1) ;
                }
            }
            __syncthreads();

            // 排序
            if(tid < DEGREE2)
                warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE2);
            __syncthreads() ;
            if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                // 将两个列表归并到一起
                merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE2 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
                __syncthreads() ;
                if(tid < DEGREE2)
                    warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
                __syncthreads() ;
                if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                    merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE2 , recycle_list_size) ;
                    __syncthreads() ;
                }
            }
        }    
        // 搜索完成, 将recycle_list中的内容合并到top_M_Cand中
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
            if(top_M_Cand[i] == 0xFFFFFFFF)
                continue ;
            bool flag = true ;
            unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
            for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
                flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
            }
            if(!flag) {
                top_M_Cand[i] = 0xFFFFFFFF ;
                top_M_Cand_dis[i] = INF_DIS ;
            }
        }
        __syncthreads() ;
        bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM);
        __syncthreads() ;
        // 若recycle_list中确实有点, 则合并
        if(recycle_list_id[0] != 0xFFFFFFFF) {
            merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
            __syncthreads() ;
        }
    }

    __shared__ unsigned prefix_sum[1024] ;
    __shared__ unsigned top_prefix[32] ;
    __shared__ unsigned indices_arr[1024] ;
    __shared__ unsigned task_num ; 

    
    // 处理剩余的小分区
    for(unsigned g = psize_block ; g < total_psize_block && little_seg_num_block == total_psize_block ; g ++) {
      
        // printf("bid : %d , l_bound : %f , r_bound : %d\n" , bid , q_l_bound[0] , q_r_bound[0]) ;
        
        unsigned gpid = pids[plist_index + g] ;
        unsigned global_idx_offset = seg_global_idx_start_idx[gpid] ;
        unsigned num_in_seg = graph_size[gpid] ;
        // unsigned num_in_seg = 62464 ;

        #pragma unroll
        for(unsigned i = tid ;i < 2000 ; i += blockDim.x * blockDim.y) 
            bit_map[i] = 0 ; 
        #pragma unroll
        for(unsigned i = tid ; i < 500 ; i += blockDim.x * blockDim.y) {
            work_mem_unsigned[i] = 0xFFFFFFFF ;
            work_mem_float[i] = INF_DIS ;
        }
        if(tid == 0)
            prefilter_pointer = 0 ; 
        __syncthreads() ;

        // 先把 total_psize_block个点分片, 分片大小可以考虑bit_map中1的个数? 
        // selectivity <= 0.1, 意味着, 每十个点里只有一个点满足条件, 找到32个点平均要遍历320个点
        //  有两种方案 :  1. 动态chunk, 尽可能的设置一组中包含32个点
        //               2. 静态chunk 设置 chunk 大小为10 , 但要维护的候选列表长度至少为320
        //  合并次数 : 1. 6250 / 32 次 ; 2. 625 / 32 次
        //  随着合并次数的增多, 方案2的待合并点数量、以及合并次数会显著减少，故采用方案2. 方案1的前置条件太复杂
        // 表示某个点是否满足属性约束条件
        for(unsigned i = tid ; i < num_in_seg ; i += blockDim.x * blockDim.y) {
            bool flag = true ;
            unsigned g_global_id = global_idx[global_idx_offset + i] ;

            for(unsigned j = 0 ; j < attr_dim ; j ++)
                flag = (flag && (attrs[g_global_id * attr_dim + j] >= q_l_bound[j]) && (attrs[g_global_id * attr_dim + j] <= q_r_bound[j])) ;
            unsigned ballot_res = __ballot_sync(__activemask() , flag) ;
            if(laneid == 0)
                bit_map[(i >> 5)] = ballot_res ; 
        }
        __syncthreads() ;

        unsigned chunk_num = (num_in_seg + 31) >> 5 ;
        unsigned chunk_batch_size = 32 ; 
        for(unsigned chunk_id = 0 ; chunk_id < chunk_num ; chunk_id += chunk_batch_size) {
            unsigned point_batch_size = (chunk_id + chunk_batch_size < chunk_num ? (chunk_batch_size << 5) : num_in_seg - (chunk_id << 5)) ;
            unsigned point_batch_start_id = (chunk_id << 5) ;


            // 扫描出该chunk所需要的前缀和数组
            // 先利用__popc() 得到长度为32的前缀和数组, 再利用0号warp, 把32个局部前缀和数组合并为一个整体
            for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                // 在bitmap中的单元格位置
                unsigned cell_id = chunk_id + (i >> 5) ;
                unsigned bit_pos = i % 32 + 1;
                unsigned mask = (bit_pos == 32 ? 0xFFFFFFFF : (1 << bit_pos) - 1) ;
                unsigned index = (bit_map[cell_id] & mask) ;
                prefix_sum[i] = __popc(index) ;
            }
            __syncthreads() ;
            if(tid < 32) {
                unsigned v = (laneid + chunk_id < chunk_num ? __popc(bit_map[laneid + chunk_id]) : 0) ;
                // 将长度大小为32的各个子prefix合并为最终的prefix_sum 
                #pragma unroll
                for(unsigned offset = 1 ; offset < 32 ; offset <<= 1) {
                    unsigned y = __shfl_up_sync(0xFFFFFFFF , v , offset) ;
                    if(laneid >= offset) v += y ;
                }
                top_prefix[laneid] = v ;
            }
            if(tid == 0) {
                task_num = top_prefix[31] ;
                // if(task_num > 0 && false) {
                //     printf("task num : %d\n" , task_num) ;

                //     printf("[") ;
                //     for(unsigned j = 0 ; j < 1024 ; j ++) {
                //         printf("%d" , prefix_sum[j]) ;
                //         if(j < 1023)
                //             printf(", ") ;
                //     }
                //     printf("]\n") ;

                //     printf("top_prefix :[") ;
                //     for(unsigned j = 0 ; j < 32 ; j ++) {
                //         printf("%d" , top_prefix[j]) ;
                //         if(j < 31)
                //             printf(", ") ;
                //     }
                //     printf("]\n") ;
                // }
            }
            __syncthreads() ;
            for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                unsigned cell_id = chunk_id + (i >> 5) ; 
                
                unsigned id_in_cell = i % 32 ;
                unsigned bit_mask = (1 << id_in_cell) ;
                if((bit_map[cell_id] & bit_mask) != 0) {
                    // 获取插入位置
                    // unsigned prefix_sum_fetch_pos = 
                    unsigned local_cell_id = (i >> 5) ;
                    unsigned put_pos = (local_cell_id == 0 ? 0 : top_prefix[local_cell_id - 1]) + (id_in_cell == 0 ? 0 : prefix_sum[i - 1]) ;
                    // if(i == 2) {
                    //     printf("i == 2 : %d , cell_id : %d , id_in_cell : %d\n" , put_pos , cell_id , id_in_cell) ;
                    // }
                    // if(put_pos >= 1024) {

                    //     // printf("bid : %d ,point_batch_size : %d\n" , bid , point_batch_size) ;
                    //     printf("bid : %d ,put_pos : %d , top_prefix[%d - 1] = %d , prefix_sum[%d - 1] = %d\n" , bid, put_pos , cell_id , top_prefix[(i >> 5) - 1] ,
                    //     i , prefix_sum[i - 1]) ;
                    // }
                    indices_arr[put_pos] = i ;
                }
            }
            __syncthreads() ;

            // if(tid == 0) {
            //     // if(task_num > 0 && false) {
            //         printf("task num : %d\n" , task_num) ;

            //         printf("[") ;
            //         for(unsigned j = 0 ; j < 1024 ; j ++) {
            //             printf("%d" , prefix_sum[j]) ;
            //             if(j < 1023)
            //                 printf(", ") ;
            //         }
            //         printf("]\n") ;

            //         printf("top_prefix :[") ;
            //         for(unsigned j = 0 ; j < 32 ; j ++) {
            //             printf("%d" , top_prefix[j]) ;
            //             if(j < 31)
            //                 printf(", ") ;
            //         }
            //         printf("]\n") ;
            //     // }
            //         printf("indices_arr: [") ;
            //         for(unsigned j = 0 ; j < 32 ; j ++) {
            //             printf("%d" , indices_arr[j]) ;
            //             if(j < 31)
            //                 printf(", ") ;
            //         }
            //         printf("]\n") ;
            // }
 
            // 计算距离, 然后用原子操作插入队列中
            for(unsigned i = threadIdx.y ; i < task_num ; i += blockDim.y) {
                unsigned cand_id = indices_arr[i] + point_batch_start_id ;
                // if(laneid == 0 && cand_id >= 62500) {
                //     printf("bid : %d , cand_id : %d , indices_arr[i] : %d\n" , bid , cand_id , indices_arr[i]) ;
                // }
                // __syncwarp() ;
                // 计算距离
                unsigned g_global_id = global_idx[global_idx_offset + cand_id] ;

                float dis = 0.0f ;

                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[g_global_id * DIM + k * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    dis = __half2float(val_res.x) + __half2float(val_res.y) ;
                    atomicAdd(dis_cal_cnt + bid , 1) ;
                }
                __syncwarp() ;
                // 存入work_mem
                if(laneid == 0 && dis < top_M_Cand_dis[K - 1]) {
                    int cur_pointer = atomicAdd(&prefilter_pointer , 1) ;
                    work_mem_unsigned[cur_pointer] = g_global_id ; 
                    work_mem_float[cur_pointer] = dis ; // 距离变量
                }
                __syncwarp() ;
            }

            __syncthreads() ;
            if(prefilter_pointer >= 32 || (chunk_id + chunk_batch_size >= chunk_num && prefilter_pointer > 0)) {
                // 先对前32个点排序, 然后再合并到topK上
                // if(tid < 32)
                //     warpSort(work_mem_float , work_mem_unsigned , tid , 32) ; 
                bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned, prefilter_pointer) ;
                __syncthreads() ;
                merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float , tid , prefilter_pointer , K) ;
                __syncthreads() ;
                unsigned limit = prefilter_pointer ;
                #pragma unroll
                for(unsigned i = tid ; i < limit ; i += blockDim.x * blockDim.y) {
                    work_mem_unsigned[i] = 0xFFFFFFFF ;
                    work_mem_float[i] = INF_DIS ;
                }
                if(tid == 0) {
                    prefilter_pointer = 0 ; 
                    atomicAdd(post_merge_cnt + bid , 1) ;
                }

                __syncthreads() ;
            }
        }
    }
    

    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x3FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    for(unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        result_k_id[bid * K + i] = (top_M_Cand[i] & 0x3FFFFFFF) ;
        results_dis[bid * K + i] = top_M_Cand_dis[i] ;
    }
}

/**
    需要补齐的数组: batch_size - 当前批次大小
                   batch_partition_ids - 陈列出当前批次的所有分区id
                   global_vec_stored_pos - 每个全局id对应当前处理批次的存储位置
**/

// batch : 当前批次的大小, batch_partition_ids : 当前batch中包含的所有partition编号
template<typename attrType = float>
__global__ void select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter_batch_process(const unsigned batch_size , const unsigned* batch_partition_ids , const unsigned* __restrict__ batch_all_graphs, const half* __restrict__ batch_values, const half* __restrict__ query_data, 
    unsigned* __restrict__ top_M_Cand_video_memory , float* __restrict__ top_M_Cand_dis_video_memory, unsigned* ent_pts , unsigned DEGREE , unsigned TOPM , unsigned DIM , const unsigned* batch_graph_start_index ,const unsigned* batch_graph_size ,
    const unsigned* __restrict__ batch_pids , const unsigned* batch_p_size ,const unsigned* batch_p_start_index ,const attrType* l_bound ,const attrType* r_bound ,const attrType* __restrict__ attrs , unsigned attr_dim , const unsigned* __restrict__ batch_global_graph , unsigned edges_num_in_global_graph,
    unsigned recycle_list_size ,const unsigned* __restrict__ global_idx ,const unsigned* __restrict__ local_idx ,const unsigned* seg_global_idx_start_idx , unsigned pnum ,unsigned K, unsigned* result_k_id  ,const unsigned* task_id_mapper , const unsigned* batch_little_seg_counts
    ,const unsigned* global_vec_stored_pos) {
    // *********   当前版本为结束时合并循环池, 后续可酌情实现不合并循环池的版本   ***********
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数

    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    unsigned* recycle_list_id = (unsigned*) shared_mem ;
    float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    unsigned* top_M_Cand = (unsigned*) (recycle_list_dis + recycle_list_size) ;
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE * 2) ;
    attrType* q_l_bound = (attrType*)(top_M_Cand_dis + TOPM + DEGREE * 2) ;
    attrType* q_r_bound = (attrType*)(q_l_bound + attr_dim) ;

    // 这里应该怎么处理8字节的对齐问题？
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) +
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(attrType)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[1024 + 32] ;
    __shared__ float work_mem_float[1024 + 32] ;
    // 可用投票一次插入32位
    __shared__ unsigned bit_map[2000] ;
    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = task_id_mapper[blockIdx.x], laneid = threadIdx.x , warpId = threadIdx.y ;
    // 若线程块发射顺序与blockIdx.x的编号无关, 则可以手动实现按到达顺序取任务的排队器
    /**
        在批处理环境下, 以下变量的语义更改为:
        psize_block - 当前批次中, 查询的正常分区数量, plist_index - 当前批次中, 当前查询的分区列表在总的分区列表中的起始位置
        total_psize_block - 当前批次中, 查询总的分区数量, little_seg_num_block - 当前批次中, 查询的小分区数量
    **/
    unsigned psize_block = batch_p_size[bid] , plist_index = batch_p_start_index[bid] , 
    total_psize_block = batch_p_size[bid] , little_seg_num_block = batch_little_seg_counts[bid] ;

    // if(tid == 0 && bid == 0) {
    //     printf("psize : %d\n" , psize_block) ;
    //     printf("plist : [") ;
    //     for(unsigned i = 0 ; i < total_psize_block ; i ++) 
    //         printf("%d ," , batch_pids[i + plist_index]) ;
    //     printf("]\n") ;
    //     printf("partition_id_list : [") ;
    //     for(unsigned i = 0 ; i < batch_size ; i ++)
    //         printf("%d ," , batch_partition_ids[i]) ;
    //     printf("]\n") ;
    //     printf("batch graph start index : [") ;
    //     for(unsigned i = 0 ; i < batch_size ; i ++)
    //         printf("%d ," , batch_graph_start_index[i]) ;
    //     printf("]\n") ; 
    //     printf("batch partition start index : [") ;
    //     for(unsigned i = 0 ; i < gridDim.x  ; i ++)
    //         printf("%d ," , batch_p_start_index[i]) ;
    //     printf("]\n") ;
    // }
    // __syncthreads() ;
    // 25 不通过 23 不通过 21 不通过 15 不通过 8 通过 11 通过 13 不通过 12 不通过
    // 1 2 5 6 8 通过
    // if(bid != 10)
    //     return ;
    if(total_psize_block == 0) {
        // if(tid == 0) {
        //     printf("当前处理批次为空 bid : %d\n" , bid) ;
        // }
        return ;
    } 
    // else {
    //     if(tid == 0) {
    //         printf("当前处理批次非空 bid : %d\n" , bid) ;
    //     }
    // }

    __shared__ unsigned cur_visit_p , main_channel , turn ;
    __shared__ unsigned to_explore[2];
    __shared__ unsigned prefilter_pointer ;

    __shared__ unsigned hash_table[HASHLEN];
    // 从全局内存中恢复topm
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
        // top_M_Cand[i] = top_M_Cand_video_memory[bid * TOPM + i] | 0xCF000000 ;
        // top_M_Cand_dis[i] = top_M_Cand_dis_video_memory[bid * TOPM + i] ;
    }
    for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
        q_l_bound[i] = l_bound[bid * attr_dim + i] ;
        q_r_bound[i] = r_bound[bid * attr_dim + i] ;
    }
    // 没必要在每个分区开始时都把recycle_list复原, 每个分区得到的入口点肯定是越近越好
    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        recycle_list_id[i] = 0xFFFFFFFF ;
        recycle_list_dis[i] = INF_DIS ;
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
        tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
    }
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    __syncthreads() ;

    // if(tid == 0) {

    //     unsigned gt_list[] = {747523 ,3806148 ,4341654 ,3992639 ,82713 ,5593568 ,5855579 ,4733809 ,3100036 ,5833647} ;
    //     for(unsigned i = 0 ; i < 10 ; i ++)
    //         work_mem_unsigned[i] = gt_list[i] ;

    //     printf("ent_pts[") ;
    //     for(unsigned i = 0 ; i < TOPM  ;i  ++) {
    //         printf("%d" , ent_pts[i]) ;
    //         if(i < TOPM - 1)
    //             printf(" ,") ;
    //     }
    //     printf("]\n") ;
    // }
    // __syncthreads() ;

    // #pragma unroll
    // for(unsigned i = threadIdx.y; i < TOPM ; i += blockDim.y) {
        
        
    //     unsigned pure_id = work_mem_unsigned[i] ;
    //     // unsigned stored_pos = batch_point_start_index[batch_pid[(top_M_Cand[i] >> 31)]] + local_idx[pure_id] ;
    //     unsigned stored_pos = global_vec_stored_pos[pure_id] ;
    //     half2 val_res ;
    //     val_res.x = 0.0 ; val_res.y = 0.0 ;
    //     for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x) {
    //         // warp中的每个子线程负责8个字节 即4个half类型分量的计算
    //         half4 val2 = Load(&batch_values[stored_pos * DIM + j * 4]);
    //         val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
    //         val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
    //     }
    //     #pragma unroll
    //     for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2) {   
    //         // 将warp内所有子线程的计算结果归并到一起
    //         val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
    //     }
    //     if(laneid == 0) {
    //         work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
    //         // atomicAdd(dis_cal_cnt + bid , 1) ;
    //     }
    // }
    // __syncthreads() ;
    // if(tid == 0) {
    //     printf("gt id and dis[") ;
    //     for(unsigned i = 0 ; i < 10 ; i ++) {

    //         printf("%d-%f" , work_mem_unsigned[i] , work_mem_float[i]) ;
    //         if(i < 9)
    //             printf(" ,") ;
    //     }
    //     printf("]\n") ;
    // }
    // __syncthreads() ;

    
    if(psize_block > 0 && little_seg_num_block == 0) {
        psize_block = total_psize_block ;
        /**
            batch_pids[i] 第i个分区, 用的是 0 - batchSize 之间的局部id
            batch_graph_start_index[i] 当前batch 第i个分区所代表的图
        **/
        // if(plist_index >= 12) {
        //     if(tid == 0)
        //         printf("error at %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;
        unsigned batch_pid[2] = {batch_pids[plist_index] , (psize_block >= 2 ? batch_pids[plist_index + 1] : batch_pids[plist_index])} ;
        unsigned pid[2] = {batch_partition_ids[batch_pid[0]] , batch_partition_ids[batch_pid[1]]} ;
        // pid[0] = 9 ; 
        const unsigned* graph[2] = {batch_all_graphs + batch_graph_start_index[batch_pid[0]] , batch_all_graphs + batch_graph_start_index[batch_pid[1]]};
        const unsigned DEGREE2 = DEGREE * 2 ;


        // 有两种可能, 没有局部结果的从入口点扩展, 有局部结果的先从局部结果扩展, 若局部结果中点个数不够从入口点补充

        // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
        for(unsigned i = tid , L2 = TOPM / 2 ; i < TOPM; i += blockDim.x * blockDim.y){
            unsigned shift_bits = 30 +  i / L2;
            unsigned mask = (batch_pid[0] == batch_pid[1] ? 0 : 1 << shift_bits) ; 
            // 用反向掩码, 将另一位置1
            top_M_Cand[i] = global_idx[seg_global_idx_start_idx[pid[i / L2]] + ent_pts[i]] | mask;
            // top_M_Cand[TOPM + DEGREE + i]

            // 哈希表存放的是局部id, 如何统一?  是否可以直接存放全局id? 
            hash_insert(hash_table , top_M_Cand[i] & 0x3FFFFFFF) ;
        }

        // 初始化入口节点
        #pragma unroll
        for(unsigned i = threadIdx.y; i < TOPM ; i += blockDim.y) {
            if(top_M_Cand[i] == 0xFFFFFFFF)
                continue ;
            // threadIdx.y 0-5 , blockDim.y 即 6
            
            unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
            // unsigned stored_pos = batch_point_start_index[batch_pid[(top_M_Cand[i] >> 31)]] + local_idx[pure_id] ;
            unsigned stored_pos = global_vec_stored_pos[pure_id] ;
            half2 val_res ;
            val_res.x = 0.0 ; val_res.y = 0.0 ;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x) {
                // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                half4 val2 = Load(&batch_values[stored_pos * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2) {   
                // 将warp内所有子线程的计算结果归并到一起
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }
            if(laneid == 0) {
                top_M_Cand_dis[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                atomicAdd(dis_cal_cnt + bid , 1) ;
            }
        }
        __syncthreads();
        bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis , top_M_Cand , TOPM) ;
        __syncthreads() ;
        // merge_top_with_recycle_list(top_M_Cand , work_mem_unsigned, top_M_Cand_dis , work_mem_float , tid , enter_pts_num , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
        // __syncthreads() ;
        // if(tid == 0) {
        //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;


        if(tid == 0) {
            // printf("psize block : %d\n" , psize_block) ;
            // printf("pid[0] %d , pid[1] %d\n" , pid[0] , pid[1]) ;
            // printf("[") ;
            // for(int i = 0; i < psize_block ; i ++) {
            //     printf("%d" , pids[plist_index + i]) ;
            //     if(i < psize_block - 1)
            //         printf(", ") ;
            // }
            // printf("]\n") ;

            cur_visit_p = (psize_block >= 2 ? 2 : 1) ;
            main_channel = 0 ; 
            turn = 0 ; 
            // psize_block = total_psize_block ;
            // printf("can click 2023\n") ;
            // printf("can click %d\n" , __LINE__) ;

            // printf("ent_pts-dis[") ;
            // for(unsigned i = 0 ; i < TOPM  ;i  ++) {
            //     printf("%d-%f" , top_M_Cand[i] , top_M_Cand_dis[i]) ;
            //     if(i < TOPM - 1)
            //         printf(" ,") ;
            // }
            // printf("]\n") ;
        }
        __syncthreads();
        
        
        
        

        // 无限次循环, 直到退出
        for(unsigned epoch = 0 ; epoch < MAX_ITER ; epoch ++) {

            // 每隔4轮将哈希表重置一次
            if((epoch + 1) % (HASH_RESET) == 0) {
                #pragma unroll
                for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                    hash_table[j] = 0xFFFFFFFF;
                }
                __syncthreads();
                #pragma unroll
                for(unsigned j = tid; j < TOPM + DEGREE2; j += blockDim.x * blockDim.y){
                    // 将此时的TOPM + K 个候选点插入哈希表中
                    if(top_M_Cand[j] != 0xFFFFFFFF) {
                        hash_insert(hash_table , top_M_Cand[j] & 0x3FFFFFFF) ; 
                    }
                }
            }
            __syncthreads();
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 下面进行点扩展
            if(tid < 32){
                /*
                    扩展规则: 找到当前turn通道的最近点, 然后从另一个通道找到距离该点最近的点, 进行扩展
                */
                if(laneid < 2)
                    to_explore[laneid] = 0xFFFFFFFF ;
                __syncwarp() ;

                unsigned shift_bits = (batch_pid[0] == batch_pid[1] ? 31 - (main_channel & 1)  : 31 - (turn & 1));
                unsigned mask = (1 << shift_bits) ;

            
                for(unsigned j = laneid; j < TOPM; j+=32){
                    unsigned n_p = ((top_M_Cand[j] & mask) == 0 ? 1 : 0) ;
                
                
                    unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                    if(ballot_res > 0) {

                        if(laneid == (__ffs(ballot_res) - 1)) {
                            to_explore[turn & 1] = local_idx[top_M_Cand[j] & 0x3FFFFFFF] ;
                            top_M_Cand[j] |= mask ;
                        }
                        break ;
                    }
                }
                __syncwarp() ;
                
                if(to_explore[turn] != 0xFFFFFFFF) {
                    unsigned global_to_explore = global_idx[seg_global_idx_start_idx[pid[turn]] + to_explore[turn]] ;
                    // unsigned stored_pos = point_batch_start_id[batch_pid[turn]] + to_explore[turn] ;
                    unsigned stored_pos = global_vec_stored_pos[global_to_explore] ;
                    unsigned offset = stored_pos * pnum * edges_num_in_global_graph + pid[turn ^ 1] * edges_num_in_global_graph ;
                    for(unsigned j = laneid ; j < 2; j += 32) {
                        unsigned n_p = !hash_peek(hash_table , batch_global_graph[offset + j]) ;
                        unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;

                        if(ballot_res > 0) {
                            if(laneid == (__ffs(ballot_res) - 1)) {
                                to_explore[turn ^ 1] = batch_global_graph[offset + j] ;
                            }
                            break ; 
                        }
                    }
                }
            }
            __syncthreads() ;
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;


            // 若扩展失败, 切换图栈
            if(to_explore[turn & 1] == 0xFFFFFFFF) {
                
                if(cur_visit_p == psize_block && batch_pid[0] == batch_pid[1]) {
                    // 两个图都没有扩展点, 且没有图可以切换, 退出
                    if(tid == 0) {
                        if(bid == 0)
                            printf("block %d: break in %d\n" ,bid ,  epoch) ; 
                        jump_num_cnt[bid] += epoch ; 
                    }
                    break ;
                } else if(cur_visit_p == psize_block) {
                        graph[turn] = graph[turn ^ 1] ;
                        pid[turn] = pid[turn ^ 1] ;
                        batch_pid[turn] = batch_pid[turn ^ 1] ;
                        if(tid == 0) {
                            main_channel = turn ^ 1 ; 
                        }
                    continue ;
                } else {
                    
                    unsigned next_batch_pid = batch_pids[plist_index + cur_visit_p] ;
                    unsigned next_pid = batch_partition_ids[next_batch_pid] ;
                    unsigned shift_bits = 30 + (turn & 1) ;
                    unsigned mask = 1 << shift_bits ;
                    
                    // 如何从全局id得到存储位置?

                    // 重置哈希表
                    #pragma unroll
                    for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
                
                        if(top_M_Cand[i % 64] == 0xFFFFFFFF) {
                            // 填充一个随机数
                            work_mem_unsigned[i] = 0xFFFFFFFF ;
                            work_mem_float[i] = 0.0f ;
                            continue ;
                        }
                        // 用前两个填充, 有重复的部分皆替换为随机点
                        unsigned pure_id = top_M_Cand[i % 64] & 0x3FFFFFFF ;
                        // unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / 64] ;
                        unsigned stored_pos = global_vec_stored_pos[pure_id] ;    
                        unsigned next_local_id = batch_global_graph[stored_pos * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / 64] ; 
                        unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                        work_mem_unsigned[i] = (hash_insert(hash_table , next_global_id) != 0 ? next_global_id | mask : 0xFFFFFFFF) ;
                        work_mem_float[i] = (work_mem_unsigned[i] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    }
                    
                    __syncthreads() ;
                    // 距离计算
                    #pragma unroll
                    for(unsigned i = threadIdx.y ; i < 128 ; i += blockDim.y) {
                        if(work_mem_unsigned[i] == 0xFFFFFFFF) {
                            continue ;  
                        }
                        
                        half2 val_res;
                        unsigned pure_id = work_mem_unsigned[i] & 0x3FFFFFFF ;
                        unsigned stored_pos = global_vec_stored_pos[pure_id] ;
                        val_res.x = 0.0; val_res.y = 0.0 ;
                        for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                            half4 val2 = Load(&batch_values[stored_pos * DIM + k * 4]);
                            val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                            val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                        }
                        #pragma unroll
                        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                        }
        
                        if(laneid == 0){
                            work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                            atomicAdd(dis_cal_cnt + bid , 1) ;
                        }
                    }
                    __syncthreads() ;
                    bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                    __syncthreads() ;
                    // 此处需调整, 这里topm中被淘汰的点没有进入top_M_Cand + TOPM 中
                    merge_top_with_recycle_list<attrType>(top_M_Cand , work_mem_unsigned, top_M_Cand_dis , work_mem_float , tid , 128 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
                
                    __syncthreads() ;
                
                    bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                    __syncthreads() ;
                    if(work_mem_unsigned[0] != 0xFFFFFFFF) {
                        merge_top(recycle_list_id , work_mem_unsigned , recycle_list_dis , work_mem_float , tid , 128 , recycle_list_size) ;
                        __syncthreads() ;
                    }
                    graph[turn] = batch_all_graphs + batch_graph_start_index[next_batch_pid] ;
                    pid[turn] = next_pid ;
                    batch_pid[turn] = next_batch_pid ; 
                    if(tid == 0) {
                        cur_visit_p ++ ; 
                    }
                    __syncthreads() ;
                    // 处理另一个channel
                    // channel_id ^= 1 ; 
                    // if(tid == 0) {
                    //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
                    // }
                    continue ;
                }
            }
            __syncthreads() ;
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 对两个通道的扩展写进两个warp里, 防止warp分歧
            if(threadIdx.y < 2) {
                if(to_explore[threadIdx.y & 1] != 0xFFFFFFFF) {
                    unsigned channel_id = threadIdx.y ;
                    unsigned shift_bits = 30 + (channel_id & 1) ;
                    // printf("line : %d , shift_bits : %d\n" , __LINE__ , shift_bits) ;
                    unsigned mask = (batch_pid[0] == batch_pid[1] ? 0 : (1 << shift_bits)) ; 
                    #pragma unroll
                    for(unsigned j = laneid; j < DEGREE - (channel_id == turn ? 0 : 1); j += blockDim.x) {

                        const unsigned *ex_graph = graph[channel_id] ;
                        // printf("to_explore : %d\n" , to_explore[channel_id]) ;
                        // if(tid == 0) {
                        //     printf("ex_graph : [") ;
                        //     for(unsigned i = 0 ; i < 16 ; i ++)
                        //         printf("ex_graph[%d] - %d," , i , ex_graph[i]) ;
                        //     printf("]\n") ;
                        // }
                        unsigned to_append_local = ex_graph[to_explore[channel_id] * DEGREE + j];
                        // printf("to_append_local : %d\n" , to_append_local) ;
                        unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_append_local] ;
                        // printf("to_append : %d\n" , to_append) ;
                        top_M_Cand[TOPM + j + channel_id * DEGREE] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                        top_M_Cand_dis[TOPM + j + channel_id * DEGREE] = (top_M_Cand[TOPM + j + channel_id * DEGREE] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    }
                    if(channel_id != turn && laneid == 31) {
                        unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_explore[channel_id]] ;
                        top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                        top_M_Cand_dis[TOPM + (channel_id + 1) * DEGREE - 1] = (top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                        // if(hash_insert(hash_table , to_append) != 0) {
                        //     top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] = (to_append | mask) ;
                        //     top_M_Cand_dis[TOPM + (channel_id + 1) * DEGREE - 1] = 0.0f ;
                        // }
                    }
                } else {
                    unsigned channel_id = threadIdx.y ;
                    #pragma unroll
                    for(unsigned i = laneid ; i < DEGREE ; i += blockDim.x) {
                        top_M_Cand[TOPM + i + channel_id * DEGREE] = 0xFFFFFFFF ;
                        top_M_Cand_dis[TOPM + i + channel_id * DEGREE] = INF_DIS ;
                    }
                }
                if(tid == 0)
                    turn ^= 1 ; 
            } 
            __syncthreads();
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 此时无论是通道1还是通道2, 所有节点已在TOPM数组的后半段就位, 因此距离计算可以一视同仁
            #pragma unroll
            for(unsigned i = threadIdx.y; i < DEGREE2 ; i += blockDim.y) {
                if(top_M_Cand[TOPM + i] == 0xFFFFFFFF)
                    continue ;
                
                unsigned pure_id = top_M_Cand[TOPM + i] & 0x3FFFFFFF ;

                // if(pure_id >= 1000000 && laneid == 0 ) {
                //     printf("error at %d : pure_id = %d , i = %d\n" , __LINE__ , pure_id , i) ;
                    
                // }
                unsigned stored_pos = global_vec_stored_pos[pure_id] ;
                // if(stored_pos >= 62500 * 4)
                //     printf("error at %d : stored_pos = %d\n" , __LINE__ , stored_pos) ;
            
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                    // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                    half4 val2 = Load(&batch_values[stored_pos * DIM + j * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    // 将warp内所有子线程的计算结果归并到一起
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }
                if(laneid == 0){
                    top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                    atomicAdd(dis_cal_cnt + bid , 1) ;
                }
            }
            __syncthreads();
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 排序
            // if(tid < DEGREE2)
            //     warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE2);
            bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], DEGREE2) ;
            __syncthreads() ;
            if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                // 将两个列表归并到一起
                merge_top_with_recycle_list<attrType>(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE2 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
                __syncthreads() ;
                // if(tid < DEGREE2)
                //     warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
                bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], DEGREE2) ;
                __syncthreads() ;
                if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                    merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE2 , recycle_list_size) ;
                    __syncthreads() ;
                }
            }
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;
        }  
        
        // if(tid == 0) {
        //     printf("过滤前的topm[") ;
        //     for(unsigned i = 0 ; i < 10 ; i ++) {
        //         printf("%d-%f" , top_M_Cand[i] & 0x3FFFFFFF , top_M_Cand_dis[i]) ;
        //         if(i < 9)
        //             printf(" ,") ;
        //     }
        //     printf("]\n") ;
        // }
        __syncthreads() ;

        // 搜索完成, 将recycle_list中的内容合并到top_M_Cand中
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
            if(top_M_Cand[i] == 0xFFFFFFFF)
                continue ;
            bool flag = true ;
            unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
            for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
                flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
            }
            if(!flag) {
                top_M_Cand[i] = 0xFFFFFFFF ;
                top_M_Cand_dis[i] = INF_DIS ;
            }
        }
        __syncthreads() ;
        bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM) ;
        __syncthreads() ;
        // 若recycle_list中确实有点, 则合并
        if(recycle_list_id[0] != 0xFFFFFFFF) {
            merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
            __syncthreads() ;
        }
        // if(tid == 0) {
        //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;
    }

    __shared__ unsigned prefix_sum[1024] ;
    __shared__ unsigned top_prefix[32] ;
    __shared__ unsigned indices_arr[1024] ;
    __shared__ unsigned task_num ; 
    // 处理剩余的小分区
    for(unsigned g = 0 ; g < total_psize_block && little_seg_num_block ; g ++) {
        unsigned batch_gpid = batch_pids[plist_index + g] ;
        unsigned gpid = batch_partition_ids[batch_gpid] ;
        unsigned global_idx_offset = seg_global_idx_start_idx[gpid] ;
        unsigned num_in_seg = batch_graph_size[batch_gpid] ;
        // unsigned num_in_seg = 62464 ;

        #pragma unroll
        for(unsigned i = tid ;i < 2000 ; i += blockDim.x * blockDim.y) 
            bit_map[i] = 0 ; 
        #pragma unroll
        for(unsigned i = tid ; i < 500 ; i += blockDim.x * blockDim.y) {
            work_mem_unsigned[i] = 0xFFFFFFFF ;
            work_mem_float[i] = INF_DIS ;
        }
        if(tid == 0)
            prefilter_pointer = 0 ; 
        __syncthreads() ;
        // if(tid == 0) {
        //     printf("bid : %d , can click %d\n" ,bid ,  __LINE__) ;
        // }
        // __syncthreads() ;

        // 先把 total_psize_block个点分片, 分片大小可以考虑bit_map中1的个数? 
        // selectivity <= 0.1, 意味着, 每十个点里只有一个点满足条件, 找到32个点平均要遍历320个点
        //  有两种方案 :  1. 动态chunk, 尽可能的设置一组中包含32个点
        //               2. 静态chunk 设置 chunk 大小为10 , 但要维护的候选列表长度至少为320
        //  合并次数 : 1. 6250 / 32 次 ; 2. 625 / 32 次
        //  随着合并次数的增多, 方案2的待合并点数量、以及合并次数会显著减少，故采用方案2. 方案1的前置条件太复杂
        // 表示某个点是否满足属性约束条件
        // for(unsigned i = tid ; i < num_in_seg ; i += blockDim.x * blockDim.y) {
        //     bool flag = true ;
        //     unsigned g_global_id = global_idx[global_idx_offset + i] ;
        //     if(g_global_id >= 1e8) {
        //         printf("bid : %d , illegal id : %d\n" , bid , g_global_id) ;
        //     }
        //     for(unsigned j = 0 ; j < attr_dim ; j ++)
        //         flag = (flag && (attrs[g_global_id * attr_dim + j] >= q_l_bound[j]) && (attrs[g_global_id * attr_dim + j] <= q_r_bound[j])) ;
        //     unsigned ballot_res = __ballot_sync(__activemask() , flag) ;
        //     if(laneid == 0)
        //         bit_map[(i >> 5)] = ballot_res ; 
        // }
        // __syncthreads() ;
        // if(tid == 0) {
        //     printf("bid : %d , can click %d\n" , __LINE__) ;
        // }
        // __syncthreads() ;

        unsigned chunk_num = (num_in_seg + 31) >> 5 ;
        unsigned chunk_batch_size = 32 ; 
        for(unsigned chunk_id = 0 ; chunk_id < chunk_num ; chunk_id += chunk_batch_size) {
            unsigned point_batch_size = (chunk_id + chunk_batch_size < chunk_num ? (chunk_batch_size << 5) : num_in_seg - (chunk_id << 5)) ;
            unsigned point_batch_start_id = (chunk_id << 5) ;

            for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                bool flag = true ;
                unsigned g_global_id = global_idx[global_idx_offset + point_batch_start_id + i] ;
                for(unsigned j = 0 ; j < attr_dim ; j ++)
                    flag = (flag && (attrs[g_global_id * attr_dim + j] >= q_l_bound[j]) && (attrs[g_global_id * attr_dim + j] <= q_r_bound[j])) ;
                unsigned ballot_res = __ballot_sync(__activemask() , flag) ;
                if(laneid == 0)
                    bit_map[(i >> 5)] = ballot_res ; 
            }
            __syncthreads() ;


            // 扫描出该chunk所需要的前缀和数组
            // 先利用__popc() 得到长度为32的前缀和数组, 再利用0号warp, 把32个局部前缀和数组合并为一个整体
            for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                // 在bitmap中的单元格位置
                // unsigned cell_id = chunk_id + (i >> 5) ;
                unsigned cell_id = (i >> 5) ;
                unsigned bit_pos = i % 32 + 1 ;
                unsigned mask = (bit_pos == 32 ? 0xFFFFFFFF : (1 << bit_pos) - 1) ;
                unsigned index = (bit_map[cell_id] & mask) ;
                prefix_sum[i] = __popc(index) ;
            }
            __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" ,bid ,  __LINE__) ;
            // }
            // __syncthreads() ;
            if(tid < 32) {
                unsigned v = (laneid + chunk_id < chunk_num ? __popc(bit_map[laneid]) : 0) ;
                // 将长度大小为32的各个子prefix合并为最终的prefix_sum 
                #pragma unroll
                for(unsigned offset = 1 ; offset < 32 ; offset <<= 1) {
                    unsigned y = __shfl_up_sync(0xFFFFFFFF , v , offset) ;
                    if(laneid >= offset) v += y ;
                }
                top_prefix[laneid] = v ;
            }
            // __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" ,bid ,  __LINE__) ;
            // }
            // __syncthreads() ;
            if(tid == 0) {
                task_num = top_prefix[31] ;
                // if(task_num > 0 && false) {
                //     printf("task num : %d\n" , task_num) ;

                //     printf("[") ;
                //     for(unsigned j = 0 ; j < 1024 ; j ++) {
                //         printf("%d" , prefix_sum[j]) ;
                //         if(j < 1023)
                //             printf(", ") ;
                //     }
                //     printf("]\n") ;

                //     printf("top_prefix :[") ;
                //     for(unsigned j = 0 ; j < 32 ; j ++) {
                //         printf("%d" , top_prefix[j]) ;
                //         if(j < 31)
                //             printf(", ") ;
                //     }
                //     printf("]\n") ;
                // }
                // printf("bid : %d , can click : %d\n" , bid , __LINE__) ;
            }
            __syncthreads() ;
            for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                // unsigned cell_id = chunk_id + (i >> 5) ; 
                unsigned cell_id = (i >> 5) ;
                
                unsigned id_in_cell = i % 32 ;
                unsigned bit_mask = (1 << id_in_cell) ;
                if((bit_map[cell_id] & bit_mask) != 0) {
                    // 获取插入位置
                    // unsigned prefix_sum_fetch_pos = 
                    unsigned local_cell_id = (i >> 5) ;
                    unsigned put_pos = (local_cell_id == 0 ? 0 : top_prefix[local_cell_id - 1]) + (id_in_cell == 0 ? 0 : prefix_sum[i - 1]) ;
                    // if(i == 2) {
                    //     printf("i == 2 : %d , cell_id : %d , id_in_cell : %d\n" , put_pos , cell_id , id_in_cell) ;
                    // }
                    // if(put_pos >= 1024) {

                    //     // printf("bid : %d ,point_batch_size : %d\n" , bid , point_batch_size) ;
                    //     printf("bid : %d ,put_pos : %d , top_prefix[%d - 1] = %d , prefix_sum[%d - 1] = %d\n" , bid, put_pos , cell_id , top_prefix[(i >> 5) - 1] ,
                    //     i , prefix_sum[i - 1]) ;
                    // }
                    indices_arr[put_pos] = i ;
                }
            }
            __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click : %d , chunk_id : %d , chunk_num : %d  \n" , bid , __LINE__ , chunk_id , chunk_num) ;
            // }
 
            // 计算距离, 然后用原子操作插入队列中
            for(unsigned i = threadIdx.y ; i < task_num ; i += blockDim.y) {
                unsigned cand_id = indices_arr[i] + point_batch_start_id ;
                // if(laneid == 0 && chunk_id == 125920) {
                //     printf("bid : %d , cand_id : %d , indices_arr[i] : %d\n" , bid , cand_id , indices_arr[i]) ;
                // }
                // __syncwarp() ;
                // 计算距离
                unsigned g_global_id = global_idx[global_idx_offset + cand_id] ;
                unsigned stored_pos = global_vec_stored_pos[g_global_id] ;
                // if(chunk_id == 125920 ) {
                //     if(laneid == 0)
                //         printf("bid : %d , stored_pos : %d , g_global_id : %d ,DIM : %d , stored_pos * DIM : %d\n" , bid , stored_pos , g_global_id , DIM , stored_pos * DIM ) ;
                // } 
                float dis = 0.0f ;

                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&batch_values[stored_pos * DIM + k * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    dis = __half2float(val_res.x) + __half2float(val_res.y) ;
                    atomicAdd(dis_cal_cnt + bid , 1) ;
                    // printf("bid : %d , can click : %d dis cal finish!\n" , bid , __LINE__) ;
                }
                __syncwarp() ;
                // 存入work_mem
                if(laneid == 0 && dis < top_M_Cand_dis[K - 1]) {
                    int cur_pointer = atomicAdd(&prefilter_pointer , 1) ;
                    // if(cur_pointer >= 1024)
                    //     printf("bid : %d , error cur_pointer : %d\n" , bid , cur_pointer) ;
                    work_mem_unsigned[cur_pointer] = g_global_id ; 
                    work_mem_float[cur_pointer] = dis ; // 距离变量
                }
                __syncwarp() ;
            }

            __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" , bid , __LINE__) ;
            // }
            // __syncthreads() ;
            if(prefilter_pointer >= 32 || (chunk_id + chunk_batch_size >= chunk_num && prefilter_pointer > 0)) {
                // 先对前32个点排序, 然后再合并到topK上
                // if(tid < 32)
                //     warpSort(work_mem_float , work_mem_unsigned , tid , 32) ; 
                bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned, prefilter_pointer) ;
                __syncthreads() ;
                merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float , tid , prefilter_pointer , K) ;
                __syncthreads() ;
                unsigned limit = prefilter_pointer ;
                #pragma unroll
                for(unsigned i = tid ; i < limit ; i += blockDim.x * blockDim.y) {
                    work_mem_unsigned[i] = 0xFFFFFFFF ;
                    work_mem_float[i] = INF_DIS ;
                }
                if(tid == 0) {
                    prefilter_pointer = 0 ; 
                    atomicAdd(post_merge_cnt + bid , 1) ;
                }

                __syncthreads() ;
            }
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" ,bid , __LINE__) ;
            // }
            // __syncthreads() ;
        }
    }
    // if(tid == 0) {
    //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
    // }
    // __syncthreads() ;

    #pragma unroll
    for (unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        work_mem_unsigned[i] = top_M_Cand_video_memory[bid * TOPM + i] ;
        work_mem_float[i] = top_M_Cand_dis_video_memory[bid * TOPM + i] ;
    }
    // if(tid == 0) {
    //     printf("this epoch : id[") ;
    //     for(unsigned i = 0 ; i < TOPM ;i  ++) {
    //         printf("%d-%d ," , top_M_Cand[i] & 0x3FFFFFFF , (top_M_Cand[i] >> 30)) ;
    //     }
    //     printf("]\nthis epoch : dis[") ;
    //     for(unsigned i = 0 ; i < TOPM ;i  ++) {
    //         printf("%f ," , top_M_Cand_dis[i]) ;
    //     }
    //     printf("]\n") ;
    //     // --------------------------------
    //     printf("this epoch work mem : id[") ;
    //     for(unsigned i = 0 ; i < K ;i  ++) {
    //         printf("%d ," , work_mem_unsigned[i] & 0x3FFFFFFF) ;
    //     }
    //     printf("]\nthis epoch work mem : dis[") ;
    //     for(unsigned i = 0 ; i < K ;i  ++) {
    //         printf("%f ," , work_mem_float[i]) ;
    //     }
    //     printf("]\n") ;
    // }
    __syncthreads() ;
    merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float , tid , K , TOPM) ;
    __syncthreads() ;

    // 每次执行完毕, 和现有结果做合并(和前k个结果合并即可)
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        top_M_Cand_video_memory[bid * TOPM + i] = (top_M_Cand[i] & 0x3FFFFFFF);
        top_M_Cand_dis_video_memory[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    for(unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        result_k_id[bid * K + i] = (top_M_Cand[i] & 0x3FFFFFFF) ;
        // result_k_dis[bid * K + i] = top_M_Cand_dis[i] ;
    }


}


template<typename attrType = float>
__global__ void select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter_batch_process_compressed_graph_T(const unsigned batch_size , const unsigned* batch_partition_ids , const unsigned* __restrict__ batch_all_graphs, const half* __restrict__ batch_values, const half* __restrict__ query_data, 
    unsigned* __restrict__ top_M_Cand_video_memory , float* __restrict__ top_M_Cand_dis_video_memory, unsigned* ent_pts , unsigned DEGREE , unsigned TOPM , unsigned DIM , const unsigned* batch_graph_start_index ,const unsigned* batch_graph_size ,
    const unsigned* __restrict__ batch_pids , const unsigned* batch_p_size ,const unsigned* batch_p_start_index ,const attrType* l_bound ,const attrType* r_bound ,const attrType* __restrict__ attrs , unsigned attr_dim , const unsigned* __restrict__ batch_global_graph , unsigned edges_num_in_global_graph,
    unsigned recycle_list_size ,const unsigned* __restrict__ global_idx ,const unsigned* __restrict__ local_idx ,const unsigned* seg_global_idx_start_idx , unsigned pnum ,unsigned K, unsigned* result_k_id  ,const unsigned* task_id_mapper , const unsigned* batch_little_seg_counts
    ,const unsigned* global_vec_stored_pos , const unsigned batch_all_point_num) {
    // *********   当前版本为结束时合并循环池, 后续可酌情实现不合并循环池的版本   ***********
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数

    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    unsigned* recycle_list_id = (unsigned*) shared_mem ;
    float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    unsigned* top_M_Cand = (unsigned*) (recycle_list_dis + recycle_list_size) ;
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE * 2) ;
    attrType* q_l_bound = (attrType*)(top_M_Cand_dis + TOPM + DEGREE * 2) ;
    attrType* q_r_bound = (attrType*)(q_l_bound + attr_dim) ;

    // 这里应该怎么处理8字节的对齐问题？
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) +
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(attrType)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[1024 + 32] ;
    __shared__ float work_mem_float[1024 + 32] ;
    // 可用投票一次插入32位
    __shared__ unsigned bit_map[32] ;
    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = task_id_mapper[blockIdx.x], laneid = threadIdx.x , warpId = threadIdx.y ;
    // 若线程块发射顺序与blockIdx.x的编号无关, 则可以手动实现按到达顺序取任务的排队器
    /**
        在批处理环境下, 以下变量的语义更改为:
        psize_block - 当前批次中, 查询的正常分区数量, plist_index - 当前批次中, 当前查询的分区列表在总的分区列表中的起始位置
        total_psize_block - 当前批次中, 查询总的分区数量, little_seg_num_block - 当前批次中, 查询的小分区数量
    **/
    unsigned psize_block = batch_p_size[bid] , plist_index = batch_p_start_index[bid] , 
    total_psize_block = batch_p_size[bid] , little_seg_num_block = batch_little_seg_counts[bid] ;

    // if(tid == 0 && bid == 0) {
    //     printf("psize : %d\n" , psize_block) ;
    //     printf("plist : [") ;
    //     for(unsigned i = 0 ; i < total_psize_block ; i ++) 
    //         printf("%d ," , batch_pids[i + plist_index]) ;
    //     printf("]\n") ;
    //     printf("partition_id_list : [") ;
    //     for(unsigned i = 0 ; i < batch_size ; i ++)
    //         printf("%d ," , batch_partition_ids[i]) ;
    //     printf("]\n") ;
    //     printf("batch graph start index : [") ;
    //     for(unsigned i = 0 ; i < batch_size ; i ++)
    //         printf("%d ," , batch_graph_start_index[i]) ;
    //     printf("]\n") ; 
    //     printf("batch partition start index : [") ;
    //     for(unsigned i = 0 ; i < gridDim.x  ; i ++)
    //         printf("%d ," , batch_p_start_index[i]) ;
    //     printf("]\n") ;
    // }
    // __syncthreads() ;
    // 25 不通过 23 不通过 21 不通过 15 不通过 8 通过 11 通过 13 不通过 12 不通过
    // 1 2 5 6 8 通过
    // if(bid != 5)
    //     return ;
    if(total_psize_block == 0) {
        // if(tid == 0) {
        //     printf("当前处理批次为空 bid : %d\n" , bid) ;
        // }
        return ;
    } 
    // else {
    //     if(tid == 0) {
    //         printf("当前处理批次非空 bid : %d\n" , bid) ;
    //     }
    // }

    __shared__ unsigned cur_visit_p , main_channel , turn ;
    __shared__ unsigned to_explore[2];
    __shared__ unsigned prefilter_pointer ;

    __shared__ unsigned hash_table[HASHLEN];
    // 从全局内存中恢复topm
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
        // top_M_Cand[i] = top_M_Cand_video_memory[bid * TOPM + i] | 0xCF000000 ;
        // top_M_Cand_dis[i] = top_M_Cand_dis_video_memory[bid * TOPM + i] ;
    }
    for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
        q_l_bound[i] = l_bound[bid * attr_dim + i] ;
        q_r_bound[i] = r_bound[bid * attr_dim + i] ;
    }
    // 没必要在每个分区开始时都把recycle_list复原, 每个分区得到的入口点肯定是越近越好
    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        recycle_list_id[i] = 0xFFFFFFFF ;
        recycle_list_dis[i] = INF_DIS ;
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
        tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
    }
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    __syncthreads() ;

    // if(tid == 0) {

    //     unsigned gt_list[] = {747523 ,3806148 ,4341654 ,3992639 ,82713 ,5593568 ,5855579 ,4733809 ,3100036 ,5833647} ;
    //     for(unsigned i = 0 ; i < 10 ; i ++)
    //         work_mem_unsigned[i] = gt_list[i] ;

    //     printf("ent_pts[") ;
    //     for(unsigned i = 0 ; i < TOPM  ;i  ++) {
    //         printf("%d" , ent_pts[i]) ;
    //         if(i < TOPM - 1)
    //             printf(" ,") ;
    //     }
    //     printf("]\n") ;
    // }
    // __syncthreads() ;

    // #pragma unroll
    // for(unsigned i = threadIdx.y; i < TOPM ; i += blockDim.y) {
        
        
    //     unsigned pure_id = work_mem_unsigned[i] ;
    //     // unsigned stored_pos = batch_point_start_index[batch_pid[(top_M_Cand[i] >> 31)]] + local_idx[pure_id] ;
    //     unsigned stored_pos = global_vec_stored_pos[pure_id] ;
    //     half2 val_res ;
    //     val_res.x = 0.0 ; val_res.y = 0.0 ;
    //     for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x) {
    //         // warp中的每个子线程负责8个字节 即4个half类型分量的计算
    //         half4 val2 = Load(&batch_values[stored_pos * DIM + j * 4]);
    //         val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
    //         val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
    //     }
    //     #pragma unroll
    //     for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2) {   
    //         // 将warp内所有子线程的计算结果归并到一起
    //         val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
    //     }
    //     if(laneid == 0) {
    //         work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
    //         // atomicAdd(dis_cal_cnt + bid , 1) ;
    //     }
    // }
    // __syncthreads() ;
    // if(tid == 0) {
    //     printf("gt id and dis[") ;
    //     for(unsigned i = 0 ; i < 10 ; i ++) {

    //         printf("%d-%f" , work_mem_unsigned[i] , work_mem_float[i]) ;
    //         if(i < 9)
    //             printf(" ,") ;
    //     }
    //     printf("]\n") ;
    // }
    // __syncthreads() ;

    
    if(psize_block > 0 && little_seg_num_block == 0) {
        psize_block = total_psize_block ;
        /**
            batch_pids[i] 第i个分区, 用的是 0 - batchSize 之间的局部id
            batch_graph_start_index[i] 当前batch 第i个分区所代表的图
        **/
        // if(plist_index >= 12) {
        //     if(tid == 0)
        //         printf("error at %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;
        unsigned batch_pid[2] = {batch_pids[plist_index] , (psize_block >= 2 ? batch_pids[plist_index + 1] : batch_pids[plist_index])} ;
        unsigned pid[2] = {batch_partition_ids[batch_pid[0]] , batch_partition_ids[batch_pid[1]]} ;
        // pid[0] = 9 ; 
        const unsigned* graph[2] = {batch_all_graphs + batch_graph_start_index[batch_pid[0]] , batch_all_graphs + batch_graph_start_index[batch_pid[1]]};
        const unsigned DEGREE2 = DEGREE * 2 ;
        // if(tid == 0) {
        //     printf("bid : %d , batch_pid0 : %d , batch_pid1 : %d , pid0 : %d , pid1 : %d\n" , bid , batch_pid[0] , batch_pid[1] , pid[0] , pid[1]) ;
        // }

        // 有两种可能, 没有局部结果的从入口点扩展, 有局部结果的先从局部结果扩展, 若局部结果中点个数不够从入口点补充

        // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
        for(unsigned i = tid , L2 = DEGREE ; i < DEGREE2; i += blockDim.x * blockDim.y){
            unsigned shift_bits = 30 +  i / L2;
            unsigned mask = (batch_pid[0] == batch_pid[1] ? 0 : 1 << shift_bits) ; 
            // 用反向掩码, 将另一位置1
            top_M_Cand[i] = global_idx[seg_global_idx_start_idx[pid[i / L2]] + ent_pts[i]] | mask;
            // top_M_Cand[TOPM + DEGREE + i]

            // 哈希表存放的是局部id, 如何统一?  是否可以直接存放全局id? 
            hash_insert(hash_table , top_M_Cand[i] & 0x3FFFFFFF) ;
        }

        // 初始化入口节点
        #pragma unroll
        for(unsigned i = threadIdx.y; i < TOPM ; i += blockDim.y) {
            if(top_M_Cand[i] == 0xFFFFFFFF)
                continue ;
            // threadIdx.y 0-5 , blockDim.y 即 6
            
            unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
            // unsigned stored_pos = batch_point_start_index[batch_pid[(top_M_Cand[i] >> 31)]] + local_idx[pure_id] ;
            unsigned stored_pos = global_vec_stored_pos[pure_id] ;
            half2 val_res ;
            val_res.x = 0.0 ; val_res.y = 0.0 ;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x) {
                // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                half4 val2 = Load(&batch_values[stored_pos * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2) {   
                // 将warp内所有子线程的计算结果归并到一起
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }
            if(laneid == 0) {
                top_M_Cand_dis[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                atomicAdd(dis_cal_cnt + bid , 1) ;
            }
        }
        __syncthreads();
        bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis , top_M_Cand , DEGREE2) ;
        __syncthreads() ;
        // merge_top_with_recycle_list(top_M_Cand , work_mem_unsigned, top_M_Cand_dis , work_mem_float , tid , enter_pts_num , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
        // __syncthreads() ;
        // if(tid == 0) {
        //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;


        if(tid == 0) {
            // printf("psize block : %d\n" , psize_block) ;
            // printf("pid[0] %d , pid[1] %d\n" , pid[0] , pid[1]) ;
            // printf("[") ;
            // for(int i = 0; i < psize_block ; i ++) {
            //     printf("%d" , pids[plist_index + i]) ;
            //     if(i < psize_block - 1)
            //         printf(", ") ;
            // }
            // printf("]\n") ;

            cur_visit_p = (psize_block >= 2 ? 2 : 1) ;
            main_channel = 0 ; 
            turn = 0 ; 
            // psize_block = total_psize_block ;
            // printf("can click 2023\n") ;
            // printf("can click %d\n" , __LINE__) ;

            // printf("ent_pts-dis[") ;
            // for(unsigned i = 0 ; i < TOPM  ;i  ++) {
            //     printf("%d-%f" , top_M_Cand[i] , top_M_Cand_dis[i]) ;
            //     if(i < TOPM - 1)
            //         printf(" ,") ;
            // }
            // printf("]\n") ;

            // printf("ent pts : [") ;
            // for(unsigned i = 0 ; i < TOPM ; i ++) {
            //     printf("%d - %f" , top_M_Cand[i] & 0x3FFFFFFFu , top_M_Cand_dis[i]) ;
            //     if(i < TOPM -1)
            //         printf(", ") ;
            // }
            // printf("]\n") ;
        }
        __syncthreads();
        
        
        
        

        // 无限次循环, 直到退出
        for(unsigned epoch = 0 ; epoch < MAX_ITER ; epoch ++) {

            // 每隔4轮将哈希表重置一次
            if((epoch + 1) % (HASH_RESET) == 0) {
                #pragma unroll
                for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                    hash_table[j] = 0xFFFFFFFF;
                }
                __syncthreads();
                #pragma unroll
                for(unsigned j = tid; j < TOPM + DEGREE2; j += blockDim.x * blockDim.y){
                    // 将此时的TOPM + K 个候选点插入哈希表中
                    if(top_M_Cand[j] != 0xFFFFFFFF) {
                        hash_insert(hash_table , top_M_Cand[j] & 0x3FFFFFFF) ; 
                    }
                }
            }
            __syncthreads();
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 下面进行点扩展
            if(tid < 32){
                /*
                    扩展规则: 找到当前turn通道的最近点, 然后从另一个通道找到距离该点最近的点, 进行扩展
                */
                if(laneid < 2)
                    to_explore[laneid] = 0xFFFFFFFF ;
                __syncwarp() ;

                unsigned shift_bits = (batch_pid[0] == batch_pid[1] ? 31 - (main_channel & 1)  : 31 - (turn & 1));
                unsigned mask = (1 << shift_bits) ;

            
                for(unsigned j = laneid; j < TOPM; j+=32){
                    unsigned n_p = ((top_M_Cand[j] & mask) == 0 ? 1 : 0) ;
                
                
                    unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                    if(ballot_res > 0) {

                        if(laneid == (__ffs(ballot_res) - 1)) {
                            to_explore[turn & 1] = local_idx[top_M_Cand[j] & 0x3FFFFFFF] ;
                            top_M_Cand[j] |= mask ;
                        }
                        break ;
                    }
                }
                __syncwarp() ;
                
                if(to_explore[turn] != 0xFFFFFFFF) {
                    unsigned global_to_explore = global_idx[seg_global_idx_start_idx[pid[turn]] + to_explore[turn]] ;
                    // unsigned stored_pos = point_batch_start_id[batch_pid[turn]] + to_explore[turn] ;
                    unsigned stored_pos = global_vec_stored_pos[global_to_explore] ;
                    // unsigned offset = stored_pos * pnum * edges_num_in_global_graph + pid[turn ^ 1] * edges_num_in_global_graph ;
                    unsigned offset = batch_all_point_num * edges_num_in_global_graph * batch_pid[turn ^ 1] + stored_pos * edges_num_in_global_graph ;
                    // printf("bid : %d , offset : %d , line : %d\n" , bid , offset , __LINE__) ;
                    // if(offset > 112500000) {
                    //     if(laneid == 0)
                    //     printf("bid : %d , batch_all_point_num : %d , edge_num_in_global_graph : %d , batch_pid[turn ^ 1] : %d , stored_pos : %d \n" 
                    //     , bid , batch_all_point_num , edges_num_in_global_graph , batch_pid[turn ^ 1] , stored_pos) ;
                    // }
                    for(unsigned j = laneid ; j < 2; j += 32) {
                        unsigned n_p = (!hash_peek(hash_table , batch_global_graph[offset + j])) ;
                        unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;

                        if(ballot_res > 0) {
                            if(laneid == (__ffs(ballot_res) - 1)) {
                                to_explore[turn ^ 1] = batch_global_graph[offset + j] ;
                            }
                            break ; 
                        }
                    }
                }
            }
            __syncthreads() ;
            // if(tid == 0 && epoch < 3) {
            //     printf("epoch : %d , bid : %d , batch_pid0 : %d , batch_pid1 : %d\n" , epoch , bid , batch_pid[0] , batch_pid[1]) ;
            // }
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;


            // 若扩展失败, 切换图栈
            if(to_explore[turn & 1] == 0xFFFFFFFF) {
                
                if(cur_visit_p == psize_block && batch_pid[0] == batch_pid[1]) {
                    // 两个图都没有扩展点, 且没有图可以切换, 退出
                    if(tid == 0) {
                        if(bid == 0)
                            printf("block %d: break in %d\n" ,bid ,  epoch) ; 
                        jump_num_cnt[bid] = epoch ; 
                    }
                    break ;
                } else if(cur_visit_p == psize_block) {
                        graph[turn] = graph[turn ^ 1] ;
                        pid[turn] = pid[turn ^ 1] ;
                        batch_pid[turn] = batch_pid[turn ^ 1] ;
                        if(tid == 0) {
                            main_channel = turn ^ 1 ; 
                        }
                    continue ;
                } else {
                    
                    unsigned next_batch_pid = batch_pids[plist_index + cur_visit_p] ;
                    unsigned next_pid = batch_partition_ids[next_batch_pid] ;
                    unsigned shift_bits = 30 + (turn & 1) ;
                    unsigned mask = 1 << shift_bits ;
                    
                    // 如何从全局id得到存储位置?

                    // 重置哈希表
                    #pragma unroll
                    for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
                
                        if(top_M_Cand[i % 64] == 0xFFFFFFFF) {
                            // 填充一个随机数
                            work_mem_unsigned[i] = 0xFFFFFFFF ;
                            work_mem_float[i] = 0.0f ;
                            continue ;
                        }
                        // 用前两个填充, 有重复的部分皆替换为随机点
                        unsigned pure_id = top_M_Cand[i % 64] & 0x3FFFFFFF ;
                        // unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / 64] ;
                        unsigned stored_pos = global_vec_stored_pos[pure_id] ;    
                        unsigned offset = batch_all_point_num * edges_num_in_global_graph * next_batch_pid + stored_pos * edges_num_in_global_graph ;
                        // printf("bid : %d , offset : %d , line : %d\n" , bid , offset , __LINE__) ;
                        // if(offset > 112500000) {
                        //     if(laneid == 0)
                        //     printf("bid : %d , batch_all_point_num : %d , edge_num_in_global_graph : %d , batch_pid[turn ^ 1] : %d , stored_pos : %d \n" 
                        //     , bid , batch_all_point_num , edges_num_in_global_graph , batch_pid[turn ^ 1] , stored_pos) ;
                        // }
                        // unsigned next_local_id = batch_global_graph[stored_pos * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / 64] ; 
                        unsigned next_local_id = batch_global_graph[offset + i / 64] ;
                        unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                        work_mem_unsigned[i] = (hash_insert(hash_table , next_global_id) != 0 ? next_global_id | mask : 0xFFFFFFFF) ;
                        work_mem_float[i] = (work_mem_unsigned[i] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    }
                    
                    __syncthreads() ;
                    // 距离计算
                    #pragma unroll
                    for(unsigned i = threadIdx.y ; i < 128 ; i += blockDim.y) {
                        if(work_mem_unsigned[i] == 0xFFFFFFFF) {
                            continue ;  
                        }
                        
                        half2 val_res;
                        unsigned pure_id = work_mem_unsigned[i] & 0x3FFFFFFF ;
                        unsigned stored_pos = global_vec_stored_pos[pure_id] ;
                        val_res.x = 0.0; val_res.y = 0.0 ;
                        for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                            half4 val2 = Load(&batch_values[stored_pos * DIM + k * 4]);
                            val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                            val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                        }
                        #pragma unroll
                        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                        }
        
                        if(laneid == 0){
                            work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                            atomicAdd(dis_cal_cnt + bid , 1) ;
                        }
                    }
                    __syncthreads() ;
                    bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                    __syncthreads() ;
                    // 此处需调整, 这里topm中被淘汰的点没有进入top_M_Cand + TOPM 中
                    merge_top_with_recycle_list<attrType>(top_M_Cand , work_mem_unsigned, top_M_Cand_dis , work_mem_float , tid , 128 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
                
                    __syncthreads() ;
                
                    bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                    __syncthreads() ;
                    if(work_mem_unsigned[0] != 0xFFFFFFFF) {
                        merge_top(recycle_list_id , work_mem_unsigned , recycle_list_dis , work_mem_float , tid , 128 , recycle_list_size) ;
                        __syncthreads() ;
                    }
                    graph[turn] = batch_all_graphs + batch_graph_start_index[next_batch_pid] ;
                    pid[turn] = next_pid ;
                    batch_pid[turn] = next_batch_pid ; 
                    if(tid == 0) {
                        cur_visit_p ++ ; 
                    }
                    __syncthreads() ;
                    // 处理另一个channel
                    // channel_id ^= 1 ; 
                    // if(tid == 0) {
                    //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
                    // }
                    continue ;
                }
            }
            __syncthreads() ;
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 对两个通道的扩展写进两个warp里, 防止warp分歧
            if(threadIdx.y < 2) {
                if(to_explore[threadIdx.y & 1] != 0xFFFFFFFF) {
                    unsigned channel_id = threadIdx.y ;
                    unsigned shift_bits = 30 + (channel_id & 1) ;
                    // printf("line : %d , shift_bits : %d\n" , __LINE__ , shift_bits) ;
                    unsigned mask = (batch_pid[0] == batch_pid[1] ? 0 : (1 << shift_bits)) ; 
                    #pragma unroll
                    for(unsigned j = laneid; j < DEGREE - (channel_id == turn ? 0 : 1); j += blockDim.x) {

                        const unsigned *ex_graph = graph[channel_id] ;
                        // printf("to_explore : %d\n" , to_explore[channel_id]) ;
                        // if(tid == 0) {
                        //     printf("ex_graph : [") ;
                        //     for(unsigned i = 0 ; i < 16 ; i ++)
                        //         printf("ex_graph[%d] - %d," , i , ex_graph[i]) ;
                        //     printf("]\n") ;
                        // }
                        unsigned to_append_local = ex_graph[to_explore[channel_id] * DEGREE + j];
                        // printf("to_append_local : %d\n" , to_append_local) ;
                        unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_append_local] ;
                        // printf("to_append : %d\n" , to_append) ;
                        top_M_Cand[TOPM + j + channel_id * DEGREE] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                        top_M_Cand_dis[TOPM + j + channel_id * DEGREE] = (top_M_Cand[TOPM + j + channel_id * DEGREE] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    }
                    if(channel_id != turn && laneid == 31) {
                        unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_explore[channel_id]] ;
                        top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                        top_M_Cand_dis[TOPM + (channel_id + 1) * DEGREE - 1] = (top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                        // if(hash_insert(hash_table , to_append) != 0) {
                        //     top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] = (to_append | mask) ;
                        //     top_M_Cand_dis[TOPM + (channel_id + 1) * DEGREE - 1] = 0.0f ;
                        // }
                    }
                } else {
                    unsigned channel_id = threadIdx.y ;
                    #pragma unroll
                    for(unsigned i = laneid ; i < DEGREE ; i += blockDim.x) {
                        top_M_Cand[TOPM + i + channel_id * DEGREE] = 0xFFFFFFFF ;
                        top_M_Cand_dis[TOPM + i + channel_id * DEGREE] = INF_DIS ;
                    }
                }
                if(tid == 0)
                    turn ^= 1 ; 
            } 
            __syncthreads();
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 此时无论是通道1还是通道2, 所有节点已在TOPM数组的后半段就位, 因此距离计算可以一视同仁
            #pragma unroll
            for(unsigned i = threadIdx.y; i < DEGREE2 ; i += blockDim.y) {
                if(top_M_Cand[TOPM + i] == 0xFFFFFFFF)
                    continue ;
                
                unsigned pure_id = top_M_Cand[TOPM + i] & 0x3FFFFFFF ;

                // if(pure_id >= 1000000 && laneid == 0 ) {
                //     printf("error at %d : pure_id = %d , i = %d\n" , __LINE__ , pure_id , i) ;
                    
                // }
                unsigned stored_pos = global_vec_stored_pos[pure_id] ;
                // if(stored_pos >= 62500 * 4)
                //     printf("error at %d : stored_pos = %d\n" , __LINE__ , stored_pos) ;
            
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                    // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                    half4 val2 = Load(&batch_values[stored_pos * DIM + j * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    // 将warp内所有子线程的计算结果归并到一起
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }
                if(laneid == 0){
                    top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                    atomicAdd(dis_cal_cnt + bid , 1) ;
                }
            }
            __syncthreads();

            // if(tid == 0 && epoch < 10) {
            //     // printf("bid : %d , epoch : %d , degree2 : %d , Tturn : %d\n, " 
            //     // , bid , epoch , DEGREE2, turn) ;
            //     // printf("to_explore[turn] : %d, to_explore[turn ^ 1] : %d, graph0 : %d , graph1 : %d  , pid0 : %d  , pid1 : %d \n前10轮扩展: [" , 
            //     // to_explore[turn] , to_explore[turn ^ 1]  , graph[0] , graph[1] , batch_pid[0] , batch_pid[1]) ;
            //     printf("bid[") ;
            //     for(unsigned i = 0 ; i < DEGREE2  ;i ++) {
            //         printf("(%d)%d - %f" ,i ,  (top_M_Cand[i + TOPM] == 0xFFFFFFFFu ? -1 : top_M_Cand[i + TOPM] & 0x3FFFFFFFu) , (top_M_Cand[i + TOPM] == 0xFFFFFFFFu ? -1.0f : top_M_Cand_dis[i + TOPM])) ;
            //         if(i < DEGREE2 - 1)
            //             printf(", ") ;
            //     }
            //     printf("]\n") ;
            //     printf("bid : %d ,graph0 : %lld , graph1 : %lld , batch_pid[0] : %d , batch_pid[1] : %d, pid[0] : %d , pid[1] : %d\n" , bid , graph[0] , graph[1] ,
            //     batch_pid[0] , batch_pid[1] , pid[0] , pid[1]) ;

            // }
            // __syncthreads() ;
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 排序
            // if(tid < DEGREE2)
            //     warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE2);
            bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], DEGREE2) ;
            __syncthreads() ;
            if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                // 将两个列表归并到一起
                merge_top_with_recycle_list<attrType>(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE2 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
                __syncthreads() ;
                // if(tid < DEGREE2)
                //     warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
                bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], DEGREE2) ;
                __syncthreads() ;
                if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                    merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE2 , recycle_list_size) ;
                    __syncthreads() ;
                }
            }
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;
        }  
        
        // if(tid == 0) {
        //     printf("过滤前的topm[") ;
        //     for(unsigned i = 0 ; i < 10 ; i ++) {
        //         printf("%d-%f" , top_M_Cand[i] & 0x3FFFFFFF , top_M_Cand_dis[i]) ;
        //         if(i < 9)
        //             printf(" ,") ;
        //     }
        //     printf("]\n") ;
        // }
        // if(tid == 0) {
        //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;

        // 搜索完成, 将recycle_list中的内容合并到top_M_Cand中
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
            if(top_M_Cand[i] == 0xFFFFFFFF)
                continue ;
            bool flag = true ;
            unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
            for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
                flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
            }
            if(!flag) {
                top_M_Cand[i] = 0xFFFFFFFF ;
                top_M_Cand_dis[i] = INF_DIS ;
            }
        }
        __syncthreads() ;
        bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM) ;
        __syncthreads() ;
        // 若recycle_list中确实有点, 则合并
        if(recycle_list_id[0] != 0xFFFFFFFF) {
            merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
            __syncthreads() ;
        }
        // if(tid == 0) {
        //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;
    }

    __shared__ unsigned prefix_sum[1024] ;
    __shared__ unsigned top_prefix[32] ;
    __shared__ unsigned indices_arr[1024] ;
    __shared__ unsigned task_num ; 
    // 处理剩余的小分区
    for(unsigned g = 0 ; g < total_psize_block && little_seg_num_block ; g ++) {
        unsigned batch_gpid = batch_pids[plist_index + g] ;
        unsigned gpid = batch_partition_ids[batch_gpid] ;
        unsigned global_idx_offset = seg_global_idx_start_idx[gpid] ;
        unsigned num_in_seg = batch_graph_size[batch_gpid] ;
        // unsigned num_in_seg = 62464 ;

        #pragma unroll
        for(unsigned i = tid ;i < 32 ; i += blockDim.x * blockDim.y) 
            bit_map[i] = 0 ; 
        #pragma unroll
        for(unsigned i = tid ; i < 1056 ; i += blockDim.x * blockDim.y) {
            work_mem_unsigned[i] = 0xFFFFFFFF ;
            work_mem_float[i] = INF_DIS ;
        }
        if(tid == 0)
            prefilter_pointer = 0 ; 
        __syncthreads() ;
        // if(tid == 0) {
        //     printf("bid : %d , can click %d\n" ,bid ,  __LINE__) ;
        // }
        // __syncthreads() ;

        // 先把 total_psize_block个点分片, 分片大小可以考虑bit_map中1的个数? 
        // selectivity <= 0.1, 意味着, 每十个点里只有一个点满足条件, 找到32个点平均要遍历320个点
        //  有两种方案 :  1. 动态chunk, 尽可能的设置一组中包含32个点
        //               2. 静态chunk 设置 chunk 大小为10 , 但要维护的候选列表长度至少为320
        //  合并次数 : 1. 6250 / 32 次 ; 2. 625 / 32 次
        //  随着合并次数的增多, 方案2的待合并点数量、以及合并次数会显著减少，故采用方案2. 方案1的前置条件太复杂
        // 表示某个点是否满足属性约束条件
        // for(unsigned i = tid ; i < num_in_seg ; i += blockDim.x * blockDim.y) {
        //     bool flag = true ;
        //     unsigned g_global_id = global_idx[global_idx_offset + i] ;
        //     if(g_global_id >= 1e8) {
        //         printf("bid : %d , illegal id : %d\n" , bid , g_global_id) ;
        //     }
        //     for(unsigned j = 0 ; j < attr_dim ; j ++)
        //         flag = (flag && (attrs[g_global_id * attr_dim + j] >= q_l_bound[j]) && (attrs[g_global_id * attr_dim + j] <= q_r_bound[j])) ;
        //     unsigned ballot_res = __ballot_sync(__activemask() , flag) ;
        //     if(laneid == 0)
        //         bit_map[(i >> 5)] = ballot_res ; 
        // }
        // __syncthreads() ;
        // if(tid == 0) {
        //     printf("bid : %d , can click %d\n" , __LINE__) ;
        // }
        // __syncthreads() ;

        unsigned chunk_num = (num_in_seg + 31) >> 5 ;
        unsigned chunk_batch_size = 32 ; 
        for(unsigned chunk_id = 0 ; chunk_id < chunk_num ; chunk_id += chunk_batch_size) {
            unsigned point_batch_size = (chunk_id + chunk_batch_size < chunk_num ? (chunk_batch_size << 5) : num_in_seg - (chunk_id << 5)) ;
            unsigned point_batch_start_id = (chunk_id << 5) ;

            for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                bool flag = true ;
                unsigned g_global_id = global_idx[global_idx_offset + point_batch_start_id + i] ;
                for(unsigned j = 0 ; j < attr_dim ; j ++)
                    flag = (flag && (attrs[g_global_id * attr_dim + j] >= q_l_bound[j]) && (attrs[g_global_id * attr_dim + j] <= q_r_bound[j])) ;
                unsigned ballot_res = __ballot_sync(__activemask() , flag) ;
                if(laneid == 0)
                    bit_map[(i >> 5)] = ballot_res ; 
            }
            __syncthreads() ;


            // 扫描出该chunk所需要的前缀和数组
            // 先利用__popc() 得到长度为32的前缀和数组, 再利用0号warp, 把32个局部前缀和数组合并为一个整体
            for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                // 在bitmap中的单元格位置
                // unsigned cell_id = chunk_id + (i >> 5) ;
                unsigned cell_id = (i >> 5) ;
                unsigned bit_pos = i % 32 + 1 ;
                unsigned mask = (bit_pos == 32 ? 0xFFFFFFFF : (1 << bit_pos) - 1) ;
                unsigned index = (bit_map[cell_id] & mask) ;
                prefix_sum[i] = __popc(index) ;
            }
            __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" ,bid ,  __LINE__) ;
            // }
            // __syncthreads() ;
            if(tid < 32) {
                unsigned v = (laneid + chunk_id < chunk_num ? __popc(bit_map[laneid]) : 0) ;
                // 将长度大小为32的各个子prefix合并为最终的prefix_sum 
                #pragma unroll
                for(unsigned offset = 1 ; offset < 32 ; offset <<= 1) {
                    unsigned y = __shfl_up_sync(0xFFFFFFFF , v , offset) ;
                    if(laneid >= offset) v += y ;
                }
                top_prefix[laneid] = v ;
            }
            // __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" ,bid ,  __LINE__) ;
            // }
            // __syncthreads() ;
            if(tid == 0) {
                task_num = top_prefix[31] ;
                // if(task_num > 0 && false) {
                //     printf("task num : %d\n" , task_num) ;

                //     printf("[") ;
                //     for(unsigned j = 0 ; j < 1024 ; j ++) {
                //         printf("%d" , prefix_sum[j]) ;
                //         if(j < 1023)
                //             printf(", ") ;
                //     }
                //     printf("]\n") ;

                //     printf("top_prefix :[") ;
                //     for(unsigned j = 0 ; j < 32 ; j ++) {
                //         printf("%d" , top_prefix[j]) ;
                //         if(j < 31)
                //             printf(", ") ;
                //     }
                //     printf("]\n") ;
                // }
                // printf("bid : %d , can click : %d\n" , bid , __LINE__) ;
            }
            __syncthreads() ;
            for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                // unsigned cell_id = chunk_id + (i >> 5) ; 
                unsigned cell_id = (i >> 5) ;
                
                unsigned id_in_cell = i % 32 ;
                unsigned bit_mask = (1 << id_in_cell) ;
                if((bit_map[cell_id] & bit_mask) != 0) {
                    // 获取插入位置
                    // unsigned prefix_sum_fetch_pos = 
                    unsigned local_cell_id = (i >> 5) ;
                    unsigned put_pos = (local_cell_id == 0 ? 0 : top_prefix[local_cell_id - 1]) + (id_in_cell == 0 ? 0 : prefix_sum[i - 1]) ;
                    // if(i == 2) {
                    //     printf("i == 2 : %d , cell_id : %d , id_in_cell : %d\n" , put_pos , cell_id , id_in_cell) ;
                    // }
                    // if(put_pos >= 1024) {

                    //     // printf("bid : %d ,point_batch_size : %d\n" , bid , point_batch_size) ;
                    //     printf("bid : %d ,put_pos : %d , top_prefix[%d - 1] = %d , prefix_sum[%d - 1] = %d\n" , bid, put_pos , cell_id , top_prefix[(i >> 5) - 1] ,
                    //     i , prefix_sum[i - 1]) ;
                    // }
                    indices_arr[put_pos] = i ;
                }
            }
            __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click : %d , chunk_id : %d , chunk_num : %d  \n" , bid , __LINE__ , chunk_id , chunk_num) ;
            // }
 
            // 计算距离, 然后用原子操作插入队列中
            for(unsigned i = threadIdx.y ; i < task_num ; i += blockDim.y) {
                unsigned cand_id = indices_arr[i] + point_batch_start_id ;
                // if(laneid == 0 && chunk_id == 125920) {
                //     printf("bid : %d , cand_id : %d , indices_arr[i] : %d\n" , bid , cand_id , indices_arr[i]) ;
                // }
                // __syncwarp() ;
                // 计算距离
                unsigned g_global_id = global_idx[global_idx_offset + cand_id] ;
                unsigned stored_pos = global_vec_stored_pos[g_global_id] ;
                // if(chunk_id == 125920 ) {
                //     if(laneid == 0)
                //         printf("bid : %d , stored_pos : %d , g_global_id : %d ,DIM : %d , stored_pos * DIM : %d\n" , bid , stored_pos , g_global_id , DIM , stored_pos * DIM ) ;
                // } 
                float dis = 0.0f ;

                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&batch_values[stored_pos * DIM + k * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    dis = __half2float(val_res.x) + __half2float(val_res.y) ;
                    atomicAdd(dis_cal_cnt + bid , 1) ;
                    // printf("bid : %d , can click : %d dis cal finish!\n" , bid , __LINE__) ;
                }
                __syncwarp() ;
                // 存入work_mem
                if(laneid == 0 && dis < top_M_Cand_dis[K - 1]) {
                    int cur_pointer = atomicAdd(&prefilter_pointer , 1) ;
                    // if(cur_pointer >= 1024)
                    //     printf("bid : %d , error cur_pointer : %d\n" , bid , cur_pointer) ;
                    work_mem_unsigned[cur_pointer] = g_global_id ; 
                    work_mem_float[cur_pointer] = dis ; // 距离变量
                }
                __syncwarp() ;
            }

            __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" , bid , __LINE__) ;
            // }
            // __syncthreads() ;
            if(prefilter_pointer >= 32 || (chunk_id + chunk_batch_size >= chunk_num && prefilter_pointer > 0)) {
                // 先对前32个点排序, 然后再合并到topK上
                // if(tid < 32)
                //     warpSort(work_mem_float , work_mem_unsigned , tid , 32) ; 
                bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned, prefilter_pointer) ;
                __syncthreads() ;
                merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float , tid , prefilter_pointer , K) ;
                __syncthreads() ;
                unsigned limit = prefilter_pointer ;
                #pragma unroll
                for(unsigned i = tid ; i < limit ; i += blockDim.x * blockDim.y) {
                    work_mem_unsigned[i] = 0xFFFFFFFF ;
                    work_mem_float[i] = INF_DIS ;
                }
                if(tid == 0) {
                    prefilter_pointer = 0 ; 
                    atomicAdd(post_merge_cnt + bid , 1) ;
                }

                __syncthreads() ;
            }
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" ,bid , __LINE__) ;
            // }
            // __syncthreads() ;
        }
    }
    // if(tid == 0) {
    //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
    // }
    // __syncthreads() ;

    #pragma unroll
    for (unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        work_mem_unsigned[i] = top_M_Cand_video_memory[bid * TOPM + i] ;
        work_mem_float[i] = top_M_Cand_dis_video_memory[bid * TOPM + i] ;
    }
    // if(tid == 0) {
    //     printf("this epoch : id[") ;
    //     for(unsigned i = 0 ; i < TOPM ;i  ++) {
    //         printf("%d-%d ," , top_M_Cand[i] & 0x3FFFFFFF , (top_M_Cand[i] >> 30)) ;
    //     }
    //     printf("]\nthis epoch : dis[") ;
    //     for(unsigned i = 0 ; i < TOPM ;i  ++) {
    //         printf("%f ," , top_M_Cand_dis[i]) ;
    //     }
    //     printf("]\n") ;
    //     // --------------------------------
    //     printf("this epoch work mem : id[") ;
    //     for(unsigned i = 0 ; i < K ;i  ++) {
    //         printf("%d ," , work_mem_unsigned[i] & 0x3FFFFFFF) ;
    //     }
    //     printf("]\nthis epoch work mem : dis[") ;
    //     for(unsigned i = 0 ; i < K ;i  ++) {
    //         printf("%f ," , work_mem_float[i]) ;
    //     }
    //     printf("]\n") ;
    // }
    __syncthreads() ;
    merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float , tid , K , TOPM) ;
    __syncthreads() ;

    // 每次执行完毕, 和现有结果做合并(和前k个结果合并即可)
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        top_M_Cand_video_memory[bid * TOPM + i] = (top_M_Cand[i] & 0x3FFFFFFF);
        top_M_Cand_dis_video_memory[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    for(unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        result_k_id[bid * K + i] = (top_M_Cand[i] & 0x3FFFFFFF) ;
        // result_k_dis[bid * K + i] = top_M_Cand_dis[i] ;
    }
}

__device__ double dis_cal_cost_total_outer[10000] = {0.0};
template<typename attrType = float>
__global__ void select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter_batch_process_compressed_graph_T_dataset_resident(const unsigned batch_size , const unsigned* batch_partition_ids , const unsigned* __restrict__ batch_all_graphs, const half* __restrict__ batch_values, const half* __restrict__ query_data, 
    unsigned* __restrict__ top_M_Cand_video_memory , float* __restrict__ top_M_Cand_dis_video_memory, unsigned* ent_pts , unsigned DEGREE , unsigned TOPM , unsigned DIM , const unsigned* batch_graph_start_index ,const unsigned* batch_graph_size ,
    const unsigned* __restrict__ batch_pids , const unsigned* batch_p_size ,const unsigned* batch_p_start_index ,const attrType* l_bound ,const attrType* r_bound ,const attrType* __restrict__ attrs , unsigned attr_dim , const unsigned* __restrict__ batch_global_graph , unsigned edges_num_in_global_graph,
    unsigned recycle_list_size ,const unsigned* __restrict__ global_idx ,const unsigned* __restrict__ local_idx ,const unsigned* seg_global_idx_start_idx , unsigned pnum ,unsigned K, unsigned* result_k_id  ,const unsigned* task_id_mapper , const unsigned* batch_little_seg_counts
    ,const unsigned* global_vec_stored_pos , const unsigned batch_all_point_num , const unsigned OP_TYPE = 0) {
    // *********   当前版本为结束时合并循环池, 后续可酌情实现不合并循环池的版本   ***********
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数

    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    unsigned* recycle_list_id = (unsigned*) shared_mem ;
    float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    unsigned* top_M_Cand = (unsigned*) (recycle_list_dis + recycle_list_size) ;
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE * 2) ;
    attrType* q_l_bound = (attrType*)(top_M_Cand_dis + TOPM + DEGREE * 2) ;
    attrType* q_r_bound = (attrType*)(q_l_bound + attr_dim) ;

    // 这里应该怎么处理8字节的对齐问题？
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) +
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(attrType)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[1024 + 32] ;
    __shared__ float work_mem_float[1024 + 32] ;
    // 可用投票一次插入32位
    __shared__ unsigned bit_map[32] ;
    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = task_id_mapper[blockIdx.x], laneid = threadIdx.x , warpId = threadIdx.y ;
    // 若线程块发射顺序与blockIdx.x的编号无关, 则可以手动实现按到达顺序取任务的排队器
    /**
        在批处理环境下, 以下变量的语义更改为:
        psize_block - 当前批次中, 查询的正常分区数量, plist_index - 当前批次中, 当前查询的分区列表在总的分区列表中的起始位置
        total_psize_block - 当前批次中, 查询总的分区数量, little_seg_num_block - 当前批次中, 查询的小分区数量
    **/
    unsigned psize_block = batch_p_size[bid] , plist_index = batch_p_start_index[bid] , 
    total_psize_block = batch_p_size[bid] , little_seg_num_block = batch_little_seg_counts[bid] ;

    // if(tid == 0 && bid == 0) {
    //     printf("psize : %d\n" , psize_block) ;
    //     printf("plist : [") ;
    //     for(unsigned i = 0 ; i < total_psize_block ; i ++) 
    //         printf("%d ," , batch_pids[i + plist_index]) ;
    //     printf("]\n") ;
    //     printf("partition_id_list : [") ;
    //     for(unsigned i = 0 ; i < batch_size ; i ++)
    //         printf("%d ," , batch_partition_ids[i]) ;
    //     printf("]\n") ;
    //     printf("batch graph start index : [") ;
    //     for(unsigned i = 0 ; i < batch_size ; i ++)
    //         printf("%d ," , batch_graph_start_index[i]) ;
    //     printf("]\n") ; 
    //     printf("batch partition start index : [") ;
    //     for(unsigned i = 0 ; i < gridDim.x  ; i ++)
    //         printf("%d ," , batch_p_start_index[i]) ;
    //     printf("]\n") ;
    // }
    // __syncthreads() ;
    // 25 不通过 23 不通过 21 不通过 15 不通过 8 通过 11 通过 13 不通过 12 不通过
    // 1 2 5 6 8 通过
    // if(bid != 5)
    //     return ;
    if(total_psize_block == 0) {
        // if(tid == 0) {
        //     printf("当前处理批次为空 bid : %d\n" , bid) ;
        // }
        return ;
    } 
    // else {
    //     if(tid == 0) {
    //         printf("当前处理批次非空 bid : %d\n" , bid) ;
    //     }
    // }

    __shared__ unsigned cur_visit_p , main_channel , turn ;
    __shared__ unsigned to_explore[2];
    __shared__ unsigned prefilter_pointer ;
    __shared__ long long start ;
    __shared__ double dis_cal_cost ;


    __shared__ unsigned hash_table[HASHLEN];
    // 从全局内存中恢复topm
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
        // top_M_Cand[i] = top_M_Cand_video_memory[bid * TOPM + i] | 0xCF000000 ;
        // top_M_Cand_dis[i] = top_M_Cand_dis_video_memory[bid * TOPM + i] ;
    }
    for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
        q_l_bound[i] = l_bound[bid * attr_dim + i] ;
        q_r_bound[i] = r_bound[bid * attr_dim + i] ;
    }
    // 没必要在每个分区开始时都把recycle_list复原, 每个分区得到的入口点肯定是越近越好
    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        recycle_list_id[i] = 0xFFFFFFFF ;
        recycle_list_dis[i] = INF_DIS ;
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
        tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
    }
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }

    if(tid == 0) {
        dis_cal_cost = 0.0 ;
    }
    __syncthreads() ;

    // if(tid == 0) {

    //     unsigned gt_list[] = {1337510 ,15321394 ,41130812 ,49986347 ,53500708 ,66236865 ,68479532 ,72095672 ,74280217 ,86258417} ;
    //     for(unsigned i = 0 ; i < 10 ; i ++)
    //         work_mem_unsigned[i] = gt_list[i] ;

    //     printf("ent_pts[") ;
    //     for(unsigned i = 0 ; i < TOPM  ;i  ++) {
    //         printf("%d" , ent_pts[i]) ;
    //         if(i < TOPM - 1)
    //             printf(" ,") ;
    //     }
    //     printf("]\n") ;
    // }
    // __syncthreads() ;

    // #pragma unroll
    // for(unsigned i = threadIdx.y; i < TOPM ; i += blockDim.y) {
        
        
    //     unsigned pure_id = work_mem_unsigned[i] ;
    //     // unsigned stored_pos = batch_point_start_index[batch_pid[(top_M_Cand[i] >> 31)]] + local_idx[pure_id] ;
    //     // unsigned stored_pos = global_vec_stored_pos[pure_id] ;
    //     size_t vec_offset = static_cast<size_t>(pure_id) * static_cast<size_t>(DIM) ;
    //     half2 val_res ;
    //     val_res.x = 0.0 ; val_res.y = 0.0 ;
    //     for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x) {
    //         // warp中的每个子线程负责8个字节 即4个half类型分量的计算
    //         half4 val2 = Load(&batch_values[vec_offset + static_cast<size_t>(j * 4)]);
    //         val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
    //         val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
    //     }
    //     #pragma unroll
    //     for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2) {   
    //         // 将warp内所有子线程的计算结果归并到一起
    //         val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
    //     }
    //     if(laneid == 0) {
    //         work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
    //         // atomicAdd(dis_cal_cnt + bid , 1) ;
    //     }
    // }
    // __syncthreads() ;
    // if(tid == 0) {
    //     printf("gt id and dis[") ;
    //     for(unsigned i = 0 ; i < 10 ; i ++) {

    //         printf("%d-%f" , work_mem_unsigned[i] , work_mem_float[i]) ;
    //         if(i < 9)
    //             printf(" ,") ;
    //     }
    //     printf("]\n") ;
    // }
    // __syncthreads() ;

    
    if(psize_block > 0 && little_seg_num_block == 0) {
        psize_block = total_psize_block ;
        /**
            batch_pids[i] 第i个分区, 用的是 0 - batchSize 之间的局部id
            batch_graph_start_index[i] 当前batch 第i个分区所代表的图
        **/
        // if(plist_index >= 12) {
        //     if(tid == 0)
        //         printf("error at %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;
        unsigned batch_pid[2] = {batch_pids[plist_index] , (psize_block >= 2 ? batch_pids[plist_index + 1] : batch_pids[plist_index])} ;
        unsigned pid[2] = {batch_partition_ids[batch_pid[0]] , batch_partition_ids[batch_pid[1]]} ;
        // pid[0] = 9 ; 
        const unsigned* graph[2] = {batch_all_graphs + batch_graph_start_index[batch_pid[0]] , batch_all_graphs + batch_graph_start_index[batch_pid[1]]};
        const unsigned DEGREE2 = DEGREE * 2 ;
        // if(tid == 0) {
        //     printf("bid : %d , batch_pid0 : %d , batch_pid1 : %d , pid0 : %d , pid1 : %d\n" , bid , batch_pid[0] , batch_pid[1] , pid[0] , pid[1]) ;
        // }

        // 有两种可能, 没有局部结果的从入口点扩展, 有局部结果的先从局部结果扩展, 若局部结果中点个数不够从入口点补充

        // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
        for(unsigned i = tid , L2 = DEGREE ; i < DEGREE2; i += blockDim.x * blockDim.y){
            unsigned shift_bits = 30 +  i / L2;
            unsigned mask = (batch_pid[0] == batch_pid[1] ? 0 : 1 << shift_bits) ; 
            // 用反向掩码, 将另一位置1
            top_M_Cand[i] = global_idx[seg_global_idx_start_idx[pid[i / L2]] + ent_pts[i]] | mask;
            // top_M_Cand[TOPM + DEGREE + i]

            // 哈希表存放的是局部id, 如何统一?  是否可以直接存放全局id? 
            hash_insert(hash_table , top_M_Cand[i] & 0x3FFFFFFF) ;
        }

        __syncthreads() ;
        if(tid == 0) {
            start = clock64() ;
        }
        __syncthreads() ;
        // 初始化入口节点
        #pragma unroll
        for(unsigned i = threadIdx.y; i < TOPM ; i += blockDim.y) {
            if(top_M_Cand[i] == 0xFFFFFFFF)
                continue ;
            // threadIdx.y 0-5 , blockDim.y 即 6
            
            unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
            size_t vec_offset = static_cast<size_t>(pure_id) * static_cast<size_t>(DIM) ;
            // unsigned stored_pos = batch_point_start_index[batch_pid[(top_M_Cand[i] >> 31)]] + local_idx[pure_id] ;
            // unsigned stored_pos = global_vec_stored_pos[pure_id] ;
            half2 val_res ;
            val_res.x = 0.0 ; val_res.y = 0.0 ;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x) {
                // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                half4 val2 = Load(&batch_values[vec_offset + static_cast<size_t>(j * 4)]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2) {   
                // 将warp内所有子线程的计算结果归并到一起
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }
            if(laneid == 0) {
                top_M_Cand_dis[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                atomicAdd(dis_cal_cnt + bid , 1) ;
            }
        }
        __syncthreads();
        if(tid == 0) {
            dis_cal_cost += ((double)(clock64() - start) / 1582000) ;
        }
        __syncthreads() ;
        bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis , top_M_Cand , DEGREE2) ;
        __syncthreads() ;
        // merge_top_with_recycle_list(top_M_Cand , work_mem_unsigned, top_M_Cand_dis , work_mem_float , tid , enter_pts_num , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
        // __syncthreads() ;
        // if(tid == 0) {
        //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;


        if(tid == 0) {
            // printf("psize block : %d\n" , psize_block) ;
            // printf("pid[0] %d , pid[1] %d\n" , pid[0] , pid[1]) ;
            // printf("[") ;
            // for(int i = 0; i < psize_block ; i ++) {
            //     printf("%d" , pids[plist_index + i]) ;
            //     if(i < psize_block - 1)
            //         printf(", ") ;
            // }
            // printf("]\n") ;

            cur_visit_p = (psize_block >= 2 ? 2 : 1) ;
            main_channel = 0 ; 
            turn = 0 ; 
            // psize_block = total_psize_block ;
            // printf("can click 2023\n") ;
            // printf("can click %d\n" , __LINE__) ;

            // printf("ent_pts-dis[") ;
            // for(unsigned i = 0 ; i < TOPM  ;i  ++) {
            //     printf("%d-%f" , top_M_Cand[i] , top_M_Cand_dis[i]) ;
            //     if(i < TOPM - 1)
            //         printf(" ,") ;
            // }
            // printf("]\n") ;

            // printf("ent pts : [") ;
            // for(unsigned i = 0 ; i < TOPM ; i ++) {
            //     printf("%d - %f" , top_M_Cand[i] & 0x3FFFFFFFu , top_M_Cand_dis[i]) ;
            //     if(i < TOPM -1)
            //         printf(", ") ;
            // }
            // printf("]\n") ;
        }
        __syncthreads();
        
        
        
        

        // 无限次循环, 直到退出
        for(unsigned epoch = 0 ; epoch < MAX_ITER ; epoch ++) {

            // 每隔4轮将哈希表重置一次
            if((epoch + 1) % (HASH_RESET) == 0) {
                #pragma unroll
                for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                    hash_table[j] = 0xFFFFFFFF;
                }
                __syncthreads();
                #pragma unroll
                for(unsigned j = tid; j < TOPM + DEGREE2; j += blockDim.x * blockDim.y){
                    // 将此时的TOPM + K 个候选点插入哈希表中
                    if(top_M_Cand[j] != 0xFFFFFFFF) {
                        hash_insert(hash_table , top_M_Cand[j] & 0x3FFFFFFF) ; 
                    }
                }
            }
            __syncthreads();
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 下面进行点扩展
            if(tid < 32){
                /*
                    扩展规则: 找到当前turn通道的最近点, 然后从另一个通道找到距离该点最近的点, 进行扩展
                */
                if(laneid < 2)
                    to_explore[laneid] = 0xFFFFFFFF ;
                __syncwarp() ;

                unsigned shift_bits = (batch_pid[0] == batch_pid[1] ? 31 - (main_channel & 1)  : 31 - (turn & 1));
                unsigned mask = (1 << shift_bits) ;

            
                for(unsigned j = laneid; j < TOPM; j+=32){
                    unsigned n_p = ((top_M_Cand[j] & mask) == 0 ? 1 : 0) ;
                
                
                    unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                    if(ballot_res > 0) {

                        if(laneid == (__ffs(ballot_res) - 1)) {
                            to_explore[turn & 1] = local_idx[top_M_Cand[j] & 0x3FFFFFFF] ;
                            top_M_Cand[j] |= mask ;
                        }
                        break ;
                    }
                }
                __syncwarp() ;
                
                if(to_explore[turn] != 0xFFFFFFFF) {
                    unsigned global_to_explore = global_idx[seg_global_idx_start_idx[pid[turn]] + to_explore[turn]] ;
                    // unsigned stored_pos = point_batch_start_id[batch_pid[turn]] + to_explore[turn] ;
                    unsigned stored_pos = global_vec_stored_pos[global_to_explore] ;
                    // unsigned offset = stored_pos * pnum * edges_num_in_global_graph + pid[turn ^ 1] * edges_num_in_global_graph ;
                    unsigned offset = batch_all_point_num * edges_num_in_global_graph * batch_pid[turn ^ 1] + stored_pos * edges_num_in_global_graph ;
                    // printf("bid : %d , offset : %d , line : %d\n" , bid , offset , __LINE__) ;
                    // if(offset > 112500000) {
                    //     if(laneid == 0)
                    //     printf("bid : %d , batch_all_point_num : %d , edge_num_in_global_graph : %d , batch_pid[turn ^ 1] : %d , stored_pos : %d \n" 
                    //     , bid , batch_all_point_num , edges_num_in_global_graph , batch_pid[turn ^ 1] , stored_pos) ;
                    // }
                    for(unsigned j = laneid ; j < 2; j += 32) {
                        unsigned n_p = (!hash_peek(hash_table , batch_global_graph[offset + j])) ;
                        unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;

                        if(ballot_res > 0) {
                            if(laneid == (__ffs(ballot_res) - 1)) {
                                to_explore[turn ^ 1] = batch_global_graph[offset + j] ;
                            }
                            break ; 
                        }
                    }
                }
            }
            __syncthreads() ;
            // if(tid == 0 && epoch < 3) {
            //     printf("epoch : %d , bid : %d , batch_pid0 : %d , batch_pid1 : %d\n" , epoch , bid , batch_pid[0] , batch_pid[1]) ;
            // }
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;


            // 若扩展失败, 切换图栈
            if(to_explore[turn & 1] == 0xFFFFFFFF) {
                
                if(cur_visit_p == psize_block && batch_pid[0] == batch_pid[1]) {
                    // 两个图都没有扩展点, 且没有图可以切换, 退出
                    if(tid == 0) {
                        if(bid == 0)
                            printf("block %d: break in %d\n" ,bid ,  epoch) ; 
                        jump_num_cnt[bid] = epoch ; 
                    }
                    break ;
                } else if(cur_visit_p == psize_block) {
                        graph[turn] = graph[turn ^ 1] ;
                        pid[turn] = pid[turn ^ 1] ;
                        batch_pid[turn] = batch_pid[turn ^ 1] ;
                        if(tid == 0) {
                            main_channel = turn ^ 1 ; 
                        }
                    continue ;
                } else {
                    
                    unsigned next_batch_pid = batch_pids[plist_index + cur_visit_p] ;
                    unsigned next_pid = batch_partition_ids[next_batch_pid] ;
                    unsigned shift_bits = 30 + (turn & 1) ;
                    unsigned mask = 1 << shift_bits ;
                    
                    // 如何从全局id得到存储位置?

                    // 重置哈希表
                    #pragma unroll
                    for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
                
                        if(top_M_Cand[i % 64] == 0xFFFFFFFF) {
                            // 填充一个随机数
                            work_mem_unsigned[i] = 0xFFFFFFFF ;
                            work_mem_float[i] = 0.0f ;
                            continue ;
                        }
                        // 用前两个填充, 有重复的部分皆替换为随机点
                        unsigned pure_id = top_M_Cand[i % 64] & 0x3FFFFFFF ;
                        // unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / 64] ;
                        unsigned stored_pos = global_vec_stored_pos[pure_id] ;    
                        unsigned offset = batch_all_point_num * edges_num_in_global_graph * next_batch_pid + stored_pos * edges_num_in_global_graph ;
                        // printf("bid : %d , offset : %d , line : %d\n" , bid , offset , __LINE__) ;
                        // if(offset > 112500000) {
                        //     if(laneid == 0)
                        //     printf("bid : %d , batch_all_point_num : %d , edge_num_in_global_graph : %d , batch_pid[turn ^ 1] : %d , stored_pos : %d \n" 
                        //     , bid , batch_all_point_num , edges_num_in_global_graph , batch_pid[turn ^ 1] , stored_pos) ;
                        // }
                        // unsigned next_local_id = batch_global_graph[stored_pos * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / 64] ; 
                        unsigned next_local_id = batch_global_graph[offset + i / 64] ;
                        unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                        work_mem_unsigned[i] = (hash_insert(hash_table , next_global_id) != 0 ? next_global_id | mask : 0xFFFFFFFF) ;
                        work_mem_float[i] = (work_mem_unsigned[i] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    }
                    
                    __syncthreads() ;

                    if(tid == 0) {
                        start = clock64() ;
                    }
                    __syncthreads() ;
                    // 距离计算
                    #pragma unroll
                    for(unsigned i = threadIdx.y ; i < 128 ; i += blockDim.y) {
                        if(work_mem_unsigned[i] == 0xFFFFFFFF) {
                            continue ;  
                        }
                        
                        half2 val_res;
                        unsigned pure_id = work_mem_unsigned[i] & 0x3FFFFFFF ;
                        size_t vec_offset = static_cast<size_t>(pure_id) * static_cast<size_t>(DIM) ;
                        // unsigned stored_pos = global_vec_stored_pos[pure_id] ;
                        val_res.x = 0.0; val_res.y = 0.0 ;
                        for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                            half4 val2 = Load(&batch_values[vec_offset + static_cast<size_t>(k * 4)]);
                            val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                            val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                        }
                        #pragma unroll
                        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                        }
        
                        if(laneid == 0){
                            work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                            atomicAdd(dis_cal_cnt + bid , 1) ;
                        }
                    }
                    __syncthreads() ;
                    if(tid == 0) {
                        dis_cal_cost += ((double)(clock64() - start) / 1582000) ;
                    }
                    __syncthreads() ;
                    bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                    __syncthreads() ;
                    // 此处需调整, 这里topm中被淘汰的点没有进入top_M_Cand + TOPM 中
                    merge_top_with_recycle_list<attrType>(top_M_Cand , work_mem_unsigned, top_M_Cand_dis , work_mem_float , tid , 128 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim , OP_TYPE) ;
                
                    __syncthreads() ;
                
                    bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                    __syncthreads() ;
                    if(work_mem_unsigned[0] != 0xFFFFFFFF) {
                        merge_top(recycle_list_id , work_mem_unsigned , recycle_list_dis , work_mem_float , tid , 128 , recycle_list_size) ;
                        __syncthreads() ;
                    }
                    graph[turn] = batch_all_graphs + batch_graph_start_index[next_batch_pid] ;
                    pid[turn] = next_pid ;
                    batch_pid[turn] = next_batch_pid ; 
                    if(tid == 0) {
                        cur_visit_p ++ ; 
                    }
                    __syncthreads() ;
                    // 处理另一个channel
                    // channel_id ^= 1 ; 
                    // if(tid == 0) {
                    //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
                    // }
                    continue ;
                }
            }
            __syncthreads() ;
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 对两个通道的扩展写进两个warp里, 防止warp分歧
            if(threadIdx.y < 2) {
                if(to_explore[threadIdx.y & 1] != 0xFFFFFFFF) {
                    unsigned channel_id = threadIdx.y ;
                    unsigned shift_bits = 30 + (channel_id & 1) ;
                    // printf("line : %d , shift_bits : %d\n" , __LINE__ , shift_bits) ;
                    unsigned mask = (batch_pid[0] == batch_pid[1] ? 0 : (1 << shift_bits)) ; 
                    #pragma unroll
                    for(unsigned j = laneid; j < DEGREE - (channel_id == turn ? 0 : 1); j += blockDim.x) {

                        const unsigned *ex_graph = graph[channel_id] ;
                        // printf("to_explore : %d\n" , to_explore[channel_id]) ;
                        // if(tid == 0) {
                        //     printf("ex_graph : [") ;
                        //     for(unsigned i = 0 ; i < 16 ; i ++)
                        //         printf("ex_graph[%d] - %d," , i , ex_graph[i]) ;
                        //     printf("]\n") ;
                        // }
                        unsigned to_append_local = ex_graph[to_explore[channel_id] * DEGREE + j];
                        // printf("to_append_local : %d\n" , to_append_local) ;
                        unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_append_local] ;
                        // printf("to_append : %d\n" , to_append) ;
                        top_M_Cand[TOPM + j + channel_id * DEGREE] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                        top_M_Cand_dis[TOPM + j + channel_id * DEGREE] = (top_M_Cand[TOPM + j + channel_id * DEGREE] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    }
                    if(channel_id != turn && laneid == 31) {
                        unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_explore[channel_id]] ;
                        top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                        top_M_Cand_dis[TOPM + (channel_id + 1) * DEGREE - 1] = (top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                        // if(hash_insert(hash_table , to_append) != 0) {
                        //     top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] = (to_append | mask) ;
                        //     top_M_Cand_dis[TOPM + (channel_id + 1) * DEGREE - 1] = 0.0f ;
                        // }
                    }
                } else {
                    unsigned channel_id = threadIdx.y ;
                    #pragma unroll
                    for(unsigned i = laneid ; i < DEGREE ; i += blockDim.x) {
                        top_M_Cand[TOPM + i + channel_id * DEGREE] = 0xFFFFFFFF ;
                        top_M_Cand_dis[TOPM + i + channel_id * DEGREE] = INF_DIS ;
                    }
                }
                if(tid == 0)
                    turn ^= 1 ; 
            } 
            __syncthreads();
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;
            if(tid == 0) {
                start = clock64() ; 
            }
            __syncthreads() ;

            // 此时无论是通道1还是通道2, 所有节点已在TOPM数组的后半段就位, 因此距离计算可以一视同仁
            #pragma unroll
            for(unsigned i = threadIdx.y; i < DEGREE2 ; i += blockDim.y) {
                if(top_M_Cand[TOPM + i] == 0xFFFFFFFF)
                    continue ;
                
                unsigned pure_id = top_M_Cand[TOPM + i] & 0x3FFFFFFF ;
                size_t vec_offset = static_cast<size_t>(pure_id) * static_cast<size_t>(DIM) ;
                // if(pure_id >= 1000000 && laneid == 0 ) {
                //     printf("error at %d : pure_id = %d , i = %d\n" , __LINE__ , pure_id , i) ;
                    
                // }
                // unsigned stored_pos = global_vec_stored_pos[pure_id] ;
                // if(stored_pos >= 62500 * 4)
                //     printf("error at %d : stored_pos = %d\n" , __LINE__ , stored_pos) ;
            
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                    // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                    half4 val2 = Load(&batch_values[vec_offset + static_cast<size_t>(j * 4)]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    // 将warp内所有子线程的计算结果归并到一起
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }
                if(laneid == 0){
                    top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                    atomicAdd(dis_cal_cnt + bid , 1) ;
                }
            }
            __syncthreads();

            if(tid == 0) {
                dis_cal_cost += ((double)(clock64() - start) / 1582000) ;
            }
            __syncthreads() ;
            // if(tid == 0 && epoch < 10) {
            //     // printf("bid : %d , epoch : %d , degree2 : %d , Tturn : %d\n, " 
            //     // , bid , epoch , DEGREE2, turn) ;
            //     // printf("to_explore[turn] : %d, to_explore[turn ^ 1] : %d, graph0 : %d , graph1 : %d  , pid0 : %d  , pid1 : %d \n前10轮扩展: [" , 
            //     // to_explore[turn] , to_explore[turn ^ 1]  , graph[0] , graph[1] , batch_pid[0] , batch_pid[1]) ;
            //     printf("bid[") ;
            //     for(unsigned i = 0 ; i < DEGREE2  ;i ++) {
            //         printf("(%d)%d - %f" ,i ,  (top_M_Cand[i + TOPM] == 0xFFFFFFFFu ? -1 : top_M_Cand[i + TOPM] & 0x3FFFFFFFu) , (top_M_Cand[i + TOPM] == 0xFFFFFFFFu ? -1.0f : top_M_Cand_dis[i + TOPM])) ;
            //         if(i < DEGREE2 - 1)
            //             printf(", ") ;
            //     }
            //     printf("]\n") ;
            //     printf("bid : %d ,graph0 : %lld , graph1 : %lld , batch_pid[0] : %d , batch_pid[1] : %d, pid[0] : %d , pid[1] : %d\n" , bid , graph[0] , graph[1] ,
            //     batch_pid[0] , batch_pid[1] , pid[0] , pid[1]) ;

            // }
            // __syncthreads() ;
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 排序
            // if(tid < DEGREE2)
            //     warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE2);
            bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], DEGREE2) ;
            __syncthreads() ;
            if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                // 将两个列表归并到一起
                merge_top_with_recycle_list<attrType>(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE2 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim , OP_TYPE) ;
                __syncthreads() ;
                // if(tid < DEGREE2)
                //     warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
                bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], DEGREE2) ;
                __syncthreads() ;
                if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                    merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE2 , recycle_list_size) ;
                    __syncthreads() ;
                }
            }
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;
        }  
        
        // if(tid == 0) {
        //     printf("过滤前的topm[") ;
        //     for(unsigned i = 0 ; i < 10 ; i ++) {
        //         printf("%d-%f" , top_M_Cand[i] & 0x3FFFFFFF , top_M_Cand_dis[i]) ;
        //         if(i < 9)
        //             printf(" ,") ;
        //     }
        //     printf("]\n") ;
        // }
        // if(tid == 0) {
        //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;

        // 搜索完成, 将recycle_list中的内容合并到top_M_Cand中
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
            if(top_M_Cand[i] == 0xFFFFFFFF)
                continue ;
            // bool flag = true ;
            bool flag = (OP_TYPE == 0 ? true : false) ;
            unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
            if(OP_TYPE == 0) {
                // 逻辑与
                for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
                    flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
                }
            } else if(OP_TYPE == 1) {
                // 逻辑或 
                for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
                    flag = (flag || ((attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j]))) ;
                }
            }
            if(!flag) {
                top_M_Cand[i] = 0xFFFFFFFF ;
                top_M_Cand_dis[i] = INF_DIS ;
            }
        }
        __syncthreads() ;
        bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM) ;
        __syncthreads() ;
        // 若recycle_list中确实有点, 则合并
        if(recycle_list_id[0] != 0xFFFFFFFF) {
            merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
            __syncthreads() ;
        }
        // if(tid == 0) {
        //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;
    }

    __shared__ unsigned prefix_sum[1024] ;
    __shared__ unsigned top_prefix[32] ;
    __shared__ unsigned indices_arr[1024] ;
    __shared__ unsigned task_num ; 
    // 处理剩余的小分区
    for(unsigned g = 0 ; g < total_psize_block && little_seg_num_block ; g ++) {
        unsigned batch_gpid = batch_pids[plist_index + g] ;
        unsigned gpid = batch_partition_ids[batch_gpid] ;
        unsigned global_idx_offset = seg_global_idx_start_idx[gpid] ;
        unsigned num_in_seg = batch_graph_size[batch_gpid] ;
        // unsigned num_in_seg = 62464 ;

        #pragma unroll
        for(unsigned i = tid ;i < 32 ; i += blockDim.x * blockDim.y) 
            bit_map[i] = 0 ; 
        #pragma unroll
        for(unsigned i = tid ; i < 1056 ; i += blockDim.x * blockDim.y) {
            work_mem_unsigned[i] = 0xFFFFFFFF ;
            work_mem_float[i] = INF_DIS ;
        }
        if(tid == 0)
            prefilter_pointer = 0 ; 
        __syncthreads() ;
        // if(tid == 0) {
        //     printf("bid : %d , can click %d\n" ,bid ,  __LINE__) ;
        // }
        // __syncthreads() ;

        // 先把 total_psize_block个点分片, 分片大小可以考虑bit_map中1的个数? 
        // selectivity <= 0.1, 意味着, 每十个点里只有一个点满足条件, 找到32个点平均要遍历320个点
        //  有两种方案 :  1. 动态chunk, 尽可能的设置一组中包含32个点
        //               2. 静态chunk 设置 chunk 大小为10 , 但要维护的候选列表长度至少为320
        //  合并次数 : 1. 6250 / 32 次 ; 2. 625 / 32 次
        //  随着合并次数的增多, 方案2的待合并点数量、以及合并次数会显著减少，故采用方案2. 方案1的前置条件太复杂
        // 表示某个点是否满足属性约束条件
        // for(unsigned i = tid ; i < num_in_seg ; i += blockDim.x * blockDim.y) {
        //     bool flag = true ;
        //     unsigned g_global_id = global_idx[global_idx_offset + i] ;
        //     if(g_global_id >= 1e8) {
        //         printf("bid : %d , illegal id : %d\n" , bid , g_global_id) ;
        //     }
        //     for(unsigned j = 0 ; j < attr_dim ; j ++)
        //         flag = (flag && (attrs[g_global_id * attr_dim + j] >= q_l_bound[j]) && (attrs[g_global_id * attr_dim + j] <= q_r_bound[j])) ;
        //     unsigned ballot_res = __ballot_sync(__activemask() , flag) ;
        //     if(laneid == 0)
        //         bit_map[(i >> 5)] = ballot_res ; 
        // }
        // __syncthreads() ;
        // if(tid == 0) {
        //     printf("bid : %d , can click %d\n" , __LINE__) ;
        // }
        // __syncthreads() ;

        unsigned chunk_num = (num_in_seg + 31) >> 5 ;
        unsigned chunk_batch_size = 32 ; 
        for(unsigned chunk_id = 0 ; chunk_id < chunk_num ; chunk_id += chunk_batch_size) {
            unsigned point_batch_size = (chunk_id + chunk_batch_size < chunk_num ? (chunk_batch_size << 5) : num_in_seg - (chunk_id << 5)) ;
            unsigned point_batch_start_id = (chunk_id << 5) ;

            for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                // bool flag = true ;
                bool flag = (OP_TYPE == 0 ? true : false) ;
                unsigned g_global_id = global_idx[global_idx_offset + point_batch_start_id + i] ;
                if(OP_TYPE == 0) {
                    for(unsigned j = 0 ; j < attr_dim ; j ++)
                        flag = (flag && (attrs[g_global_id * attr_dim + j] >= q_l_bound[j]) && (attrs[g_global_id * attr_dim + j] <= q_r_bound[j])) ;
                } else if(OP_TYPE == 1) {
                    for(unsigned j = 0 ; j < attr_dim ; j ++)
                    flag = (flag || ((attrs[g_global_id * attr_dim + j] >= q_l_bound[j]) && (attrs[g_global_id * attr_dim + j] <= q_r_bound[j]))) ;
                }
                unsigned ballot_res = __ballot_sync(__activemask() , flag) ;
                if(laneid == 0)
                    bit_map[(i >> 5)] = ballot_res ; 
            }
            __syncthreads() ;


            // 扫描出该chunk所需要的前缀和数组
            // 先利用__popc() 得到长度为32的前缀和数组, 再利用0号warp, 把32个局部前缀和数组合并为一个整体
            for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                // 在bitmap中的单元格位置
                // unsigned cell_id = chunk_id + (i >> 5) ;
                unsigned cell_id = (i >> 5) ;
                unsigned bit_pos = i % 32 + 1 ;
                unsigned mask = (bit_pos == 32 ? 0xFFFFFFFF : (1 << bit_pos) - 1) ;
                unsigned index = (bit_map[cell_id] & mask) ;
                prefix_sum[i] = __popc(index) ;
            }
            __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" ,bid ,  __LINE__) ;
            // }
            // __syncthreads() ;
            if(tid < 32) {
                unsigned v = (laneid + chunk_id < chunk_num ? __popc(bit_map[laneid]) : 0) ;
                // 将长度大小为32的各个子prefix合并为最终的prefix_sum 
                #pragma unroll
                for(unsigned offset = 1 ; offset < 32 ; offset <<= 1) {
                    unsigned y = __shfl_up_sync(0xFFFFFFFF , v , offset) ;
                    if(laneid >= offset) v += y ;
                }
                top_prefix[laneid] = v ;
            }
            // __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" ,bid ,  __LINE__) ;
            // }
            // __syncthreads() ;
            if(tid == 0) {
                task_num = top_prefix[31] ;
                // if(task_num > 0 && false) {
                //     printf("task num : %d\n" , task_num) ;

                //     printf("[") ;
                //     for(unsigned j = 0 ; j < 1024 ; j ++) {
                //         printf("%d" , prefix_sum[j]) ;
                //         if(j < 1023)
                //             printf(", ") ;
                //     }
                //     printf("]\n") ;

                //     printf("top_prefix :[") ;
                //     for(unsigned j = 0 ; j < 32 ; j ++) {
                //         printf("%d" , top_prefix[j]) ;
                //         if(j < 31)
                //             printf(", ") ;
                //     }
                //     printf("]\n") ;
                // }
                // printf("bid : %d , can click : %d\n" , bid , __LINE__) ;
            }
            __syncthreads() ;
            for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                // unsigned cell_id = chunk_id + (i >> 5) ; 
                unsigned cell_id = (i >> 5) ;
                
                unsigned id_in_cell = i % 32 ;
                unsigned bit_mask = (1 << id_in_cell) ;
                if((bit_map[cell_id] & bit_mask) != 0) {
                    // 获取插入位置
                    // unsigned prefix_sum_fetch_pos = 
                    unsigned local_cell_id = (i >> 5) ;
                    unsigned put_pos = (local_cell_id == 0 ? 0 : top_prefix[local_cell_id - 1]) + (id_in_cell == 0 ? 0 : prefix_sum[i - 1]) ;
                    // if(i == 2) {
                    //     printf("i == 2 : %d , cell_id : %d , id_in_cell : %d\n" , put_pos , cell_id , id_in_cell) ;
                    // }
                    // if(put_pos >= 1024) {

                    //     // printf("bid : %d ,point_batch_size : %d\n" , bid , point_batch_size) ;
                    //     printf("bid : %d ,put_pos : %d , top_prefix[%d - 1] = %d , prefix_sum[%d - 1] = %d\n" , bid, put_pos , cell_id , top_prefix[(i >> 5) - 1] ,
                    //     i , prefix_sum[i - 1]) ;
                    // }
                    indices_arr[put_pos] = i ;
                }
            }
            __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click : %d , chunk_id : %d , chunk_num : %d  \n" , bid , __LINE__ , chunk_id , chunk_num) ;
            // }
            if(tid == 0) {
                start = clock64() ;
            }
            __syncthreads() ;
            // 计算距离, 然后用原子操作插入队列中
            for(unsigned i = threadIdx.y ; i < task_num ; i += blockDim.y) {
                unsigned cand_id = indices_arr[i] + point_batch_start_id ;
                // if(laneid == 0 && chunk_id == 125920) {
                //     printf("bid : %d , cand_id : %d , indices_arr[i] : %d\n" , bid , cand_id , indices_arr[i]) ;
                // }
                // __syncwarp() ;
                // 计算距离
                unsigned g_global_id = global_idx[global_idx_offset + cand_id] ;
                size_t vec_offset = static_cast<size_t>(g_global_id) * static_cast<size_t>(DIM) ;
                // unsigned stored_pos = global_vec_stored_pos[g_global_id] ;
                // if(chunk_id == 125920 ) {
                //     if(laneid == 0)
                //         printf("bid : %d , stored_pos : %d , g_global_id : %d ,DIM : %d , stored_pos * DIM : %d\n" , bid , stored_pos , g_global_id , DIM , stored_pos * DIM ) ;
                // } 
                float dis = 0.0f ;

                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&batch_values[vec_offset + static_cast<size_t>(k * 4)]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    dis = __half2float(val_res.x) + __half2float(val_res.y) ;
                    atomicAdd(dis_cal_cnt + bid , 1) ;
                    // printf("bid : %d , can click : %d dis cal finish!\n" , bid , __LINE__) ;
                }
                __syncwarp() ;
                // 存入work_mem
                if(laneid == 0 && dis < top_M_Cand_dis[K - 1]) {
                    int cur_pointer = atomicAdd(&prefilter_pointer , 1) ;
                    // if(cur_pointer >= 1024)
                    //     printf("bid : %d , error cur_pointer : %d\n" , bid , cur_pointer) ;
                    work_mem_unsigned[cur_pointer] = g_global_id ; 
                    work_mem_float[cur_pointer] = dis ; // 距离变量
                }
                __syncwarp() ;
            }

            __syncthreads() ;
            if(tid == 0) {
                dis_cal_cost += ((double)(clock64() - start) / 1582000) ;
            }
            __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" , bid , __LINE__) ;
            // }
            // __syncthreads() ;
            if(prefilter_pointer >= 32 || (chunk_id + chunk_batch_size >= chunk_num && prefilter_pointer > 0)) {
                // 先对前32个点排序, 然后再合并到topK上
                // if(tid < 32)
                //     warpSort(work_mem_float , work_mem_unsigned , tid , 32) ; 
                bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned, prefilter_pointer) ;
                __syncthreads() ;
                merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float , tid , prefilter_pointer , K) ;
                __syncthreads() ;
                unsigned limit = prefilter_pointer ;
                #pragma unroll
                for(unsigned i = tid ; i < limit ; i += blockDim.x * blockDim.y) {
                    work_mem_unsigned[i] = 0xFFFFFFFF ;
                    work_mem_float[i] = INF_DIS ;
                }
                if(tid == 0) {
                    prefilter_pointer = 0 ; 
                    atomicAdd(post_merge_cnt + bid , 1) ;
                }

                __syncthreads() ;
            }
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" ,bid , __LINE__) ;
            // }
            // __syncthreads() ;
        }
    }
    // if(tid == 0) {
    //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
    // }
    // __syncthreads() ;

    #pragma unroll
    for (unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        work_mem_unsigned[i] = top_M_Cand_video_memory[bid * TOPM + i] ;
        work_mem_float[i] = top_M_Cand_dis_video_memory[bid * TOPM + i] ;
    }
    // if(tid == 0) {
    //     printf("this epoch : id[") ;
    //     for(unsigned i = 0 ; i < TOPM ;i  ++) {
    //         printf("%d-%d ," , top_M_Cand[i] & 0x3FFFFFFF , (top_M_Cand[i] >> 30)) ;
    //     }
    //     printf("]\nthis epoch : dis[") ;
    //     for(unsigned i = 0 ; i < TOPM ;i  ++) {
    //         printf("%f ," , top_M_Cand_dis[i]) ;
    //     }
    //     printf("]\n") ;
    //     // --------------------------------
    //     printf("this epoch work mem : id[") ;
    //     for(unsigned i = 0 ; i < K ;i  ++) {
    //         printf("%d ," , work_mem_unsigned[i] & 0x3FFFFFFF) ;
    //     }
    //     printf("]\nthis epoch work mem : dis[") ;
    //     for(unsigned i = 0 ; i < K ;i  ++) {
    //         printf("%f ," , work_mem_float[i]) ;
    //     }
    //     printf("]\n") ;
    // }
    __syncthreads() ;
    merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float , tid , K , TOPM) ;
    __syncthreads() ;

    // 每次执行完毕, 和现有结果做合并(和前k个结果合并即可)
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        top_M_Cand_video_memory[bid * TOPM + i] = (top_M_Cand[i] & 0x3FFFFFFF);
        top_M_Cand_dis_video_memory[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    for(unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        result_k_id[bid * K + i] = (top_M_Cand[i] & 0x3FFFFFFF) ;
        // result_k_dis[bid * K + i] = top_M_Cand_dis[i] ;
    }
    if(tid == 0) {
        // printf("bid : %d , dis cal cost : %f\n" , bid , dis_cal_cost ) ;
        dis_cal_cost_total_outer[bid] = dis_cal_cost ;
    }
}



template<typename attrType = float>
__global__ void select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter_batch_process_extreme_compressed_graph(const unsigned batch_size , const unsigned* batch_partition_ids , const unsigned* __restrict__ batch_all_graphs, const half* __restrict__ batch_values, const half* __restrict__ query_data, 
    unsigned* __restrict__ top_M_Cand_video_memory , float* __restrict__ top_M_Cand_dis_video_memory, unsigned* ent_pts , unsigned DEGREE , unsigned TOPM , unsigned DIM , const unsigned* batch_graph_start_index ,const unsigned* batch_graph_size ,
    const unsigned* __restrict__ batch_pids , const unsigned* batch_p_size ,const unsigned* batch_p_start_index ,const attrType* l_bound ,const attrType* r_bound ,const attrType* __restrict__ attrs , unsigned attr_dim , const unsigned* __restrict__ batch_global_graph , unsigned edges_num_in_global_graph,
    unsigned recycle_list_size ,const unsigned* __restrict__ global_idx ,const unsigned* __restrict__ local_idx ,const unsigned* seg_global_idx_start_idx , unsigned pnum ,unsigned K, unsigned* result_k_id  ,const unsigned* task_id_mapper , const unsigned* batch_little_seg_counts
    ,const unsigned* global_vec_stored_pos , const unsigned batch_all_point_num , const unsigned* spokesman , const unsigned* spokesman_stored_pos , const unsigned spokesman_num) {
    // *********   当前版本为结束时合并循环池, 后续可酌情实现不合并循环池的版本   ***********
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数

    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    unsigned* recycle_list_id = (unsigned*) shared_mem ;
    float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    unsigned* top_M_Cand = (unsigned*) (recycle_list_dis + recycle_list_size) ;
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE * 2) ;
    attrType* q_l_bound = (attrType*)(top_M_Cand_dis + TOPM + DEGREE * 2) ;
    attrType* q_r_bound = (attrType*)(q_l_bound + attr_dim) ;

    // 这里应该怎么处理8字节的对齐问题？
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem + recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) +
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(attrType)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[1024 + 32] ;
    __shared__ float work_mem_float[1024 + 32] ;
    // 可用投票一次插入32位
    __shared__ unsigned bit_map[32] ;
    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = task_id_mapper[blockIdx.x], laneid = threadIdx.x , warpId = threadIdx.y ;
    // 若线程块发射顺序与blockIdx.x的编号无关, 则可以手动实现按到达顺序取任务的排队器
    /**
        在批处理环境下, 以下变量的语义更改为:
        psize_block - 当前批次中, 查询的正常分区数量, plist_index - 当前批次中, 当前查询的分区列表在总的分区列表中的起始位置
        total_psize_block - 当前批次中, 查询总的分区数量, little_seg_num_block - 当前批次中, 查询的小分区数量
    **/
    unsigned psize_block = batch_p_size[bid] , plist_index = batch_p_start_index[bid] , 
    total_psize_block = batch_p_size[bid] , little_seg_num_block = batch_little_seg_counts[bid] ;

    // if(tid == 0 && bid == 0) {
    //     printf("psize : %d\n" , psize_block) ;
    //     printf("plist : [") ;
    //     for(unsigned i = 0 ; i < total_psize_block ; i ++) 
    //         printf("%d ," , batch_pids[i + plist_index]) ;
    //     printf("]\n") ;
    //     printf("partition_id_list : [") ;
    //     for(unsigned i = 0 ; i < batch_size ; i ++)
    //         printf("%d ," , batch_partition_ids[i]) ;
    //     printf("]\n") ;
    //     printf("batch graph start index : [") ;
    //     for(unsigned i = 0 ; i < batch_size ; i ++)
    //         printf("%d ," , batch_graph_start_index[i]) ;
    //     printf("]\n") ; 
    //     printf("batch partition start index : [") ;
    //     for(unsigned i = 0 ; i < gridDim.x  ; i ++)
    //         printf("%d ," , batch_p_start_index[i]) ;
    //     printf("]\n") ;
    // }
    // __syncthreads() ;
    // 25 不通过 23 不通过 21 不通过 15 不通过 8 通过 11 通过 13 不通过 12 不通过
    // 1 2 5 6 8 通过
    // if(bid != 5)
    //     return ;
    if(total_psize_block == 0) {
        // if(tid == 0) {
        //     printf("当前处理批次为空 bid : %d\n" , bid) ;
        // }
        return ;
    } 
    // else {
    //     if(tid == 0) {
    //         printf("当前处理批次非空 bid : %d\n" , bid) ;
    //     }
    // }

    __shared__ unsigned cur_visit_p , main_channel , turn ;
    __shared__ unsigned to_explore[2];
    __shared__ unsigned prefilter_pointer ;

    __shared__ unsigned hash_table[HASHLEN];
    // 从全局内存中恢复topm
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
        // top_M_Cand[i] = top_M_Cand_video_memory[bid * TOPM + i] | 0xCF000000 ;
        // top_M_Cand_dis[i] = top_M_Cand_dis_video_memory[bid * TOPM + i] ;
    }
    for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
        q_l_bound[i] = l_bound[bid * attr_dim + i] ;
        q_r_bound[i] = r_bound[bid * attr_dim + i] ;
    }
    // 没必要在每个分区开始时都把recycle_list复原, 每个分区得到的入口点肯定是越近越好
    for(unsigned i = tid ; i < recycle_list_size ; i += blockDim.x * blockDim.y) {
        recycle_list_id[i] = 0xFFFFFFFF ;
        recycle_list_dis[i] = INF_DIS ;
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
        tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
    }
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    __syncthreads() ;

    // if(tid == 0) {

    //     // unsigned gt_list[] = {747523 ,3806148 ,4341654 ,3992639 ,82713 ,5593568 ,5855579 ,4733809 ,3100036 ,5833647} ;
    //     unsigned gt_list[] {484111 ,956339 ,1280685 ,1328871 ,1337510 ,1447866 ,2085619 ,2211775 ,2265112 ,2559870} ;
    //     for(unsigned i = 0 ; i < 10 ; i ++)
    //         work_mem_unsigned[i] = gt_list[i] ;

    //     printf("ent_pts[") ;
    //     for(unsigned i = 0 ; i < TOPM  ;i  ++) {
    //         printf("%d" , ent_pts[i]) ;
    //         if(i < TOPM - 1)
    //             printf(" ,") ;
    //     }
    //     printf("]\n") ;
    // }
    // __syncthreads() ;

    // #pragma unroll
    // for(unsigned i = threadIdx.y; i < TOPM ; i += blockDim.y) {
        
        
    //     unsigned pure_id = work_mem_unsigned[i] ;
    //     // unsigned stored_pos = batch_point_start_index[batch_pid[(top_M_Cand[i] >> 31)]] + local_idx[pure_id] ;
    //     unsigned stored_pos = global_vec_stored_pos[pure_id] ;
    //     half2 val_res ;
    //     val_res.x = 0.0 ; val_res.y = 0.0 ;
    //     for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x) {
    //         // warp中的每个子线程负责8个字节 即4个half类型分量的计算
    //         half4 val2 = Load(&batch_values[stored_pos * DIM + j * 4]);
    //         val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
    //         val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
    //     }
    //     #pragma unroll
    //     for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2) {   
    //         // 将warp内所有子线程的计算结果归并到一起
    //         val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
    //     }
    //     if(laneid == 0) {
    //         work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
    //         // atomicAdd(dis_cal_cnt + bid , 1) ;
    //     }
    // }
    // __syncthreads() ;
    // if(tid == 0) {
    //     printf("gt id and dis[") ;
    //     for(unsigned i = 0 ; i < 10 ; i ++) {

    //         printf("%d-%f" , work_mem_unsigned[i] , work_mem_float[i]) ;
    //         if(i < 9)
    //             printf(" ,") ;
    //     }
    //     printf("]\n") ;
    // }
    // __syncthreads() ;

    
    if(psize_block > 0 && little_seg_num_block == 0) {
        psize_block = total_psize_block ;
        /**
            batch_pids[i] 第i个分区, 用的是 0 - batchSize 之间的局部id
            batch_graph_start_index[i] 当前batch 第i个分区所代表的图
        **/
        // if(plist_index >= 12) {
        //     if(tid == 0)
        //         printf("error at %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;
        unsigned batch_pid[2] = {batch_pids[plist_index] , (psize_block >= 2 ? batch_pids[plist_index + 1] : batch_pids[plist_index])} ;
        unsigned pid[2] = {batch_partition_ids[batch_pid[0]] , batch_partition_ids[batch_pid[1]]} ;
        // pid[0] = 9 ; 
        const unsigned* graph[2] = {batch_all_graphs + batch_graph_start_index[batch_pid[0]] , batch_all_graphs + batch_graph_start_index[batch_pid[1]]};
        const unsigned DEGREE2 = DEGREE * 2 ;
        // if(tid == 0) {
        //     printf("bid : %d , batch_pid0 : %d , batch_pid1 : %d , pid0 : %d , pid1 : %d\n" , bid , batch_pid[0] , batch_pid[1] , pid[0] , pid[1]) ;
        // }

        // 有两种可能, 没有局部结果的从入口点扩展, 有局部结果的先从局部结果扩展, 若局部结果中点个数不够从入口点补充

        // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
        for(unsigned i = tid , L2 = TOPM / 2 ; i < TOPM; i += blockDim.x * blockDim.y){
            unsigned shift_bits = 30 +  i / L2;
            unsigned mask = (batch_pid[0] == batch_pid[1] ? 0 : 1 << shift_bits) ; 
            // 用反向掩码, 将另一位置1
            top_M_Cand[i] = global_idx[seg_global_idx_start_idx[pid[i / L2]] + ent_pts[i]] | mask;
            // top_M_Cand[TOPM + DEGREE + i]

            // 哈希表存放的是局部id, 如何统一?  是否可以直接存放全局id? 
            hash_insert(hash_table , top_M_Cand[i] & 0x3FFFFFFF) ;
        }

        // 初始化入口节点
        #pragma unroll
        for(unsigned i = threadIdx.y; i < TOPM ; i += blockDim.y) {
            if(top_M_Cand[i] == 0xFFFFFFFF)
                continue ;
            // threadIdx.y 0-5 , blockDim.y 即 6
            
            unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
            // unsigned stored_pos = batch_point_start_index[batch_pid[(top_M_Cand[i] >> 31)]] + local_idx[pure_id] ;
            unsigned stored_pos = global_vec_stored_pos[pure_id] ;
            half2 val_res ;
            val_res.x = 0.0 ; val_res.y = 0.0 ;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x) {
                // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                half4 val2 = Load(&batch_values[stored_pos * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2) {   
                // 将warp内所有子线程的计算结果归并到一起
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }
            if(laneid == 0) {
                top_M_Cand_dis[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                atomicAdd(dis_cal_cnt + bid , 1) ;
            }
        }
        __syncthreads();
        bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis , top_M_Cand , TOPM) ;
        __syncthreads() ;
        // merge_top_with_recycle_list(top_M_Cand , work_mem_unsigned, top_M_Cand_dis , work_mem_float , tid , enter_pts_num , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
        // __syncthreads() ;
        // if(tid == 0) {
        //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;


        if(tid == 0) {
            // printf("psize block : %d\n" , psize_block) ;
            // printf("pid[0] %d , pid[1] %d\n" , pid[0] , pid[1]) ;
            // printf("[") ;
            // for(int i = 0; i < psize_block ; i ++) {
            //     printf("%d" , pids[plist_index + i]) ;
            //     if(i < psize_block - 1)
            //         printf(", ") ;
            // }
            // printf("]\n") ;

            cur_visit_p = (psize_block >= 2 ? 2 : 1) ;
            main_channel = 0 ; 
            turn = 0 ; 
            // psize_block = total_psize_block ;
            // printf("can click 2023\n") ;
            // printf("can click %d\n" , __LINE__) ;

            // printf("ent_pts-dis[") ;
            // for(unsigned i = 0 ; i < TOPM  ;i  ++) {
            //     printf("%d-%f" , top_M_Cand[i] , top_M_Cand_dis[i]) ;
            //     if(i < TOPM - 1)
            //         printf(" ,") ;
            // }
            // printf("]\n") ;

            // printf("ent pts : [") ;
            // for(unsigned i = 0 ; i < TOPM ; i ++) {
            //     printf("%d - %f" , top_M_Cand[i] & 0x3FFFFFFFu , top_M_Cand_dis[i]) ;
            //     if(i < TOPM -1)
            //         printf(", ") ;
            // }
            // printf("]\n") ;
        }
        __syncthreads();
        
        
        
        

        // 无限次循环, 直到退出
        for(unsigned epoch = 0 ; epoch < MAX_ITER ; epoch ++) {

            // 每隔4轮将哈希表重置一次
            if((epoch + 1) % (HASH_RESET) == 0) {
                #pragma unroll
                for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                    hash_table[j] = 0xFFFFFFFF;
                }
                __syncthreads();
                #pragma unroll
                for(unsigned j = tid; j < TOPM + DEGREE2; j += blockDim.x * blockDim.y){
                    // 将此时的TOPM + K 个候选点插入哈希表中
                    if(top_M_Cand[j] != 0xFFFFFFFF) {
                        hash_insert(hash_table , top_M_Cand[j] & 0x3FFFFFFF) ; 
                    }
                }
            }
            __syncthreads();
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 下面进行点扩展
            if(tid < 32){
                /*
                    扩展规则: 找到当前turn通道的最近点, 然后从另一个通道找到距离该点最近的点, 进行扩展
                */
                if(laneid < 2)
                    to_explore[laneid] = 0xFFFFFFFF ;
                __syncwarp() ;

                unsigned shift_bits = (batch_pid[0] == batch_pid[1] ? 31 - (main_channel & 1)  : 31 - (turn & 1));
                unsigned mask = (1 << shift_bits) ;

            
                for(unsigned j = laneid; j < TOPM; j+=32){
                    unsigned n_p = ((top_M_Cand[j] & mask) == 0 ? 1 : 0) ;
                
                
                    unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
                    if(ballot_res > 0) {

                        if(laneid == (__ffs(ballot_res) - 1)) {
                            to_explore[turn & 1] = local_idx[top_M_Cand[j] & 0x3FFFFFFF] ;
                            top_M_Cand[j] |= mask ;
                        }
                        break ;
                    }
                }
                __syncwarp() ;
                
                if(to_explore[turn] != 0xFFFFFFFF) {
                    /**
                        unsigned global_to_explore = global_idx[seg_global_idx_start_idx[pid[turn]] + to_explore[turn]] ;
                        unsigned stored_pos = spokesman_stored_pos[spokesman[global_to_explore]] ;
                        unsigned offset = spokesman_num * edges_num_in_global_graph * batch_pid[turn ^ 1] + stored_pos * edges_num_in_global_graph ; 
                    
                        for(unsigned j = laneid ; j < 2; j += 32) {
                            unsigned n_p = (!hash_peek(hash_table , batch_global_graph[offset + j])) ;
                            unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;

                            if(ballot_res > 0) {
                                if(laneid == (__ffs(ballot_res) - 1)) {
                                    to_explore[turn ^ 1] = batch_global_graph[offset + j] ;
                                }
                                break ; 
                            }
                        }
                    **/ 
                    for(int l = -1 ; l < (int) DEGREE ; l ++) {

                        unsigned explore_pos = (l == -1 ? to_explore[turn] : graph[turn][to_explore[turn] * DEGREE + l]) ;
                        // unsigned global_to_explore = global_idx[seg_global_idx_start_idx[pid[turn]] + to_explore[turn]] ;
                        unsigned global_to_explore = global_idx[seg_global_idx_start_idx[pid[turn]] + explore_pos] ;
                        unsigned stored_pos = spokesman_stored_pos[spokesman[global_to_explore]] ;
                        unsigned offset = spokesman_num * edges_num_in_global_graph * batch_pid[turn ^ 1] + stored_pos * edges_num_in_global_graph ; 
                        
                        unsigned exit_flag = false ;
                        for(unsigned j = laneid ; j < 2; j += 32) {
                            unsigned n_p = (!hash_peek(hash_table , batch_global_graph[offset + j])) ;
                            unsigned ballot_res = __ballot_sync(__activemask() , n_p) ;
    
                            if(ballot_res > 0) {
                                exit_flag = true ; 
                                if(laneid == (__ffs(ballot_res) - 1)) {
                                    to_explore[turn ^ 1] = batch_global_graph[offset + j] ;
                                }
                                break ; 
                            }
                        }
                        if(exit_flag)
                            break ;
                    }
                }
            }
            __syncthreads() ;
            // if(tid == 0 && epoch < 3) {
            //     printf("epoch : %d , bid : %d , batch_pid0 : %d , batch_pid1 : %d\n" , epoch , bid , batch_pid[0] , batch_pid[1]) ;
            // }
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;


            // 若扩展失败, 切换图栈
            if(to_explore[turn & 1] == 0xFFFFFFFF) {
                
                if(cur_visit_p == psize_block && batch_pid[0] == batch_pid[1]) {
                    // 两个图都没有扩展点, 且没有图可以切换, 退出
                    if(tid == 0) {
                        if(bid == 0)
                            printf("block %d: break in %d\n" ,bid ,  epoch) ; 
                        jump_num_cnt[bid] = epoch ; 
                    }
                    break ;
                } else if(cur_visit_p == psize_block) {
                        graph[turn] = graph[turn ^ 1] ;
                        pid[turn] = pid[turn ^ 1] ;
                        batch_pid[turn] = batch_pid[turn ^ 1] ;
                        if(tid == 0) {
                            main_channel = turn ^ 1 ; 
                        }
                    continue ;
                } else {
                    
                    unsigned next_batch_pid = batch_pids[plist_index + cur_visit_p] ;
                    unsigned next_pid = batch_partition_ids[next_batch_pid] ;
                    unsigned shift_bits = 30 + (turn & 1) ;
                    unsigned mask = 1 << shift_bits ;
                    
                    // 如何从全局id得到存储位置?
                    /**
                    // 重置哈希表
                    #pragma unroll
                    for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
                
                        if(top_M_Cand[i % 64] == 0xFFFFFFFF) {
                            // 填充一个随机数
                            work_mem_unsigned[i] = 0xFFFFFFFF ;
                            work_mem_float[i] = 0.0f ;
                            continue ;
                        }
                        // 用前两个填充, 有重复的部分皆替换为随机点
                        unsigned pure_id = top_M_Cand[i % 64] & 0x3FFFFFFF ;
                        // unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / 64] ;
                        // unsigned stored_pos = global_vec_stored_pos[pure_id] ;    
                        // unsigned stored_pos = global_vec_stored_pos[spokesman[pure_id]] ;
                        unsigned stored_pos = spokesman_stored_pos[spokesman[pure_id]] ;
                        // unsigned offset = batch_all_point_num * edges_num_in_global_graph * next_batch_pid + stored_pos * edges_num_in_global_graph ;
                        unsigned offset = spokesman_num * edges_num_in_global_graph * next_batch_pid + stored_pos * edges_num_in_global_graph ;
                        // printf("bid : %d , offset : %d , line : %d\n" , bid , offset , __LINE__) ;
                        // if(offset > 112500000) {
                        //     if(laneid == 0)
                        //     printf("bid : %d , batch_all_point_num : %d , edge_num_in_global_graph : %d , batch_pid[turn ^ 1] : %d , stored_pos : %d \n" 
                        //     , bid , batch_all_point_num , edges_num_in_global_graph , batch_pid[turn ^ 1] , stored_pos) ;
                        // }
                        // unsigned next_local_id = batch_global_graph[stored_pos * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / 64] ; 
                        unsigned next_local_id = batch_global_graph[offset + i / 64] ;
                        unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                        work_mem_unsigned[i] = (hash_insert(hash_table , next_global_id) != 0 ? next_global_id | mask : 0xFFFFFFFF) ;
                        work_mem_float[i] = (work_mem_unsigned[i] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    } **/
                    
                    // 重置哈希表
                    #pragma unroll
                    for(unsigned i = tid ; i < 64 ; i += blockDim.x * blockDim.y) { 
                
                        // if(top_M_Cand[i % 64] == 0xFFFFFFFF) {
                        //     // 填充一个随机数
                        //     work_mem_unsigned[i] = 0xFFFFFFFF ;
                        //     work_mem_float[i] = 0.0f ;
                        //     continue ;
                        // }
                        work_mem_unsigned[i * 2] = work_mem_unsigned[i * 2 + 1] = 0xFFFFFFFFu ;
                        work_mem_float[i * 2] = work_mem_float[i * 2 + 1] = INF_DIS ;

                        // 从邻居的列表中找, 找到为止
                        unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
                        for(int l = -1 ; l < (int) DEGREE ; l ++) {
                            // printf("1111111\n") ;
                            // unsigned spokesman_id = (l == -1 ? spokesman[pure_id] : global_idx[seg_global_idx_start_idx[]])
                            unsigned spokesman_id ;
                            if(l == -1)
                                spokesman_id = spokesman[pure_id] ;
                            else {
                                // 扩展top_M_Cand[i] 的邻居
                                unsigned o_stored_pos = global_vec_stored_pos[pure_id] ;
                                unsigned o_pid = o_stored_pos / batch_graph_size[batch_pid[turn]] ;
                                if(o_pid >= batch_size)
                                    o_pid = batch_size - 1 ;
                                unsigned o_local_id = local_idx[pure_id] ; 
                                spokesman_id = spokesman[global_idx[seg_global_idx_start_idx[o_pid] + (batch_all_graphs + batch_graph_start_index[o_pid])[o_local_id * DEGREE + l]]] ;
                                printf("bid : %d , l : %d , cur_visit_p : %d , vid : %d , spokesman_id : %d\n" , bid , l , cur_visit_p , o_local_id , spokesman_id) ;
                            }


                            unsigned stored_pos = spokesman_stored_pos[spokesman_id] ;
                            unsigned offset = spokesman_num * edges_num_in_global_graph * next_batch_pid + stored_pos * edges_num_in_global_graph ;
                            for(unsigned nt = 0 ; nt < 2 ; nt ++) {
                                unsigned next_local_id = batch_global_graph[offset + nt] ;
                                unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                                work_mem_unsigned[i * 2 + nt] = (hash_insert(hash_table , next_global_id) != 0 ? next_global_id | mask : 0xFFFFFFFF) ;
                                work_mem_float[i * 2 + nt] = (work_mem_unsigned[i] != 0xFFFFFFFF ? 0.0f : INF_DIS) ; 
                            }
                            if(work_mem_unsigned[i * 2] != 0xFFFFFFFFu || work_mem_unsigned[i * 2 + 1] != 0xFFFFFFFFu)
                                break ;
                        }
                    }

                    __syncthreads() ;

                    // 距离计算
                    #pragma unroll
                    for(unsigned i = threadIdx.y ; i < 128 ; i += blockDim.y) {
                        if(work_mem_unsigned[i] == 0xFFFFFFFF) {
                            continue ;  
                        }
                        
                        half2 val_res;
                        unsigned pure_id = work_mem_unsigned[i] & 0x3FFFFFFF ;
                        unsigned stored_pos = global_vec_stored_pos[pure_id] ;
                        val_res.x = 0.0; val_res.y = 0.0 ;
                        for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                            half4 val2 = Load(&batch_values[stored_pos * DIM + k * 4]);
                            val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                            val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                        }
                        #pragma unroll
                        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                        }
        
                        if(laneid == 0){
                            work_mem_float[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                            atomicAdd(dis_cal_cnt + bid , 1) ;
                        }
                    }
                    __syncthreads() ;
                    bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                    __syncthreads() ;


                    if(tid == 0 && bid == 0) {
                        unsigned cnt =0 ;
                        for(unsigned c = 0 ; c < 128 ; c ++)
                            if(work_mem_unsigned[c] != 0xFFFFFFFFu && work_mem_float[c] < top_M_Cand_dis[TOPM - 1])
                                cnt ++ ;
                        printf("bid : %d , cur_visit_p : %d , 本轮扩展 : %d\n" , bid , cur_visit_p , cnt) ;
                    }
                    __syncthreads() ;

                    // 此处需调整, 这里topm中被淘汰的点没有进入top_M_Cand + TOPM 中
                    merge_top_with_recycle_list<attrType>(top_M_Cand , work_mem_unsigned, top_M_Cand_dis , work_mem_float , tid , 128 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
                
                    __syncthreads() ;
                
                    bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned , 128) ;
                    __syncthreads() ;
                    if(work_mem_unsigned[0] != 0xFFFFFFFF) {
                        merge_top(recycle_list_id , work_mem_unsigned , recycle_list_dis , work_mem_float , tid , 128 , recycle_list_size) ;
                        __syncthreads() ;
                    }
                    graph[turn] = batch_all_graphs + batch_graph_start_index[next_batch_pid] ;
                    pid[turn] = next_pid ;
                    batch_pid[turn] = next_batch_pid ; 
                    if(tid == 0) {
                        cur_visit_p ++ ; 
                    }
                    __syncthreads() ;
                    // 处理另一个channel
                    // channel_id ^= 1 ; 
                    // if(tid == 0) {
                    //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
                    // }
                    continue ;
                }
            }
            __syncthreads() ;
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 对两个通道的扩展写进两个warp里, 防止warp分歧
            if(threadIdx.y < 2) {
                if(to_explore[threadIdx.y & 1] != 0xFFFFFFFF) {
                    unsigned channel_id = threadIdx.y ;
                    unsigned shift_bits = 30 + (channel_id & 1) ;
                    // printf("line : %d , shift_bits : %d\n" , __LINE__ , shift_bits) ;
                    unsigned mask = (batch_pid[0] == batch_pid[1] ? 0 : (1 << shift_bits)) ; 
                    #pragma unroll
                    for(unsigned j = laneid; j < DEGREE - (channel_id == turn ? 0 : 1); j += blockDim.x) {

                        const unsigned *ex_graph = graph[channel_id] ;
                        // printf("to_explore : %d\n" , to_explore[channel_id]) ;
                        // if(tid == 0) {
                        //     printf("ex_graph : [") ;
                        //     for(unsigned i = 0 ; i < 16 ; i ++)
                        //         printf("ex_graph[%d] - %d," , i , ex_graph[i]) ;
                        //     printf("]\n") ;
                        // }
                        unsigned to_append_local = ex_graph[to_explore[channel_id] * DEGREE + j];
                        // printf("to_append_local : %d\n" , to_append_local) ;
                        unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_append_local] ;
                        // printf("to_append : %d\n" , to_append) ;
                        top_M_Cand[TOPM + j + channel_id * DEGREE] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                        top_M_Cand_dis[TOPM + j + channel_id * DEGREE] = (top_M_Cand[TOPM + j + channel_id * DEGREE] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    }
                    if(channel_id != turn && laneid == 31) {
                        unsigned to_append = global_idx[seg_global_idx_start_idx[pid[channel_id]] + to_explore[channel_id]] ;
                        top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] = (hash_insert(hash_table , to_append) == 0 ? 0xFFFFFFFF : (to_append | mask)) ;
                        top_M_Cand_dis[TOPM + (channel_id + 1) * DEGREE - 1] = (top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                        // if(hash_insert(hash_table , to_append) != 0) {
                        //     top_M_Cand[TOPM + (channel_id + 1) * DEGREE - 1] = (to_append | mask) ;
                        //     top_M_Cand_dis[TOPM + (channel_id + 1) * DEGREE - 1] = 0.0f ;
                        // }
                    }
                } else {
                    unsigned channel_id = threadIdx.y ;
                    #pragma unroll
                    for(unsigned i = laneid ; i < DEGREE ; i += blockDim.x) {
                        top_M_Cand[TOPM + i + channel_id * DEGREE] = 0xFFFFFFFF ;
                        top_M_Cand_dis[TOPM + i + channel_id * DEGREE] = INF_DIS ;
                    }
                }
                if(tid == 0)
                    turn ^= 1 ; 
            } 
            __syncthreads();
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 此时无论是通道1还是通道2, 所有节点已在TOPM数组的后半段就位, 因此距离计算可以一视同仁
            #pragma unroll
            for(unsigned i = threadIdx.y; i < DEGREE2 ; i += blockDim.y) {
                if(top_M_Cand[TOPM + i] == 0xFFFFFFFF)
                    continue ;
                
                unsigned pure_id = top_M_Cand[TOPM + i] & 0x3FFFFFFF ;

                // if(pure_id >= 1000000 && laneid == 0 ) {
                //     printf("error at %d : pure_id = %d , i = %d\n" , __LINE__ , pure_id , i) ;
                    
                // }
                unsigned stored_pos = global_vec_stored_pos[pure_id] ;
                // if(stored_pos >= 62500 * 4)
                //     printf("error at %d : stored_pos = %d\n" , __LINE__ , stored_pos) ;
            
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                    // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                    half4 val2 = Load(&batch_values[stored_pos * DIM + j * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    // 将warp内所有子线程的计算结果归并到一起
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }
                if(laneid == 0){
                    top_M_Cand_dis[TOPM + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
                    atomicAdd(dis_cal_cnt + bid , 1) ;
                }
            }
            __syncthreads();

            // if(tid == 0 && epoch < 10) {
            //     // printf("bid : %d , epoch : %d , degree2 : %d , Tturn : %d\n, " 
            //     // , bid , epoch , DEGREE2, turn) ;
            //     // printf("to_explore[turn] : %d, to_explore[turn ^ 1] : %d, graph0 : %d , graph1 : %d  , pid0 : %d  , pid1 : %d \n前10轮扩展: [" , 
            //     // to_explore[turn] , to_explore[turn ^ 1]  , graph[0] , graph[1] , batch_pid[0] , batch_pid[1]) ;
            //     printf("bid[") ;
            //     for(unsigned i = 0 ; i < DEGREE2  ;i ++) {
            //         printf("(%d)%d - %f" ,i ,  (top_M_Cand[i + TOPM] == 0xFFFFFFFFu ? -1 : top_M_Cand[i + TOPM] & 0x3FFFFFFFu) , (top_M_Cand[i + TOPM] == 0xFFFFFFFFu ? -1.0f : top_M_Cand_dis[i + TOPM])) ;
            //         if(i < DEGREE2 - 1)
            //             printf(", ") ;
            //     }
            //     printf("]\n") ;
            //     printf("bid : %d ,graph0 : %lld , graph1 : %lld , batch_pid[0] : %d , batch_pid[1] : %d, pid[0] : %d , pid[1] : %d\n" , bid , graph[0] , graph[1] ,
            //     batch_pid[0] , batch_pid[1] , pid[0] , pid[1]) ;

            // }
            // __syncthreads() ;
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;

            // 排序
            // if(tid < DEGREE2)
            //     warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE2);
            bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], DEGREE2) ;
            __syncthreads() ;
            if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                // 将两个列表归并到一起
                merge_top_with_recycle_list<attrType>(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE2 , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
                __syncthreads() ;
                // if(tid < DEGREE2)
                //     warpSort(top_M_Cand_dis + TOPM , top_M_Cand + TOPM , tid , DEGREE2) ; 
                bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], DEGREE2) ;
                __syncthreads() ;
                if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                    merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE2 , recycle_list_size) ;
                    __syncthreads() ;
                }
            }
            // if(tid == 0) {
            //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
            // }
            // __syncthreads() ;
        }  
        
        // if(tid == 0) {
        //     printf("----this epoch : id[") ;
        //     for(unsigned i = 0 ; i < TOPM ;i  ++) {
        //         printf("%d-%d ," , top_M_Cand[i] & 0x3FFFFFFF , (top_M_Cand[i] >> 30)) ;
        //     }
        //     printf("]\n----this epoch : dis[") ;
        //     for(unsigned i = 0 ; i < TOPM ;i  ++) {
        //         printf("%f ," , top_M_Cand_dis[i]) ;
        //     }
        //     printf("]\n") ;
        // }
        // __syncthreads() ;

        // if(tid == 0) {
        //     printf("过滤前的topm[") ;
        //     for(unsigned i = 0 ; i < 10 ; i ++) {
        //         printf("%d-%f" , top_M_Cand[i] & 0x3FFFFFFF , top_M_Cand_dis[i]) ;
        //         if(i < 9)
        //             printf(" ,") ;
        //     }
        //     printf("]\n") ;
        // }
        // if(tid == 0) {
        //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;

        // 搜索完成, 将recycle_list中的内容合并到top_M_Cand中
        for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y) {
            if(top_M_Cand[i] == 0xFFFFFFFF)
                continue ;
            bool flag = true ;
            unsigned pure_id = top_M_Cand[i] & 0x3FFFFFFF ;
            for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
                flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
            }
            if(!flag) {
                top_M_Cand[i] = 0xFFFFFFFF ;
                top_M_Cand_dis[i] = INF_DIS ;
            }
        }
        __syncthreads() ;
        bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM) ;
        __syncthreads() ;
        // 若recycle_list中确实有点, 则合并
        if(recycle_list_id[0] != 0xFFFFFFFF) {
            merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
            __syncthreads() ;
        }
        // if(tid == 0) {
        //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
        // }
        // __syncthreads() ;
    }

    __shared__ unsigned prefix_sum[1024] ;
    __shared__ unsigned top_prefix[32] ;
    __shared__ unsigned indices_arr[1024] ;
    __shared__ unsigned task_num ; 
    // 处理剩余的小分区
    for(unsigned g = 0 ; g < total_psize_block && little_seg_num_block ; g ++) {
        unsigned batch_gpid = batch_pids[plist_index + g] ;
        unsigned gpid = batch_partition_ids[batch_gpid] ;
        unsigned global_idx_offset = seg_global_idx_start_idx[gpid] ;
        unsigned num_in_seg = batch_graph_size[batch_gpid] ;
        // unsigned num_in_seg = 62464 ;

        #pragma unroll
        for(unsigned i = tid ;i < 32 ; i += blockDim.x * blockDim.y) 
            bit_map[i] = 0 ; 
        #pragma unroll
        for(unsigned i = tid ; i < 1056 ; i += blockDim.x * blockDim.y) {
            work_mem_unsigned[i] = 0xFFFFFFFF ;
            work_mem_float[i] = INF_DIS ;
        }
        if(tid == 0)
            prefilter_pointer = 0 ; 
        __syncthreads() ;
        // if(tid == 0) {
        //     printf("bid : %d , can click %d\n" ,bid ,  __LINE__) ;
        // }
        // __syncthreads() ;

        // 先把 total_psize_block个点分片, 分片大小可以考虑bit_map中1的个数? 
        // selectivity <= 0.1, 意味着, 每十个点里只有一个点满足条件, 找到32个点平均要遍历320个点
        //  有两种方案 :  1. 动态chunk, 尽可能的设置一组中包含32个点
        //               2. 静态chunk 设置 chunk 大小为10 , 但要维护的候选列表长度至少为320
        //  合并次数 : 1. 6250 / 32 次 ; 2. 625 / 32 次
        //  随着合并次数的增多, 方案2的待合并点数量、以及合并次数会显著减少，故采用方案2. 方案1的前置条件太复杂
        // 表示某个点是否满足属性约束条件
        // for(unsigned i = tid ; i < num_in_seg ; i += blockDim.x * blockDim.y) {
        //     bool flag = true ;
        //     unsigned g_global_id = global_idx[global_idx_offset + i] ;
        //     if(g_global_id >= 1e8) {
        //         printf("bid : %d , illegal id : %d\n" , bid , g_global_id) ;
        //     }
        //     for(unsigned j = 0 ; j < attr_dim ; j ++)
        //         flag = (flag && (attrs[g_global_id * attr_dim + j] >= q_l_bound[j]) && (attrs[g_global_id * attr_dim + j] <= q_r_bound[j])) ;
        //     unsigned ballot_res = __ballot_sync(__activemask() , flag) ;
        //     if(laneid == 0)
        //         bit_map[(i >> 5)] = ballot_res ; 
        // }
        // __syncthreads() ;
        // if(tid == 0) {
        //     printf("bid : %d , can click %d\n" , __LINE__) ;
        // }
        // __syncthreads() ;

        unsigned chunk_num = (num_in_seg + 31) >> 5 ;
        unsigned chunk_batch_size = 32 ; 
        for(unsigned chunk_id = 0 ; chunk_id < chunk_num ; chunk_id += chunk_batch_size) {
            unsigned point_batch_size = (chunk_id + chunk_batch_size < chunk_num ? (chunk_batch_size << 5) : num_in_seg - (chunk_id << 5)) ;
            unsigned point_batch_start_id = (chunk_id << 5) ;

            for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                bool flag = true ;
                unsigned g_global_id = global_idx[global_idx_offset + point_batch_start_id + i] ;
                for(unsigned j = 0 ; j < attr_dim ; j ++)
                    flag = (flag && (attrs[g_global_id * attr_dim + j] >= q_l_bound[j]) && (attrs[g_global_id * attr_dim + j] <= q_r_bound[j])) ;
                unsigned ballot_res = __ballot_sync(__activemask() , flag) ;
                if(laneid == 0)
                    bit_map[(i >> 5)] = ballot_res ; 
            }
            __syncthreads() ;


            // 扫描出该chunk所需要的前缀和数组
            // 先利用__popc() 得到长度为32的前缀和数组, 再利用0号warp, 把32个局部前缀和数组合并为一个整体
            for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                // 在bitmap中的单元格位置
                // unsigned cell_id = chunk_id + (i >> 5) ;
                unsigned cell_id = (i >> 5) ;
                unsigned bit_pos = i % 32 + 1 ;
                unsigned mask = (bit_pos == 32 ? 0xFFFFFFFF : (1 << bit_pos) - 1) ;
                unsigned index = (bit_map[cell_id] & mask) ;
                prefix_sum[i] = __popc(index) ;
            }
            __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" ,bid ,  __LINE__) ;
            // }
            // __syncthreads() ;
            if(tid < 32) {
                unsigned v = (laneid + chunk_id < chunk_num ? __popc(bit_map[laneid]) : 0) ;
                // 将长度大小为32的各个子prefix合并为最终的prefix_sum 
                #pragma unroll
                for(unsigned offset = 1 ; offset < 32 ; offset <<= 1) {
                    unsigned y = __shfl_up_sync(0xFFFFFFFF , v , offset) ;
                    if(laneid >= offset) v += y ;
                }
                top_prefix[laneid] = v ;
            }
            // __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" ,bid ,  __LINE__) ;
            // }
            // __syncthreads() ;
            if(tid == 0) {
                task_num = top_prefix[31] ;
                // if(task_num > 0 && false) {
                //     printf("task num : %d\n" , task_num) ;

                //     printf("[") ;
                //     for(unsigned j = 0 ; j < 1024 ; j ++) {
                //         printf("%d" , prefix_sum[j]) ;
                //         if(j < 1023)
                //             printf(", ") ;
                //     }
                //     printf("]\n") ;

                //     printf("top_prefix :[") ;
                //     for(unsigned j = 0 ; j < 32 ; j ++) {
                //         printf("%d" , top_prefix[j]) ;
                //         if(j < 31)
                //             printf(", ") ;
                //     }
                //     printf("]\n") ;
                // }
                // printf("bid : %d , can click : %d\n" , bid , __LINE__) ;
            }
            __syncthreads() ;
            for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                // unsigned cell_id = chunk_id + (i >> 5) ; 
                unsigned cell_id = (i >> 5) ;
                
                unsigned id_in_cell = i % 32 ;
                unsigned bit_mask = (1 << id_in_cell) ;
                if((bit_map[cell_id] & bit_mask) != 0) {
                    // 获取插入位置
                    // unsigned prefix_sum_fetch_pos = 
                    unsigned local_cell_id = (i >> 5) ;
                    unsigned put_pos = (local_cell_id == 0 ? 0 : top_prefix[local_cell_id - 1]) + (id_in_cell == 0 ? 0 : prefix_sum[i - 1]) ;
                    // if(i == 2) {
                    //     printf("i == 2 : %d , cell_id : %d , id_in_cell : %d\n" , put_pos , cell_id , id_in_cell) ;
                    // }
                    // if(put_pos >= 1024) {

                    //     // printf("bid : %d ,point_batch_size : %d\n" , bid , point_batch_size) ;
                    //     printf("bid : %d ,put_pos : %d , top_prefix[%d - 1] = %d , prefix_sum[%d - 1] = %d\n" , bid, put_pos , cell_id , top_prefix[(i >> 5) - 1] ,
                    //     i , prefix_sum[i - 1]) ;
                    // }
                    indices_arr[put_pos] = i ;
                }
            }
            __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click : %d , chunk_id : %d , chunk_num : %d  \n" , bid , __LINE__ , chunk_id , chunk_num) ;
            // }
 
            // 计算距离, 然后用原子操作插入队列中
            for(unsigned i = threadIdx.y ; i < task_num ; i += blockDim.y) {
                unsigned cand_id = indices_arr[i] + point_batch_start_id ;
                // if(laneid == 0 && chunk_id == 125920) {
                //     printf("bid : %d , cand_id : %d , indices_arr[i] : %d\n" , bid , cand_id , indices_arr[i]) ;
                // }
                // __syncwarp() ;
                // 计算距离
                unsigned g_global_id = global_idx[global_idx_offset + cand_id] ;
                unsigned stored_pos = global_vec_stored_pos[g_global_id] ;
                // if(chunk_id == 125920 ) {
                //     if(laneid == 0)
                //         printf("bid : %d , stored_pos : %d , g_global_id : %d ,DIM : %d , stored_pos * DIM : %d\n" , bid , stored_pos , g_global_id , DIM , stored_pos * DIM ) ;
                // } 
                float dis = 0.0f ;

                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&batch_values[stored_pos * DIM + k * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    dis = __half2float(val_res.x) + __half2float(val_res.y) ;
                    atomicAdd(dis_cal_cnt + bid , 1) ;
                    // printf("bid : %d , can click : %d dis cal finish!\n" , bid , __LINE__) ;
                }
                __syncwarp() ;
                // 存入work_mem
                if(laneid == 0 && dis < top_M_Cand_dis[K - 1]) {
                    int cur_pointer = atomicAdd(&prefilter_pointer , 1) ;
                    // if(cur_pointer >= 1024)
                    //     printf("bid : %d , error cur_pointer : %d\n" , bid , cur_pointer) ;
                    work_mem_unsigned[cur_pointer] = g_global_id ; 
                    work_mem_float[cur_pointer] = dis ; // 距离变量
                }
                __syncwarp() ;
            }

            __syncthreads() ;
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" , bid , __LINE__) ;
            // }
            // __syncthreads() ;
            if(prefilter_pointer >= 32 || (chunk_id + chunk_batch_size >= chunk_num && prefilter_pointer > 0)) {
                // 先对前32个点排序, 然后再合并到topK上
                // if(tid < 32)
                //     warpSort(work_mem_float , work_mem_unsigned , tid , 32) ; 
                bitonic_sort_id_by_dis_no_explore(work_mem_float , work_mem_unsigned, prefilter_pointer) ;
                __syncthreads() ;
                merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float , tid , prefilter_pointer , K) ;
                __syncthreads() ;
                unsigned limit = prefilter_pointer ;
                #pragma unroll
                for(unsigned i = tid ; i < limit ; i += blockDim.x * blockDim.y) {
                    work_mem_unsigned[i] = 0xFFFFFFFF ;
                    work_mem_float[i] = INF_DIS ;
                }
                if(tid == 0) {
                    prefilter_pointer = 0 ; 
                    atomicAdd(post_merge_cnt + bid , 1) ;
                }

                __syncthreads() ;
            }
            // if(tid == 0) {
            //     printf("bid : %d , can click %d\n" ,bid , __LINE__) ;
            // }
            // __syncthreads() ;
        }
    }
    // if(tid == 0) {
    //     printf("can click %d , bid : %d\n" , __LINE__ , bid) ;
    // }
    // __syncthreads() ;

    #pragma unroll
    for (unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        work_mem_unsigned[i] = top_M_Cand_video_memory[bid * TOPM + i] ;
        work_mem_float[i] = top_M_Cand_dis_video_memory[bid * TOPM + i] ;
    }
    // if(tid == 0) {
    //     printf("this epoch : id[") ;
    //     for(unsigned i = 0 ; i < TOPM ;i  ++) {
    //         printf("%d-%d ," , top_M_Cand[i] & 0x3FFFFFFF , (top_M_Cand[i] >> 30)) ;
    //     }
    //     printf("]\nthis epoch : dis[") ;
    //     for(unsigned i = 0 ; i < TOPM ;i  ++) {
    //         printf("%f ," , top_M_Cand_dis[i]) ;
    //     }
    //     printf("]\n") ;
    //     // --------------------------------
    //     printf("this epoch work mem : id[") ;
    //     for(unsigned i = 0 ; i < K ;i  ++) {
    //         printf("%d ," , work_mem_unsigned[i] & 0x3FFFFFFF) ;
    //     }
    //     printf("]\nthis epoch work mem : dis[") ;
    //     for(unsigned i = 0 ; i < K ;i  ++) {
    //         printf("%f ," , work_mem_float[i]) ;
    //     }
    //     printf("]\n") ;
    // }
    __syncthreads() ;
    merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float , tid , K , TOPM) ;
    __syncthreads() ;

    // 每次执行完毕, 和现有结果做合并(和前k个结果合并即可)
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        top_M_Cand_video_memory[bid * TOPM + i] = (top_M_Cand[i] & 0x3FFFFFFF);
        top_M_Cand_dis_video_memory[bid * TOPM + i] = top_M_Cand_dis[i];
    }
    for(unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
        result_k_id[bid * K + i] = (top_M_Cand[i] & 0x3FFFFFFF) ;
        // result_k_dis[bid * K + i] = top_M_Cand_dis[i] ;
    }
}



typedef pair<float , unsigned> pfu ;
typedef priority_queue<pfu> pqfu ;
vector<unsigned> naive_brute_force(float* q , float* data_ , unsigned dim , unsigned num , unsigned k) {
    pqfu pq ; 
    for(unsigned i = 0 ; i < num ; i ++) {
        float dis = 0.0 ;
        for(unsigned j = 0 ;j  < dim ; j ++)
            dis += (q[j] - data_[i * dim + j]) * (q[j] - data_[i * dim + j]) ;
        dis = sqrt(dis) ;
        if(pq.size() < k || pq.top().first > dis) {
            pq.push({dis , i}) ;
            if(pq.size() > k)
                pq.pop() ; 
        }
    }
    vector<unsigned> res ;
    while(!pq.empty()) {
        res.push_back(pq.top().second) ;
        pq.pop() ;
    }
    return res ;
}

vector<unsigned> naive_brute_force_with_filter(float* q , float* data_ , unsigned dim , unsigned num , unsigned k ,
    float* l_bound , float* r_bound , float* attrs , unsigned attr_dim) {
    pqfu pq ; 
    for(unsigned i = 0 ; i < num ; i ++) {
        // 先判断一下是否满足过滤条件
        bool flag = true ;
        for(unsigned j =0 ; j < attr_dim && flag ; j ++) {
            flag = (flag && (attrs[i * attr_dim + j] >= l_bound[j]) && (attrs[i * attr_dim + j] <= r_bound[j])) ;
        }
        if(!flag)
            continue ;
        float dis = 0.0 ;
        for(unsigned j = 0 ;j  < dim ; j ++)
            dis += (q[j] - data_[i * dim + j]) * (q[j] - data_[i * dim + j]) ;
        dis = sqrt(dis) ;
        if(pq.size() < k || pq.top().first > dis) {
            pq.push({dis , i}) ;
            if(pq.size() > k)
                pq.pop() ; 
        }
    }
    vector<unsigned> res ;
    while(!pq.empty()) {
        res.push_back(pq.top().second) ;
        pq.pop() ;
    }
    // while(res.size() < k) {
    //     res.push_back(0xFFFFFFFFu) ;
    // }
    return res ;
}

template<typename attrType = float>
vector<unsigned> naive_brute_force_with_filter_store_dis(float* q , float* data_ , unsigned dim , unsigned num , unsigned k ,
    attrType* l_bound , attrType* r_bound , attrType* attrs , unsigned attr_dim , vector<float>& result_dis) {
    pqfu pq ; 
    for(unsigned i = 0 ; i < num ; i ++) {
        // 先判断一下是否满足过滤条件
        bool flag = true ;
        for(unsigned j =0 ; j < attr_dim && flag ; j ++) {
            flag = (flag && (attrs[i * attr_dim + j] >= l_bound[j]) && (attrs[i * attr_dim + j] <= r_bound[j])) ;
        }
        if(!flag)
            continue ;
        float dis = 0.0 ;
        size_t offset = static_cast<size_t>(i) * static_cast<size_t>(dim) ;
        for(unsigned j = 0 ;j  < dim ; j ++)
            dis += (q[j] - data_[offset + static_cast<size_t>(j)]) * (q[j] - data_[offset + static_cast<size_t>(j)]) ;
        dis = sqrt(dis) ;
        if(pq.size() < k || pq.top().first > dis) {
            pq.push({dis , i}) ;
            if(pq.size() > k)
                pq.pop() ; 
        }
    }
    vector<unsigned> res ;
    while(!pq.empty()) {
        res.push_back(pq.top().second) ;
        result_dis.push_back(pq.top().first) ;
        pq.pop() ;
    }
    return res ;
}

// 计算recall
float cal_recall(vector<vector<unsigned>>& target_res , vector<vector<unsigned>>& gt , bool flag = true) {
    float total_recall = 0.0 ;

    for(unsigned i = 0 ; i < gt.size() ; i ++) {

        vector<unsigned> vv1 = target_res[i] , vv2 = gt[i] ;
        sort(vv1.begin(),vv1.end());   
        sort(vv2.begin(),vv2.end());
        // sort(target_res[i].begin() , target_res[i].end()) ;
        // sort(gt[i].begin() , gt[i].end()) ;
        vector<unsigned> result ; 
        set_intersection(vv1.begin() , vv1.end() , vv2.begin() , vv2.end() , back_inserter(result)) ;
        total_recall += (result.size() * 1.0) / (gt[i].size() * 1.0) ;

        float recall = (result.size() * 1.0) / (gt[i].size() * 1.0) ; 
        if(isnan(recall) || gt[i].size() == 0) {
            cout <<   "tar_res.size() : " << target_res[i].size() <<  ", result.size() : " << result.size() << ", gt.size() : " << gt[i].size() << endl ;
        }

        if(recall < 0.6 && flag) {
        // if(true) {
            cout << "q" << i << ": recall : " << recall << endl ;
            cout << "A[" ;
            for(unsigned j = 0 ;j < target_res[i].size() ; j ++) {
                cout << target_res[i][j] ;
                if(j < target_res[i].size() - 1)
                    cout << " ," ;
            }
            cout << "]" << endl ;
            cout << "B[" ;
            for(unsigned j = 0 ;j < gt[i].size() ; j ++) {
                cout << gt[i][j] ;
                if(j < gt[i].size() - 1)
                    cout << " ," ;
            }
            cout << "]" << endl ;
        }

        
    }
    // if(isnan(total_recall))
    //     cout << "3155 !!!" << endl ; 
    total_recall /= gt.size() ;
    // if(isnan(total_recall))
    //     cout << "3158!!!" << endl ;
    // cout << "in function cal_recall : " << total_recall << endl
    return total_recall ;
}

float cal_recall_show_partition(vector<vector<unsigned>>& target_res , vector<vector<unsigned>>& gt , vector<pair<unsigned,unsigned>>& query_range_x , 
    vector<pair<unsigned,unsigned>>& query_range_y , float x_width , float y_width, bool flag = true) {
    float total_recall = 0.0 ;

    for(unsigned i = 0 ; i < gt.size() ; i ++) {

        sort(target_res[i].begin() , target_res[i].end()) ;
        sort(gt[i].begin() , gt[i].end()) ;
        vector<unsigned> result ; 
        set_intersection(target_res[i].begin() , target_res[i].end() , gt[i].begin() , gt[i].end() , back_inserter(result)) ;
        // total_recall += (result.size() * 1.0) / (gt[i].size() * 1.0) ;

        float recall = (gt[i].size() > 0 ? (result.size() * 1.0) / (gt[i].size() * 1.0) : 1.0f) ; 
        total_recall += recall ;
        if(isnan(recall) || gt[i].size() == 0) {
            cout << "q" << i << ": " ;
            cout <<   "tar_res.size() : " << target_res[i].size() <<  ", result.size() : " << result.size() << ", gt.size() : " << gt[i].size() << endl ;
            cout << "A[" ;
            for(unsigned j = 0 ;j < target_res[i].size() ; j ++) {
                cout << target_res[i][j] ;
                if(j < target_res[i].size() - 1)
                    cout << " ," ;
            }
            cout << "]" << endl ;
        }
        if(recall < 0.6 && flag) {
        // if(true) {
            unsigned begin_par = query_range_x[i].first / x_width, end_par = min((unsigned)3, (unsigned)(query_range_x[i].second / x_width));
            unsigned y_begin_par = query_range_y[i].first / y_width , y_end_par = min((unsigned) 3 ,(unsigned)(query_range_y[i].second / y_width)) ;
            cout << "q" << i << ": recall : " << recall << endl ;
            cout << "x(" << begin_par << "," << end_par << ") , y(" << y_begin_par << "," << y_end_par << ")" << endl ;
            cout << "A[" ;
            for(unsigned j = 0 ;j < target_res[i].size() ; j ++) {
                cout << target_res[i][j] ;
                if(j < target_res[i].size() - 1)
                    cout << " ," ;
            }
            cout << "]" << endl ;
            cout << "B[" ;
            for(unsigned j = 0 ;j < gt[i].size() ; j ++) {
                cout << gt[i][j] ;
                if(j < gt[i].size() - 1)
                    cout << " ," ;
            }
            cout << "]" << endl ;
        }

        
    }
    if(isnan(total_recall))
        cout << __FILE__ << ": 3155 !!! nan come !!!!" << endl ; 
    total_recall /= gt.size() ;
    if(isnan(total_recall))
        cout << __FILE__ << ": 3158!!! nan come !!!!" << endl ;
    cout << "in function cal_recall : " << total_recall << endl ;
    return total_recall ;
}

/*
    根据属性建图代码(后面完成):
    1. 统计出某一维属性的最大值和最小值, 然后将这一维度均匀的分割成l个分区
    2. 根据维度, 分配某一点所在的分区
    
    搜索大致流程如下: 
    1. 根据范围约束, 筛选出几个分区
    2. 按照聚类统计, 得到分区的访问顺序
    3. 逐次访问分区, 第一个分区入口点随机(或由聚类得到), 后续分区为前序分区结果扩展
*/

__device__ __forceinline__ unsigned get_partition_id(unsigned* partition_indexes , unsigned* attr_grid_size , unsigned attr_dim) {
    unsigned pid = 0 ;
    unsigned factor = 1 ;
    for(int i = attr_dim - 1 ; i >= 0 ; i --) {
        pid += partition_indexes[i] * factor ;
        factor *= attr_grid_size[i] ;
    }
    return pid ; 
}


// 返回bool类型数组, 然后使用thrust规约
template<unsigned ATTR_DIM=2>
__global__ void intersection_area_batch(float* l_bound, float* r_bound , float* attr_min , float* attr_width , unsigned* attr_grid_size , unsigned attr_dim , 
    unsigned qnum , bool* is_selected , unsigned pnum) {
    // 每个线程处理一个查询
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x  , total_num = blockDim.x * gridDim.x ;
    
    for(unsigned i = tid ; i < qnum * pnum ; i += total_num)
        is_selected[i] = false ;
    __syncthreads() ;
    // unsigned* partition_indexes = new unsigned[attr_dim] ;
    // unsigned* partition_l = new unsigned[attr_dim] ;
    // unsigned* partition_r = new unsigned[attr_dim] ;
    unsigned partition_indexes[ATTR_DIM] ;
    unsigned partition_l[ATTR_DIM] ;
    unsigned partition_r[ATTR_DIM] ;
    unsigned attr_grid_size_register[ATTR_DIM] ;
    for(unsigned i = 0 ; i < ATTR_DIM ;i  ++)
        attr_grid_size_register[i] = attr_grid_size[i] ;
    for(unsigned i = tid ; i < qnum ; i += total_num) {
        //得出每个维度的范围
        for(unsigned j = 0 ; j < attr_dim ; j ++) {
            partition_l[j] = (unsigned)((l_bound[i * attr_dim + j] - attr_min[j]) / attr_width[j]) ;
            partition_r[j] = (unsigned)((r_bound[i * attr_dim + j] - attr_min[j]) / attr_width[j]) ;
            if(partition_r[j] >= attr_grid_size[j])
                partition_r[j] = attr_grid_size[j] - 1 ;    
            partition_indexes[j] = partition_l[j] ;
        }

        // 枚举attr_dim个范围中的所有组合
        while(true) {
            // printf("tid : %d, attr_grid_size[0] : %d , attr_grid_size[1] : %d , attr_dim : %d\n" , tid , attr_grid_size[0] , attr_grid_size[1] , attr_dim) ;
            unsigned pid = get_partition_id(partition_indexes , attr_grid_size_register , attr_dim) ;
            
            is_selected[i * pnum + pid] = true ; 
            int pos = 0 ; 
            while(pos < attr_dim) {
                partition_indexes[pos] ++ ;
                if(partition_indexes[pos] <= partition_r[pos])
                    break ; 
                else {
                    partition_indexes[pos] = partition_l[pos] ;
                    pos ++ ;
                }
            }
            if(pos >= attr_dim)
                break ; 
        }
    }
    // delete [] partition_indexes ;
    // delete [] partition_l ;
    // delete [] partition_r ; 
}

// 返回bool类型数组, 然后使用thrust规约
template<unsigned ATTR_DIM=2>
__global__ void intersection_area_batch_logicalor(float* l_bound, float* r_bound , float* attr_min , float* attr_width , unsigned* attr_grid_size , unsigned attr_dim , 
    unsigned qnum , bool* is_selected , unsigned pnum) {
    // 每个线程处理一个查询
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x  , total_num = blockDim.x * gridDim.x ;
    
    for(unsigned i = tid ; i < qnum * pnum ; i += total_num)
        is_selected[i] = false ;
    __syncthreads() ;
    // unsigned* partition_indexes = new unsigned[attr_dim] ;
    // unsigned* partition_l = new unsigned[attr_dim] ;
    // unsigned* partition_r = new unsigned[attr_dim] ;
    unsigned partition_indexes[ATTR_DIM] ;
    unsigned partition_l[ATTR_DIM] ;
    unsigned partition_r[ATTR_DIM] ;
    unsigned attr_grid_size_register[ATTR_DIM] ;
    unsigned partition_l_mask[ATTR_DIM] ;
    unsigned partition_r_mask[ATTR_DIM] ;
    for(unsigned i = 0 ; i < ATTR_DIM ;i  ++)
        attr_grid_size_register[i] = attr_grid_size[i] ;
    for(unsigned i = tid ; i < qnum ; i += total_num) {
        //得出每个维度的范围
        for(unsigned j = 0 ; j < attr_dim ; j ++) {
            partition_l[j] = (unsigned)((l_bound[i * attr_dim + j] - attr_min[j]) / attr_width[j]) ;
            partition_r[j] = (unsigned)((r_bound[i * attr_dim + j] - attr_min[j]) / attr_width[j]) ;
            if(partition_r[j] >= attr_grid_size[j])
                partition_r[j] = attr_grid_size[j] - 1 ;    
            partition_indexes[j] = partition_l[j] ;
        }

        // partition_l 每个维度上的左界,  partition_r 每个维度上的右界
        for(unsigned j = 0 ;j < attr_dim ; j ++) {
            for(unsigned k = 0 ; k < attr_dim ; k ++)
                partition_l_mask[k] = 0 , partition_r_mask[k] = attr_grid_size[k] - 1 ;
            partition_l_mask[j] = partition_l[j] , partition_r_mask[j] = partition_r[j] ;
            for(unsigned k = 0 ; k < attr_dim ; k ++)
                partition_indexes[k] = partition_l_mask[k] ;
            // 枚举attr_dim个范围中的所有组合
            while(true) {
                // printf("tid : %d, attr_grid_size[0] : %d , attr_grid_size[1] : %d , attr_dim : %d\n" , tid , attr_grid_size[0] , attr_grid_size[1] , attr_dim) ;
                unsigned pid = get_partition_id(partition_indexes , attr_grid_size_register , attr_dim) ;
                
                is_selected[i * pnum + pid] = true ; 
                int pos = 0 ; 
                while(pos < attr_dim) {
                    partition_indexes[pos] ++ ;
                    if(partition_indexes[pos] <= partition_r_mask[pos])
                        break ; 
                    else {
                        partition_indexes[pos] = partition_l_mask[pos] ;
                        pos ++ ;
                    }
                }
                if(pos >= attr_dim)
                    break ; 
            }
        }
    }
    // delete [] partition_indexes ;
    // delete [] partition_l ;
    // delete [] partition_r ; 
}

// 返回bool类型数组, 然后使用thrust规约
template<unsigned ATTR_DIM=2 , typename attrType = float>
__global__ void intersection_area_batch_mark_little_seg(attrType* l_bound, attrType* r_bound , attrType* attr_min , attrType* attr_width , unsigned* attr_grid_size , unsigned attr_dim , 
    unsigned qnum , bool* is_selected , unsigned pnum, bool* is_little_seg , double little_seg_threshold) {
    // 每个线程处理一个查询
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x  , total_num = blockDim.x * gridDim.x ;
    
    for(unsigned i = tid ; i < qnum * pnum ; i += total_num) 
       is_little_seg[i] = is_selected[i] = false ;
    __syncthreads() ;
    // unsigned* partition_indexes = new unsigned[attr_dim] ;
    // unsigned* partition_l = new unsigned[attr_dim] ;
    // unsigned* partition_r = new unsigned[attr_dim] ;
    unsigned partition_indexes[ATTR_DIM] ;
    unsigned partition_l[ATTR_DIM] ;
    unsigned partition_r[ATTR_DIM] ;
    unsigned attr_grid_size_register[ATTR_DIM] ;
    // unsigned l_bound_register[ATTR_DIM] ;
    // unsigned r_bound_register[ATTR_DIM] ;
    for(unsigned i = 0 ; i < ATTR_DIM ;i  ++) {
        attr_grid_size_register[i] = attr_grid_size[i] ;
    }
    __syncthreads() ;
    // if(tid == 0) {
    //     printf("r_bound[") ;
    //     for(unsigned i = 0 ; i < attr_dim ; i ++) {
    //         printf("%f" , r_bound[i]) ;
    //         if(i < attr_dim - 1)
    //             printf(" ,") ;
    //     }
    //     printf("]\n") ;
    //     printf("attr_width[") ;
    //     for(unsigned i = 0 ; i < attr_dim ; i ++) {
    //         printf("%d" , attr_width[i]) ;
    //         if(i < attr_dim - 1)
    //             printf(" ,") ;
    //     }
    //     printf("]\n") ;
    //     printf("(rb - am) / attr_width[") ;
    //     for(unsigned i = 0 ; i < attr_dim ; i ++) {
    //         unsigned iidx = (unsigned)((r_bound[i] - attr_min[i]) / attr_width[i]) ;
    //         printf("(%f - %f) / %f = %d" ,r_bound[i] , attr_min[i] ,  attr_width[i] ,iidx ) ;
    //         if(i < attr_dim - 1)
    //             printf(" ,") ;
    //     }
    //     printf("]\n") ;
    // }
    // __syncthreads() ;
    for(unsigned i = tid ; i < qnum ; i += total_num) {
        //得出每个维度的范围
        for(unsigned j = 0 ; j < attr_dim ; j ++) {
            partition_l[j] = (unsigned)((l_bound[i * attr_dim + j] - attr_min[j]) / attr_width[j]) ;
            partition_r[j] = (unsigned)((r_bound[i * attr_dim + j] - attr_min[j]) / attr_width[j]) ;
            if(partition_r[j] >= attr_grid_size[j])
                partition_r[j] = attr_grid_size[j] - 1 ;    
            partition_indexes[j] = partition_l[j] ;
        }

        // 枚举attr_dim个范围中的所有组合
        while(true) {
            // printf("tid : %d, attr_grid_size[0] : %d , attr_grid_size[1] : %d , attr_dim : %d\n" , tid , attr_grid_size[0] , attr_grid_size[1] , attr_dim) ;
            unsigned pid = get_partition_id(partition_indexes , attr_grid_size_register , attr_dim) ;
            
            is_selected[i * pnum + pid] = true ; 
            double area = 1.0f ;
            // 计算得出当前分区与查询的交叉面积
            for(unsigned j = 0 ; j < attr_dim ;  j++) {
                double step = (double) partition_indexes[j] ;
                // 若在左界的右边或者在
                double lb = max((double)attr_min[j] + step * (double)attr_width[j] , (double)l_bound[i * attr_dim + j]) ;
                double rb = min((double)attr_min[j] + (step + 1.0) * (double)attr_width[j] , (double)r_bound[i * attr_dim + j]) ;
                area *= (rb - lb) / attr_width[j] ;
            }

            is_little_seg[i * pnum + pid] = (area <= little_seg_threshold) ;

            // 得到下一个分区号
            int pos = 0 ; 
            while(pos < attr_dim) {
                partition_indexes[pos] ++ ;
                if(partition_indexes[pos] <= partition_r[pos])
                    break ; 
                else {
                    partition_indexes[pos] = partition_l[pos] ;
                    pos ++ ;
                }
            }
            if(pos >= attr_dim)
                break ; 
        }
    }
    // delete [] partition_indexes ;
    // delete [] partition_l ;
    // delete [] partition_r ; 
}

// 返回bool类型数组, 然后使用thrust规约
template<unsigned ATTR_DIM=2 , typename attrType = float>
__global__ void intersection_area_batch_mark_little_seg_logicalor(attrType* l_bound, attrType* r_bound , attrType* attr_min , attrType* attr_width , unsigned* attr_grid_size , unsigned attr_dim , 
    unsigned qnum , bool* is_selected , unsigned pnum, bool* is_little_seg , double little_seg_threshold) {
    // 每个线程处理一个查询
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x  , total_num = blockDim.x * gridDim.x ;
    
    for(unsigned i = tid ; i < qnum * pnum ; i += total_num) 
       is_little_seg[i] = is_selected[i] = false ;
    __syncthreads() ;
    // unsigned* partition_indexes = new unsigned[attr_dim] ;
    // unsigned* partition_l = new unsigned[attr_dim] ;
    // unsigned* partition_r = new unsigned[attr_dim] ;
    unsigned partition_indexes[ATTR_DIM] ;
    unsigned partition_l[ATTR_DIM] ;
    unsigned partition_r[ATTR_DIM] ;
    unsigned attr_grid_size_register[ATTR_DIM] ;
    unsigned partition_l_mask[ATTR_DIM] ;
    unsigned partition_r_mask[ATTR_DIM] ;
    // unsigned l_bound_register[ATTR_DIM] ;
    // unsigned r_bound_register[ATTR_DIM] ;
    for(unsigned i = 0 ; i < ATTR_DIM ;i  ++) {
        attr_grid_size_register[i] = attr_grid_size[i] ;
    }
    __syncthreads() ;
    // if(tid == 0) {
    //     printf("r_bound[") ;
    //     for(unsigned i = 0 ; i < attr_dim ; i ++) {
    //         printf("%f" , r_bound[i]) ;
    //         if(i < attr_dim - 1)
    //             printf(" ,") ;
    //     }
    //     printf("]\n") ;
    //     printf("attr_width[") ;
    //     for(unsigned i = 0 ; i < attr_dim ; i ++) {
    //         printf("%d" , attr_width[i]) ;
    //         if(i < attr_dim - 1)
    //             printf(" ,") ;
    //     }
    //     printf("]\n") ;
    //     printf("(rb - am) / attr_width[") ;
    //     for(unsigned i = 0 ; i < attr_dim ; i ++) {
    //         unsigned iidx = (unsigned)((r_bound[i] - attr_min[i]) / attr_width[i]) ;
    //         printf("(%f - %f) / %f = %d" ,r_bound[i] , attr_min[i] ,  attr_width[i] ,iidx ) ;
    //         if(i < attr_dim - 1)
    //             printf(" ,") ;
    //     }
    //     printf("]\n") ;
    // }
    // __syncthreads() ;
    for(unsigned i = tid ; i < qnum ; i += total_num) {
        //得出每个维度的范围
        for(unsigned j = 0 ; j < attr_dim ; j ++) {
            partition_l[j] = (unsigned)((l_bound[i * attr_dim + j] - attr_min[j]) / attr_width[j]) ;
            partition_r[j] = (unsigned)((r_bound[i * attr_dim + j] - attr_min[j]) / attr_width[j]) ;
            if(partition_r[j] >= attr_grid_size[j])
                partition_r[j] = attr_grid_size[j] - 1 ;    
            partition_indexes[j] = partition_l[j] ;
        }

        // partition_l 每个维度上的左界,  partition_r 每个维度上的右界
        for(unsigned j = 0 ;j < attr_dim ; j ++) {
            for(unsigned k = 0 ; k < attr_dim ; k ++)
                partition_l_mask[k] = 0 , partition_r_mask[k] = attr_grid_size[k] - 1 ;
            partition_l_mask[j] = partition_l[j] , partition_r_mask[j] = partition_r[j] ;
            for(unsigned k = 0 ; k < attr_dim ; k ++)
                partition_indexes[k] = partition_l_mask[k] ;
            // 枚举attr_dim个范围中的所有组合
            while(true) {
                // printf("tid : %d, attr_grid_size[0] : %d , attr_grid_size[1] : %d , attr_dim : %d\n" , tid , attr_grid_size[0] , attr_grid_size[1] , attr_dim) ;
                unsigned pid = get_partition_id(partition_indexes , attr_grid_size_register , attr_dim) ;
                
                is_selected[i * pnum + pid] = true ; 

                attrType area = 1.0f ;
                // 计算得出当前分区与查询的交叉面积
                for(unsigned k = 0 ; k < attr_dim ;  k++) {
                    attrType step = (attrType) partition_indexes[k] ;
                    // 若在左界的右边或者在
                    attrType lb = max(attr_min[k] + step * attr_width[k] , l_bound[i * attr_dim + k]) ;
                    attrType rb = min(attr_min[k] + (step + 1) * attr_width[k] , r_bound[i * attr_dim + k]) ;
                    area *= (rb - lb) / attr_width[k] ;
                }
    
                is_little_seg[i * pnum + pid] = (area <= little_seg_threshold) ;

                int pos = 0 ; 
                while(pos < attr_dim) {
                    partition_indexes[pos] ++ ;
                    if(partition_indexes[pos] <= partition_r_mask[pos])
                        break ; 
                    else {
                        partition_indexes[pos] = partition_l_mask[pos] ;
                        pos ++ ;
                    }
                }
                if(pos >= attr_dim)
                    break ; 
            }
        }
    }
    // delete [] partition_indexes ;
    // delete [] partition_l ;
    // delete [] partition_r ; 
}



// 根据聚类的信息, 对分区的访问顺序进行排序
__global__ void sort_by_clustering_info(unsigned* partitions , unsigned* p_counts , unsigned* p_offsets , unsigned pnum , unsigned cnum , unsigned qnum , 
    unsigned* cluster_p_info , unsigned* topk_clusters , unsigned topk_cluster_num) {
        // 存放分区的id号, 和分区和聚类相交点个数
        extern __shared__ unsigned char shared_mem[] ;
        unsigned *pids = (unsigned*) shared_mem ;
        unsigned *pids_nums = (unsigned*) (pids + pnum) ;
        unsigned *topk_clusters_shared = (unsigned*) (pids_nums + pnum) ;
        unsigned tid = blockDim.x * threadIdx.y + threadIdx.x , total_num = blockDim.x * blockDim.y ;
        unsigned cur_partition_pnum = p_counts[blockIdx.x] , cur_partition_offset = p_offsets[blockIdx.x] ;
        // 将分区号导入共享内存
        for(unsigned i = tid ; i < cur_partition_pnum ; i += total_num)  {
            pids[i] = partitions[cur_partition_offset + i] ;
            pids_nums[i] = 0 ;
        }
        for(unsigned i = tid ; i < topk_cluster_num ; i += total_num)
            topk_clusters_shared[i] = topk_clusters[blockIdx.x * topk_cluster_num + i] ;
        __syncthreads() ;

        // 将数量累加到pids_nums 上
        // unsigned v = 0 ; 
        // i 为 第i个分区
        for(unsigned i = threadIdx.y ; i < cur_partition_pnum ; i += blockDim.y) {
            unsigned v = 0 ; 
            for(unsigned j = threadIdx.x ; j < topk_cluster_num ; j += blockDim.x) {
                v += cluster_p_info[pids[i] * cnum + topk_clusters_shared[j]] ;
            }
            __syncwarp() ;
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                v += __shfl_down_sync(0xffffffff, v, lane_mask);
            }
            if(threadIdx.x == 0)
                pids_nums[i] = v ; 
            __syncwarp() ;
        }
        __syncthreads() ;
        // sort by key 
        bitonic_sort_id_by_num(pids_nums, pids, cur_partition_pnum) ;
        __syncthreads() ;
        for(unsigned i = tid ;i < cur_partition_pnum ; i += total_num) {
            partitions[cur_partition_offset + i] = pids[i] ;
            // partition_nums_device[cur_partition_offset + i] = pids_nums[i] ;
        }
}

// 根据聚类的信息, 对分区的访问顺序进行排序
__global__ void sort_by_clustering_info_weighted(unsigned* partitions , unsigned* p_counts , unsigned* p_offsets , unsigned* litlte_seg_counts , bool* is_little_seg,
 unsigned pnum , unsigned cnum , unsigned qnum , 
    unsigned* cluster_p_info , unsigned* topk_clusters , unsigned topk_cluster_num , unsigned cardinality , unsigned* p_ls_counts) {
        // 存放分区的id号, 和分区和聚类相交点个数
        extern __shared__ unsigned char shared_mem[] ;
        unsigned *pids = (unsigned*) shared_mem ;
        int *pids_nums = (int*) (pids + pnum) ;
        unsigned *topk_clusters_shared = (unsigned*) (pids_nums + pnum) ;
        unsigned tid = blockDim.x * threadIdx.y + threadIdx.x , total_num = blockDim.x * blockDim.y , bid = blockIdx.x ;
       

        unsigned cur_partition_pnum = p_counts[blockIdx.x] , cur_partition_offset = p_offsets[blockIdx.x] ;
        unsigned cur_little_seg_num = p_ls_counts[blockIdx.x] ;

       
        // 将分区号导入共享内存
        for(unsigned i = tid ; i < cur_partition_pnum ; i += total_num)  {
            pids[i] = partitions[cur_partition_offset + i] ;
            pids_nums[i] = 0 ;
        }
 
        for(unsigned i = tid ; i < topk_cluster_num ; i += total_num)
            topk_clusters_shared[i] = topk_clusters[blockIdx.x * topk_cluster_num + i] ;
        __syncthreads() ;
       
   
     
        bool flag = (cur_little_seg_num == cur_partition_pnum) ; 
       
        for(unsigned i = threadIdx.y ; i < cur_partition_pnum ; i += blockDim.y) {
           
            int v = 0 ; 
            for(unsigned j = threadIdx.x ; j < topk_cluster_num ; j += blockDim.x) {
                v += cluster_p_info[pids[i] * cnum + topk_clusters_shared[j]] ;
            }
            // __syncwarp() ;
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                v += __shfl_down_sync(0xffffffff, v, lane_mask);
            }
            if(threadIdx.x == 0) {
                pids_nums[i] = v - (int)((flag && is_little_seg[cur_partition_offset + i]) ? cardinality : 0) ; 
            }
            // __syncwarp() ;
        }
        __syncthreads() ;
        bitonic_sort_id_by_num_int(pids_nums, pids, cur_partition_pnum) ;
        __syncthreads() ;
        
        for(unsigned i = tid ;i < cur_partition_pnum ; i += total_num) {
            partitions[cur_partition_offset + i] = pids[i] ;
            // partition_nums_device[cur_partition_offset + i] = pids_nums[i] ;
        }
}

__global__ void bitonic_sort_groups(float* __restrict__ dis_arr ,unsigned* __restrict__ ids_arr, const unsigned group_size , const unsigned group_num) {
    extern __shared__ unsigned char shared_mem[] ;
    float *dis = (float*) shared_mem ;
    unsigned *ids = (unsigned*) (dis + group_size) ;

    unsigned tid = threadIdx.y * blockDim.x + threadIdx.x ;
    // #pragma unroll
    for(unsigned i = tid ; i < group_size ;  i ++) {
        dis[i] = dis_arr[blockIdx.x * group_size + i] ;
        ids[i] = ids_arr[blockIdx.x * group_size + i] ;
    }
    __syncthreads() ;
    bitonic_sort_id_by_dis_no_explore(dis, ids, group_size) ;
    __syncthreads() ;
    // #pragma unroll
    for(unsigned i = tid ; i < group_size ;i  ++) {
        dis_arr[blockIdx.x * group_size + i] = dis[i] ;
        ids_arr[blockIdx.x * group_size + i] = ids[i] ;
    }
}

void get_partitions_access_order(array<unsigned*,3>& partition_infos , unsigned pnum , unsigned cnum , unsigned qnum , unsigned* cluster_p_info
     , unsigned topk_cluster_num , __half* cluster_center_half_dev , __half* q_vectors , unsigned dim , float* query_load_host , float* cluster_centers ,
    float* q_norm , float* c_norm) {
    cout << partition_infos[0] << "," <<  partition_infos[1] << "," <<  partition_infos[2] << endl ;
    // vector<unsigned> info1_host(qnum) , info2_host(qnum) ;
    // cudaMemcpy(info1_host.data() , partition_infos[1] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // cudaMemcpy(info2_host.data() , partition_infos[2] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // for(unsigned i = 0 ; i < qnum ;i  ++) {
    //     cout << i << ": size-" << info1_host[i] << ", offset-" << info2_host[i] << endl ; 
    // }
    // 先得到前k个聚类中心
    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f GB / %.2f GB\n",
       freeB / 1e9, totalB / 1e9);
    float mem_manage_cost = 0.0 , thrust_initial_cost = 0.0 ;

    auto total_s = chrono::high_resolution_clock::now() ;
    unsigned *indices ; 
    auto s1 = chrono::high_resolution_clock::now() ;
    cudaMalloc((void**) &indices , cnum * qnum * sizeof(unsigned)) ;
    auto e1 = chrono::high_resolution_clock::now() ;
    mem_manage_cost += chrono::duration<double>(e1 - s1).count() ;
    // thrust::sequence(thrust::device , indices , indices + cnum * qnum) ;
    auto cit_first = thrust::make_counting_iterator<unsigned>(0) ;
    // 按行生成原始id
    s1 = chrono::high_resolution_clock::now() ;
    thrust::transform(
        thrust::device , 
        cit_first , cit_first + cnum * qnum ,
        indices ,
        [cnum] __host__ __device__ (unsigned i) { return i % cnum ; }
    ) ;
    e1 = chrono::high_resolution_clock::now() ;
    thrust_initial_cost += chrono::duration<double>(e1 - s1).count() ;
    float* ret_dis ; 

    s1 = chrono::high_resolution_clock::now() ;
    cudaMalloc((void**) &ret_dis , cnum * qnum * sizeof(float)) ;
    e1 = chrono::high_resolution_clock::now() ;
    mem_manage_cost += chrono::duration<double>(e1 - s1).count() ;
    auto s = chrono::high_resolution_clock::now() ;
    // 猜测这里有些结果可能变成了NAN
    dis_cal_tensor_core<16><<<(qnum + 15) / 16 , dim3(32 , 16 , 1) >>>(q_vectors , cluster_center_half_dev, qnum , 
        cnum , dim , q_norm , c_norm , ret_dis) ;
    // 先做距离计算
    // dis_cal_matrix<<<qnum * 10 , dim3(32 , 6 , 1) , dim * sizeof(__half)>>>(cluster_center_half_dev , q_vectors , qnum ,cnum, dim , ret_dis) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__  , __FILE__) ;
    auto e = chrono::high_resolution_clock::now() ;
    float dis_cal_cost_old = chrono::duration<double>(e - s).count() ;
    // **************************
    // 检查是否有些内容变为了nan
    /**
        float* ret_dis_host = new float[qnum * cnum] ;
        cudaMemcpy(ret_dis_host , ret_dis , qnum * cnum * sizeof(float) , cudaMemcpyDeviceToHost) ;
        // 暴力计算前16个元素的交叉距离
        float* check_dis_host = new float[qnum * cnum] ;
        for(unsigned i = 0 ; i < qnum ; i ++)
            for(unsigned j = 0 ; j < cnum ; j++) {
                float dis = 0.0f ;
                for(unsigned d=  0 ; d < dim ; d++) 
                    dis += (query_load_host[i * dim + d] - cluster_centers[j * dim + d]) * (query_load_host[i * dim + d] - cluster_centers[j * dim + d]) ;
                check_dis_host[i * cnum + j] = dis / 10000.0f ; 
            }
        float diff_ret_dis = 0.0f ;
        for(unsigned i = 0 ; i < 16 ; i ++) {
            for(unsigned j =0 ; j < 10 ; j ++) {
                // printf("%f : %f, " , ret_dis_host[i * cnum + j] , check_dis_host[i * cnum + j]) ;
                diff_ret_dis += abs(ret_dis_host[i*cnum + j] - check_dis_host[i* cnum+ j]) ;
            }    
            // printf("\n") ;
        }
        cout << "diff return dis : " << diff_ret_dis << endl ; 
    **/
    // **************************
        


    // GT (Np x Nq) = P_row (Np x d) * Q_row^T (d x Nq)  [行主序逻辑]
    // cuBLAS 列主序视角：GT_col = P_col^T * Q_col
    
    // cublasHandle_t handle;
    // cublasCreate(&handle);
    // float dis_cal_cost = 0.0 ;

    // float alpha = 1.0f;
    // float beta  = 0.0f;
    // float* q_norm , *p_norm , *D2_f32 ;
    // cudaMalloc((void**) &q_norm , qnum * sizeof(float)) ;
    // cudaMalloc((void**) &p_norm , cnum * sizeof(float)) ;
    // cudaMalloc((void**) &D2_f32 , qnum * cnum * sizeof(float)) ;
    // s = chrono::high_resolution_clock::now() ;
    // gemm_GT_rowmajor_f16_to_f32(handle , cluster_center_half_dev , q_vectors, ret_dis , cnum, qnum , dim) ;
    // cudaDeviceSynchronize() ;
    // e = chrono::high_resolution_clock::now() ;
    // row_norm2_f16_rowmajor<<<cnum, 256>>>(cluster_center_half_dev, cnum, dim, p_norm);
    // row_norm2_f16_rowmajor<<<qnum, 256>>>(q_vectors, qnum, dim, q_norm);
    // cudaDeviceSynchronize() ;
    
    // dis_cal_cost += chrono::duration<double>(e - s).count() ;
    // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // dim3 block(16, 16);
    // dim3 grid((cnum + block.x - 1)/block.x, (qnum + block.y - 1)/block.y);
    // s = chrono::high_resolution_clock::now() ;
    // fuse_dist2_from_GT<<<grid, block>>>(q_norm, p_norm, ret_dis, qnum, cnum, D2_f32) ;
    // cudaDeviceSynchronize() ;
    // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // e = chrono::high_resolution_clock::now() ;
    // dis_cal_cost += chrono::duration<double>(e - s).count() ;
    // cudaFree(ret_dis) ;
    // ret_dis = D2_f32 ;
    // cudaFree(q_norm) ;
    // cudaFree(p_norm) ;
    
    // dis_cal_cost = chrono::duration<double>(e - s).count() ;

    // 按分组排序
    // unsigned *d_offsets ;
    // s1 = chrono::high_resolution_clock::now() ;
    // cudaMalloc((void**) &d_offsets , (qnum + 1) * sizeof(unsigned)) ;
    // e1 = chrono::high_resolution_clock::now() ;
    // mem_manage_cost += chrono::duration<double>(e1 - s1).count() ;
    // s1 = chrono::high_resolution_clock::now() ;
    // thrust::transform(
    //     thrust::device , 
    //     cit_first , cit_first + qnum + 1 ,
    //     d_offsets ,
    //     [cnum] __host__ __device__ (unsigned i) { return i * cnum ; }
    // ) ;
    // e1 = chrono::high_resolution_clock::now() ;
    // thrust_initial_cost += chrono::duration<double>(e1 - s1).count() ;

    // unsigned * d_ofst_host = new unsigned[qnum + 1] ;
    // cudaMemcpy(d_ofst_host , d_offsets , (qnum + 1) * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // for(unsigned i = 0 ; i < qnum + 1 ; i ++)
    //     cout << "--" << d_ofst_host[i] ;
    // float* dis_key_b ;
    // unsigned* id_value_b ;
    // s1 = chrono::high_resolution_clock::now() ;
    // cudaMalloc((void**) &dis_key_b , cnum * qnum * sizeof(float)) ;
    // cudaMalloc((void**) &id_value_b , cnum * qnum * sizeof(float)) ; 
    // e1 = chrono::high_resolution_clock::now() ;
    // mem_manage_cost += chrono::duration<double>(e1 - s1).count() ;
    // cub::DoubleBuffer<float> keys(ret_dis , dis_key_b) ;
    // cub::DoubleBuffer<unsigned> vals(indices , id_value_b) ;
    // void* d_temp = nullptr ; 
    // size_t temp_bytes = 0  ;

    // s = chrono::high_resolution_clock::now() ;
    // cub::DeviceSegmentedRadixSort::SortPairs(
    //     d_temp , temp_bytes ,
    //     keys , vals , 
    //     qnum * cnum , qnum ,
    //     d_offsets , d_offsets + 1 ,
    //     0 , sizeof(float) * 8
    // ) ;

    // s1 = chrono::high_resolution_clock::now() ;
    // cudaMalloc(&d_temp , temp_bytes) ;
    // e1 = chrono::high_resolution_clock::now() ;
    // mem_manage_cost += chrono::duration<double>(e1 - s1).count() ;
    // cout << "temp_bytes : " << temp_bytes << endl ;
    // cub::DeviceSegmentedRadixSort::SortPairs(
    //     d_temp , temp_bytes ,
    //     keys , vals , 
    //     qnum * cnum , qnum ,
    //     d_offsets , d_offsets + 1 ,
    //     0 , sizeof(float) * 8
    // ) ;
    // bitonic_sort_groups<<<dim3(qnum , 1 , 1) ,dim3(32 , 6 , 1) , cnum * sizeof(float) + cnum * sizeof(unsigned)>>>
    //     (ret_dis ,indices , cnum, qnum) ;


    // cudaDeviceSynchronize() ;
    // e = chrono::high_resolution_clock::now() ;
    // cudaFreeAsync(d_temp , cudaStreamDefault) ;
    // if(indices != vals.Current()) {
    //     cudaFreeAsync(indices , cudaStreamDefault) ;
    //     indices = vals.Current() ;
    // } else  {
    //     cudaFreeAsync(id_value_b , cudaStreamDefault) ;
    // }

    // float sort_by_group_cost = chrono::duration<double>(e - s).count() ;
    // float* ret_dis_host = new float[cnum * qnum] ;
    // cudaMemcpy(ret_dis_host , keys.Current() , cnum * qnum * sizeof(float) , cudaMemcpyDeviceToHost) ;
    // for(unsigned i = 0 ; i < cnum * qnum ; i ++)
    //     cout << "**-" << ret_dis_host[i] ;
    // cudaFreeAsync(dis_key_b , cudaStreamDefault) ;

    // cudaFreeAsync(d_offsets , cudaStreamDefault) ;
    
    auto trans_it = thrust::make_transform_iterator(
        cit_first ,
        [topk_cluster_num , cnum] __host__ __device__ (unsigned i) {
            return (i / topk_cluster_num) * cnum + (i % topk_cluster_num) ;
            // return i ;
        }
    ) ;
    unsigned* topk_clusters ;
    s1 = chrono::high_resolution_clock::now() ;
    cudaMalloc((void**) &topk_clusters , topk_cluster_num * qnum * sizeof(unsigned)) ;
    e1 = chrono::high_resolution_clock::now() ;
    mem_manage_cost += chrono::duration<double>(e1 - s1).count() ; 

    // thrust::gather(thrust::device , trans_it , trans_it + topk_cluster_num * qnum , indices , topk_clusters) ;
    // cudaEvent_t cs , ce ;
    // cudaEventCreate(&cs) ;
    // cudaEventCreate(&ce) ;
    // cudaDeviceSynchronize() ;
    // s1 = chrono::high_resolution_clock::now() ;
    // cudaEventRecord(cs) ;
    // gather_topk_element_from_per_group<<<(qnum + 31) /32 , 32>>>(cnum ,qnum , topk_cluster_num , indices , topk_clusters) ;
    // cudaEventRecord(ce) ;
    // cudaEventSynchronize(ce) ;
    // // cudaDeviceSynchronize() ;
    // CHECK(cudaGetLastError() , __LINE__ ,  __FILE__ ) ;
    // e1 = chrono::high_resolution_clock::now() ;
    // thrust_initial_cost += chrono::duration<double>(e1 - s1).count() ;
    // // 此时万事俱备, 只欠东风
    // float kernal_exe_time = 0.0 ;
    // cudaEventElapsedTime(&kernal_exe_time , cs , ce) ;
    // kernal_exe_time /= 1000 ;
    // 测试一下此时的topk是否正确




    // unsigned* partition_nums_device , *partition_nums_host = new unsigned[pnum * qnum] ;
    // cudaMalloc((void**) &partition_nums_device , pnum * qnum * sizeof(unsigned)) ;
    s = chrono::high_resolution_clock::now() ;
    // select_topk_parallel<<<qnum , dim3(32 , 2 , 1)>>>(cnum , qnum , ret_dis , indices , topk_cluster_num, topk_clusters , INF_DIS) ;
    // size_t shmem = 1024 * (sizeof(float) + sizeof(unsigned));
    // bitonic_topk_only_per_group_1000<16,256><<<qnum, 256, shmem>>>(
    //     ret_dis, indices, topk_clusters, 
    //     qnum, topk_cluster_num, 16
    // );
    select_topk_by_reduce_min<4 , 16><<<qnum , dim3(32 , 16 , 1)>>>(cnum , qnum , ret_dis , indices , topk_clusters) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    e = chrono::high_resolution_clock::now() ;
    float collect_cost = chrono::duration<double>(e - s).count() ;
    /**
    vector<vector<unsigned>> vvvvvv(qnum) ;
    for(unsigned i = 0 ; i < qnum ; i ++) {
        vvvvvv[i] = naive_brute_force(query_load_host + i * dim , cluster_centers , dim , cnum , 4) ; 
        reverse(vvvvvv[i].begin(), vvvvvv[i].end()) ;
    }
    cout << "暴力计算完成" << endl ;
    // unsigned *indices_host = new unsigned[pnum * qnum] ;
    // cudaMemcpy(indices_host , partition_infos[0] , pnum * qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    unsigned *topk_clusters_host = new unsigned[qnum * 4] ;
    cudaMemcpy(topk_clusters_host , topk_clusters , qnum * topk_cluster_num * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    cout << "内存复制完成" << endl ;
    unsigned top_misaligned_num = 0 ;
    for(unsigned i = 0 ; i < qnum ; i ++) {
        cout << "query " << i << ": " ;
        for(unsigned j = 0 ; j < 4 ; j ++) {
            top_misaligned_num += (topk_clusters_host[i *4 + j] != vvvvvv[i][j]) ;
            cout << "top" << j <<" : " << topk_clusters_host[i * 4 + j] << "-" << vvvvvv[i][j] << ", " ; 
        }
        cout << endl ;
    }
    cout << "top misaligned num : " << top_misaligned_num << endl ;
    **/
    // -------------------------------------------------------------------------------------


    s = chrono::high_resolution_clock::now() ;
    dim3 grid_s(qnum, 1, 1);
    dim3 block_s(32, 4, 1); // 6是block中的warp数量，可以调整
    unsigned byteSize = pnum * 2 * sizeof(unsigned) + topk_cluster_num * sizeof(unsigned) ;
    sort_by_clustering_info<<<grid_s , block_s , byteSize>>>(partition_infos[0] , partition_infos[1] , partition_infos[2] , pnum , cnum , 
        qnum , cluster_p_info , topk_clusters , topk_cluster_num) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    e = chrono::high_resolution_clock::now() ;
    float sort_cost = chrono::duration<double>(e - s).count() ;
    // cudaFreeAsync(topk_clusters , cudaStreamDefault) ;
    // cudaFreeAsync(indices , cudaStreamDefault) ;
    
    auto total_e = chrono::high_resolution_clock::now() ;
    float total_access_order_cost = chrono::duration<double>(total_e - total_s).count() ;
    
    
    // cudaMemcpy(partition_nums_host , partition_nums_device , pnum * qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // cout << "show num : " << endl ;
    // for(unsigned i = 0 ; i < pnum * qnum ; i ++)
    //     cout << partition_nums_host[i] << "," ;

    // vector<unsigned> slsl(pnum) ;
    
    // for(unsigned i = 0 ; i < 9 ; i ++) {
    //     slsl[i] = 0 ; 
        
    //     for(unsigned j = 0 ;j  < topk_cluster_num ; j ++)
    //         slsl[i] += num_c_p[indices_host[i]][topk_clusters_host[j]] ;
    // }
    // cout << endl ;
    // for(unsigned i = 0 ; i < 9 ; i ++)
    //     cout << slsl[i] << "," ;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f GB / %.2f GB\n",
       freeB / 1e9, totalB / 1e9);
    cout << "和聚类中心做距离计算用时:" << dis_cal_cost_old << endl ;
    // cout << "和聚类中心做距离计算用时(矩阵乘法) :" << dis_cal_cost << endl ;
    cout << "将各分区按照聚类中心相交点个数排序:" << sort_cost << endl ;
    // cout << "分组排序成本:" << sort_by_group_cost << endl ;
    // cout << "内存管理成本: " << mem_manage_cost << ", thrust数组操作成本: " << thrust_initial_cost << endl ;
    // cout << "gather核函数执行时间: " << kernal_exe_time << endl ;
    cout << "collect 操作耗时: " << collect_cost << endl ;
    cout << "整体成本:" << total_access_order_cost << endl ;
    
    cout << "generate access order finish!" << endl ;

}

namespace COST{
    vector<float> cost1 , cost2 , cost3 ;
}


void get_partitions_access_order_V2(array<unsigned*,5>& partition_infos , unsigned pnum , unsigned cnum , unsigned qnum , unsigned* cluster_p_info
    , unsigned topk_cluster_num , __half* cluster_center_half_dev , __half* q_vectors , unsigned dim , float* query_load_host , float* cluster_centers ,
   float* q_norm , float* c_norm , unsigned cardinality) {
   cout << partition_infos[0] << "," <<  partition_infos[1] << "," <<  partition_infos[2] << endl ;
   // vector<unsigned> info1_host(qnum) , info2_host(qnum) ;
   // cudaMemcpy(info1_host.data() , partition_infos[1] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
   // cudaMemcpy(info2_host.data() , partition_infos[2] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
   // for(unsigned i = 0 ; i < qnum ;i  ++) {
   //     cout << i << ": size-" << info1_host[i] << ", offset-" << info2_host[i] << endl ; 
   // }
   // 先得到前k个聚类中心
   size_t freeB, totalB;
   cudaMemGetInfo(&freeB, &totalB);
   printf("Free: %.2f GB / %.2f GB\n",
      freeB / 1e9, totalB / 1e9);
   float mem_manage_cost = 0.0 , thrust_initial_cost = 0.0 ;

   auto total_s = chrono::high_resolution_clock::now() ;
   unsigned *indices ; 
   auto s1 = chrono::high_resolution_clock::now() ;
   cudaMalloc((void**) &indices , cnum * qnum * sizeof(unsigned)) ;
   auto e1 = chrono::high_resolution_clock::now() ;
   mem_manage_cost += chrono::duration<double>(e1 - s1).count() ;
   // thrust::sequence(thrust::device , indices , indices + cnum * qnum) ;
   auto cit_first = thrust::make_counting_iterator<unsigned>(0) ;
   // 按行生成原始id
   s1 = chrono::high_resolution_clock::now() ;
   thrust::transform(
       thrust::device , 
       cit_first , cit_first + cnum * qnum ,
       indices ,
       [cnum] __host__ __device__ (unsigned i) { return i % cnum ; }
   ) ;
   e1 = chrono::high_resolution_clock::now() ;
   thrust_initial_cost += chrono::duration<double>(e1 - s1).count() ;
   float* ret_dis ; 

   s1 = chrono::high_resolution_clock::now() ;
   cudaMalloc((void**) &ret_dis , cnum * qnum * sizeof(float)) ;
   e1 = chrono::high_resolution_clock::now() ;
   mem_manage_cost += chrono::duration<double>(e1 - s1).count() ;
   auto s = chrono::high_resolution_clock::now() ;
   // 猜测这里有些结果可能变成了NAN
   dis_cal_tensor_core<16><<<(qnum + 15) / 16 , dim3(32 , 16 , 1) >>>(q_vectors , cluster_center_half_dev, qnum , 
       cnum , dim , q_norm , c_norm , ret_dis) ;
   // 先做距离计算
   // dis_cal_matrix<<<qnum * 10 , dim3(32 , 6 , 1) , dim * sizeof(__half)>>>(cluster_center_half_dev , q_vectors , qnum ,cnum, dim , ret_dis) ;
   cudaDeviceSynchronize() ;
   CHECK(cudaGetLastError() , __LINE__  , __FILE__) ;
   auto e = chrono::high_resolution_clock::now() ;
   float dis_cal_cost_old = chrono::duration<double>(e - s).count() ;
   
   auto trans_it = thrust::make_transform_iterator(
       cit_first ,
       [topk_cluster_num , cnum] __host__ __device__ (unsigned i) {
           return (i / topk_cluster_num) * cnum + (i % topk_cluster_num) ;
           // return i ;
       }
   ) ;
   unsigned* topk_clusters ;
   s1 = chrono::high_resolution_clock::now() ;
   cudaMalloc((void**) &topk_clusters , topk_cluster_num * qnum * sizeof(unsigned)) ;
   e1 = chrono::high_resolution_clock::now() ;
   mem_manage_cost += chrono::duration<double>(e1 - s1).count() ; 

   s = chrono::high_resolution_clock::now() ;
   select_topk_by_reduce_min<4 , 16><<<qnum , dim3(32 , 16 , 1)>>>(cnum , qnum , ret_dis , indices , topk_clusters) ;
   cudaDeviceSynchronize() ;
   CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
   e = chrono::high_resolution_clock::now() ;
   float collect_cost = chrono::duration<double>(e - s).count() ;

   s = chrono::high_resolution_clock::now() ;
   dim3 grid_s(qnum, 1, 1);
   dim3 block_s(32, 4, 1); // 6是block中的warp数量，可以调整
   unsigned byteSize = pnum * 2 * sizeof(unsigned) + topk_cluster_num * sizeof(unsigned) ;
   sort_by_clustering_info_weighted<<<grid_s , block_s , byteSize>>>(partition_infos[0] , partition_infos[1] , partition_infos[2] , partition_infos[3] , 
        (bool*) partition_infos[4] , pnum , cnum , 
       qnum , cluster_p_info , topk_clusters , topk_cluster_num , cardinality , partition_infos[4]) ;
   cudaDeviceSynchronize() ;
   CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
   e = chrono::high_resolution_clock::now() ;
   float sort_cost = chrono::duration<double>(e - s).count() ;
   
   auto total_e = chrono::high_resolution_clock::now() ;
   float total_access_order_cost = chrono::duration<double>(total_e - total_s).count() ;
   
   cudaMemGetInfo(&freeB, &totalB);
   printf("Free: %.2f GB / %.2f GB\n",
      freeB / 1e9, totalB / 1e9);
   cout << "和聚类中心做距离计算用时:" << dis_cal_cost_old << endl ;
   // cout << "和聚类中心做距离计算用时(矩阵乘法) :" << dis_cal_cost << endl ;
   cout << "将各分区按照聚类中心相交点个数排序:" << sort_cost << endl ;
   // cout << "分组排序成本:" << sort_by_group_cost << endl ;
   // cout << "内存管理成本: " << mem_manage_cost << ", thrust数组操作成本: " << thrust_initial_cost << endl ;
   // cout << "gather核函数执行时间: " << kernal_exe_time << endl ;
   cout << "collect 操作耗时: " << collect_cost << endl ;
   cout << "整体成本:" << total_access_order_cost << endl ;
   
   cout << "generate access order finish!" << endl ;

   COST::cost2.push_back(dis_cal_cost_old + sort_cost + collect_cost + thrust_initial_cost) ;

}

// 该单元通过测试
template<unsigned ATTR_DIM = 2>
array<unsigned*, 3> intersection_area(float* l_bound , float* r_bound , float* attr_min , float* attr_width , unsigned* attr_grid_size , unsigned attr_dim , unsigned qnum , unsigned pnum) {
    // unsigned pnum = thrust::reduce(
    //     thrust::device ,
    //     attr_grid_size ,
    //     attr_grid_size + attr_dim ,
    //     1 ,
    //     thrust::multiplies<unsigned>()
    // ) ;
    cout << "pnum : " << pnum << endl ;
    bool *is_selected ;
    cudaMalloc((void**) &is_selected , qnum * pnum * sizeof(bool)) ;
    auto s = chrono::high_resolution_clock::now() ;
    intersection_area_batch<ATTR_DIM><<<(qnum + 31) / 32 , 32>>>(l_bound, r_bound , attr_min , attr_width , attr_grid_size , attr_dim , qnum ,  is_selected ,pnum) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    auto e = chrono::high_resolution_clock::now() ;
    float kernal1_cost = chrono::duration<double>(e - s).count() ;
    // 得到索引数组和前缀数组
    // 需要三个信息, 即每个查询的区域个数,其在列表中的开始位置, 以及总的区域列表
    // 总的区域列表 可以使用copy_if + transform做, 每个查询的区域个数, 利用is_selected做规约？
    unsigned* indices ;
    cudaMalloc((void**) &indices , qnum * pnum * sizeof(unsigned)) ;

    s = chrono::high_resolution_clock::now() ;
    thrust::sequence(thrust::device , indices , indices + qnum * pnum) ;
    auto end_it = thrust::copy_if(
        thrust::device , 
        indices , indices + qnum * pnum ,
        is_selected , 
        indices , 
        thrust::identity<bool>() 
    ) ;
    unsigned selected_num = end_it - indices ; 
    e = chrono::high_resolution_clock::now() ;
    float copyif_cost = chrono::duration<double>(e - s).count() ;


    cout << "selected_part_num : " << selected_num << endl ;
    thrust::transform(
        thrust::device , 
        indices , indices + selected_num ,
        indices ,
        [pnum] __device__ (unsigned idx) -> unsigned {
            return idx % pnum ; 
        }
    ) ;
    // 此时得到一段连续的索引数组

    // 按行规约
    unsigned * p_counts , *p_start_index ;
    cudaMalloc((void**) & p_counts , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) & p_start_index , qnum * sizeof(unsigned) );

    auto row_key_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0) ,
        [pnum] __host__ __device__ (unsigned idx) -> unsigned {return idx / pnum ;}
    ) ;
    auto val_begin = thrust::make_transform_iterator(
        is_selected , 
        [] __host__ __device__ (const bool flag) -> unsigned {
            return (flag ? 1 : 0) ;
        }
    ) ;

    s = chrono::high_resolution_clock::now() ;
    thrust::reduce_by_key(
        thrust::device ,
        row_key_begin , row_key_begin + pnum * qnum ,
        val_begin , 
        thrust::make_discard_iterator() ,
        p_counts ,
        thrust::equal_to<unsigned>() ,
        thrust::plus<unsigned>()
    ) ;
    e = chrono::high_resolution_clock::now() ;
    float reduce_cost = chrono::duration<double>(e - s).count() ;
    thrust::exclusive_scan(thrust::device , p_counts , p_counts + qnum , p_start_index , 0) ;

    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cudaFree(is_selected) ;


    // 在这里对分区的访问顺序进行排序, 每个线程块处理一个查询
    cout << "判断有哪些分区相交用时:" << kernal1_cost << endl ;
    cout << "flag arr 2 index arr 用时:" << copyif_cost << endl ;
    cout << "分组reduce用时:" << reduce_cost << endl ;

    return {indices , p_counts , p_start_index} ;
}

// 该单元通过测试
template<unsigned ATTR_DIM = 2>
array<unsigned*, 5> intersection_area_mark_little_seg(float* l_bound , float* r_bound , float* attr_min , float* attr_width , unsigned* attr_grid_size , unsigned attr_dim , unsigned qnum , unsigned pnum , double little_seg_threshold , unsigned OP_TYPE = 0) {
    // unsigned pnum = thrust::reduce(
    //     thrust::device ,
    //     attr_grid_size ,
    //     attr_grid_size + attr_dim ,
    //     1 ,
    //     thrust::multiplies<unsigned>()
    // ) ;
    cout << "pnum : " << pnum << endl ;
    bool *is_selected , *is_little_seg ;
    cudaMalloc((void**) &is_selected , qnum * pnum * sizeof(bool)) ;
    cudaMalloc((void**) &is_little_seg , qnum * pnum * sizeof(bool)) ;
    auto s = chrono::high_resolution_clock::now() ;
    // 逻辑与
    if(OP_TYPE == 0)
        intersection_area_batch_mark_little_seg<ATTR_DIM><<<(qnum + 31) / 32 , 32>>>(l_bound, r_bound , attr_min , attr_width , attr_grid_size , attr_dim , qnum ,  is_selected ,pnum , is_little_seg , little_seg_threshold) ;
    // 逻辑或
    else if(OP_TYPE == 1)
        intersection_area_batch_mark_little_seg_logicalor<ATTR_DIM><<<(qnum + 31) / 32 , 32>>>(l_bound, r_bound , attr_min , attr_width , attr_grid_size , attr_dim , qnum ,  is_selected ,pnum , is_little_seg , little_seg_threshold) ;
    
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    auto e = chrono::high_resolution_clock::now() ;
    float kernal1_cost = chrono::duration<double>(e - s).count() ;
    // 得到索引数组和前缀数组
    // 需要三个信息, 即每个查询的区域个数,其在列表中的开始位置, 以及总的区域列表
    // 总的区域列表 可以使用copy_if + transform做, 每个查询的区域个数, 利用is_selected做规约？
    unsigned* indices ;
    bool* indices_is_little_seg ;
    cudaMalloc((void**) &indices , qnum * pnum * sizeof(unsigned)) ;
    s = chrono::high_resolution_clock::now() ;
    thrust::sequence(thrust::device , indices , indices + qnum * pnum) ;
    auto end_it = thrust::copy_if(
        thrust::device , 
        indices , indices + qnum * pnum ,
        is_selected , 
        indices , 
        thrust::identity<bool>() 
    ) ;
    unsigned selected_num = end_it - indices ; 
    e = chrono::high_resolution_clock::now() ;
    float copyif_cost = chrono::duration<double>(e - s).count() ;
    cudaMalloc((void**) &indices_is_little_seg , selected_num * sizeof(bool)) ;
    
    s = chrono::high_resolution_clock::now() ;
    thrust::gather(
        thrust::device ,
        indices , indices + selected_num ,
        is_little_seg , 
        indices_is_little_seg 
    ) ;

    cout << "selected_part_num : " << selected_num << endl ;
    thrust::transform(
        thrust::device , 
        indices , indices + selected_num ,
        indices ,
        [pnum] __device__ (unsigned idx) -> unsigned {
            return idx % pnum ; 
        }
    ) ;
    e = chrono::high_resolution_clock::now() ;
    float gather_cost = chrono::duration<float>(e - s).count() ;
    // 此时得到一段连续的索引数组

    // 按行规约
    unsigned * p_counts , *p_start_index , * little_seg_counts ;
    cudaMalloc((void**) & p_counts , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) & p_start_index , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) & little_seg_counts , qnum * sizeof(unsigned)) ;
    auto row_key_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0) ,
        [pnum] __host__ __device__ (unsigned idx) -> unsigned {return idx / pnum ;}
    ) ;
    auto val_begin = thrust::make_transform_iterator(
        is_selected , 
        [] __host__ __device__ (const bool flag) -> unsigned {
            return (flag ? 1 : 0) ;
        }
    ) ;
    auto val_begin2 = thrust::make_transform_iterator(
        is_little_seg ,
        [] __host__ __device__ (const bool flag) -> unsigned {
            return (flag ? 1 : 0) ;
        }
    ) ;

    s = chrono::high_resolution_clock::now() ;
    thrust::reduce_by_key(
        thrust::device ,
        row_key_begin , row_key_begin + pnum * qnum ,
        val_begin , 
        thrust::make_discard_iterator() ,
        p_counts ,
        thrust::equal_to<unsigned>() ,
        thrust::plus<unsigned>()
    ) ;
    thrust::reduce_by_key(
        thrust::device , 
        row_key_begin, row_key_begin + pnum * qnum ,
        val_begin2 ,
        thrust::make_discard_iterator() ,
        little_seg_counts ,
        thrust::equal_to<unsigned>() ,
        thrust::plus<unsigned>() 
    ) ;
    thrust::exclusive_scan(thrust::device , p_counts , p_counts + qnum , p_start_index , 0) ;
    e = chrono::high_resolution_clock::now() ;
    float reduce_cost = chrono::duration<double>(e - s).count() ;

    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cudaFree(is_selected) ;
    cudaFree(is_little_seg) ;


    // 在这里对分区的访问顺序进行排序, 每个线程块处理一个查询
    cout << "判断有哪些分区相交用时:" << kernal1_cost << endl ;
    cout << "flag arr 2 index arr 用时:" << copyif_cost << endl ;
    cout << "分组reduce用时:" << reduce_cost << endl ;
    COST::cost1.push_back(kernal1_cost + copyif_cost + reduce_cost + gather_cost) ;

    return {indices , p_counts , p_start_index , little_seg_counts , (unsigned*) indices_is_little_seg} ;
}

// 该单元通过测试
template<unsigned ATTR_DIM = 2 , typename attrType = float>
array<unsigned*, 5> intersection_area_mark_little_seg_batch_process(attrType* l_bound , attrType* r_bound , attrType* attr_min , attrType* attr_width , unsigned* attr_grid_size , unsigned attr_dim , unsigned qnum , unsigned pnum , double little_seg_threshold , bool* &is_selected , unsigned OP_TYPE = 0) {
    // unsigned pnum = thrust::reduce(
    //     thrust::device ,
    //     attr_grid_size ,
    //     attr_grid_size + attr_dim ,
    //     1 ,
    //     thrust::multiplies<unsigned>()
    // ) ;
    cout << "pnum : " << pnum << endl ;
    bool *is_little_seg ;
    cudaMalloc((void**) &is_selected , qnum * pnum * sizeof(bool)) ;
    cudaMalloc((void**) &is_little_seg , qnum * pnum * sizeof(bool)) ;
    auto s = chrono::high_resolution_clock::now() ;
    // 逻辑与
    if(OP_TYPE == 0)
        intersection_area_batch_mark_little_seg<ATTR_DIM, attrType><<<(qnum + 31) / 32 , 32>>>(l_bound, r_bound , attr_min , attr_width , attr_grid_size , attr_dim , qnum ,  is_selected ,pnum , is_little_seg , little_seg_threshold) ;
    // 逻辑或
    else if(OP_TYPE == 1)
        intersection_area_batch_mark_little_seg_logicalor<ATTR_DIM, attrType><<<(qnum + 31) / 32 , 32>>>(l_bound, r_bound , attr_min , attr_width , attr_grid_size , attr_dim , qnum ,  is_selected ,pnum , is_little_seg , little_seg_threshold) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    auto e = chrono::high_resolution_clock::now() ;
    float kernal1_cost = chrono::duration<double>(e - s).count() ;
    // 得到索引数组和前缀数组
    // 需要三个信息, 即每个查询的区域个数,其在列表中的开始位置, 以及总的区域列表
    // 总的区域列表 可以使用copy_if + transform做, 每个查询的区域个数, 利用is_selected做规约？
    unsigned* indices ;
    bool* indices_is_little_seg ;
    cudaMalloc((void**) &indices , qnum * pnum * sizeof(unsigned)) ;
    s = chrono::high_resolution_clock::now() ;
    thrust::sequence(thrust::device , indices , indices + qnum * pnum) ;
    auto end_it = thrust::copy_if(
        thrust::device , 
        indices , indices + qnum * pnum ,
        is_selected , 
        indices , 
        thrust::identity<bool>() 
    ) ;
    unsigned selected_num = end_it - indices ; 
    e = chrono::high_resolution_clock::now() ;
    float copyif_cost = chrono::duration<double>(e - s).count() ;
    cudaMalloc((void**) &indices_is_little_seg , selected_num * sizeof(bool)) ;
    
    s = chrono::high_resolution_clock::now() ;
    thrust::gather(
        thrust::device ,
        indices , indices + selected_num ,
        is_little_seg , 
        indices_is_little_seg 
    ) ;

    cout << "selected_part_num : " << selected_num << endl ;
    thrust::transform(
        thrust::device , 
        indices , indices + selected_num ,
        indices ,
        [pnum] __device__ (unsigned idx) -> unsigned {
            return idx % pnum ; 
        }
    ) ;
    e = chrono::high_resolution_clock::now() ;
    float gather_cost = chrono::duration<double>(e - s).count() ;
    // 此时得到一段连续的索引数组

    // 按行规约
    unsigned * p_counts , *p_start_index , * little_seg_counts ;
    cudaMalloc((void**) & p_counts , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) & p_start_index , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) & little_seg_counts , qnum * sizeof(unsigned)) ;
    auto row_key_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0) ,
        [pnum] __host__ __device__ (unsigned idx) -> unsigned {return idx / pnum ;}
    ) ;
    auto val_begin = thrust::make_transform_iterator(
        is_selected , 
        [] __host__ __device__ (const bool flag) -> unsigned {
            return (flag ? 1 : 0) ;
        }
    ) ;
    auto val_begin2 = thrust::make_transform_iterator(
        is_little_seg ,
        [] __host__ __device__ (const bool flag) -> unsigned {
            return (flag ? 1 : 0) ;
        }
    ) ;

    s = chrono::high_resolution_clock::now() ;
    thrust::reduce_by_key(
        thrust::device ,
        row_key_begin , row_key_begin + pnum * qnum ,
        val_begin , 
        thrust::make_discard_iterator() ,
        p_counts ,
        thrust::equal_to<unsigned>() ,
        thrust::plus<unsigned>()
    ) ;
    thrust::reduce_by_key(
        thrust::device , 
        row_key_begin, row_key_begin + pnum * qnum ,
        val_begin2 ,
        thrust::make_discard_iterator() ,
        little_seg_counts ,
        thrust::equal_to<unsigned>() ,
        thrust::plus<unsigned>() 
    ) ;
    thrust::exclusive_scan(thrust::device , p_counts , p_counts + qnum , p_start_index , 0) ;
    e = chrono::high_resolution_clock::now() ;
    float reduce_cost = chrono::duration<double>(e - s).count() ;

    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // cudaFree(is_selected) ;
    cudaFree(indices_is_little_seg) ;


    // 在这里对分区的访问顺序进行排序, 每个线程块处理一个查询
    cout << "判断有哪些分区相交用时:" << kernal1_cost << endl ;
    cout << "flag arr 2 index arr 用时:" << copyif_cost << endl ;
    cout << "分组reduce用时:" << reduce_cost << endl ;
    COST::cost1.push_back(kernal1_cost + copyif_cost + reduce_cost + gather_cost) ;
    return {indices , p_counts , p_start_index , little_seg_counts , (unsigned*) is_little_seg} ;
}

// 分组去重
__global__ void unique_by_group(unsigned* arr , unsigned group_size , unsigned group_num) {
    extern __shared__ unsigned char shared_mem[] ;
    unsigned *s = (unsigned*) shared_mem ;
    unsigned *s_is_first = (unsigned*) (s + group_size) ;
    unsigned *s_pos = (unsigned*) (s_is_first + group_size) ;

    for(unsigned i = threadIdx.x ; i < group_size ; i += blockDim.x) {
        s[i] = arr[blockIdx.x * group_size + i] ;
        arr[blockIdx.x * group_size + i] = 0xFFFFFFFF ;
    }
    __syncthreads() ;

    // 先填充 s_is_first 
    for(unsigned i = threadIdx.x ; i < group_size ; i += blockDim.x) {
        s_is_first[i] = (i == 0 || s[i] != s[i - 1]) ;
    }
    __syncthreads() ;

    // 扫描出前缀和数组, 串行
    if(threadIdx.x == 0) {
        s_pos[0] = 0 ;
        for(unsigned i = 1 ; i < group_size ;i  ++) {
            s_pos[i] = s_pos[i - 1] + s_is_first[i - 1] ;
        }
    }
    __syncthreads() ;

    for(unsigned i = threadIdx.x ; i < group_size ; i += blockDim.x) 
        if(s_is_first[i])
            arr[blockIdx.x * group_size + s_pos[i]] = s[i] ;

}
float select_path_kernal_cost = 0.0 ;
vector<vector<unsigned>> batch_graph_search_gpu(array<unsigned* , 3>& graph_infos , unsigned degree ,  __half* data_ , __half* query ,unsigned dim , unsigned qnum ,  float* l_bound , float* r_bound , float* attrs , 
    unsigned attr_dim  , array<unsigned*,3>& idx_mapper , array<unsigned* , 3>& partition_infos , unsigned L , unsigned K , 
    unsigned* global_edges , unsigned edge_num_contain , unsigned pnum , unsigned recycle_list_size) {
    // global_idx[subgraph_offset[subgraph_id] + local_id] = global_id 
    // 输入: 图, data_, query,  l_bound , r_bound , attr_dim , attrs , 每个子图中的点个数, 每个分区的全局id开始索引, 每个分区第i个点对应的全局id 
    // 每个查询所对应的分区列表, 分区个数, 分区起始索引
    unsigned DIM = dim , TOPM = L ,  DEGREE = degree ;
    unsigned *results_id ; 
    float *results_dis ; 
    unsigned *result_k_id ; 
    unsigned *task_id_mapper ; 


    auto ms = chrono::high_resolution_clock::now() ;
    cudaMalloc((void**) &results_id , qnum * L * sizeof(unsigned)) ;
    cudaMalloc((void**) &results_dis , qnum * L * sizeof(unsigned)) ;
    cudaMalloc((void**) &result_k_id , qnum * K * sizeof(unsigned)) ;
    cudaMalloc((void**) &task_id_mapper , qnum * sizeof(unsigned)) ;

    // 将任务按照分区的个数重新排序
    auto sorts = chrono::high_resolution_clock::now() ; 
    thrust::sequence(thrust::device , task_id_mapper , task_id_mapper + qnum) ;
    unsigned* task_size = partition_infos[1] ;
    thrust::sort(
        thrust::device , 
        task_id_mapper , task_id_mapper + qnum ,
        [task_size] __host__ __device__ (unsigned x , unsigned y) {
            return task_size[x] > task_size[y] ;
        }
    ) ;
    auto sorte = chrono::high_resolution_clock::now() ;
    float sort_cost = chrono::duration<double>(sorte - sorts).count() ;

    // ent_pts 先设置为前degree个点, 后续修改为每个查询第一个分区随机degree个
    unsigned *ent_pts ;
    cudaMalloc((void**) &ent_pts , L *  sizeof(unsigned)) ;
    auto me = chrono::high_resolution_clock::now() ;
    float m_cost = chrono::duration<double>(me - ms).count() ;

    // thrust::sequence(thrust::device , ent_pts , ent_pts + degree) ;
    auto es = chrono::high_resolution_clock::now() ;
    vector<unsigned> ent_pts_random(62500) ;
    for(unsigned i = 0 ; i < 62500 ; i ++)
        ent_pts_random[i] = i ;
    random_shuffle(ent_pts_random.begin() , ent_pts_random.end()) ;
    cudaMemcpy(ent_pts , ent_pts_random.data() , L *  sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    auto ee = chrono::high_resolution_clock::now() ;
    float find_ep_cost = chrono::duration<double>(ee - es).count() ;

    dim3 grid_s(qnum , 1 , 1) , block_s(32 ,4 , 1) ;
    unsigned byteSize = recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) + 
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float) + (DIM/4) * sizeof(half4) + alignof(half4) ;
    cout << "byteSize : " << (byteSize * 1.0 / 1024) << "KB" << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn<<<grid_s , block_s , byteSize>>> (graph_infos[0], data_, query, 
        results_id, results_dis, ent_pts , degree , L , dim , graph_infos[2] , graph_infos[1] ,
        partition_infos[0] , partition_infos[1] , partition_infos[2] , l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain,
        recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] ,  pnum , K , result_k_id , task_id_mapper) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    auto e = chrono::high_resolution_clock::now() ;
    select_path_kernal_cost = chrono::duration<double>(e - s).count() ;
    // auto cit = thrust::make_counting_iterator<unsigned>(0) ; `
    // auto trans_it = thrust::make_transform_iterator(cit , [L] __host__ __device__ (unsigned x) { return x / L ;}) ;
    // auto unique_end = thrust::unique_by_key(
    //     thrust::device , 
    //     trans_it , trans_it + L * qnum ,
    //     results_id 
    // ) ;
    // unique_by_group<<<qnum , 32 , 3 * L * sizeof(unsigned)>>>(results_id , L , qnum) ;
    // cudaDeviceSynchronize() ;
    // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // 将结果转换成vector形式
    vector<vector<unsigned>> results(qnum , vector<unsigned>(K)) ;
    // for(unsigned i = 0 ; i < qnum ; i ++) {
    //     cudaMemcpy(results[i].data() , results_id + i * L , K * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // }
    s = chrono::high_resolution_clock::now() ;
    vector<unsigned> result_source(K * qnum) ;
    cudaMemcpy(result_source.data() , result_k_id , qnum * K * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    for(unsigned i = 0 ; i < qnum ;i  ++) {
        results[i] = vector<unsigned>(result_source.begin() + i * K  , result_source.begin() + (i + 1) * K) ;
    }
    e = chrono::high_resolution_clock::now() ;
    float result_load_cost = chrono::duration<double>(e - s).count() ;
    // cout << "搜索核函数运行时间:" << cost << endl ;

    s = chrono::high_resolution_clock::now() ;
    unsigned* symbol_address , *symbol_address_dcnt ;
    cudaGetSymbolAddress((void**) &symbol_address , jump_num_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_dcnt , dis_cal_cnt) ;
    unsigned total_j_num = thrust::reduce(thrust::device , symbol_address , symbol_address + qnum , 0) ;
    unsigned total_c_num = thrust::reduce(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    e = chrono::high_resolution_clock::now() ;
    float jump_cal_cost = chrono::duration<double>(e - s).count() ; 
    cout << "总的跳转次数: " << total_j_num << " , 平均跳转次数: " << (total_j_num * 1.0 / qnum) << endl ;
    cout << "总的距离计算次数: " << total_c_num << ", 平均距离计算次数: " << (total_c_num * 1.0 / qnum) << endl ;
    cout << "结果传递时间: " << result_load_cost << endl ;
    cout << "计算跳数时间: " << jump_cal_cost << endl ;
    cout << "获取入口点时间: " << find_ep_cost << endl ; 
    cout << "排序时间: " << sort_cost << endl ;
    thrust::fill(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    ms = chrono::high_resolution_clock::now() ;
    cudaFree(results_id) ;
    cudaFree(results_dis) ;
    cudaFree(ent_pts) ;
    cudaFree(result_k_id) ;
    me = chrono::high_resolution_clock::now() ;
    m_cost += chrono::duration<double>(me - ms).count() ;
    cout << "管理内存耗时: " << m_cost << endl ;
    return results ;
}


vector<vector<unsigned>> batch_graph_search_gpu_with_prefilter(array<unsigned* , 3>& graph_infos , unsigned degree ,  __half* data_ , __half* query ,unsigned dim , unsigned qnum ,  float* l_bound , float* r_bound , float* attrs , 
    unsigned attr_dim  , array<unsigned*,3>& idx_mapper , array<unsigned* , 5>& partition_infos , unsigned L , unsigned K , 
    unsigned* global_edges , unsigned edge_num_contain , unsigned pnum , unsigned recycle_list_size) {
    // global_idx[subgraph_offset[subgraph_id] + local_id] = global_id 
    // 输入: 图, data_, query,  l_bound , r_bound , attr_dim , attrs , 每个子图中的点个数, 每个分区的全局id开始索引, 每个分区第i个点对应的全局id 
    // 每个查询所对应的分区列表, 分区个数, 分区起始索引
    unsigned DIM = dim , TOPM = L ,  DEGREE = degree ;
    unsigned *results_id ; 
    float *results_dis ; 
    unsigned *result_k_id ; 
    unsigned *task_id_mapper ; 


    auto ms = chrono::high_resolution_clock::now() ;
    cudaMalloc((void**) &results_id , qnum * L * sizeof(unsigned)) ;
    cudaMalloc((void**) &results_dis , qnum * L * sizeof(unsigned)) ;
    cudaMalloc((void**) &result_k_id , qnum * K * sizeof(unsigned)) ;
    cudaMalloc((void**) &task_id_mapper , qnum * sizeof(unsigned)) ;

    // 将任务按照分区的个数重新排序
    auto sorts = chrono::high_resolution_clock::now() ; 
    thrust::sequence(thrust::device , task_id_mapper , task_id_mapper + qnum) ;
    unsigned* task_size = partition_infos[1] ;
    thrust::sort(
        thrust::device , 
        task_id_mapper , task_id_mapper + qnum ,
        [task_size] __host__ __device__ (unsigned x , unsigned y) {
            return task_size[x] > task_size[y] ;
        }
    ) ;
    auto sorte = chrono::high_resolution_clock::now() ;
    float sort_cost = chrono::duration<double>(sorte - sorts).count() ;

    // ent_pts 先设置为前degree个点, 后续修改为每个查询第一个分区随机degree个
    unsigned *ent_pts ;
    cudaMalloc((void**) &ent_pts , L *  sizeof(unsigned)) ;
    auto me = chrono::high_resolution_clock::now() ;
    float m_cost = chrono::duration<double>(me - ms).count() ;
    mt19937 rng ;
    uniform_int_distribution<> intd(0 ,62500 - 1) ;

    // thrust::sequence(thrust::device , ent_pts , ent_pts + degree) ;
    auto es = chrono::high_resolution_clock::now() ;
    vector<unsigned> ent_pts_random(L * 10) ;
    // for(unsigned i = 0 ; i < 62500 ; i ++)
    //     ent_pts_random[i] = i ;
    // random_shuffle(ent_pts_random.begin() , ent_pts_random.end()) ;
    unordered_set<unsigned> ent_pts_contains ;
    unsigned ent_pts_ofst = 0 ;
    while(ent_pts_ofst < L * 10) {
        unsigned t = intd(rng) ;
        if(!ent_pts_contains.count(t)) {
            ent_pts_contains.insert(t) ;
            ent_pts_random[ent_pts_ofst ++] = t ;
        }
    }
    random_shuffle(ent_pts_random.begin() , ent_pts_random.end()) ;

    cudaMemcpy(ent_pts , ent_pts_random.data() , L *  sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    auto ee = chrono::high_resolution_clock::now() ;
    float find_ep_cost = chrono::duration<double>(ee - es).count() ;

    dim3 grid_s(qnum , 1 , 1) , block_s(32 ,6 , 1) ;
    unsigned byteSize = recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) + 
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float) + (DIM/4) * sizeof(half4) + alignof(half4) ;
    cout << "byteSize : " << (byteSize * 1.0 / 1024) << "KB" << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    // select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter<<<grid_s , block_s , byteSize>>> (graph_infos[0], data_, query, 
    //     results_id, results_dis, ent_pts , degree , L , dim , graph_infos[2] , graph_infos[1] ,
    //     partition_infos[0] , partition_infos[1] , partition_infos[2] , l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain,
    //     recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] ,  pnum , K , result_k_id , task_id_mapper , partition_infos[3]) ;

    select_path_contain_attr_filter_V2_without_output<<<grid_s , block_s , byteSize>>>(graph_infos[0], data_, query, 
        results_id, results_dis, ent_pts , degree , L , dim , graph_infos[2] ,graph_infos[1] ,
        partition_infos[0] , partition_infos[1] ,partition_infos[2] , l_bound , r_bound , attrs ,  attr_dim , global_edges , edge_num_contain,
        recycle_list_size ,idx_mapper[0] , idx_mapper[1] ,idx_mapper[2] , pnum , K, result_k_id , task_id_mapper) ;
    cudaDeviceSynchronize() ;

    auto e = chrono::high_resolution_clock::now() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    select_path_kernal_cost = chrono::duration<double>(e - s).count() ;

    vector<vector<unsigned>> results(qnum , vector<unsigned>(K)) ;
    s = chrono::high_resolution_clock::now() ;
    vector<unsigned> result_source(K * qnum) ;
    cudaMemcpy(result_source.data() , result_k_id , qnum * K * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    for(unsigned i = 0 ; i < qnum ;i  ++) {
        results[i] = vector<unsigned>(result_source.begin() + i * K  , result_source.begin() + (i + 1) * K) ;
    }
    e = chrono::high_resolution_clock::now() ;
    float result_load_cost = chrono::duration<double>(e - s).count() ;
    // cout << "搜索核函数运行时间:" << cost << endl ;

    s = chrono::high_resolution_clock::now() ;
    unsigned* symbol_address , *symbol_address_dcnt , *symbol_address_pmcnt ; 
    cudaGetSymbolAddress((void**) &symbol_address , jump_num_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_dcnt , dis_cal_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_pmcnt , post_merge_cnt) ;
    unsigned total_j_num = thrust::reduce(thrust::device , symbol_address , symbol_address + qnum , 0) ;
    unsigned total_c_num = thrust::reduce(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    unsigned total_m_num = thrust::reduce(thrust::device , symbol_address_pmcnt , symbol_address_pmcnt + qnum , 0) ;
    e = chrono::high_resolution_clock::now() ;
    float jump_cal_cost = chrono::duration<double>(e - s).count() ; 
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "总的跳转次数: " << total_j_num << " , 平均跳转次数: " << (total_j_num * 1.0 / qnum) << endl ;
    cout << "总的距离计算次数: " << total_c_num << ", 平均距离计算次数: " << (total_c_num * 1.0 / qnum) << endl ;
    cout << "总的后合并次数: " << total_m_num << ", 平均后合并次数: " << (total_m_num * 1.0 / qnum) << endl ; 
    cout << "结果传递时间: " << result_load_cost << endl ;
    cout << "计算跳数时间: " << jump_cal_cost << endl ;
    cout << "获取入口点时间: " << find_ep_cost << endl ; 
    cout << "排序时间: " << sort_cost << endl ;

    COST::cost3.push_back(select_path_kernal_cost) ;
    thrust::fill(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    ms = chrono::high_resolution_clock::now() ;
    cudaFree(results_id) ;
    cudaFree(results_dis) ;
    cudaFree(ent_pts) ;
    cudaFree(result_k_id) ;
    me = chrono::high_resolution_clock::now() ;
    m_cost += chrono::duration<double>(me - ms).count() ;
    cout << "管理内存耗时: " << m_cost << endl ;
    return results ;
}


template<typename T = float>
vector<vector<unsigned>> batch_graph_search_gpu_with_prefilter_and_postfilter(array<unsigned* , 3>& graph_infos , unsigned degree ,  __half* data_ , __half* query ,unsigned dim , unsigned qnum ,  T* l_bound , T* r_bound , T* attrs , 
    unsigned attr_dim  , array<unsigned*,3>& idx_mapper , array<unsigned* , 5>& partition_infos , unsigned L , unsigned K , 
    unsigned* global_edges , unsigned edge_num_contain , unsigned pnum , unsigned recycle_list_size , unsigned postfilter_threshold , const unsigned* bigGraph , const unsigned bigDegree) {
    // global_idx[subgraph_offset[subgraph_id] + local_id] = global_id 
    // 输入: 图, data_, query,  l_bound , r_bound , attr_dim , attrs , 每个子图中的点个数, 每个分区的全局id开始索引, 每个分区第i个点对应的全局id 
    // 每个查询所对应的分区列表, 分区个数, 分区起始索引
    unsigned DIM = dim , TOPM = L ,  DEGREE = degree ;
    unsigned *results_id ; 
    float *results_dis ; 
    unsigned *result_k_id ; 
    unsigned *task_id_mapper ; 


    auto ms = chrono::high_resolution_clock::now() ;
    cudaMalloc((void**) &results_id , qnum * L * sizeof(unsigned)) ;
    cudaMalloc((void**) &results_dis , qnum * L * sizeof(unsigned)) ;
    cudaMalloc((void**) &result_k_id , qnum * K * sizeof(unsigned)) ;
    cudaMalloc((void**) &task_id_mapper , qnum * sizeof(unsigned)) ;

    // 将任务按照分区的个数重新排序
    auto sorts = chrono::high_resolution_clock::now() ; 
    thrust::sequence(thrust::device , task_id_mapper , task_id_mapper + qnum) ;
    unsigned* task_size = partition_infos[1] ;
    thrust::sort(
        thrust::device , 
        task_id_mapper , task_id_mapper + qnum ,
        [task_size] __host__ __device__ (unsigned x , unsigned y) {
            return task_size[x] > task_size[y] ;
        }
    ) ;
    auto sorte = chrono::high_resolution_clock::now() ;
    float sort_cost = chrono::duration<double>(sorte - sorts).count() ;

    // ent_pts 先设置为前degree个点, 后续修改为每个查询第一个分区随机degree个
    unsigned *ent_pts , *ent_pts_post ;
    cudaMalloc((void**) &ent_pts , degree *  sizeof(unsigned)) ;
    cudaMalloc((void**) &ent_pts_post , bigDegree * sizeof(unsigned)) ;
    auto me = chrono::high_resolution_clock::now() ;
    float m_cost = chrono::duration<double>(me - ms).count() ;
    unsigned avg_num = (1000000 + pnum - 1) / pnum ; 
    unsigned rand_max = 1000000 - (pnum - 1) * avg_num ;
    cout << "rand max : " << rand_max << endl ;
    mt19937 rng ;
    uniform_int_distribution<> intd(0 ,rand_max - 1) ;

    // thrust::sequence(thrust::device , ent_pts , ent_pts + degree) ;
    auto es = chrono::high_resolution_clock::now() ;
    vector<unsigned> ent_pts_random(bigDegree * 10) ;
    // for(unsigned i = 0 ; i < 62500 ; i ++)
    //     ent_pts_random[i] = i ;
    // random_shuffle(ent_pts_random.begin() , ent_pts_random.end()) ;
    unordered_set<unsigned> ent_pts_contains ;
    unsigned ent_pts_ofst = 0 ;
    while(ent_pts_ofst < degree * 10) {
        unsigned t = intd(rng) ;
        if(!ent_pts_contains.count(t)) {
            ent_pts_contains.insert(t) ;
            ent_pts_random[ent_pts_ofst ++] = t ;
        }
    }
    random_shuffle(ent_pts_random.begin() , ent_pts_random.begin() + degree * 10) ;

    cudaMemcpy(ent_pts , ent_pts_random.data() , degree *  sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    ent_pts_contains.clear() ;
    // ent_pts_random.clear() ;
    intd = uniform_int_distribution<>(0 , 1000000 - 1) ;
    ent_pts_ofst = 0 ;
    while(ent_pts_ofst < bigDegree * 10) {
        unsigned t = intd(rng) ;
        if(!ent_pts_contains.count(t)) {
            ent_pts_contains.insert(t) ;
            ent_pts_random[ent_pts_ofst ++] = t ; 
        }
    }
    random_shuffle(ent_pts_random.begin() , ent_pts_random.begin() + bigDegree * 10) ;
    cudaMemcpy(ent_pts_post , ent_pts_random.data() , bigDegree * sizeof(unsigned) , cudaMemcpyHostToDevice) ;

    auto ee = chrono::high_resolution_clock::now() ;
    float find_ep_cost = chrono::duration<double>(ee - es).count() ;

    dim3 grid_s(qnum , 1 , 1) , block_s(32 ,6 , 1) ;
    unsigned byteSize = recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) + 
     (TOPM + DEGREE + bigDegree) * sizeof(unsigned) + (TOPM + DEGREE + bigDegree) * sizeof(float) + 2 * attr_dim * sizeof(T) + (DIM/4) * sizeof(half4) + alignof(half4) ;
    cout << "byteSize : " << (byteSize * 1.0 / 1024) << "KB" << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    // select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter<<<grid_s , block_s , byteSize>>> (graph_infos[0], data_, query, 
    //     results_id, results_dis, ent_pts , degree , L , dim , graph_infos[2] , graph_infos[1] ,
    //     partition_infos[0] , partition_infos[1] , partition_infos[2] , l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain,
    //     recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] ,  pnum , K , result_k_id , task_id_mapper , partition_infos[3]) ;
    // select_path_contain_attr_filter_V2_without_output_with_postfilter<<<grid_s , block_s , byteSize>>>(graph_infos[0], data_, query, 
    //     results_id, results_dis, ent_pts , ent_pts_post , degree , L , dim , graph_infos[2] , graph_infos[1] ,
    //     partition_infos[0] , partition_infos[1] , partition_infos[2] , l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain ,
    //     recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , pnum , K , result_k_id , bigGraph, bigDegree 
    //     , postfilter_threshold, task_id_mapper) ;
    
    select_path_contain_attr_filter_V2_without_output_with_post_pre_filter<T><<<grid_s , block_s , byteSize>>>(graph_infos[0], data_, query, 
        results_id, results_dis, ent_pts , ent_pts_post , degree , L , dim , graph_infos[2] , graph_infos[1] ,
        partition_infos[0] , partition_infos[1] , partition_infos[2] , l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain ,
        recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , pnum , K , result_k_id , bigGraph, bigDegree 
        , postfilter_threshold, task_id_mapper, partition_infos[3]) ;
    // select_path_contain_attr_filter_V2_without_output<<<grid_s , block_s , byteSize>>>(graph_infos[0], data_, query, 
    //     results_id, results_dis, ent_pts , degree , L , dim , graph_infos[2] ,graph_infos[1] ,
    //     partition_infos[0] , partition_infos[1] ,partition_infos[2] , l_bound , r_bound , attrs ,  attr_dim , global_edges , edge_num_contain,
    //     recycle_list_size ,idx_mapper[0] , idx_mapper[1] ,idx_mapper[2] , pnum , K, result_k_id , task_id_mapper) ;
    cudaDeviceSynchronize() ;

    auto e = chrono::high_resolution_clock::now() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    select_path_kernal_cost = chrono::duration<double>(e - s).count() ;

    vector<vector<unsigned>> results(qnum , vector<unsigned>(K)) ;
    s = chrono::high_resolution_clock::now() ;
    vector<unsigned> result_source(K * qnum) ;
    cudaMemcpy(result_source.data() , result_k_id , qnum * K * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    for(unsigned i = 0 ; i < qnum ;i  ++) {
        results[i] = vector<unsigned>(result_source.begin() + i * K  , result_source.begin() + (i + 1) * K) ;
    }
    e = chrono::high_resolution_clock::now() ;
    float result_load_cost = chrono::duration<double>(e - s).count() ;
    // cout << "搜索核函数运行时间:" << cost << endl ;

    s = chrono::high_resolution_clock::now() ;
    unsigned* symbol_address , *symbol_address_dcnt , *symbol_address_pmcnt ; 
    cudaGetSymbolAddress((void**) &symbol_address , jump_num_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_dcnt , dis_cal_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_pmcnt , post_merge_cnt) ;
    unsigned total_j_num = thrust::reduce(thrust::device , symbol_address , symbol_address + qnum , 0) ;
    unsigned total_c_num = thrust::reduce(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    unsigned total_m_num = thrust::reduce(thrust::device , symbol_address_pmcnt , symbol_address_pmcnt + qnum , 0) ;
    e = chrono::high_resolution_clock::now() ;
    float jump_cal_cost = chrono::duration<double>(e - s).count() ; 
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "总的跳转次数: " << total_j_num << " , 平均跳转次数: " << (total_j_num * 1.0 / qnum) << endl ;
    cout << "总的距离计算次数: " << total_c_num << ", 平均距离计算次数: " << (total_c_num * 1.0 / qnum) << endl ;
    cout << "总的后合并次数: " << total_m_num << ", 平均后合并次数: " << (total_m_num * 1.0 / qnum) << endl ; 
    cout << "结果传递时间: " << result_load_cost << endl ;
    cout << "计算跳数时间: " << jump_cal_cost << endl ;
    cout << "获取入口点时间: " << find_ep_cost << endl ; 
    cout << "排序时间: " << sort_cost << endl ;

    COST::cost3.push_back(select_path_kernal_cost) ;
    thrust::fill(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    ms = chrono::high_resolution_clock::now() ;
    cudaFree(results_id) ;
    cudaFree(results_dis) ;
    cudaFree(ent_pts) ;
    cudaFree(result_k_id) ;
    cudaFree(ent_pts_post) ;
    me = chrono::high_resolution_clock::now() ;
    m_cost += chrono::duration<double>(me - ms).count() ;
    cout << "管理内存耗时: " << m_cost << endl ;
    return results ;
}


vector<vector<unsigned>> batch_graph_search_gpu_with_postfilter(array<unsigned* , 3>& graph_infos , unsigned degree ,  __half* data_ , __half* query ,unsigned dim , unsigned qnum ,  float* l_bound , float* r_bound , float* attrs , 
    unsigned attr_dim  , array<unsigned*,3>& idx_mapper , array<unsigned* , 5>& partition_infos , unsigned L , unsigned K , 
    unsigned* global_edges , unsigned edge_num_contain , unsigned pnum , unsigned recycle_list_size , unsigned postfilter_threshold , const unsigned* bigGraph , const unsigned bigDegree) {
    // global_idx[subgraph_offset[subgraph_id] + local_id] = global_id 
    // 输入: 图, data_, query,  l_bound , r_bound , attr_dim , attrs , 每个子图中的点个数, 每个分区的全局id开始索引, 每个分区第i个点对应的全局id 
    // 每个查询所对应的分区列表, 分区个数, 分区起始索引
    unsigned DIM = dim , TOPM = L ,  DEGREE = degree ;
    unsigned *results_id ; 
    float *results_dis ; 
    unsigned *result_k_id ; 
    unsigned *task_id_mapper ; 


    auto ms = chrono::high_resolution_clock::now() ;
    cudaMalloc((void**) &results_id , qnum * L * sizeof(unsigned)) ;
    cudaMalloc((void**) &results_dis , qnum * L * sizeof(unsigned)) ;
    cudaMalloc((void**) &result_k_id , qnum * K * sizeof(unsigned)) ;
    cudaMalloc((void**) &task_id_mapper , qnum * sizeof(unsigned)) ;

    // 将任务按照分区的个数重新排序
    auto sorts = chrono::high_resolution_clock::now() ; 
    thrust::sequence(thrust::device , task_id_mapper , task_id_mapper + qnum) ;
    unsigned* task_size = partition_infos[1] ;
    thrust::sort(
        thrust::device , 
        task_id_mapper , task_id_mapper + qnum ,
        [task_size] __host__ __device__ (unsigned x , unsigned y) {
            return task_size[x] > task_size[y] ;
        }
    ) ;
    auto sorte = chrono::high_resolution_clock::now() ;
    float sort_cost = chrono::duration<double>(sorte - sorts).count() ;

    // ent_pts 先设置为前degree个点, 后续修改为每个查询第一个分区随机degree个
    unsigned *ent_pts , *ent_pts_post ;
    cudaMalloc((void**) &ent_pts , degree *  sizeof(unsigned)) ;
    cudaMalloc((void**) &ent_pts_post , bigDegree * sizeof(unsigned)) ;
    auto me = chrono::high_resolution_clock::now() ;
    float m_cost = chrono::duration<double>(me - ms).count() ;
    mt19937 rng ;
    uniform_int_distribution<> intd(0 ,62500 - 1) ;

    // thrust::sequence(thrust::device , ent_pts , ent_pts + degree) ;
    auto es = chrono::high_resolution_clock::now() ;
    vector<unsigned> ent_pts_random(bigDegree * 10) ;
    // for(unsigned i = 0 ; i < 62500 ; i ++)
    //     ent_pts_random[i] = i ;
    // random_shuffle(ent_pts_random.begin() , ent_pts_random.end()) ;
    unordered_set<unsigned> ent_pts_contains ;
    unsigned ent_pts_ofst = 0 ;
    while(ent_pts_ofst < degree * 10) {
        unsigned t = intd(rng) ;
        if(!ent_pts_contains.count(t)) {
            ent_pts_contains.insert(t) ;
            ent_pts_random[ent_pts_ofst ++] = t ;
        }
    }
    random_shuffle(ent_pts_random.begin() , ent_pts_random.begin() + degree * 10) ;

    cudaMemcpy(ent_pts , ent_pts_random.data() , degree *  sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    ent_pts_contains.clear() ;
    // ent_pts_random.clear() ;
    intd = uniform_int_distribution<>(0 , 1000000 - 1) ;
    ent_pts_ofst = 0 ;
    while(ent_pts_ofst < bigDegree * 10) {
        unsigned t = intd(rng) ;
        if(!ent_pts_contains.count(t)) {
            ent_pts_contains.insert(t) ;
            ent_pts_random[ent_pts_ofst ++] = t ; 
        }
    }
    random_shuffle(ent_pts_random.begin() , ent_pts_random.begin() + bigDegree * 10) ;
    cudaMemcpy(ent_pts_post , ent_pts_random.data() , bigDegree * sizeof(unsigned) , cudaMemcpyHostToDevice) ;

    auto ee = chrono::high_resolution_clock::now() ;
    float find_ep_cost = chrono::duration<double>(ee - es).count() ;

    dim3 grid_s(qnum , 1 , 1) , block_s(32 ,6 , 1) ;
    unsigned byteSize = recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) + 
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float) + (DIM/4) * sizeof(half4) + alignof(half4) ;
    cout << "byteSize : " << (byteSize * 1.0 / 1024) << "KB" << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    // select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter<<<grid_s , block_s , byteSize>>> (graph_infos[0], data_, query, 
    //     results_id, results_dis, ent_pts , degree , L , dim , graph_infos[2] , graph_infos[1] ,
    //     partition_infos[0] , partition_infos[1] , partition_infos[2] , l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain,
    //     recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] ,  pnum , K , result_k_id , task_id_mapper , partition_infos[3]) ;
    select_path_contain_attr_filter_V2_without_output_with_postfilter<<<grid_s , block_s , byteSize>>>(graph_infos[0], data_, query, 
        results_id, results_dis, ent_pts , ent_pts_post , degree , L , dim , graph_infos[2] , graph_infos[1] ,
        partition_infos[0] , partition_infos[1] , partition_infos[2] , l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain ,
        recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , pnum , K , result_k_id , bigGraph, bigDegree 
        , postfilter_threshold, task_id_mapper) ;
    
    // select_path_contain_attr_filter_V2_without_output_with_post_pre_filter<<<grid_s , block_s , byteSize>>>(graph_infos[0], data_, query, 
    //     results_id, results_dis, ent_pts , ent_pts_post , degree , L , dim , graph_infos[2] , graph_infos[1] ,
    //     partition_infos[0] , partition_infos[1] , partition_infos[2] , l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain ,
    //     recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , pnum , K , result_k_id , bigGraph, bigDegree 
    //     , postfilter_threshold, task_id_mapper, partition_infos[3]) ;
    // select_path_contain_attr_filter_V2_without_output<<<grid_s , block_s , byteSize>>>(graph_infos[0], data_, query, 
    //     results_id, results_dis, ent_pts , degree , L , dim , graph_infos[2] ,graph_infos[1] ,
    //     partition_infos[0] , partition_infos[1] ,partition_infos[2] , l_bound , r_bound , attrs ,  attr_dim , global_edges , edge_num_contain,
    //     recycle_list_size ,idx_mapper[0] , idx_mapper[1] ,idx_mapper[2] , pnum , K, result_k_id , task_id_mapper) ;
    cudaDeviceSynchronize() ;

    auto e = chrono::high_resolution_clock::now() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    select_path_kernal_cost = chrono::duration<double>(e - s).count() ;

    vector<vector<unsigned>> results(qnum , vector<unsigned>(K)) ;
    s = chrono::high_resolution_clock::now() ;
    vector<unsigned> result_source(K * qnum) ;
    cudaMemcpy(result_source.data() , result_k_id , qnum * K * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    for(unsigned i = 0 ; i < qnum ;i  ++) {
        results[i] = vector<unsigned>(result_source.begin() + i * K  , result_source.begin() + (i + 1) * K) ;
    }
    e = chrono::high_resolution_clock::now() ;
    float result_load_cost = chrono::duration<double>(e - s).count() ;
    // cout << "搜索核函数运行时间:" << cost << endl ;

    s = chrono::high_resolution_clock::now() ;
    unsigned* symbol_address , *symbol_address_dcnt , *symbol_address_pmcnt ; 
    cudaGetSymbolAddress((void**) &symbol_address , jump_num_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_dcnt , dis_cal_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_pmcnt , post_merge_cnt) ;
    unsigned total_j_num = thrust::reduce(thrust::device , symbol_address , symbol_address + qnum , 0) ;
    unsigned total_c_num = thrust::reduce(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    unsigned total_m_num = thrust::reduce(thrust::device , symbol_address_pmcnt , symbol_address_pmcnt + qnum , 0) ;
    e = chrono::high_resolution_clock::now() ;
    float jump_cal_cost = chrono::duration<double>(e - s).count() ; 
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "总的跳转次数: " << total_j_num << " , 平均跳转次数: " << (total_j_num * 1.0 / qnum) << endl ;
    cout << "总的距离计算次数: " << total_c_num << ", 平均距离计算次数: " << (total_c_num * 1.0 / qnum) << endl ;
    cout << "总的后合并次数: " << total_m_num << ", 平均后合并次数: " << (total_m_num * 1.0 / qnum) << endl ; 
    cout << "结果传递时间: " << result_load_cost << endl ;
    cout << "计算跳数时间: " << jump_cal_cost << endl ;
    cout << "获取入口点时间: " << find_ep_cost << endl ; 
    cout << "排序时间: " << sort_cost << endl ;

    COST::cost3.push_back(select_path_kernal_cost) ;
    thrust::fill(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    ms = chrono::high_resolution_clock::now() ;
    cudaFree(results_id) ;
    cudaFree(results_dis) ;
    cudaFree(ent_pts) ;
    cudaFree(result_k_id) ;
    cudaFree(ent_pts_post) ;
    me = chrono::high_resolution_clock::now() ;
    m_cost += chrono::duration<double>(me - ms).count() ;
    cout << "管理内存耗时: " << m_cost << endl ;
    return results ;
}

/**
    需要补齐的数组: batch_size - 当前批次大小
                   batch_partition_ids - 陈列出当前批次的所有分区id
                   global_vec_stored_pos - 每个全局id对应当前处理批次的存储位置
    以上三个数组为函数内部生成
    batch_partition_ids 可以写进分批时的代码里
**/
float other_kernal_cost = 0.0 ;
vector<float> batch_process_cost , batch_data_trans_cost ;
float mem_trans_cost = 0.0 ;
template<typename attrType = float>
vector<vector<unsigned>> batch_graph_search_gpu_with_prefilter_batch_process(unsigned batch_num , vector<vector<unsigned>>& batch_info , unsigned degree ,  __half* data_ , array<__half*,2>& vec_buffer, array<unsigned*,2>& graph_buffer , array<unsigned*,2>& global_graph_buffer ,
    array<unsigned*,2>& generic_buffer , __half* query , unsigned dim , unsigned qnum ,  attrType* l_bound , attrType* r_bound , attrType* attrs , 
    unsigned attr_dim  ,array<unsigned*,3>& graph_info , array<unsigned*,4>& idx_mapper , vector<unsigned>& seg_partition_point_start_index  , array<unsigned* , 5> partition_infos , unsigned L , unsigned K , 
    unsigned* global_edges , unsigned edge_num_contain , unsigned pnum , bool *partition_is_selected , unsigned recycle_list_size , unsigned MAX_BATCH_SIZE = 16) {
    // ******************* 传输进来的数据都当成主机端数据
    // global_idx[subgraph_offset[subgraph_id] + local_id] = global_id 
    // 输入: 图, data_, query,  l_bound , r_bound , attr_dim , attrs , 每个子图中的点个数, 每个分区的全局id开始索引, 每个分区第i个点对应的全局id 
    // 每个查询所对应的分区列表, 分区个数, 分区起始索引
    cout << "dim : " << dim << ", L : " << L << ", DEGREE : " << degree << endl ;
    unsigned DIM = dim , TOPM = L ,  DEGREE = degree ;
    unsigned *results_id ; 
    float *results_dis ; 
    unsigned *result_k_id ; 

    unsigned *task_id_mapper ; 
    cudaStream_t stream_non_block , stream_non_block2 ;  
    cudaStreamCreateWithFlags(&stream_non_block , cudaStreamNonBlocking) ;
    cudaStreamCreateWithFlags(&stream_non_block2 , cudaStreamNonBlocking) ;
    

    auto ms = chrono::high_resolution_clock::now() ;
    cudaMalloc((void**) &results_id , qnum * L * sizeof(unsigned)) ;
    cudaMalloc((void**) &results_dis , qnum * L * sizeof(float)) ;
    cudaMalloc((void**) &result_k_id , qnum * K * sizeof(unsigned)) ;
    cudaMalloc((void**) &task_id_mapper , qnum * sizeof(unsigned)) ;
    thrust::fill(thrust::device , results_id , results_id + qnum * L , 0xFFFFFFFF) ;
    thrust::fill(thrust::device , results_dis , results_dis + qnum * L , INF_DIS) ;

    array<array<unsigned* , 3> , 2> partition_buffer ; 
    cudaMalloc((void**) &partition_buffer[0][0] , qnum * pnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[1][0] , qnum * pnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[0][1] , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[1][1] , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[0][2] , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[1][2] , qnum * sizeof(unsigned)) ;

    // 将任务按照分区的个数重新排序
    auto sorts = chrono::high_resolution_clock::now() ; 
    thrust::sequence(thrust::device , task_id_mapper , task_id_mapper + qnum) ;
    unsigned* task_size = partition_infos[1] ;
    thrust::sort(
        thrust::device , 
        task_id_mapper , task_id_mapper + qnum ,
        [task_size] __host__ __device__ (unsigned x , unsigned y) {
            return task_size[x] > task_size[y] ;
        }
    ) ;
    auto sorte = chrono::high_resolution_clock::now() ;
    float sort_cost = chrono::duration<double>(sorte - sorts).count() ;

    // ent_pts 先设置为前degree个点, 后续修改为每个查询第一个分区随机degree个
    unsigned *ent_pts ;
    cudaMalloc((void**) &ent_pts , L *  sizeof(unsigned)) ;
    auto me = chrono::high_resolution_clock::now() ;
    float m_cost = chrono::duration<double>(me - ms).count() ;

    // thrust::sequence(thrust::device , ent_pts , ent_pts + degree) ;
    auto es = chrono::high_resolution_clock::now() ;
    vector<unsigned> ent_pts_random(L) ;
    unordered_set<unsigned> ent_pts_contains ;
    // for(unsigned i = 0 ; i < 62500 ; i ++)
        // ent_pts_random[i] = i ;
    // random_shuffle(ent_pts_random.begin() , ent_pts_random.end()) ;
    mt19937 rng ;
    uniform_int_distribution<> intd(0 , graph_info[1][0] - 1) ;
    // unsigned qids_ofst = 0 ;
    unsigned ent_pts_ofst = 0 ;
    while(ent_pts_ofst < L) {
        unsigned t = intd(rng) ;
        if(!ent_pts_contains.count(t)) {
            ent_pts_contains.insert(t) ;
            ent_pts_random[ent_pts_ofst ++] = t ;
        }
    }
    cudaMemcpy(ent_pts , ent_pts_random.data() , L *  sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    auto ee = chrono::high_resolution_clock::now() ;
    float find_ep_cost = chrono::duration<double>(ee - es).count() ;

    // 将小分区的数量改为标记信息
    auto cit_first = thrust::make_counting_iterator<unsigned>(0) ;
    // unsigned* p_counts = partition_info[1] , * little_p_counts = partition_info[3] ;
    thrust::transform(
        thrust::device ,
        partition_infos[1] , partition_infos[1] + qnum ,
        partition_infos[3] , 
        partition_infos[3] ,
        [] __host__ __device__ (const unsigned a , const unsigned b) -> unsigned {
            return a == b ;
        }
    ) ;
    // vector<unsigned> ptif3(qnum) ;
    // cudaMemcpy(ptif3.data() , partition_infos[3] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // cout << "partition info 3 : [" ;
    // for(unsigned i = 0 ; i < qnum ; i ++) {
    //     cout << ptif3[i] ;
    //     if(i < qnum - 1)
    //         cout << " ," ; 
    // }
    // cout << "]" << endl ;
    // cout << "seg_partition_point_start_index[" ;
    // for(unsigned i = 0 ; i < 16 ; i ++) {
    //     cout << seg_partition_point_start_index[i] ;
    //     if(i < 15)
    //         cout << " ," ;
    // }
    // cout << "]" << endl ;

    vector<unsigned> batch_graph_start_index_host(MAX_BATCH_SIZE) , batch_graph_size_host(MAX_BATCH_SIZE) , batch_partition_ids_host(MAX_BATCH_SIZE) ;
    dim3 grid_s(qnum , 1 , 1) , block_s(32 ,6 , 1) ;
    unsigned turn = 0 ; 
    cout << "进入批处理函数" << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    // 执行批次处理
    for(unsigned batch = 0 ; batch < batch_num ; batch ++) {
        size_t start_pointer[3] = {0} ; 
        size_t mapper_start = 0 ; 
        unsigned batch_size = batch_info[batch].size() ;
        // thrust::fill(thrust::device , idx_mapper[3] , idx_mapper[3] + 1000000 , 0xFFFFFFFFu) ;
        cout << "batch" << batch << "[" ;
        for(unsigned i = 0 ; i < batch_size ; i ++) {
            cout << batch_info[batch][i] ;
            if(i < batch_size - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;

        auto mps = chrono::high_resolution_clock::now() ;
        // 先把当前批次的上下文载入显存 (包括向量、图、全局图) 
        for(unsigned i = 0 ; i < batch_size ; i ++) {
            unsigned pid = batch_info[batch][i] ;
            size_t data_offset = static_cast<size_t>(seg_partition_point_start_index[pid]) * static_cast<size_t>(dim) ;
            size_t data_byteSize = static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(dim) * sizeof(__half) ;
            // cout << "data_offset : " << data_offset << endl ;
            // cout << "data_byteSize : " << data_byteSize << endl ;
            cudaMemcpyAsync(vec_buffer[turn] + start_pointer[0], data_ + data_offset , data_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
            // cudaMemcpy(vec_buffer[turn] + start_pointer[0], data_ + data_offset , data_byteSize , cudaMemcpyHostToDevice) ;
            CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            size_t graph_byteSize = static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(degree) * sizeof(unsigned) ;
            cudaMemcpyAsync(graph_buffer[turn] + start_pointer[1] , graph_info[0] + graph_info[2][pid] , graph_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
            // cudaMemcpy(graph_buffer[turn] + start_pointer[1] , graph_info[0] + graph_info[2][pid] , graph_byteSize , cudaMemcpyHostToDevice) ;
            CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            // unsigned* graph_buffer_host = new unsigned[graph_info[1][pid] * degree] ;
            // cudaMemcpy(graph_buffer_host , graph_buffer[turn] + start_pointer[1] , graph_byteSize , cudaMemcpyDeviceToHost) ;
            // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            // for(unsigned d = 0 ; d < graph_info[1][pid] * degree ;  d++) {
            //     if(graph_buffer_host[d] != graph_info[0][graph_info[2][pid] + d]) {
            //         cout << "error : " << graph_info[0][graph_info[2][pid] + d] << "-" << graph_buffer_host[d] << endl ;
            //     }
            // }
            // cout << "检查完成-------" << endl ;
            
            size_t global_graph_offset = static_cast<size_t>(seg_partition_point_start_index[pid]) * static_cast<size_t>(pnum) * static_cast<size_t>(edge_num_contain) ;
            size_t global_graph_byteSize = static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(pnum) * static_cast<size_t>(edge_num_contain) * sizeof(unsigned) ;
            cudaMemcpyAsync(global_graph_buffer[turn] + start_pointer[2] , global_edges + global_graph_offset , 
            global_graph_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
            // 映射所涉及全局id的存储位置
            thrust::transform(
                thrust::cuda::par.on(stream_non_block) , 
                // idx_mapper[3] + seg_partition_point_start_index[pid] , idx_mapper[3] + seg_partition_point_start_index[pid] + graph_info[1][pid] ,
                cit_first  , cit_first + graph_info[1][pid] ,
                idx_mapper[3] + seg_partition_point_start_index[pid] ,
                [mapper_start] __host__ __device__ (const unsigned idx) {
                    return mapper_start + idx ; 
                }
            ) ;
            batch_graph_start_index_host[i] = start_pointer[1] ;
            batch_graph_size_host[i] = graph_info[1][pid] ; 
            start_pointer[0] += static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(dim) ;
            start_pointer[1] += static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(degree) ;
            start_pointer[2] += static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(pnum) * static_cast<size_t>(edge_num_contain) ;
            mapper_start += static_cast<size_t>(graph_info[1][pid]) ;
            // cout << "start pointer0 : " << start_pointer[0] << endl  ;
            // cout << "start pointer1 : " << start_pointer[1] << endl  ;
            // cout << "start pointer2 : " << start_pointer[2] << endl  ;
            // cout << "mapper_start : " << mapper_start << endl ;
        }
        cout << "内存导入完毕 batch : " << batch <<  endl ;
        unsigned* batch_global_graph_start_index_address_pointer = generic_buffer[turn] ;
        unsigned* batch_graph_size_address_pointer = batch_global_graph_start_index_address_pointer + batch_size ;
        unsigned* batch_partition_ids_address_pointer = batch_graph_size_address_pointer + batch_size ;
        // 以下变量后面要修改成传参
        // unsigned* batch_little_seg_counts_address_pointer = batch_partition_ids_address_pointer + batch_size ;
        
        // thrust::transform(
        //     thrust::cuda::par.on(stream_non_block) , 
        //     // partition_infos[0] , partition_infos[0] + qnum * pnum ,
        //     cit_first , cit_first + qnum * pnum ,
        //     // partition_infos[0] ,
        //     partition_buffer[turn][0] ,
        //     [] __host__ __device__ (const unsigned idx) {
        //         // return (idx % pnum) % 4 ;
        //         return idx % 4 ;
        //     }
        // ) ;
        
        // thrust::transform(
        //     thrust::cuda::par.on(stream_non_block) , 
        //     // partition_infos[2] , partition_infos[2] + qnum ,
        //     cit_first , cit_first + qnum ,
        //     // partition_infos[2] ,
        //     partition_buffer[turn][2] ,
        //     [] __host__ __device__ (const unsigned idx) {
        //         return idx * 4 ; 
        //     }
        // ) ;
        // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // thrust::fill(thrust::cuda::par.on(stream_non_block) ,  partition_buffer[turn][1] , partition_buffer[turn][1] + qnum , batch_size) ;
        cudaMemcpyAsync(batch_global_graph_start_index_address_pointer , batch_graph_start_index_host.data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
        cudaMemcpyAsync(batch_graph_size_address_pointer , batch_graph_size_host.data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
        cudaMemcpyAsync(batch_partition_ids_address_pointer , batch_info[batch].data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
        // thrust::fill(thrust::cuda::par.on(stream_non_block) , batch_little_seg_counts_address_pointer , batch_little_seg_counts_address_pointer + qnum , 0) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        auto row_key_begin = thrust::make_transform_iterator(
            thrust::make_counting_iterator(0) ,
            [batch_size] __host__ __device__ (unsigned idx) -> unsigned {return idx / batch_size ;}
        ) ;
        auto val_begin = thrust::make_transform_iterator(
            // partition_is_selected , 
            thrust::make_counting_iterator(0) ,
            [partition_is_selected , batch_partition_ids_address_pointer , pnum , batch_size] __host__ __device__ (unsigned idx) -> unsigned {
                // return (flag ? 1 : 0) ;
                unsigned qid = idx / batch_size , pid = idx % batch_size ; 
                bool flag = partition_is_selected[qid * pnum + batch_partition_ids_address_pointer[pid]] ;
                return (flag ? 1 : 0) ;
            }
        ) ;
        thrust::reduce_by_key(
            thrust::cuda::par.on(stream_non_block) , 
            row_key_begin , row_key_begin + batch_size * qnum ,
            val_begin , 
            thrust::make_discard_iterator() ,
            partition_buffer[turn][1] ,
            thrust::equal_to<unsigned>() ,
            thrust::plus<unsigned>()
        ) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // 根据长度信息, 得到partition_buffer[turn][2] 
        thrust::exclusive_scan(thrust::cuda::par.on(stream_non_block) , partition_buffer[turn][1] , partition_buffer[turn][1] + qnum , partition_buffer[turn][2] , 0) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // 填充 partition_buffer[0] 
        fill_partition_info0<<<(qnum + 31) / 32 , 32 ,0 ,  stream_non_block>>>(batch_size , batch_partition_ids_address_pointer , partition_buffer[turn][0] ,partition_buffer[turn][1] ,partition_buffer[turn][2] ,
            partition_is_selected , qnum , pnum) ;
        cudaDeviceSynchronize() ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        auto mpe = chrono::high_resolution_clock::now() ;
        mem_trans_cost += chrono::duration<double>(mpe - mps).count() ;
        batch_data_trans_cost.push_back(chrono::duration<double>(mpe - mps).count()) ;
        // int t1 ;
        // cin >> t1 ; 
        // 发射核函数, 开始执行
        auto kernal_s = chrono::high_resolution_clock::now() ;
        unsigned byteSize = recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) + 
        (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float) + (DIM/4) * sizeof(half4) + alignof(half4) ;
        cout << "byteSize : " << (byteSize * 1.0 / 1024) << "KB" << endl ;
        // auto inner_s = chrono::high_resolution_clock::now() ;
        select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter_batch_process<attrType><<<grid_s , block_s , byteSize>>>(batch_size , batch_partition_ids_address_pointer , graph_buffer[turn], vec_buffer[turn], query, 
            results_id , results_dis , ent_pts , DEGREE , TOPM , DIM , batch_global_graph_start_index_address_pointer , batch_graph_size_address_pointer ,
            partition_buffer[turn][0] , partition_buffer[turn][1] , partition_buffer[turn][2] , l_bound , r_bound , attrs , attr_dim , global_graph_buffer[turn] , edge_num_contain ,
            recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , pnum , K, result_k_id  , task_id_mapper , partition_infos[3]
            , idx_mapper[3]) ;
        // cudaDeviceSynchronize() ;
        // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // int t ;
        // cin >> t ;
        auto kernal_e = chrono::high_resolution_clock::now() ;
        select_path_kernal_cost += chrono::duration<double>(kernal_e - kernal_s).count() ;
        batch_process_cost.push_back(chrono::duration<double>(kernal_e - kernal_s).count()) ;
        // auto inner_e = chrono::high_resolution_clock::now() ;
        // select_path_kernal_cost += chrono::duration<double>(inner_e - inner_s).count() ;
        turn ^= 1 ; 
    }
    cudaDeviceSynchronize() ;
    auto e = chrono::high_resolution_clock::now() ;
    other_kernal_cost = chrono::duration<double>(e - s).count() ;
    
    // auto s = chrono::high_resolution_clock::now() ;
    // select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter<<<grid_s , block_s , byteSize>>> (graph_infos[0], data_, query, 
    //     results_id, results_dis, ent_pts , degree , L , dim , graph_infos[2] , graph_infos[1] ,
    //     partition_infos[0] , partition_infos[1] , partition_infos[2] , l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain,
    //     recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] ,  pnum , K , result_k_id , task_id_mapper , partition_infos[3]) ;
    // cudaDeviceSynchronize() ;
    // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // auto e = chrono::high_resolution_clock::now() ;
    // select_path_kernal_cost = chrono::duration<double>(e - s).count() ;

    vector<vector<unsigned>> results(qnum , vector<unsigned>(K)) ;
    vector<vector<float>> results_vec_dis(qnum , vector<float>(K)) ;
    s = chrono::high_resolution_clock::now() ;
    vector<unsigned> result_source(K * qnum) ;
    vector<float> result_source_dis(L * qnum) ;
    cudaMemcpy(result_source.data() , result_k_id , qnum * K * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    cudaMemcpy(result_source_dis.data() , results_dis , qnum * L * sizeof(float) , cudaMemcpyDeviceToHost) ;
    for(unsigned i = 0 ; i < qnum ;i  ++) {
        results[i] = vector<unsigned>(result_source.begin() + i * K  , result_source.begin() + (i + 1) * K) ;
    }
    for(unsigned i = 0 ; i < qnum ; i ++) {
        results_vec_dis[i] = vector<float> (result_source_dis.begin() + i * L , result_source_dis.begin() + i * L + K) ;
    }
    e = chrono::high_resolution_clock::now() ;
    float result_load_cost = chrono::duration<double>(e - s).count() ;
    // cout << "搜索核函数运行时间:" << cost << endl ;

    s = chrono::high_resolution_clock::now() ;
    unsigned* symbol_address , *symbol_address_dcnt , *symbol_address_pmcnt ; 
    cudaGetSymbolAddress((void**) &symbol_address , jump_num_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_dcnt , dis_cal_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_pmcnt , post_merge_cnt) ;
    unsigned total_j_num = thrust::reduce(thrust::device , symbol_address , symbol_address + qnum , 0) ;
    unsigned total_c_num = thrust::reduce(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    unsigned total_m_num = thrust::reduce(thrust::device , symbol_address_pmcnt , symbol_address_pmcnt + qnum , 0) ;
    e = chrono::high_resolution_clock::now() ;
    float jump_cal_cost = chrono::duration<double>(e - s).count() ; 
    // cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "总体执行时间: " << other_kernal_cost << endl ;
    cout << "总的跳转次数: " << total_j_num << " , 平均跳转次数: " << (total_j_num * 1.0 / qnum) << endl ;
    cout << "总的距离计算次数: " << total_c_num << ", 平均距离计算次数: " << (total_c_num * 1.0 / qnum) << endl ;
    cout << "总的后合并次数: " << total_m_num << ", 平均后合并次数: " << (total_m_num * 1.0 / qnum) << endl ; 
    cout << "结果传递时间: " << result_load_cost << endl ;
    cout << "计算跳数时间: " << jump_cal_cost << endl ;
    cout << "获取入口点时间: " << find_ep_cost << endl ; 
    cout << "排序时间: " << sort_cost << endl ;
    thrust::fill(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    ms = chrono::high_resolution_clock::now() ;
    cudaFree(results_id) ;
    cudaFree(results_dis) ;
    cudaFree(ent_pts) ;
    cudaFree(result_k_id) ;
    cudaFree(task_id_mapper) ;
    for(unsigned i = 0 ; i < 2 ; i  ++)
        for(unsigned j = 0 ; j < 3 ; j ++)
            cudaFree(partition_buffer[i][j]) ;
    me = chrono::high_resolution_clock::now() ;
    m_cost += chrono::duration<double>(me - ms).count() ;
    cout << "管理内存耗时: " << m_cost << endl ;

    // cout << "result dis : [" ;
    // for(unsigned i = 0 ; i < K ;i  ++) {
    //     cout << results_vec_dis[0][i]  ;
    //     if(i < K - 1)
    //         cout << " ," ;
    // }
    // cout << "]" << endl ;
    return results ;
}

template<typename attrType = float>
vector<vector<unsigned>> batch_graph_search_gpu_with_prefilter_batch_process_compressed_graph_T(unsigned batch_num , vector<vector<unsigned>>& batch_info , unsigned degree ,  __half* data_ , array<__half*,2>& vec_buffer, array<unsigned*,2>& graph_buffer , array<unsigned*,2>& global_graph_buffer ,
    array<unsigned*,2>& generic_buffer , __half* query , unsigned dim , unsigned qnum ,  attrType* l_bound , attrType* r_bound , attrType* attrs , 
    unsigned attr_dim  ,array<unsigned*,3>& graph_info , array<unsigned*,4>& idx_mapper , vector<unsigned>& seg_partition_point_start_index  , array<unsigned* , 5> partition_infos , unsigned L , unsigned K , 
    unsigned* global_edges , unsigned edge_num_contain , unsigned pnum , bool *partition_is_selected , unsigned recycle_list_size , const unsigned cardinality , unsigned MAX_BATCH_SIZE = 16) {
    // ******************* 传输进来的数据都当成主机端数据
    // global_idx[subgraph_offset[subgraph_id] + local_id] = global_id 
    // 输入: 图, data_, query,  l_bound , r_bound , attr_dim , attrs , 每个子图中的点个数, 每个分区的全局id开始索引, 每个分区第i个点对应的全局id 
    // 每个查询所对应的分区列表, 分区个数, 分区起始索引
    cout << "dim : " << dim << ", L : " << L << ", DEGREE : " << degree << endl ;
    unsigned DIM = dim , TOPM = L ,  DEGREE = degree ;
    unsigned *results_id ; 
    float *results_dis ; 
    unsigned *result_k_id ; 

    unsigned *task_id_mapper ; 
    cudaStream_t stream_non_block ; 
    cudaStreamCreateWithFlags(&stream_non_block , cudaStreamNonBlocking) ;


    auto ms = chrono::high_resolution_clock::now() ;
    cudaMalloc((void**) &results_id , qnum * L * sizeof(unsigned)) ;
    cudaMalloc((void**) &results_dis , qnum * L * sizeof(float)) ;
    cudaMalloc((void**) &result_k_id , qnum * K * sizeof(unsigned)) ;
    cudaMalloc((void**) &task_id_mapper , qnum * sizeof(unsigned)) ;
    thrust::fill(thrust::device , results_id , results_id + qnum * L , 0xFFFFFFFF) ;
    thrust::fill(thrust::device , results_dis , results_dis + qnum * L , INF_DIS) ;

    array<array<unsigned* , 3> , 2> partition_buffer ; 
    cudaMalloc((void**) &partition_buffer[0][0] , qnum * pnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[1][0] , qnum * pnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[0][1] , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[1][1] , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[0][2] , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[1][2] , qnum * sizeof(unsigned)) ;

    // 将任务按照分区的个数重新排序
    auto sorts = chrono::high_resolution_clock::now() ; 
    thrust::sequence(thrust::device , task_id_mapper , task_id_mapper + qnum) ;
    unsigned* task_size = partition_infos[1] ;
    thrust::sort(
        thrust::device , 
        task_id_mapper , task_id_mapper + qnum ,
        [task_size] __host__ __device__ (unsigned x , unsigned y) {
            return task_size[x] > task_size[y] ;
        }
    ) ;
    auto sorte = chrono::high_resolution_clock::now() ;
    float sort_cost = chrono::duration<double>(sorte - sorts).count() ;

    // ent_pts 先设置为前degree个点, 后续修改为每个查询第一个分区随机degree个
    unsigned *ent_pts ;
    cudaMalloc((void**) &ent_pts , L *  sizeof(unsigned)) ;
    auto me = chrono::high_resolution_clock::now() ;
    float m_cost = chrono::duration<double>(me - ms).count() ;

    // thrust::sequence(thrust::device , ent_pts , ent_pts + degree) ;
    auto es = chrono::high_resolution_clock::now() ;
    vector<unsigned> ent_pts_random(L * 10) ;
    unordered_set<unsigned> ent_pts_contains ;
    // for(unsigned i = 0 ; i < 62500 ; i ++)
        // ent_pts_random[i] = i ;
    // random_shuffle(ent_pts_random.begin() , ent_pts_random.end()) ;
    unsigned rand_max = *min_element(graph_info[1] , graph_info[1] + pnum) ;
    mt19937 rng ;
    uniform_int_distribution<> intd(0 , rand_max - 1) ;
    // unsigned qids_ofst = 0 ;
    unsigned ent_pts_ofst = 0 ;
    while(ent_pts_ofst < L * 10) {
        unsigned t = intd(rng) ;
        if(!ent_pts_contains.count(t)) {
            ent_pts_contains.insert(t) ;
            ent_pts_random[ent_pts_ofst ++] = t ;
        }
    }
    random_shuffle(ent_pts_random.begin() , ent_pts_random.end()) ;
    cudaMemcpy(ent_pts , ent_pts_random.data() , L *  sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    auto ee = chrono::high_resolution_clock::now() ;
    float find_ep_cost = chrono::duration<double>(ee - es).count() ;

    // 将小分区的数量改为标记信息
    auto cit_first = thrust::make_counting_iterator<unsigned>(0) ;
    // unsigned* p_counts = partition_info[1] , * little_p_counts = partition_info[3] ;
    thrust::transform(
        thrust::device ,
        partition_infos[1] , partition_infos[1] + qnum ,
        partition_infos[3] , 
        partition_infos[3] ,
        [] __host__ __device__ (const unsigned a , const unsigned b) -> unsigned {
            return a == b ;
        }
    ) ;
    // vector<unsigned> ptif3(qnum) ;
    // cudaMemcpy(ptif3.data() , partition_infos[3] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // cout << "partition info 3 : [" ;
    // for(unsigned i = 0 ; i < qnum ; i ++) {
    //     cout << ptif3[i] ;
    //     if(i < qnum - 1)
    //         cout << " ," ; 
    // }
    // cout << "]" << endl ;
    // cout << "seg_partition_point_start_index[" ;
    // for(unsigned i = 0 ; i < 16 ; i ++) {
    //     cout << seg_partition_point_start_index[i] ;
    //     if(i < 15)
    //         cout << " ," ;
    // }
    // cout << "]" << endl ;

    vector<unsigned> batch_graph_start_index_host(MAX_BATCH_SIZE) , batch_graph_size_host(MAX_BATCH_SIZE) , batch_partition_ids_host(MAX_BATCH_SIZE) ;
    dim3 grid_s(qnum , 1 , 1) , block_s(32 ,6 , 1) ;
    unsigned turn = 0 ; 
    cout << "进入批处理函数" << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // 执行批次处理
    for(unsigned batch = 0 ; batch < batch_num ; batch ++) {
        size_t start_pointer[3] = {0} ; 
        size_t mapper_start = 0 ; 
        unsigned batch_size = batch_info[batch].size() ;
        // thrust::fill(thrust::device , idx_mapper[3] , idx_mapper[3] + 100000000 , 0xFFFFFFFFu) ;
        cout << "batch" << batch << "[" ;
        for(unsigned i = 0 ; i < batch_size ; i ++) {
            cout << batch_info[batch][i] ;
            if(i < batch_size - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;

        auto mps = chrono::high_resolution_clock::now() ;
        // 先把当前批次的上下文载入显存 (包括向量、图、全局图) 
        for(unsigned i = 0 ; i < batch_size ; i ++) {
            unsigned pid = batch_info[batch][i] ;
            size_t data_offset = static_cast<size_t>(seg_partition_point_start_index[pid]) * static_cast<size_t>(dim) ;
            size_t data_byteSize = static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(dim) * sizeof(__half) ;
            // cout << "data_offset : " << data_offset << endl ;
            // cout << "data_byteSize : " << data_byteSize << endl ;
            cudaMemcpyAsync(vec_buffer[turn] + start_pointer[0], data_ + data_offset , data_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
            // cudaMemcpy(vec_buffer[turn] + start_pointer[0], data_ + data_offset , data_byteSize , cudaMemcpyHostToDevice) ;
            CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            size_t graph_byteSize = static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(degree) * sizeof(unsigned) ;
            cudaMemcpyAsync(graph_buffer[turn] + start_pointer[1] , graph_info[0] + graph_info[2][pid] , graph_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
            // cudaMemcpy(graph_buffer[turn] + start_pointer[1] , graph_info[0] + graph_info[2][pid] , graph_byteSize , cudaMemcpyHostToDevice) ;
            CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            // unsigned* graph_buffer_host = new unsigned[graph_info[1][pid] * degree] ;
            // cudaMemcpy(graph_buffer_host , graph_buffer[turn] + start_pointer[1] , graph_byteSize , cudaMemcpyDeviceToHost) ;
            // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            // for(unsigned d = 0 ; d < graph_info[1][pid] * degree ;  d++) {
            //     if(graph_buffer_host[d] != graph_info[0][graph_info[2][pid] + d]) {
            //         cout << "error : " << graph_info[0][graph_info[2][pid] + d] << "-" << graph_buffer_host[d] << endl ;
            //     }
            // }
            // cout << "检查完成-------" << endl ;
            
            // size_t global_graph_offset = static_cast<size_t>(seg_partition_point_start_index[pid]) * static_cast<size_t>(pnum) * static_cast<size_t>(edge_num_contain) ;
            // size_t global_graph_byteSize = static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(pnum) * static_cast<size_t>(edge_num_contain) * sizeof(unsigned) ;
            // cudaMemcpyAsync(global_graph_buffer[turn] + start_pointer[2] , global_edges + global_graph_offset , 
            // global_graph_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
            for(unsigned j = 0 ; j < batch_size ; j ++) {
                unsigned jpid = batch_info[batch][j] ;
                // 行开始索引 + 列索引
                size_t global_graph_offset = static_cast<size_t>(pid) * static_cast<size_t>(cardinality) * static_cast<size_t>(edge_num_contain) + 
                static_cast<size_t>(seg_partition_point_start_index[jpid]) * static_cast<size_t>(edge_num_contain) ;
                size_t global_graph_byteSize = static_cast<size_t>(graph_info[1][jpid]) * static_cast<size_t>(edge_num_contain) * sizeof(unsigned) ;
                cudaMemcpyAsync(global_graph_buffer[turn] + start_pointer[2] , global_edges + global_graph_offset , global_graph_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
                start_pointer[2] += static_cast<size_t>(graph_info[1][jpid]) * static_cast<size_t>(edge_num_contain) ;
            //     
                // cout <<"pid : " << pid << ", jpid : " << jpid ;
                // cout << ", global_graph_offset : " << global_graph_offset ;
                // cout << ", global_graph_byteSize : " << global_graph_byteSize  ;
                // cout << ", start_pointer[2] : " << start_pointer[2] << endl ;
            }

            // 映射所涉及全局id的存储位置
            thrust::transform(
                thrust::cuda::par.on(stream_non_block) , 
                // idx_mapper[3] + seg_partition_point_start_index[pid] , idx_mapper[3] + seg_partition_point_start_index[pid] + graph_info[1][pid] ,
                cit_first  , cit_first + graph_info[1][pid] ,
                idx_mapper[3] + seg_partition_point_start_index[pid] ,
                [mapper_start] __host__ __device__ (const unsigned idx) {
                    return mapper_start + idx ; 
                }
            ) ;
            batch_graph_start_index_host[i] = start_pointer[1] ;
            batch_graph_size_host[i] = graph_info[1][pid] ; 
            start_pointer[0] += static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(dim) ;
            start_pointer[1] += static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(degree) ;
            // start_pointer[2] += static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(pnum) * static_cast<size_t>(edge_num_contain) ;
            mapper_start += static_cast<size_t>(graph_info[1][pid]) ;
            // cout << "start pointer0 : " << start_pointer[0] << endl  ;
            // cout << "start pointer1 : " << start_pointer[1] << endl  ;
            // cout << "start pointer2 : " << start_pointer[2] << endl  ;
            // cout << "mapper_start : " << mapper_start << endl ;
        }
        cout << "内存导入完毕 batch : " << batch <<  endl ;
        unsigned* batch_global_graph_start_index_address_pointer = generic_buffer[turn] ;
        unsigned* batch_graph_size_address_pointer = batch_global_graph_start_index_address_pointer + batch_size ;
        unsigned* batch_partition_ids_address_pointer = batch_graph_size_address_pointer + batch_size ;
        // 以下变量后面要修改成传参
        // unsigned* batch_little_seg_counts_address_pointer = batch_partition_ids_address_pointer + batch_size ;
        
        // thrust::transform(
        //     thrust::cuda::par.on(stream_non_block) , 
        //     // partition_infos[0] , partition_infos[0] + qnum * pnum ,
        //     cit_first , cit_first + qnum * pnum ,
        //     // partition_infos[0] ,
        //     partition_buffer[turn][0] ,
        //     [] __host__ __device__ (const unsigned idx) {
        //         // return (idx % pnum) % 4 ;
        //         return idx % 4 ;
        //     }
        // ) ;
        
        // thrust::transform(
        //     thrust::cuda::par.on(stream_non_block) , 
        //     // partition_infos[2] , partition_infos[2] + qnum ,
        //     cit_first , cit_first + qnum ,
        //     // partition_infos[2] ,
        //     partition_buffer[turn][2] ,
        //     [] __host__ __device__ (const unsigned idx) {
        //         return idx * 4 ; 
        //     }
        // ) ;
        // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // thrust::fill(thrust::cuda::par.on(stream_non_block) ,  partition_buffer[turn][1] , partition_buffer[turn][1] + qnum , batch_size) ;
        cudaMemcpyAsync(batch_global_graph_start_index_address_pointer , batch_graph_start_index_host.data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
        cudaMemcpyAsync(batch_graph_size_address_pointer , batch_graph_size_host.data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
        cudaMemcpyAsync(batch_partition_ids_address_pointer , batch_info[batch].data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
        // thrust::fill(thrust::cuda::par.on(stream_non_block) , batch_little_seg_counts_address_pointer , batch_little_seg_counts_address_pointer + qnum , 0) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        auto row_key_begin = thrust::make_transform_iterator(
            thrust::make_counting_iterator(0) ,
            [batch_size] __host__ __device__ (unsigned idx) -> unsigned {return idx / batch_size ;}
        ) ;
        auto val_begin = thrust::make_transform_iterator(
            // partition_is_selected , 
            thrust::make_counting_iterator(0) ,
            [partition_is_selected , batch_partition_ids_address_pointer , pnum , batch_size] __host__ __device__ (unsigned idx) -> unsigned {
                // return (flag ? 1 : 0) ;
                unsigned qid = idx / batch_size , pid = idx % batch_size ; 
                bool flag = partition_is_selected[qid * pnum + batch_partition_ids_address_pointer[pid]] ;
                return (flag ? 1 : 0) ;
            }
        ) ;
        thrust::reduce_by_key(
            thrust::cuda::par.on(stream_non_block) , 
            row_key_begin , row_key_begin + batch_size * qnum ,
            val_begin , 
            thrust::make_discard_iterator() ,
            partition_buffer[turn][1] ,
            thrust::equal_to<unsigned>() ,
            thrust::plus<unsigned>()
        ) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // 根据长度信息, 得到partition_buffer[turn][2] 
        thrust::exclusive_scan(thrust::cuda::par.on(stream_non_block) , partition_buffer[turn][1] , partition_buffer[turn][1] + qnum , partition_buffer[turn][2] , 0) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // 填充 partition_buffer[0] 
        fill_partition_info0<<<(qnum + 31) / 32 , 32 ,0 ,  stream_non_block>>>(batch_size , batch_partition_ids_address_pointer , partition_buffer[turn][0] ,partition_buffer[turn][1] ,partition_buffer[turn][2] ,
            partition_is_selected , qnum , pnum) ;
        cudaDeviceSynchronize() ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        auto mpe = chrono::high_resolution_clock::now() ;
        mem_trans_cost += chrono::duration<double>(mpe - mps).count() ;
        batch_data_trans_cost.push_back(chrono::duration<double>(mpe - mps).count()) ;
        // int t1 ;
        // cin >> t1 ; 
        // 发射核函数, 开始执行
        auto kernal_s = chrono::high_resolution_clock::now() ;
        unsigned byteSize = recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) + 
        (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float) + (DIM/4) * sizeof(half4) + alignof(half4) ;
        cout << "byteSize : " << (byteSize * 1.0 / 1024) << "KB" << endl ;
        // auto inner_s = chrono::high_resolution_clock::now() ;
        select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter_batch_process_compressed_graph_T<attrType><<<grid_s , block_s , byteSize>>>(batch_size , batch_partition_ids_address_pointer , graph_buffer[turn], vec_buffer[turn], query, 
            results_id , results_dis , ent_pts , DEGREE , TOPM , DIM , batch_global_graph_start_index_address_pointer , batch_graph_size_address_pointer ,
            partition_buffer[turn][0] , partition_buffer[turn][1] , partition_buffer[turn][2] , l_bound , r_bound , attrs , attr_dim , global_graph_buffer[turn] , edge_num_contain ,
            recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , pnum , K, result_k_id  , task_id_mapper , partition_infos[3]
            , idx_mapper[3] , mapper_start) ;
        cudaDeviceSynchronize() ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // int t ;
        // cin >> t ;
        auto kernal_e = chrono::high_resolution_clock::now() ;
        select_path_kernal_cost += chrono::duration<double>(kernal_e - kernal_s).count() ;
        batch_process_cost.push_back(chrono::duration<double>(kernal_e - kernal_s).count()) ;
        // auto inner_e = chrono::high_resolution_clock::now() ;
        // select_path_kernal_cost += chrono::duration<double>(inner_e - inner_s).count() ;
        turn ^= 1 ; 
    }
    cudaDeviceSynchronize() ;
    auto e = chrono::high_resolution_clock::now() ;
    other_kernal_cost = chrono::duration<double>(e - s).count() ;
    
    // auto s = chrono::high_resolution_clock::now() ;
    // select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter<<<grid_s , block_s , byteSize>>> (graph_infos[0], data_, query, 
    //     results_id, results_dis, ent_pts , degree , L , dim , graph_infos[2] , graph_infos[1] ,
    //     partition_infos[0] , partition_infos[1] , partition_infos[2] , l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain,
    //     recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] ,  pnum , K , result_k_id , task_id_mapper , partition_infos[3]) ;
    // cudaDeviceSynchronize() ;
    // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // auto e = chrono::high_resolution_clock::now() ;
    // select_path_kernal_cost = chrono::duration<double>(e - s).count() ;

    vector<vector<unsigned>> results(qnum , vector<unsigned>(K)) ;
    vector<vector<float>> results_vec_dis(qnum , vector<float>(K)) ;
    s = chrono::high_resolution_clock::now() ;
    vector<unsigned> result_source(K * qnum) ;
    vector<float> result_source_dis(L * qnum) ;
    cudaMemcpy(result_source.data() , result_k_id , qnum * K * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    cudaMemcpy(result_source_dis.data() , results_dis , qnum * L * sizeof(float) , cudaMemcpyDeviceToHost) ;
    for(unsigned i = 0 ; i < qnum ;i  ++) {
        results[i] = vector<unsigned>(result_source.begin() + i * K  , result_source.begin() + (i + 1) * K) ;
    }
    for(unsigned i = 0 ; i < qnum ; i ++) {
        results_vec_dis[i] = vector<float> (result_source_dis.begin() + i * L , result_source_dis.begin() + i * L + K) ;
    }
    e = chrono::high_resolution_clock::now() ;
    float result_load_cost = chrono::duration<double>(e - s).count() ;
    // cout << "搜索核函数运行时间:" << cost << endl ;

    s = chrono::high_resolution_clock::now() ;
    unsigned* symbol_address , *symbol_address_dcnt , *symbol_address_pmcnt ; 
    cudaGetSymbolAddress((void**) &symbol_address , jump_num_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_dcnt , dis_cal_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_pmcnt , post_merge_cnt) ;
    unsigned total_j_num = thrust::reduce(thrust::device , symbol_address , symbol_address + qnum , 0) ;
    unsigned total_c_num = thrust::reduce(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    unsigned total_m_num = thrust::reduce(thrust::device , symbol_address_pmcnt , symbol_address_pmcnt + qnum , 0) ;
    e = chrono::high_resolution_clock::now() ;
    float jump_cal_cost = chrono::duration<double>(e - s).count() ; 
    // cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "总体执行时间: " << other_kernal_cost << endl ;
    cout << "总的跳转次数: " << total_j_num << " , 平均跳转次数: " << (total_j_num * 1.0 / qnum) << endl ;
    cout << "总的距离计算次数: " << total_c_num << ", 平均距离计算次数: " << (total_c_num * 1.0 / qnum) << endl ;
    cout << "总的后合并次数: " << total_m_num << ", 平均后合并次数: " << (total_m_num * 1.0 / qnum) << endl ; 
    cout << "结果传递时间: " << result_load_cost << endl ;
    cout << "计算跳数时间: " << jump_cal_cost << endl ;
    cout << "获取入口点时间: " << find_ep_cost << endl ; 
    cout << "排序时间: " << sort_cost << endl ;
    thrust::fill(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    ms = chrono::high_resolution_clock::now() ;
    cudaFree(results_id) ;
    cudaFree(results_dis) ;
    cudaFree(ent_pts) ;
    cudaFree(result_k_id) ;
    cudaFree(task_id_mapper) ;
    for(unsigned i = 0 ; i < 2 ; i  ++)
        for(unsigned j = 0 ; j < 3 ; j ++)
            cudaFree(partition_buffer[i][j]) ;
    me = chrono::high_resolution_clock::now() ;
    m_cost += chrono::duration<double>(me - ms).count() ;
    cout << "管理内存耗时: " << m_cost << endl ;

    // cout << "result dis : [" ;
    // for(unsigned i = 0 ; i < K ;i  ++) {
    //     cout << results_vec_dis[0][i]  ;
    //     if(i < K - 1)
    //         cout << " ," ;
    // }
    // cout << "]" << endl ;
    return results ;
}

template<typename attrType = float>
vector<vector<unsigned>> batch_graph_search_gpu_with_prefilter_batch_process_compressed_graph_T_dataset_resident(unsigned batch_num , vector<vector<unsigned>>& batch_info , unsigned degree ,  __half* data_ , array<__half*,2>& vec_buffer, array<unsigned*,2>& graph_buffer , array<unsigned*,2>& global_graph_buffer ,
    array<unsigned*,2>& generic_buffer , __half* query , unsigned dim , unsigned qnum ,  attrType* l_bound , attrType* r_bound , attrType* attrs , 
    unsigned attr_dim  ,array<unsigned*,3>& graph_info , array<unsigned*,4>& idx_mapper , vector<unsigned>& seg_partition_point_start_index  , array<unsigned* , 5> partition_infos , unsigned L , unsigned K , 
    unsigned* global_edges , unsigned edge_num_contain , unsigned pnum , bool *partition_is_selected , unsigned recycle_list_size , const unsigned cardinality , unsigned MAX_BATCH_SIZE = 16) {
    // ******************* 传输进来的数据都当成主机端数据
    // global_idx[subgraph_offset[subgraph_id] + local_id] = global_id 
    // 输入: 图, data_, query,  l_bound , r_bound , attr_dim , attrs , 每个子图中的点个数, 每个分区的全局id开始索引, 每个分区第i个点对应的全局id 
    // 每个查询所对应的分区列表, 分区个数, 分区起始索引
    cout << "dim : " << dim << ", L : " << L << ", DEGREE : " << degree << endl ;
    unsigned DIM = dim , TOPM = L ,  DEGREE = degree ;
    unsigned *results_id ; 
    float *results_dis ; 
    unsigned *result_k_id ; 

    unsigned *task_id_mapper ; 
    cudaStream_t stream_non_block ; 
    cudaStreamCreateWithFlags(&stream_non_block , cudaStreamNonBlocking) ;


    auto ms = chrono::high_resolution_clock::now() ;
    cudaMalloc((void**) &results_id , qnum * L * sizeof(unsigned)) ;
    cudaMalloc((void**) &results_dis , qnum * L * sizeof(float)) ;
    cudaMalloc((void**) &result_k_id , qnum * K * sizeof(unsigned)) ;
    cudaMalloc((void**) &task_id_mapper , qnum * sizeof(unsigned)) ;
    thrust::fill(thrust::device , results_id , results_id + qnum * L , 0xFFFFFFFF) ;
    thrust::fill(thrust::device , results_dis , results_dis + qnum * L , INF_DIS) ;

    array<array<unsigned* , 3> , 2> partition_buffer ; 
    cudaMalloc((void**) &partition_buffer[0][0] , qnum * pnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[1][0] , qnum * pnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[0][1] , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[1][1] , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[0][2] , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[1][2] , qnum * sizeof(unsigned)) ;

    // 将任务按照分区的个数重新排序
    auto sorts = chrono::high_resolution_clock::now() ; 
    thrust::sequence(thrust::device , task_id_mapper , task_id_mapper + qnum) ;
    unsigned* task_size = partition_infos[1] ;
    thrust::sort(
        thrust::device , 
        task_id_mapper , task_id_mapper + qnum ,
        [task_size] __host__ __device__ (unsigned x , unsigned y) {
            return task_size[x] > task_size[y] ;
        }
    ) ;
    auto sorte = chrono::high_resolution_clock::now() ;
    float sort_cost = chrono::duration<double>(sorte - sorts).count() ;

    // ent_pts 先设置为前degree个点, 后续修改为每个查询第一个分区随机degree个
    unsigned *ent_pts ;
    cudaMalloc((void**) &ent_pts , L *  sizeof(unsigned)) ;
    auto me = chrono::high_resolution_clock::now() ;
    float m_cost = chrono::duration<double>(me - ms).count() ;

    // thrust::sequence(thrust::device , ent_pts , ent_pts + degree) ;
    auto es = chrono::high_resolution_clock::now() ;
    vector<unsigned> ent_pts_random(L * 10) ;
    unordered_set<unsigned> ent_pts_contains ;
    // for(unsigned i = 0 ; i < 62500 ; i ++)
        // ent_pts_random[i] = i ;
    // random_shuffle(ent_pts_random.begin() , ent_pts_random.end()) ;
    unsigned rand_max = *min_element(graph_info[1] , graph_info[1] + pnum) ;
    mt19937 rng ;
    uniform_int_distribution<> intd(0 , rand_max - 1) ;
    // unsigned qids_ofst = 0 ;
    unsigned ent_pts_ofst = 0 ;
    while(ent_pts_ofst < L * 10) {
        unsigned t = intd(rng) ;
        if(!ent_pts_contains.count(t)) {
            ent_pts_contains.insert(t) ;
            ent_pts_random[ent_pts_ofst ++] = t ;
        }
    }
    random_shuffle(ent_pts_random.begin() , ent_pts_random.end()) ;
    cudaMemcpy(ent_pts , ent_pts_random.data() , L *  sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    auto ee = chrono::high_resolution_clock::now() ;
    float find_ep_cost = chrono::duration<double>(ee - es).count() ;

    // 将小分区的数量改为标记信息
    auto cit_first = thrust::make_counting_iterator<unsigned>(0) ;
    // unsigned* p_counts = partition_info[1] , * little_p_counts = partition_info[3] ;
    thrust::transform(
        thrust::device ,
        partition_infos[1] , partition_infos[1] + qnum ,
        partition_infos[3] , 
        partition_infos[3] ,
        [] __host__ __device__ (const unsigned a , const unsigned b) -> unsigned {
            return a == b ;
        }
    ) ;
    // vector<unsigned> ptif3(qnum) ;
    // cudaMemcpy(ptif3.data() , partition_infos[3] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // cout << "partition info 3 : [" ;
    // for(unsigned i = 0 ; i < qnum ; i ++) {
    //     cout << ptif3[i] ;
    //     if(i < qnum - 1)
    //         cout << " ," ; 
    // }
    // cout << "]" << endl ;
    // cout << "seg_partition_point_start_index[" ;
    // for(unsigned i = 0 ; i < 16 ; i ++) {
    //     cout << seg_partition_point_start_index[i] ;
    //     if(i < 15)
    //         cout << " ," ;
    // }
    // cout << "]" << endl ;

    vector<unsigned> batch_graph_start_index_host(MAX_BATCH_SIZE) , batch_graph_size_host(MAX_BATCH_SIZE) , batch_partition_ids_host(MAX_BATCH_SIZE) ;
    dim3 grid_s(qnum , 1 , 1) , block_s(32 ,6 , 1) ;
    unsigned turn = 0 ; 
    cout << "进入批处理函数" << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // 执行批次处理
    for(unsigned batch = 0 ; batch < batch_num ; batch ++) {
        size_t start_pointer[3] = {0} ; 
        size_t mapper_start = 0 ; 
        unsigned batch_size = batch_info[batch].size() ;
        // thrust::fill(thrust::device , idx_mapper[3] , idx_mapper[3] + 100000000 , 0xFFFFFFFFu) ;
        cout << "batch" << batch << "[" ;
        for(unsigned i = 0 ; i < batch_size ; i ++) {
            cout << batch_info[batch][i] ;
            if(i < batch_size - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;

        auto mps = chrono::high_resolution_clock::now() ;
        // 先把当前批次的上下文载入显存 (包括向量、图、全局图) 
        for(unsigned i = 0 ; i < batch_size ; i ++) {
            unsigned pid = batch_info[batch][i] ;
            size_t data_offset = static_cast<size_t>(seg_partition_point_start_index[pid]) * static_cast<size_t>(dim) ;
            size_t data_byteSize = static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(dim) * sizeof(__half) ;
            // cout << "data_offset : " << data_offset << endl ;
            // cout << "data_byteSize : " << data_byteSize << endl ;
            // cudaMemcpyAsync(vec_buffer[turn] + start_pointer[0], data_ + data_offset , data_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
            // cudaMemcpy(vec_buffer[turn] + start_pointer[0], data_ + data_offset , data_byteSize , cudaMemcpyHostToDevice) ;
            // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            size_t graph_byteSize = static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(degree) * sizeof(unsigned) ;
            cudaMemcpyAsync(graph_buffer[turn] + start_pointer[1] , graph_info[0] + graph_info[2][pid] , graph_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
            // cudaMemcpy(graph_buffer[turn] + start_pointer[1] , graph_info[0] + graph_info[2][pid] , graph_byteSize , cudaMemcpyHostToDevice) ;
            CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            // unsigned* graph_buffer_host = new unsigned[graph_info[1][pid] * degree] ;
            // cudaMemcpy(graph_buffer_host , graph_buffer[turn] + start_pointer[1] , graph_byteSize , cudaMemcpyDeviceToHost) ;
            // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            // for(unsigned d = 0 ; d < graph_info[1][pid] * degree ;  d++) {
            //     if(graph_buffer_host[d] != graph_info[0][graph_info[2][pid] + d]) {
            //         cout << "error : " << graph_info[0][graph_info[2][pid] + d] << "-" << graph_buffer_host[d] << endl ;
            //     }
            // }
            // cout << "检查完成-------" << endl ;
            
            // size_t global_graph_offset = static_cast<size_t>(seg_partition_point_start_index[pid]) * static_cast<size_t>(pnum) * static_cast<size_t>(edge_num_contain) ;
            // size_t global_graph_byteSize = static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(pnum) * static_cast<size_t>(edge_num_contain) * sizeof(unsigned) ;
            // cudaMemcpyAsync(global_graph_buffer[turn] + start_pointer[2] , global_edges + global_graph_offset , 
            // global_graph_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
            for(unsigned j = 0 ; j < batch_size ; j ++) {
                unsigned jpid = batch_info[batch][j] ;
                // 行开始索引 + 列索引
                size_t global_graph_offset = static_cast<size_t>(pid) * static_cast<size_t>(cardinality) * static_cast<size_t>(edge_num_contain) + 
                static_cast<size_t>(seg_partition_point_start_index[jpid]) * static_cast<size_t>(edge_num_contain) ;
                size_t global_graph_byteSize = static_cast<size_t>(graph_info[1][jpid]) * static_cast<size_t>(edge_num_contain) * sizeof(unsigned) ;
                cudaMemcpyAsync(global_graph_buffer[turn] + start_pointer[2] , global_edges + global_graph_offset , global_graph_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
                start_pointer[2] += static_cast<size_t>(graph_info[1][jpid]) * static_cast<size_t>(edge_num_contain) ;
            //     
                // cout <<"pid : " << pid << ", jpid : " << jpid ;
                // cout << ", global_graph_offset : " << global_graph_offset ;
                // cout << ", global_graph_byteSize : " << global_graph_byteSize  ;
                // cout << ", start_pointer[2] : " << start_pointer[2] << endl ;
            }

            // 映射所涉及全局id的存储位置
            thrust::transform(
                thrust::cuda::par.on(stream_non_block) , 
                // idx_mapper[3] + seg_partition_point_start_index[pid] , idx_mapper[3] + seg_partition_point_start_index[pid] + graph_info[1][pid] ,
                cit_first  , cit_first + graph_info[1][pid] ,
                idx_mapper[3] + seg_partition_point_start_index[pid] ,
                [mapper_start] __host__ __device__ (const unsigned idx) {
                    return mapper_start + idx ; 
                }
            ) ;
            batch_graph_start_index_host[i] = start_pointer[1] ;
            batch_graph_size_host[i] = graph_info[1][pid] ; 
            start_pointer[0] += static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(dim) ;
            start_pointer[1] += static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(degree) ;
            // start_pointer[2] += static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(pnum) * static_cast<size_t>(edge_num_contain) ;
            mapper_start += static_cast<size_t>(graph_info[1][pid]) ;
            // cout << "start pointer0 : " << start_pointer[0] << endl  ;
            // cout << "start pointer1 : " << start_pointer[1] << endl  ;
            // cout << "start pointer2 : " << start_pointer[2] << endl  ;
            // cout << "mapper_start : " << mapper_start << endl ;
        }
        cout << "内存导入完毕 batch : " << batch <<  endl ;
        unsigned* batch_global_graph_start_index_address_pointer = generic_buffer[turn] ;
        unsigned* batch_graph_size_address_pointer = batch_global_graph_start_index_address_pointer + batch_size ;
        unsigned* batch_partition_ids_address_pointer = batch_graph_size_address_pointer + batch_size ;
        // 以下变量后面要修改成传参
        // unsigned* batch_little_seg_counts_address_pointer = batch_partition_ids_address_pointer + batch_size ;
        
        // thrust::transform(
        //     thrust::cuda::par.on(stream_non_block) , 
        //     // partition_infos[0] , partition_infos[0] + qnum * pnum ,
        //     cit_first , cit_first + qnum * pnum ,
        //     // partition_infos[0] ,
        //     partition_buffer[turn][0] ,
        //     [] __host__ __device__ (const unsigned idx) {
        //         // return (idx % pnum) % 4 ;
        //         return idx % 4 ;
        //     }
        // ) ;
        
        // thrust::transform(
        //     thrust::cuda::par.on(stream_non_block) , 
        //     // partition_infos[2] , partition_infos[2] + qnum ,
        //     cit_first , cit_first + qnum ,
        //     // partition_infos[2] ,
        //     partition_buffer[turn][2] ,
        //     [] __host__ __device__ (const unsigned idx) {
        //         return idx * 4 ; 
        //     }
        // ) ;
        // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // thrust::fill(thrust::cuda::par.on(stream_non_block) ,  partition_buffer[turn][1] , partition_buffer[turn][1] + qnum , batch_size) ;
        cudaMemcpyAsync(batch_global_graph_start_index_address_pointer , batch_graph_start_index_host.data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
        cudaMemcpyAsync(batch_graph_size_address_pointer , batch_graph_size_host.data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
        cudaMemcpyAsync(batch_partition_ids_address_pointer , batch_info[batch].data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
        // thrust::fill(thrust::cuda::par.on(stream_non_block) , batch_little_seg_counts_address_pointer , batch_little_seg_counts_address_pointer + qnum , 0) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        auto row_key_begin = thrust::make_transform_iterator(
            thrust::make_counting_iterator(0) ,
            [batch_size] __host__ __device__ (unsigned idx) -> unsigned {return idx / batch_size ;}
        ) ;
        auto val_begin = thrust::make_transform_iterator(
            // partition_is_selected , 
            thrust::make_counting_iterator(0) ,
            [partition_is_selected , batch_partition_ids_address_pointer , pnum , batch_size] __host__ __device__ (unsigned idx) -> unsigned {
                // return (flag ? 1 : 0) ;
                unsigned qid = idx / batch_size , pid = idx % batch_size ; 
                bool flag = partition_is_selected[qid * pnum + batch_partition_ids_address_pointer[pid]] ;
                return (flag ? 1 : 0) ;
            }
        ) ;
        thrust::reduce_by_key(
            thrust::cuda::par.on(stream_non_block) , 
            row_key_begin , row_key_begin + batch_size * qnum ,
            val_begin , 
            thrust::make_discard_iterator() ,
            partition_buffer[turn][1] ,
            thrust::equal_to<unsigned>() ,
            thrust::plus<unsigned>()
        ) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // 根据长度信息, 得到partition_buffer[turn][2] 
        thrust::exclusive_scan(thrust::cuda::par.on(stream_non_block) , partition_buffer[turn][1] , partition_buffer[turn][1] + qnum , partition_buffer[turn][2] , 0) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // 填充 partition_buffer[0] 
        fill_partition_info0<<<(qnum + 31) / 32 , 32 ,0 ,  stream_non_block>>>(batch_size , batch_partition_ids_address_pointer , partition_buffer[turn][0] ,partition_buffer[turn][1] ,partition_buffer[turn][2] ,
            partition_is_selected , qnum , pnum) ;
        cudaDeviceSynchronize() ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        auto mpe = chrono::high_resolution_clock::now() ;
        mem_trans_cost += chrono::duration<double>(mpe - mps).count() ;
        batch_data_trans_cost.push_back(chrono::duration<double>(mpe - mps).count()) ;
        // int t1 ;
        // cin >> t1 ; 
        // 发射核函数, 开始执行
        auto kernal_s = chrono::high_resolution_clock::now() ;
        unsigned byteSize = recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) + 
        (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float) + (DIM/4) * sizeof(half4) + alignof(half4) ;
        cout << "byteSize : " << (byteSize * 1.0 / 1024) << "KB" << endl ;
        // auto inner_s = chrono::high_resolution_clock::now() ;
        select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter_batch_process_compressed_graph_T_dataset_resident<attrType><<<grid_s , block_s , byteSize>>>(batch_size , batch_partition_ids_address_pointer , graph_buffer[turn], vec_buffer[turn], query, 
            results_id , results_dis , ent_pts , DEGREE , TOPM , DIM , batch_global_graph_start_index_address_pointer , batch_graph_size_address_pointer ,
            partition_buffer[turn][0] , partition_buffer[turn][1] , partition_buffer[turn][2] , l_bound , r_bound , attrs , attr_dim , global_graph_buffer[turn] , edge_num_contain ,
            recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , pnum , K, result_k_id  , task_id_mapper , partition_infos[3]
            , idx_mapper[3] , mapper_start) ;
        cudaDeviceSynchronize() ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // int t ;
        // cin >> t ;
        auto kernal_e = chrono::high_resolution_clock::now() ;
        select_path_kernal_cost += chrono::duration<double>(kernal_e - kernal_s).count() ;
        batch_process_cost.push_back(chrono::duration<double>(kernal_e - kernal_s).count()) ;
        // auto inner_e = chrono::high_resolution_clock::now() ;
        // select_path_kernal_cost += chrono::duration<double>(inner_e - inner_s).count() ;
        turn ^= 1 ; 
    }
    cudaDeviceSynchronize() ;
    auto e = chrono::high_resolution_clock::now() ;
    other_kernal_cost = chrono::duration<double>(e - s).count() ;
    
    // auto s = chrono::high_resolution_clock::now() ;
    // select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter<<<grid_s , block_s , byteSize>>> (graph_infos[0], data_, query, 
    //     results_id, results_dis, ent_pts , degree , L , dim , graph_infos[2] , graph_infos[1] ,
    //     partition_infos[0] , partition_infos[1] , partition_infos[2] , l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain,
    //     recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] ,  pnum , K , result_k_id , task_id_mapper , partition_infos[3]) ;
    // cudaDeviceSynchronize() ;
    // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // auto e = chrono::high_resolution_clock::now() ;
    // select_path_kernal_cost = chrono::duration<double>(e - s).count() ;

    vector<vector<unsigned>> results(qnum , vector<unsigned>(K)) ;
    vector<vector<float>> results_vec_dis(qnum , vector<float>(K)) ;
    s = chrono::high_resolution_clock::now() ;
    vector<unsigned> result_source(K * qnum) ;
    vector<float> result_source_dis(L * qnum) ;
    cudaMemcpy(result_source.data() , result_k_id , qnum * K * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    cudaMemcpy(result_source_dis.data() , results_dis , qnum * L * sizeof(float) , cudaMemcpyDeviceToHost) ;
    for(unsigned i = 0 ; i < qnum ;i  ++) {
        results[i] = vector<unsigned>(result_source.begin() + i * K  , result_source.begin() + (i + 1) * K) ;
    }
    for(unsigned i = 0 ; i < qnum ; i ++) {
        results_vec_dis[i] = vector<float> (result_source_dis.begin() + i * L , result_source_dis.begin() + i * L + K) ;
    }
    e = chrono::high_resolution_clock::now() ;
    float result_load_cost = chrono::duration<double>(e - s).count() ;
    // cout << "搜索核函数运行时间:" << cost << endl ;

    s = chrono::high_resolution_clock::now() ;
    unsigned* symbol_address , *symbol_address_dcnt , *symbol_address_pmcnt ;
    double * symbol_address_dcct ; 
    cudaGetSymbolAddress((void**) &symbol_address , jump_num_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_dcnt , dis_cal_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_pmcnt , post_merge_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_dcct , dis_cal_cost_total_outer) ;
    unsigned total_j_num = thrust::reduce(thrust::device , symbol_address , symbol_address + qnum , 0) ;
    unsigned total_c_num = thrust::reduce(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    unsigned total_m_num = thrust::reduce(thrust::device , symbol_address_pmcnt , symbol_address_pmcnt + qnum , 0) ;
    double dis_cal_cost_total = thrust::reduce(thrust::device , symbol_address_dcct , symbol_address_dcct + qnum , 0.0) ;
    e = chrono::high_resolution_clock::now() ;
    float jump_cal_cost = chrono::duration<double>(e - s).count() ; 
    // cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "总体执行时间: " << other_kernal_cost << endl ;
    cout << "总的跳转次数: " << total_j_num << " , 平均跳转次数: " << (total_j_num * 1.0 / qnum) << endl ;
    cout << "总的距离计算次数: " << total_c_num << ", 平均距离计算次数: " << (total_c_num * 1.0 / qnum) << endl ;
    cout << "总的后合并次数: " << total_m_num << ", 平均后合并次数: " << (total_m_num * 1.0 / qnum) << endl ; 
    cout << "总的距离计算成本: " << dis_cal_cost_total << ", 平均距离计算成本: " << (dis_cal_cost_total / qnum) << endl ;
    cout << "结果传递时间: " << result_load_cost << endl ;
    cout << "计算跳数时间: " << jump_cal_cost << endl ;
    cout << "获取入口点时间: " << find_ep_cost << endl ; 
    cout << "排序时间: " << sort_cost << endl ;
    thrust::fill(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    ms = chrono::high_resolution_clock::now() ;
    cudaFree(results_id) ;
    cudaFree(results_dis) ;
    cudaFree(ent_pts) ;
    cudaFree(result_k_id) ;
    cudaFree(task_id_mapper) ;
    for(unsigned i = 0 ; i < 2 ; i  ++)
        for(unsigned j = 0 ; j < 3 ; j ++)
            cudaFree(partition_buffer[i][j]) ;
    me = chrono::high_resolution_clock::now() ;
    m_cost += chrono::duration<double>(me - ms).count() ;
    cout << "管理内存耗时: " << m_cost << endl ;

    // cout << "result dis : [" ;
    // for(unsigned i = 0 ; i < K ;i  ++) {
    //     cout << results_vec_dis[0][i]  ;
    //     if(i < K - 1)
    //         cout << " ," ;
    // }
    // cout << "]" << endl ;
    return results ;
}



template<typename attrType = float>
vector<vector<unsigned>> batch_graph_search_gpu_with_prefilter_batch_process_extreme_compressed_graph(unsigned batch_num , vector<vector<unsigned>>& batch_info , unsigned degree ,  __half* data_ , array<__half*,2>& vec_buffer, array<unsigned*,2>& graph_buffer , array<unsigned*,2>& global_graph_buffer ,
    array<unsigned*,2>& generic_buffer , __half* query , unsigned dim , unsigned qnum ,  attrType* l_bound , attrType* r_bound , attrType* attrs , 
    unsigned attr_dim  ,array<unsigned*,3>& graph_info , array<unsigned*,4>& idx_mapper , vector<unsigned>& seg_partition_point_start_index  , array<unsigned* , 5> partition_infos , unsigned L , unsigned K , 
    unsigned* global_edges , unsigned edge_num_contain , unsigned pnum , bool *partition_is_selected , unsigned recycle_list_size , const unsigned cardinality , array<unsigned*,5>& compressed_global_graph ,const unsigned total_spokesman_num , unsigned MAX_BATCH_SIZE = 16) {
    // ******************* 传输进来的数据都当成主机端数据
    // global_idx[subgraph_offset[subgraph_id] + local_id] = global_id 
    // 输入: 图, data_, query,  l_bound , r_bound , attr_dim , attrs , 每个子图中的点个数, 每个分区的全局id开始索引, 每个分区第i个点对应的全局id 
    // 每个查询所对应的分区列表, 分区个数, 分区起始索引
    cout << "dim : " << dim << ", L : " << L << ", DEGREE : " << degree << endl ;
    unsigned DIM = dim , TOPM = L ,  DEGREE = degree ;
    unsigned *results_id ; 
    float *results_dis ; 
    unsigned *result_k_id ; 

    unsigned *task_id_mapper ; 
    cudaStream_t stream_non_block ; 
    cudaStreamCreateWithFlags(&stream_non_block , cudaStreamNonBlocking) ;


    auto ms = chrono::high_resolution_clock::now() ;
    cudaMalloc((void**) &results_id , qnum * L * sizeof(unsigned)) ;
    cudaMalloc((void**) &results_dis , qnum * L * sizeof(float)) ;
    cudaMalloc((void**) &result_k_id , qnum * K * sizeof(unsigned)) ;
    cudaMalloc((void**) &task_id_mapper , qnum * sizeof(unsigned)) ;
    thrust::fill(thrust::device , results_id , results_id + qnum * L , 0xFFFFFFFF) ;
    thrust::fill(thrust::device , results_dis , results_dis + qnum * L , INF_DIS) ;

    array<array<unsigned* , 3> , 2> partition_buffer ; 
    cudaMalloc((void**) &partition_buffer[0][0] , qnum * pnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[1][0] , qnum * pnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[0][1] , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[1][1] , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[0][2] , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &partition_buffer[1][2] , qnum * sizeof(unsigned)) ;

    // 将任务按照分区的个数重新排序
    auto sorts = chrono::high_resolution_clock::now() ; 
    thrust::sequence(thrust::device , task_id_mapper , task_id_mapper + qnum) ;
    unsigned* task_size = partition_infos[1] ;
    thrust::sort(
        thrust::device , 
        task_id_mapper , task_id_mapper + qnum ,
        [task_size] __host__ __device__ (unsigned x , unsigned y) {
            return task_size[x] > task_size[y] ;
        }
    ) ;
    auto sorte = chrono::high_resolution_clock::now() ;
    float sort_cost = chrono::duration<double>(sorte - sorts).count() ;

    // ent_pts 先设置为前degree个点, 后续修改为每个查询第一个分区随机degree个
    unsigned *ent_pts ;
    cudaMalloc((void**) &ent_pts , L *  sizeof(unsigned)) ;
    auto me = chrono::high_resolution_clock::now() ;
    float m_cost = chrono::duration<double>(me - ms).count() ;

    // thrust::sequence(thrust::device , ent_pts , ent_pts + degree) ;
    auto es = chrono::high_resolution_clock::now() ;
    vector<unsigned> ent_pts_random(L * 10) ;
    unordered_set<unsigned> ent_pts_contains ;
    // for(unsigned i = 0 ; i < 62500 ; i ++)
        // ent_pts_random[i] = i ;
    // random_shuffle(ent_pts_random.begin() , ent_pts_random.end()) ;
    unsigned rand_max = *min_element(graph_info[1] , graph_info[1] + pnum) ;
    mt19937 rng ;
    uniform_int_distribution<> intd(0 , rand_max - 1) ;
    // unsigned qids_ofst = 0 ;
    unsigned ent_pts_ofst ;
    while(ent_pts_ofst < L * 10) {
        unsigned t = intd(rng) ;
        if(!ent_pts_contains.count(t)) {
            ent_pts_contains.insert(t) ;
            ent_pts_random[ent_pts_ofst ++] = t ;
        }
    }
    random_shuffle(ent_pts_random.begin() , ent_pts_random.end()) ;
    cudaMemcpy(ent_pts , ent_pts_random.data() , L *  sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    auto ee = chrono::high_resolution_clock::now() ;
    float find_ep_cost = chrono::duration<double>(ee - es).count() ;

    // 将小分区的数量改为标记信息
    auto cit_first = thrust::make_counting_iterator<unsigned>(0) ;
    // unsigned* p_counts = partition_info[1] , * little_p_counts = partition_info[3] ;
    thrust::transform(
        thrust::device ,
        partition_infos[1] , partition_infos[1] + qnum ,
        partition_infos[3] , 
        partition_infos[3] ,
        [] __host__ __device__ (const unsigned a , const unsigned b) -> unsigned {
            return a == b ;
        }
    ) ;
    // vector<unsigned> ptif3(qnum) ;
    // cudaMemcpy(ptif3.data() , partition_infos[3] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // cout << "partition info 3 : [" ;
    // for(unsigned i = 0 ; i < qnum ; i ++) {
    //     cout << ptif3[i] ;
    //     if(i < qnum - 1)
    //         cout << " ," ; 
    // }
    // cout << "]" << endl ;
    // cout << "seg_partition_point_start_index[" ;
    // for(unsigned i = 0 ; i < 16 ; i ++) {
    //     cout << seg_partition_point_start_index[i] ;
    //     if(i < 15)
    //         cout << " ," ;
    // }
    // cout << "]" << endl ;

    vector<unsigned> batch_graph_start_index_host(MAX_BATCH_SIZE) , batch_graph_size_host(MAX_BATCH_SIZE) , batch_partition_ids_host(MAX_BATCH_SIZE) ;
    dim3 grid_s(qnum , 1 , 1) , block_s(32 ,6 , 1) ;
    unsigned turn = 0 ; 
    cout << "进入批处理函数" << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    // 执行批次处理
    for(unsigned batch = 0 ; batch < batch_num ; batch ++) {
        size_t start_pointer[3] = {0} ; 
        size_t mapper_start = 0 ; 
        unsigned spokesman_start = 0 ;
        unsigned batch_size = batch_info[batch].size() ;
        // thrust::fill(thrust::device , idx_mapper[3] , idx_mapper[3] + 100000000 , 0xFFFFFFFFu) ;
        cout << "batch" << batch << "[" ;
        for(unsigned i = 0 ; i < batch_size ; i ++) {
            cout << batch_info[batch][i] ;
            if(i < batch_size - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;

        auto mps = chrono::high_resolution_clock::now() ;
        // 先把当前批次的上下文载入显存 (包括向量、图、全局图) 
        for(unsigned i = 0 ; i < batch_size ; i ++) {
            unsigned pid = batch_info[batch][i] ;
            size_t data_offset = static_cast<size_t>(seg_partition_point_start_index[pid]) * static_cast<size_t>(dim) ;
            size_t data_byteSize = static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(dim) * sizeof(__half) ;
            // cout << "data_offset : " << data_offset << endl ;
            // cout << "data_byteSize : " << data_byteSize << endl ;
            cudaMemcpyAsync(vec_buffer[turn] + start_pointer[0], data_ + data_offset , data_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
            // cudaMemcpy(vec_buffer[turn] + start_pointer[0], data_ + data_offset , data_byteSize , cudaMemcpyHostToDevice) ;
            CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            size_t graph_byteSize = static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(degree) * sizeof(unsigned) ;
            cudaMemcpyAsync(graph_buffer[turn] + start_pointer[1] , graph_info[0] + graph_info[2][pid] , graph_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
            // cudaMemcpy(graph_buffer[turn] + start_pointer[1] , graph_info[0] + graph_info[2][pid] , graph_byteSize , cudaMemcpyHostToDevice) ;
            CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            // unsigned* graph_buffer_host = new unsigned[graph_info[1][pid] * degree] ;
            // cudaMemcpy(graph_buffer_host , graph_buffer[turn] + start_pointer[1] , graph_byteSize , cudaMemcpyDeviceToHost) ;
            // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            // for(unsigned d = 0 ; d < graph_info[1][pid] * degree ;  d++) {
            //     if(graph_buffer_host[d] != graph_info[0][graph_info[2][pid] + d]) {
            //         cout << "error : " << graph_info[0][graph_info[2][pid] + d] << "-" << graph_buffer_host[d] << endl ;
            //     }
            // }
            // cout << "检查完成-------" << endl ;
            
            // size_t global_graph_offset = static_cast<size_t>(seg_partition_point_start_index[pid]) * static_cast<size_t>(pnum) * static_cast<size_t>(edge_num_contain) ;
            // size_t global_graph_byteSize = static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(pnum) * static_cast<size_t>(edge_num_contain) * sizeof(unsigned) ;
            // cudaMemcpyAsync(global_graph_buffer[turn] + start_pointer[2] , global_edges + global_graph_offset , 
            // global_graph_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
            for(unsigned j = 0 ; j < batch_size ; j ++) {
                unsigned jpid = batch_info[batch][j] ;
                // 行开始索引 + 列索引
                // size_t global_graph_offset = static_cast<size_t>(pid) * static_cast<size_t>(cardinality) * static_cast<size_t>(edge_num_contain) + 
                // static_cast<size_t>(seg_partition_point_start_index[jpid]) * static_cast<size_t>(edge_num_contain) ;
                // size_t global_graph_byteSize = static_cast<size_t>(graph_info[1][jpid]) * static_cast<size_t>(edge_num_contain) * sizeof(unsigned) ;
                // cudaMemcpyAsync(global_graph_buffer[turn] + start_pointer[2] , global_edges + global_graph_offset , global_graph_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
                // start_pointer[2] += static_cast<size_t>(graph_info[1][jpid]) * static_cast<size_t>(edge_num_contain) ;
                size_t global_graph_offset = static_cast<size_t>(pid) * static_cast<size_t>(total_spokesman_num) * static_cast<size_t>(edge_num_contain) + 
                    static_cast<size_t>(compressed_global_graph[2][jpid]) * static_cast<size_t>(edge_num_contain) ;
                size_t global_graph_byteSize = static_cast<size_t>(compressed_global_graph[1][jpid]) * static_cast<size_t>(edge_num_contain) * sizeof(unsigned) ;
                cudaMemcpyAsync(global_graph_buffer[turn] + start_pointer[2] , global_edges + global_graph_offset , global_graph_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
                start_pointer[2] += static_cast<size_t> (compressed_global_graph[1][jpid]) * static_cast<size_t>(edge_num_contain) ;
                // cout <<"pid : " << pid << ", jpid : " << jpid ;
                // cout << ", global_graph_offset : " << global_graph_offset ;
                // cout << ", global_graph_byteSize : " << global_graph_byteSize  ;
                // cout << ", start_pointer[2] : " << start_pointer[2] << endl ;
            }

            // 映射所涉及全局id的存储位置
            thrust::transform(
                thrust::cuda::par.on(stream_non_block) , 
                // idx_mapper[3] + seg_partition_point_start_index[pid] , idx_mapper[3] + seg_partition_point_start_index[pid] + graph_info[1][pid] ,
                cit_first  , cit_first + graph_info[1][pid] ,
                idx_mapper[3] + seg_partition_point_start_index[pid] ,
                [mapper_start] __host__ __device__ (const unsigned idx) {
                    return mapper_start + idx ; 
                }
            ) ;
            thrust::transform(
                thrust::cuda::par.on(stream_non_block) ,
                cit_first , cit_first + compressed_global_graph[1][pid] ,
                compressed_global_graph[4] + compressed_global_graph[2][pid] ,
                [spokesman_start] __device__ (const unsigned idx) {
                    return spokesman_start + idx ;
                }
            ) ;
            batch_graph_start_index_host[i] = start_pointer[1] ;
            batch_graph_size_host[i] = graph_info[1][pid] ; 
            start_pointer[0] += static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(dim) ;
            start_pointer[1] += static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(degree) ;
            // start_pointer[2] += static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(pnum) * static_cast<size_t>(edge_num_contain) ;
            mapper_start += static_cast<size_t>(graph_info[1][pid]) ;
            spokesman_start += compressed_global_graph[1][pid] ;
            // cout << "start pointer0 : " << start_pointer[0] << endl  ;
            // cout << "start pointer1 : " << start_pointer[1] << endl  ;
            // cout << "start pointer2 : " << start_pointer[2] << endl  ;
            // cout << "mapper_start : " << mapper_start << endl ;
        }
        cout << "内存导入完毕 batch : " << batch <<  endl ;
        unsigned* batch_global_graph_start_index_address_pointer = generic_buffer[turn] ;
        unsigned* batch_graph_size_address_pointer = batch_global_graph_start_index_address_pointer + batch_size ;
        unsigned* batch_partition_ids_address_pointer = batch_graph_size_address_pointer + batch_size ;
        // 以下变量后面要修改成传参
        // unsigned* batch_little_seg_counts_address_pointer = batch_partition_ids_address_pointer + batch_size ;
        
        // thrust::transform(
        //     thrust::cuda::par.on(stream_non_block) , 
        //     // partition_infos[0] , partition_infos[0] + qnum * pnum ,
        //     cit_first , cit_first + qnum * pnum ,
        //     // partition_infos[0] ,
        //     partition_buffer[turn][0] ,
        //     [] __host__ __device__ (const unsigned idx) {
        //         // return (idx % pnum) % 4 ;
        //         return idx % 4 ;
        //     }
        // ) ;
        
        // thrust::transform(
        //     thrust::cuda::par.on(stream_non_block) , 
        //     // partition_infos[2] , partition_infos[2] + qnum ,
        //     cit_first , cit_first + qnum ,
        //     // partition_infos[2] ,
        //     partition_buffer[turn][2] ,
        //     [] __host__ __device__ (const unsigned idx) {
        //         return idx * 4 ; 
        //     }
        // ) ;
        // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // thrust::fill(thrust::cuda::par.on(stream_non_block) ,  partition_buffer[turn][1] , partition_buffer[turn][1] + qnum , batch_size) ;
        cudaMemcpyAsync(batch_global_graph_start_index_address_pointer , batch_graph_start_index_host.data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
        cudaMemcpyAsync(batch_graph_size_address_pointer , batch_graph_size_host.data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
        cudaMemcpyAsync(batch_partition_ids_address_pointer , batch_info[batch].data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
        // thrust::fill(thrust::cuda::par.on(stream_non_block) , batch_little_seg_counts_address_pointer , batch_little_seg_counts_address_pointer + qnum , 0) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        auto row_key_begin = thrust::make_transform_iterator(
            thrust::make_counting_iterator(0) ,
            [batch_size] __host__ __device__ (unsigned idx) -> unsigned {return idx / batch_size ;}
        ) ;
        auto val_begin = thrust::make_transform_iterator(
            // partition_is_selected , 
            thrust::make_counting_iterator(0) ,
            [partition_is_selected , batch_partition_ids_address_pointer , pnum , batch_size] __host__ __device__ (unsigned idx) -> unsigned {
                // return (flag ? 1 : 0) ;
                unsigned qid = idx / batch_size , pid = idx % batch_size ; 
                bool flag = partition_is_selected[qid * pnum + batch_partition_ids_address_pointer[pid]] ;
                return (flag ? 1 : 0) ;
            }
        ) ;
        thrust::reduce_by_key(
            thrust::cuda::par.on(stream_non_block) , 
            row_key_begin , row_key_begin + batch_size * qnum ,
            val_begin , 
            thrust::make_discard_iterator() ,
            partition_buffer[turn][1] ,
            thrust::equal_to<unsigned>() ,
            thrust::plus<unsigned>()
        ) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // 根据长度信息, 得到partition_buffer[turn][2] 
        thrust::exclusive_scan(thrust::cuda::par.on(stream_non_block) , partition_buffer[turn][1] , partition_buffer[turn][1] + qnum , partition_buffer[turn][2] , 0) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // 填充 partition_buffer[0] 
        fill_partition_info0<<<(qnum + 31) / 32 , 32 ,0 ,  stream_non_block>>>(batch_size , batch_partition_ids_address_pointer , partition_buffer[turn][0] ,partition_buffer[turn][1] ,partition_buffer[turn][2] ,
            partition_is_selected , qnum , pnum) ;
        cudaDeviceSynchronize() ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        auto mpe = chrono::high_resolution_clock::now() ;
        mem_trans_cost += chrono::duration<double>(mpe - mps).count() ;
        batch_data_trans_cost.push_back(chrono::duration<double>(mpe - mps).count()) ;
        // int t1 ;
        // cin >> t1 ; 
        // 发射核函数, 开始执行
        auto kernal_s = chrono::high_resolution_clock::now() ;
        unsigned byteSize = recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) + 
        (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float) + (DIM/4) * sizeof(half4) + alignof(half4) ;
        cout << "byteSize : " << (byteSize * 1.0 / 1024) << "KB" << endl ;
        // auto inner_s = chrono::high_resolution_clock::now() ;
        select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter_batch_process_extreme_compressed_graph<attrType><<<grid_s , block_s , byteSize>>>(batch_size , batch_partition_ids_address_pointer , graph_buffer[turn], vec_buffer[turn], query, 
            results_id , results_dis , ent_pts , DEGREE , TOPM , DIM , batch_global_graph_start_index_address_pointer , batch_graph_size_address_pointer ,
            partition_buffer[turn][0] , partition_buffer[turn][1] , partition_buffer[turn][2] , l_bound , r_bound , attrs , attr_dim , global_graph_buffer[turn] , edge_num_contain ,
            recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , pnum , K, result_k_id  , task_id_mapper , partition_infos[3]
            , idx_mapper[3] , mapper_start , compressed_global_graph[3] , compressed_global_graph[4] , spokesman_start) ;
        cudaDeviceSynchronize() ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // int t ;
        // cin >> t ;
        auto kernal_e = chrono::high_resolution_clock::now() ;
        select_path_kernal_cost += chrono::duration<double>(kernal_e - kernal_s).count() ;
        batch_process_cost.push_back(chrono::duration<double>(kernal_e - kernal_s).count()) ;
        // auto inner_e = chrono::high_resolution_clock::now() ;
        // select_path_kernal_cost += chrono::duration<double>(inner_e - inner_s).count() ;
        turn ^= 1 ; 
    }
    cudaDeviceSynchronize() ;
    auto e = chrono::high_resolution_clock::now() ;
    other_kernal_cost = chrono::duration<double>(e - s).count() ;
    
    // auto s = chrono::high_resolution_clock::now() ;
    // select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter<<<grid_s , block_s , byteSize>>> (graph_infos[0], data_, query, 
    //     results_id, results_dis, ent_pts , degree , L , dim , graph_infos[2] , graph_infos[1] ,
    //     partition_infos[0] , partition_infos[1] , partition_infos[2] , l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain,
    //     recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] ,  pnum , K , result_k_id , task_id_mapper , partition_infos[3]) ;
    // cudaDeviceSynchronize() ;
    // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // auto e = chrono::high_resolution_clock::now() ;
    // select_path_kernal_cost = chrono::duration<double>(e - s).count() ;

    vector<vector<unsigned>> results(qnum , vector<unsigned>(K)) ;
    vector<vector<float>> results_vec_dis(qnum , vector<float>(K)) ;
    s = chrono::high_resolution_clock::now() ;
    vector<unsigned> result_source(K * qnum) ;
    vector<float> result_source_dis(L * qnum) ;
    cudaMemcpy(result_source.data() , result_k_id , qnum * K * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    cudaMemcpy(result_source_dis.data() , results_dis , qnum * L * sizeof(float) , cudaMemcpyDeviceToHost) ;
    for(unsigned i = 0 ; i < qnum ;i  ++) {
        results[i] = vector<unsigned>(result_source.begin() + i * K  , result_source.begin() + (i + 1) * K) ;
    }
    for(unsigned i = 0 ; i < qnum ; i ++) {
        results_vec_dis[i] = vector<float> (result_source_dis.begin() + i * L , result_source_dis.begin() + i * L + K) ;
    }
    e = chrono::high_resolution_clock::now() ;
    float result_load_cost = chrono::duration<double>(e - s).count() ;
    // cout << "搜索核函数运行时间:" << cost << endl ;

    s = chrono::high_resolution_clock::now() ;
    unsigned* symbol_address , *symbol_address_dcnt , *symbol_address_pmcnt ; 
    cudaGetSymbolAddress((void**) &symbol_address , jump_num_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_dcnt , dis_cal_cnt) ;
    cudaGetSymbolAddress((void**) &symbol_address_pmcnt , post_merge_cnt) ;
    unsigned total_j_num = thrust::reduce(thrust::device , symbol_address , symbol_address + qnum , 0) ;
    unsigned total_c_num = thrust::reduce(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    unsigned total_m_num = thrust::reduce(thrust::device , symbol_address_pmcnt , symbol_address_pmcnt + qnum , 0) ;
    e = chrono::high_resolution_clock::now() ;
    float jump_cal_cost = chrono::duration<double>(e - s).count() ; 
    // cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "总体执行时间: " << other_kernal_cost << endl ;
    cout << "总的跳转次数: " << total_j_num << " , 平均跳转次数: " << (total_j_num * 1.0 / qnum) << endl ;
    cout << "总的距离计算次数: " << total_c_num << ", 平均距离计算次数: " << (total_c_num * 1.0 / qnum) << endl ;
    cout << "总的后合并次数: " << total_m_num << ", 平均后合并次数: " << (total_m_num * 1.0 / qnum) << endl ; 
    cout << "结果传递时间: " << result_load_cost << endl ;
    cout << "计算跳数时间: " << jump_cal_cost << endl ;
    cout << "获取入口点时间: " << find_ep_cost << endl ; 
    cout << "排序时间: " << sort_cost << endl ;
    thrust::fill(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    ms = chrono::high_resolution_clock::now() ;
    cudaFree(results_id) ;
    cudaFree(results_dis) ;
    cudaFree(ent_pts) ;
    cudaFree(result_k_id) ;
    cudaFree(task_id_mapper) ;
    for(unsigned i = 0 ; i < 2 ; i  ++)
        for(unsigned j = 0 ; j < 3 ; j ++)
            cudaFree(partition_buffer[i][j]) ;
    me = chrono::high_resolution_clock::now() ;
    m_cost += chrono::duration<double>(me - ms).count() ;
    cout << "管理内存耗时: " << m_cost << endl ;

    // cout << "result dis : [" ;
    // for(unsigned i = 0 ; i < K ;i  ++) {
    //     cout << results_vec_dis[0][i]  ;
    //     if(i < K - 1)
    //         cout << " ," ;
    // }
    // cout << "]" << endl ;
    return results ;
}


int main1(){

    cout << "sizeof(half) = " << sizeof(half) << endl ;
    unsigned DIM = 128 , DEGREE = 16 , TOPM = 40 ;
    // 先载入数据集和图索引
    unsigned num , dim , degree , num_in_graph ;
    unsigned * graph ;
    float* data_ , * query_load ;
    load_data("/home/yhr/workspace/cuda_clustering/data/sift_base.fvecs" , data_ , num , dim) ;
    cout << "num : " << num << ", dim : " << dim << endl ;
    load_result("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/sift0.cagra" , graph , degree , num_in_graph) ;
    cout << "grapH degree:" << degree << ", num : " << num_in_graph << endl ;
    float* query_data_load ;
    unsigned query_num , query_dim ;
    load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_range_query.fvecs" , query_data_load , query_num , query_dim) ;
    // query_num = 10; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;

    // TOPM = 20 , DIM = dim ; 

    unsigned * graph_dev ;
    float* data_dev ;
    half *data_half_dev ;
    float* query_data_dev ;
    half *query_data_half_dev ;
    cudaMalloc((void**) &graph_dev , degree * num_in_graph * sizeof(unsigned)) ;
    cudaMalloc((void**)&data_dev , num * dim * sizeof(float)) ;
    cudaMalloc((void**) &data_half_dev , num * dim * sizeof(half)) ;
    cudaMalloc((void**) &query_data_dev , query_num * query_dim   * sizeof(float)) ;
    cudaMalloc((void**) &query_data_half_dev , query_num * query_dim * sizeof(half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(graph_dev , graph , degree * num_in_graph * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(data_dev , data_ , num * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(query_data_dev , query_data_load , query_num * query_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num_in_graph ; 
	// 调用实例
	f2h<<<points_num, 256>>>(data_dev, data_half_dev, points_num); // 先将数据转换成fp16
    // cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    f2h<<<query_num, 256>>>(query_data_dev, query_data_half_dev, query_num); // 将查询转换成fp16
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
	dim3 grid_s(query_num, 1, 1);
    dim3 block_s(32, 6, 1); // 6是block中的warp数量，可以调整


    unsigned* res_id , * ent_pts ;
    float* res_dis ; 
    cudaMalloc((void**) &res_id , query_num * TOPM * sizeof(unsigned)) ;
    cudaMalloc((void**) &res_dis , query_num * TOPM  * sizeof(float)) ;
    cudaMalloc((void**) &ent_pts , query_num * degree * sizeof(unsigned)) ;
    // 生成随机的入口节点
    vector<unsigned> samples(num_in_graph) ;
    for(unsigned i = 0 ; i < num_in_graph ; i ++)
        samples[i] = i ; 
    random_shuffle(samples.begin() , samples.end()) ;

    for(unsigned i =0 ; i < query_num ; i ++)
        cudaMemcpy(ent_pts + i * DEGREE , samples.data() , DEGREE * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    for(unsigned i = 0 ; i < DEGREE ; i ++)
        cout << samples[i] << " ," ;
    
    // cout << 33333333 << endl ;

    unsigned byteSize = (TOPM + DEGREE) * sizeof(unsigned) + (TOPM + DEGREE) * sizeof(float) + (DIM/4) * sizeof(half4) + alignof(half4) ;

    auto s = chrono::high_resolution_clock::now() ;
	select_path_v3<<<grid_s, block_s , byteSize>>>(graph_dev, data_half_dev , query_data_half_dev , res_id, res_dis, ent_pts , 16 , 40 , 128);
    cudaDeviceSynchronize() ;
    auto e = chrono::high_resolution_clock::now() ;
    float cost = chrono::duration<float>(e - s).count() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    // 以下: 测试当前图的精度如何
    // select_path_v3 默认返回topm个值
    vector<vector<unsigned>> ground_truth(query_num , vector<unsigned>()) ;

    #pragma omp parallel for num_threads(10)
    for(unsigned i = 0 ; i < query_num ; i ++) {
        ground_truth[i] = naive_brute_force(query_data_load + dim * i , data_ , dim , num_in_graph , queryK) ;
    }

    cout << "ground truth 计算完成" << endl ;
    //  将gpu计算结果也装入一个数组
    vector<vector<unsigned>> index_result(query_num , vector<unsigned>(queryK)) ;
    for(unsigned i = 0 ; i < query_num ; i++) {
        cudaMemcpy(index_result[i].data() , res_id + i * TOPM  , queryK * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;  
    }

    // 测试暴力代码
    for(unsigned i = 0 ; i < query_num ; i ++) {
        index_result[i] = brute_force_gpu(query_data_half_dev + i * dim , data_half_dev ,queryK , dim, num_in_graph) ; 
    }




    cout << "recall : " << cal_recall(index_result , ground_truth , true) << endl  ;
    cout << "搜索用时 : " << cost << endl ;
    cudaFree(graph_dev) ;
    cudaFree(data_dev) ;
    cudaFree(data_half_dev) ;
    cudaFree(query_data_dev) ;
    cudaFree(query_data_half_dev) ;
    return 0 ;
}



