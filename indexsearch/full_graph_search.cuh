#include "tools.cuh"

namespace smart::full_graph_search {
// #define half __half
// #define half2 __half2
using half = __half ;
using half2 = __half2 ;
// #define HASHLEN 2048
// inline constexpr unsigned HASHLEN = 2048 ;
// inline constexpr unsigned HASHSIZE = 11 ;
// inline constexpr unsigned HASH_RESET = 8 ;
// inline constexpr unsigned queryK = 10 ;
// inline constexpr unsigned MERGE_CELL_SIZE = 6 ;
// inline constexpr unsigned FULL_GRAPH_SEARCH
// #define HASHSIZE 11
// #define HASH_RESET 4
// #define queryK 10
// #define MERGE_CELL_SIZE 6

// const unsigned TOPM = 40 , DIM = 128 , MAX_ITER = 1000 , DEGREE = 16 ;
// const unsigned MAX_ITER = 10000 ;
inline constexpr unsigned MAX_ITER = 1000 ;
__device__ __constant__ const float INF_DIS = 1e9f; 

// constexpr unsigned EXTEND_POINTS_NUM = 128 ;
// constexpr unsigned EXTEND_POINTS_STEP = EXTEND_POINTS_NUM / 2 ; 
__device__ void swap_dis_and_id(float &a, float &b, unsigned &c, unsigned &d){
    const float t = a;
    a = b;
    b = t;
    const unsigned s = c;
    c = d;
    d = s;
}

__device__ void swap_dis_and_id_u(unsigned &a, unsigned &b, unsigned &c, unsigned &d){
    const unsigned t = a;
    a = b;
    b = t;
    const unsigned s = c;
    c = d;
    d = s;
}

__device__ void swap_dis_and_id_iu(int &a, int &b, unsigned &c, unsigned &d){
    const int t = a;
    a = b;
    b = t;
    const unsigned s = c;
    c = d;
    d = s;
}

__device__ unsigned hash_insert(unsigned *hash_table, unsigned key){
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


__device__ void warpSort(float* key, unsigned* val, unsigned lane_id, unsigned len) {
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

__device__ void bitonic_sort_id_by_dis_no_explore(float* shared_arr, unsigned* ids, unsigned len){
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

__device__ void merge_top(unsigned* arr1, unsigned* arr2, float* arr1_val, float* arr2_val, unsigned tid , unsigned DEGREE , unsigned TOPM){
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

// TOPM是candidate队列长度，K是TOP-K也是图索引的度数，DIM是数据集维度，MAX_ITER是最大迭代次数（1000足够），INF_DIS表示无穷大的浮点数
__global__ void full_graph_search_postfiltering(const unsigned* __restrict__ all_graphs, const half* __restrict__ values, const half* __restrict__ query_data, 
    unsigned* results_id, float* results_dis, unsigned* ent_pts , unsigned DEGREE , unsigned TOPM , unsigned DIM , const unsigned* graph_start_index ,const unsigned* graph_size ,
    const float* l_bound ,const float* r_bound ,const float* __restrict__ attrs , unsigned attr_dim , const unsigned* __restrict__ global_graph , unsigned edges_num_in_global_graph,
    const unsigned* __restrict__ global_idx ,const unsigned* __restrict__ local_idx ,const unsigned* seg_global_idx_start_idx , unsigned pnum ,unsigned K, const unsigned* pid_list, unsigned* result_k_id ,const unsigned* task_id_mapper) {
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数
    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    // unsigned* recycle_list_id = (unsigned*) shared_mem ;
    // float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    // __shared__ unsigned recycle_list_id[64] ;
    // __shared__ float recycle_list_dis[64] ;
    // __shared__ unsigned top_M_Cand[64 + 16];
    unsigned* top_M_Cand = (unsigned*) shared_mem ;
    // __shared__ float top_M_Cand_dis[64 + 16];
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE) ;
    float* q_l_bound = (float*)(top_M_Cand_dis + TOPM + DEGREE) ;
    float* q_r_bound = (float*)(q_l_bound + attr_dim) ;

    // 这里应该怎么处理8字节的对齐问题？
    // __shared__ half4 tmp_val_sha[(unsigned)(128 / 4)];
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem +
     (TOPM + DEGREE) * sizeof(unsigned) + (TOPM + DEGREE) * sizeof(float) + 2 * attr_dim * sizeof(float)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[2000] ;
    __shared__ float work_mem_float[2000] ;

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    // __shared__ unsigned long long start;
    // __shared__ double dis_cal_cost , sort_cost , merge_cost , new_merge_cost  , sort_cost2 ;
    // __shared__ unsigned total_jump_num ;
    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = task_id_mapper[blockIdx.x], laneid = threadIdx.x;  
    // 若线程块发射顺序与blockIdx.x的编号无关, 则可以手动实现按到达顺序取任务的排队器
    // unsigned psize_block = p_size[bid] , plist_index = p_start_index[bid] ;

    // if(tid == 0) {
    //     dis_cal_cost = sort_cost = merge_cost = new_merge_cost = sort_cost2=  0.0 ;
    //     total_jump_num = 0 ;
    // }

    // if(tid == 0) {
    //     printf("[") ;
    //     for(unsigned i = 0 ; i < pnum ; i ++)
    //         printf("%d," , graph_start_index[i]) ;
    //     printf("]\n") ;

    //     printf("edges_num_per_seg : %d\n" , edges_num_in_global_graph) ;
        
    //     printf("打印前32个点的邻居列表:\n") ;
    //     printf("[") ;
    //     for(unsigned i = 0 ; i < 32 ;i  ++) {
    //         for(unsigned j = 0 ; j < 32 ; j ++) {
    //             if(j / 2 == 0) {
    //                 printf("%d" , all_graphs[i * DEGREE + (j % 2)]) ;
    //             } else {
    //                 printf("%d" , global_idx[seg_global_idx_start_idx[j / 2] + global_graph[i * pnum * edges_num_in_global_graph + (j / 2) * edges_num_in_global_graph + (j % 2)]]) ;
    //             }
    //             if(j < 31) 
    //                 printf(", ") ;
    //         }
    //         if(i < 31)
    //             printf("\n") ;
    //     } 
    //     printf("]\n") ;
    // }
    // __syncthreads() ;
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
    // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
    for(unsigned i = tid; i < DEGREE; i += blockDim.x * blockDim.y){
        // top_M_Cand[TOPM + i] = ent_pts[i] ;
        top_M_Cand[i] = ent_pts[i] ;
        hash_insert(hash_table , ent_pts[i]) ;
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]) ;
	}
    __syncthreads() ;

    // if(tid == 0) {
    //     printf("can click %d\n" , __LINE__) ;
    // }
    // __syncthreads() ;
 
  
    /**
        1.先明确对每个分区要做什么, 首先, 将候选点载入top_M_Cand 的末DEGREE个位置, 然后排序
        2.将循环池重置为初始值
        3.从初始点开始扩展, 每次将DEGREE个邻居存放入top_M_Cand的末位, 被淘汰的分区内的点会插入循环池中
        4.当前分区查找结束后,将top_M_Cand中不符合范围约束的点去掉,用循环池中的点替代,然后依据这些点从全局图中扩展到下一个分区的入口点

        step4 的具体实现时, 可以先把不符合范围约束的点的距离改为INF,然后对列表排序, 和recycle_list合并
    **/
    // for(unsigned p_i = 0 ; p_i < psize_block; p_i ++) {
        // 得到图的入口

    #pragma unroll
    for(unsigned i = threadIdx.y; i < DEGREE ; i += blockDim.y){
        // threadIdx.y 0-5 , blockDim.y 即 6
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        unsigned global_id = top_M_Cand[i] ;
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            // warp中的每个子线程负责8个字节 即4个half类型分量的计算
            half4 val2 = Load(&values[global_id * DIM + j * 4]);
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

    

    // if(tid == 0) {
    //     printf("can click %d\n" , __LINE__) ;
    // }
    // __syncthreads() ;

    
    // 直接对整个列表做距离计算
    // if(tid < DEGREE)
        // warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
    bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, DEGREE) ;
    __syncthreads() ;
    // 此处被挤出去的点也应该合并到recycle_list 上
    // 将两个列表归并到一起
    // merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
    // merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float, tid , TOPM , TOPM) ;
    // __syncthreads() ;

    
    // if(tid == 0) {
    //     printf("can click %d\n" , __LINE__) ;
    // }
    // __syncthreads() ;


    // if(bid == 0 && tid == 0) {
    //     printf("top_M_Cand[") ;
    //     for(unsigned i = 0 ; i < TOPM ; i ++) {
    //         printf("(%d,%f)," , top_M_Cand[i] , top_M_Cand_dis[i]) ;
    //     }
    //     printf("]\n") ;
    // }
    // __syncthreads() ;

    // begin search
    for(unsigned i = 0; i < MAX_ITER; i++){

        // 每四轮将hash表重置一次
        if((i + 1) % 4 == 0){
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();

            for(unsigned j = tid; j < TOPM ; j += blockDim.x * blockDim.y){
                // 将此时的TOPM + K 个候选点插入哈希表中
                // hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
                if(top_M_Cand[j] != 0xFFFFFFFF) {
                    hash_insert(hash_table , top_M_Cand[j] & 0x7FFFFFFF) ; 
                }
            }
            for(unsigned j = tid ; j < 32 ; j += blockDim.x * blockDim.y)
                if(work_mem_unsigned[j] != 0xFFFFFFFFu)
                    hash_insert(hash_table , top_M_Cand[j] & 0x7FFFFFFFu) ;
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
        
        unsigned to_explore_local = local_idx[to_explore] ;
        unsigned extend_pid = pid_list[to_explore] ;
        // unsigned extend_pid = to_explore / 62500 ;
        const unsigned* extend_graph = all_graphs + graph_start_index[extend_pid] ;

        // if(bid == 0 && tid == 0 && i < 10) {
        //     printf("epoch %d graph extend : [" , i) ;
        //     for(unsigned j = 0 ; j < DEGREE ; j ++) {
        //         printf("%d," , global_idx[seg_global_idx_start_idx[extend_pid] + extend_graph[to_explore_local * DEGREE + j]]) ;
        //     }
        //     printf("]\n") ;
        // }
        // __syncthreads() ;

        for(unsigned j = tid; j < 2; j += blockDim.x * blockDim.y){
            // unsigned to_append_local = graph[to_explore * DEGREE + j];
            unsigned to_append_local = extend_graph[to_explore_local * DEGREE + j] ;
            // unsigned to_append = global_idx[seg_global_idx_start_idx[pid] + to_append_local] ;
            unsigned to_append = global_idx[seg_global_idx_start_idx[extend_pid] + to_append_local] ;
            // top_M_Cand_dis[TOPM + j] = (hash_insert(hash_table , to_append_local) == 0 ? INF_DIS : 0.0) ;
            work_mem_unsigned[extend_pid * 2 + j] = ((hash_insert(hash_table , to_append) == 0) ? 0xFFFFFFFFu : to_append) ;
            // top_M_Cand[TOPM + j] = to_append;
            work_mem_float[extend_pid * 2 + j] = ((work_mem_unsigned[extend_pid* 2 + j] == 0xFFFFFFFFu) ? INF_DIS : 0.0) ;
        }

        __syncthreads() ;
        // if(tid == 0) {
        //     printf("can click %d\n" , __LINE__) ;
        // }
        // __syncthreads() ;

        for(unsigned j = tid ; j < pnum ; j += blockDim.x * blockDim.y) {
            if(j == extend_pid)
                continue ;
            for(unsigned kk = 0 ; kk < 2 ; kk ++) {
                unsigned to_append_local = global_graph[to_explore * pnum * edges_num_in_global_graph + j * edges_num_in_global_graph + kk] ;
                unsigned to_append = global_idx[seg_global_idx_start_idx[j] + to_append_local] ;

                work_mem_unsigned[j * 2 + kk] = ((hash_insert(hash_table, to_append) == 0) ? 0xFFFFFFFFu : to_append) ;
                work_mem_float[j * 2 + kk] = ((work_mem_unsigned[j * 2 + kk] == 0xFFFFFFFFu) ? INF_DIS : 0.0) ;
            }
        }
        __syncthreads();

        // if(tid == 0) {
        //     printf("can click %d\n" , __LINE__) ;
        // }
        // __syncthreads() ;
        // unsigned extend_points_num = DEGREE + pnum * edges_num_in_global_graph ;
        unsigned extend_points_num = 32 ;
        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < extend_points_num ; j += blockDim.y){
            // 6个warp, 每个warp负责一个距离计算
            if(work_mem_unsigned[j] == 0xFFFFFFFFu)
                continue ;

            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            unsigned global_id = work_mem_unsigned[j] ;
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
                work_mem_float[j] += __half2float(val_res.x) + __half2float(val_res.y) ;
            }
        }
        __syncthreads();
        // if(tid == 0) {
        //     printf("can click %d\n" , __LINE__) ;
        // }
        // __syncthreads() ;

        // if(bid == 0 && tid == 0 && i < 10) {
        //     printf("epoch : %d , work_mem : [" , i) ;
        //     for(unsigned j = 0 ; j < extend_points_num ; j ++) {
        //         printf("(%d,%f)," , work_mem_unsigned[j] , work_mem_float[j]) ;
        //     }
        //     printf("]\n") ;
        // }
        // __syncthreads() ;
        // if(bid == 0 && tid == 0 && i < 10) {
        //     printf("epoch : %d , work_mem : [" , i) ;
        //     for(unsigned j = 0 ; j < extend_points_num ; j ++) {
        //         printf("(%d,%f)," , work_mem_unsigned[j] , work_mem_float[j]) ;
        //     }
        //     printf("]\n") ;
        // }
        // __syncthreads() ;

        if(tid < 32)
            warpSort(&work_mem_float[tid], &work_mem_unsigned[tid], tid , 32);
        // bitonic_sort_id_by_dis_no_explore(work_mem_float, work_mem_unsigned, extend_points_num) ;
        __syncthreads() ;
        // if(bid == 0 && tid == 0 && i < 10) {
        //     printf("epoch : %d , work_mem : [" , i) ;
        //     for(unsigned j = 0 ; j < extend_points_num ; j ++) {
        //         printf("(%d,%f)," , work_mem_unsigned[j] , work_mem_float[j]) ;
        //     }
        //     printf("]\n") ;
        // }
        // __syncthreads() ;
        
        merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float , tid , extend_points_num , TOPM) ;
        __syncthreads() ;
        // if(bid == 0 && tid == 0 && i < 10) {
        //     printf("epoch : %d , top_M_Cand [" , i) ;
        //     for(unsigned j = 0 ; j < TOPM ; j ++)
        //         printf("(%d,%f)," , top_M_Cand[j] & 0x7FFFFFFFu, top_M_Cand_dis[j]) ;
        //     printf("]\n") ;
        // }
        // __syncthreads() ;
        // if(tid == 0) {
        //     printf("can click %d\n" , __LINE__) ;
        // }
        // __syncthreads() ;
        // 问题来了, 当图开始切换时, 当前部分结果的id应该采用局部id还是全局id? 在列表中的所有id皆采用全局id
        // 对后半段列表sort一次, 然后和recycle pool进行merge
        // bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM] , &top_M_Cand[TOPM] , DEGREE) ;
    }
    __syncthreads();

    // if(tid == 0 && bid == 0) {
    //     printf("过滤前列表:") ;
    //     for(unsigned i = 0 ; i < TOPM ; i ++)
    //         printf("(%d-%f)," , top_M_Cand[i] & 0x7FFFFFFF, top_M_Cand_dis[i]) ;
    //     printf("\n") ;
    // }
    // __syncthreads() ;

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
    bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM) ;
    __syncthreads() ;
    // 若recycle_list中确实有点, 则合并
    // if(recycle_list_id[0] != 0xFFFFFFFF) {
    //     merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
    //     __syncthreads() ;
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

    // if(tid == 0 && bid == 0) {
    //     for(unsigned i = 0 ; i < TOPM ; i ++)
    //         printf("(%d-%f)," , top_M_Cand[i] & 0x7FFFFFFF, top_M_Cand_dis[i]) ;
    //     printf("\n") ;
    // }
}

// TOPM是candidate队列长度，K是TOP-K也是图索引的度数，DIM是数据集维度，MAX_ITER是最大迭代次数（1000足够），INF_DIS表示无穷大的浮点数
__global__ void full_graph_search_postfiltering_V2(const unsigned* __restrict__ graph, const half* __restrict__ values, const half* __restrict__ query_data, 
    unsigned* results_id, float* results_dis, unsigned* ent_pts , unsigned DEGREE , unsigned TOPM , unsigned DIM , 
    const float* l_bound ,const float* r_bound ,const float* __restrict__ attrs , unsigned attr_dim , unsigned K, unsigned* result_k_id ,const unsigned* task_id_mapper) {
    // global_graph[global_index * 分区数量 * 2 + 分区号 * 2] = 到下一个分区的起始节点
    // KNN 图 , VALUES看起来像存储向量的数组, query_data存储查询向量
    // graph连续存储多个子图
    // 代码需写成动态的, 这里的TOPM 和 DEGREE都要写成可变的参数
    extern __shared__ __align__(8) unsigned char shared_mem[] ;
    // unsigned* recycle_list_id = (unsigned*) shared_mem ;
    // float* recycle_list_dis = (float*) (recycle_list_id + recycle_list_size) ;
    // __shared__ unsigned recycle_list_id[64] ;
    // __shared__ float recycle_list_dis[64] ;
    // __shared__ unsigned top_M_Cand[64 + 16];
    unsigned* top_M_Cand = (unsigned*) shared_mem ;
    // __shared__ float top_M_Cand_dis[64 + 16];
    float* top_M_Cand_dis = (float*)(top_M_Cand + TOPM + DEGREE) ;
    float* q_l_bound = (float*)(top_M_Cand_dis + TOPM + DEGREE) ;
    float* q_r_bound = (float*)(q_l_bound + attr_dim) ;

    // 这里应该怎么处理8字节的对齐问题？
    // __shared__ half4 tmp_val_sha[(unsigned)(128 / 4)];
    uintptr_t pointer8 = reinterpret_cast<uintptr_t>(shared_mem +
     (TOPM + DEGREE) * sizeof(unsigned) + (TOPM + DEGREE) * sizeof(float) + 2 * attr_dim * sizeof(float)) ;
    pointer8 = (pointer8 + alignof(half4) - 1) & ~(alignof(half4) - 1) ;
    half4* tmp_val_sha = (half4*) (reinterpret_cast<char*> (pointer8));

    __shared__ unsigned work_mem_unsigned[2000] ;
    __shared__ float work_mem_float[2000] ;

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    // __shared__ unsigned long long start;
    // __shared__ double dis_cal_cost , sort_cost , merge_cost , new_merge_cost  , sort_cost2 ;
    // __shared__ unsigned total_jump_num ;
    // laneid 为线程在warp内的编号 等同于一维的 threadIdx.x % 32 , tid为线程在当前线程块内的局部id
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = task_id_mapper[blockIdx.x], laneid = threadIdx.x;  
    // 若线程块发射顺序与blockIdx.x的编号无关, 则可以手动实现按到达顺序取任务的排队器
    // unsigned psize_block = p_size[bid] , plist_index = p_start_index[bid] ;

    // if(tid == 0) {
    //     dis_cal_cost = sort_cost = merge_cost = new_merge_cost = sort_cost2=  0.0 ;
    //     total_jump_num = 0 ;
    // }

    // if(tid == 0) {
    //     printf("[") ;
    //     for(unsigned i = 0 ; i < pnum ; i ++)
    //         printf("%d," , graph_start_index[i]) ;
    //     printf("]\n") ;

    //     printf("edges_num_per_seg : %d\n" , edges_num_in_global_graph) ;
        
    //     printf("打印前32个点的邻居列表:\n") ;
    //     printf("[") ;
    //     for(unsigned i = 0 ; i < 32 ;i  ++) {
    //         for(unsigned j = 0 ; j < 32 ; j ++) {
    //             if(j / 2 == 0) {
    //                 printf("%d" , all_graphs[i * DEGREE + (j % 2)]) ;
    //             } else {
    //                 printf("%d" , global_idx[seg_global_idx_start_idx[j / 2] + global_graph[i * pnum * edges_num_in_global_graph + (j / 2) * edges_num_in_global_graph + (j % 2)]]) ;
    //             }
    //             if(j < 31) 
    //                 printf(", ") ;
    //         }
    //         if(i < 31)
    //             printf("\n") ;
    //     } 
    //     printf("]\n") ;
    // }
    // __syncthreads() ;
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM + DEGREE; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
        q_l_bound[i] = l_bound[bid * attr_dim + i] ;
        q_r_bound[i] = r_bound[bid * attr_dim + i] ;
    }
    // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
    for(unsigned i = tid; i < DEGREE; i += blockDim.x * blockDim.y){
        // top_M_Cand[TOPM + i] = ent_pts[i] ;
        top_M_Cand[i] = ent_pts[i] ;
        hash_insert(hash_table , ent_pts[i]) ;
    }
    // 每次四个字节的去读取查询向量, 存储方式为四个捆成一捆
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]) ;
	}
    __syncthreads() ;

    // if(tid == 0) {
    //     printf("can click %d\n" , __LINE__) ;
    // }
    // __syncthreads() ;
 
  
    /**
        1.先明确对每个分区要做什么, 首先, 将候选点载入top_M_Cand 的末DEGREE个位置, 然后排序
        2.将循环池重置为初始值
        3.从初始点开始扩展, 每次将DEGREE个邻居存放入top_M_Cand的末位, 被淘汰的分区内的点会插入循环池中
        4.当前分区查找结束后,将top_M_Cand中不符合范围约束的点去掉,用循环池中的点替代,然后依据这些点从全局图中扩展到下一个分区的入口点

        step4 的具体实现时, 可以先把不符合范围约束的点的距离改为INF,然后对列表排序, 和recycle_list合并
    **/
    // for(unsigned p_i = 0 ; p_i < psize_block; p_i ++) {
        // 得到图的入口

    #pragma unroll
    for(unsigned i = threadIdx.y; i < DEGREE ; i += blockDim.y){
        // threadIdx.y 0-5 , blockDim.y 即 6
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        unsigned global_id = top_M_Cand[i] ;
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            // warp中的每个子线程负责8个字节 即4个half类型分量的计算
            half4 val2 = Load(&values[global_id * DIM + j * 4]);
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

    

    // if(tid == 0) {
    //     printf("can click %d\n" , __LINE__) ;
    // }
    // __syncthreads() ;

    
    // 直接对整个列表做距离计算
    // if(tid < DEGREE)
        // warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
    bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, DEGREE) ;
    __syncthreads() ;
    // 此处被挤出去的点也应该合并到recycle_list 上
    // 将两个列表归并到一起
    // merge_top_with_recycle_list(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
    // merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float, tid , TOPM , TOPM) ;
    // __syncthreads() ;

    
    // if(tid == 0) {
    //     printf("can click %d\n" , __LINE__) ;
    // }
    // __syncthreads() ;


    // if(bid == 0 && tid == 0) {
    //     printf("top_M_Cand[") ;
    //     for(unsigned i = 0 ; i < TOPM ; i ++) {
    //         printf("(%d,%f)," , top_M_Cand[i] , top_M_Cand_dis[i]) ;
    //     }
    //     printf("]\n") ;
    // }
    // __syncthreads() ;

    // begin search
    for(unsigned i = 0; i < MAX_ITER; i++){

        // 每四轮将hash表重置一次
        if((i + 1) % 4 == 0){
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();

            for(unsigned j = tid; j < TOPM + DEGREE ; j += blockDim.x * blockDim.y){
                // 将此时的TOPM + K 个候选点插入哈希表中
                // hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
                if(top_M_Cand[j] != 0xFFFFFFFF) {
                    hash_insert(hash_table , top_M_Cand[j] & 0x7FFFFFFF) ; 
                }
            }
            // for(unsigned j = tid ; j < 32 ; j += blockDim.x * blockDim.y)
            //     if(work_mem_unsigned[j] != 0xFFFFFFFFu)
            //         hash_insert(hash_table , top_M_Cand[j] & 0x7FFFFFFFu) ;
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

        for(unsigned j = tid; j < DEGREE; j += blockDim.x * blockDim.y){
            // unsigned to_append_local = graph[to_explore * DEGREE + j];
            // unsigned to_append_local = extend_graph[to_explore_local * DEGREE + j] ;
            // unsigned to_append = global_idx[seg_global_idx_start_idx[pid] + to_append_local] ;
            // unsigned to_append = global_idx[seg_global_idx_start_idx[extend_pid] + to_append_local] ;
            // top_M_Cand_dis[TOPM + j] = (hash_insert(hash_table , to_append_local) == 0 ? INF_DIS : 0.0) ;
            unsigned to_append = graph[to_explore * DEGREE + j] ;
            // work_mem_unsigned[extend_pid * 2 + j] = ((hash_insert(hash_table , to_append) == 0) ? 0xFFFFFFFFu : to_append) ;
            top_M_Cand[TOPM + j] = ((hash_insert(hash_table , to_append) == 0) ? 0xFFFFFFFFu : to_append) ;
            top_M_Cand_dis[TOPM + j] = (top_M_Cand[TOPM + j] == 0xFFFFFFFFu ? INF_DIS : 0.0) ;
            // top_M_Cand[TOPM + j] = to_append;
            // work_mem_float[extend_pid * 2 + j] = ((work_mem_unsigned[extend_pid* 2 + j] == 0xFFFFFFFFu) ? INF_DIS : 0.0) ;
        }

        __syncthreads() ;
        // if(tid == 0) {
        //     printf("can click %d\n" , __LINE__) ;
        // }
        // __syncthreads() ;

        // if(tid == 0) {
        //     printf("can click %d\n" , __LINE__) ;
        // }
        // __syncthreads() ;
        // unsigned extend_points_num = DEGREE + pnum * edges_num_in_global_graph ;
       
        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < DEGREE ; j += blockDim.y){
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
        // if(tid == 0) {
        //     printf("can click %d\n" , __LINE__) ;
        // }
        // __syncthreads() ;

        // if(bid == 0 && tid == 0 && i < 10) {
        //     printf("epoch : %d , work_mem : [" , i) ;
        //     for(unsigned j = 0 ; j < extend_points_num ; j ++) {
        //         printf("(%d,%f)," , work_mem_unsigned[j] , work_mem_float[j]) ;
        //     }
        //     printf("]\n") ;
        // }
        // __syncthreads() ;
        // if(bid == 0 && tid == 0 && i < 10) {
        //     printf("epoch : %d , work_mem : [" , i) ;
        //     for(unsigned j = 0 ; j < extend_points_num ; j ++) {
        //         printf("(%d,%f)," , work_mem_unsigned[j] , work_mem_float[j]) ;
        //     }
        //     printf("]\n") ;
        // }
        // __syncthreads() ;

        if(tid < 32)
            warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , 32);
        // bitonic_sort_id_by_dis_no_explore(work_mem_float, work_mem_unsigned, extend_points_num) ;
        __syncthreads() ;
        // if(bid == 0 && tid == 0 && i < 10) {
        //     printf("epoch : %d , work_mem : [" , i) ;
        //     for(unsigned j = 0 ; j < extend_points_num ; j ++) {
        //         printf("(%d,%f)," , work_mem_unsigned[j] , work_mem_float[j]) ;
        //     }
        //     printf("]\n") ;
        // }
        // __syncthreads() ;
        
        merge_top(top_M_Cand , top_M_Cand + TOPM , top_M_Cand_dis , top_M_Cand_dis + TOPM , tid , DEGREE , TOPM) ;
        __syncthreads() ;
        // if(bid == 0 && tid == 0 && i < 10) {
        //     printf("epoch : %d , top_M_Cand [" , i) ;
        //     for(unsigned j = 0 ; j < TOPM ; j ++)
        //         printf("(%d,%f)," , top_M_Cand[j] & 0x7FFFFFFFu, top_M_Cand_dis[j]) ;
        //     printf("]\n") ;
        // }
        // __syncthreads() ;
        // if(tid == 0) {
        //     printf("can click %d\n" , __LINE__) ;
        // }
        // __syncthreads() ;
        // 问题来了, 当图开始切换时, 当前部分结果的id应该采用局部id还是全局id? 在列表中的所有id皆采用全局id
        // 对后半段列表sort一次, 然后和recycle pool进行merge
        // bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM] , &top_M_Cand[TOPM] , DEGREE) ;
    }
    __syncthreads();

    // if(tid == 0 && bid == 0) {
    //     printf("过滤前列表:") ;
    //     for(unsigned i = 0 ; i < TOPM ; i ++)
    //         printf("(%d-%f)," , top_M_Cand[i] & 0x7FFFFFFF, top_M_Cand_dis[i]) ;
    //     printf("\n") ;
    // }
    // __syncthreads() ;

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
    bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis, top_M_Cand, TOPM) ;
    __syncthreads() ;
    // 若recycle_list中确实有点, 则合并
    // if(recycle_list_id[0] != 0xFFFFFFFF) {
    //     merge_top(top_M_Cand , recycle_list_id , top_M_Cand_dis , recycle_list_dis , tid , recycle_list_size , TOPM) ;
    //     __syncthreads() ;
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

    // if(tid == 0 && bid == 0) {
    //     for(unsigned i = 0 ; i < TOPM ; i ++)
    //         printf("(%d-%f)," , top_M_Cand[i] & 0x7FFFFFFF, top_M_Cand_dis[i]) ;
    //     printf("\n") ;
    // }
}


__global__ void assemble_edges(const unsigned* __restrict__ all_graphs, unsigned DEGREE , const unsigned* graph_start_index ,const unsigned* graph_size ,
    const unsigned* __restrict__ global_graph , unsigned edges_num_in_global_graph, unsigned reserved_num , const unsigned pnum , const unsigned* global_idx ,
    const unsigned* local_idx , const unsigned* seg_global_idx_start_idx , const unsigned* pid_list , unsigned* dest) {
    const unsigned dest_degree = reserved_num * pnum ; 
    const unsigned tid = threadIdx.x , bid = blockIdx.x ;
    const unsigned b_pid = pid_list[bid] , b_localidx = local_idx[bid] ;
    for(unsigned i = tid ; i < dest_degree ; i += blockDim.x) {
        unsigned pid = i / reserved_num , eid = i % reserved_num ;
        unsigned v = (b_pid == pid ? (all_graphs + graph_start_index[pid])[b_localidx * DEGREE + eid] : 
            global_graph[bid * pnum * edges_num_in_global_graph + pid * edges_num_in_global_graph + eid]) ;
        v = global_idx[seg_global_idx_start_idx[pid] + v] ;
        dest[bid * dest_degree + i] = v ;        
    }
}


vector<vector<unsigned>> smart_postfiltering(array<unsigned* , 3>& graph_infos , unsigned degree ,  __half* data_ , __half* query ,unsigned dim , unsigned qnum ,  float* l_bound , float* r_bound , float* attrs , 
    unsigned attr_dim  , array<unsigned*,3>& idx_mapper , array<unsigned* , 5>& partition_infos , unsigned* pid_list ,  unsigned L , unsigned K , 
    unsigned* global_edges , unsigned edge_num_contain , unsigned pnum) {

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
    uniform_int_distribution<> intd(0 ,1000000 - 1) ;

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
    unsigned byteSize =
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float) + (DIM/4) * sizeof(half4) + alignof(half4) ;
    cout << "byteSize : " << (byteSize * 1.0 / 1024) << "KB" << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    // select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter<<<grid_s , block_s , byteSize>>> (graph_infos[0], data_, query, 
    //     results_id, results_dis, ent_pts , degree , L , dim , graph_infos[2] , graph_infos[1] ,
    //     partition_infos[0] , partition_infos[1] , partition_infos[2] , l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain,
    //     recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] ,  pnum , K , result_k_id , task_id_mapper , partition_infos[3]) ;
    full_graph_search_postfiltering<<<grid_s , block_s , byteSize>>>(graph_infos[0], data_, query, 
        results_id, results_dis, ent_pts , degree , L , dim , graph_infos[2] ,graph_infos[1] ,
        l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain,
        idx_mapper[0] ,idx_mapper[1] ,idx_mapper[2] , pnum , K, pid_list, result_k_id , task_id_mapper) ;
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

    // s = chrono::high_resolution_clock::now() ;
    // unsigned* symbol_address , *symbol_address_dcnt , *symbol_address_pmcnt ; 
    // cudaGetSymbolAddress((void**) &symbol_address , jump_num_cnt) ;
    // cudaGetSymbolAddress((void**) &symbol_address_dcnt , dis_cal_cnt) ;
    // cudaGetSymbolAddress((void**) &symbol_address_pmcnt , post_merge_cnt) ;
    // unsigned total_j_num = thrust::reduce(thrust::device , symbol_address , symbol_address + qnum , 0) ;
    // unsigned total_c_num = thrust::reduce(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    // unsigned total_m_num = thrust::reduce(thrust::device , symbol_address_pmcnt , symbol_address_pmcnt + qnum , 0) ;
    // e = chrono::high_resolution_clock::now() ;
    // float jump_cal_cost = chrono::duration<double>(e - s).count() ; 
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    // cout << "总的跳转次数: " << total_j_num << " , 平均跳转次数: " << (total_j_num * 1.0 / qnum) << endl ;
    // cout << "总的距离计算次数: " << total_c_num << ", 平均距离计算次数: " << (total_c_num * 1.0 / qnum) << endl ;
    // cout << "总的后合并次数: " << total_m_num << ", 平均后合并次数: " << (total_m_num * 1.0 / qnum) << endl ; 
    cout << "结果传递时间: " << result_load_cost << endl ;
    // cout << "计算跳数时间: " << jump_cal_cost << endl ;
    cout << "获取入口点时间: " << find_ep_cost << endl ; 
    cout << "排序时间: " << sort_cost << endl ;
    // thrust::fill(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
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


vector<vector<unsigned>> smart_postfiltering_V2(array<unsigned* , 3>& graph_infos , unsigned degree ,  __half* data_ , __half* query ,unsigned dim , unsigned qnum ,  float* l_bound , float* r_bound , float* attrs , 
    unsigned attr_dim  , array<unsigned*,3>& idx_mapper , array<unsigned* , 5>& partition_infos , unsigned* pid_list ,  unsigned L , unsigned K , 
    unsigned* global_edges , unsigned edge_num_contain , unsigned pnum) {

    unsigned DIM = dim , TOPM = L ,  DEGREE = degree ;
    unsigned *results_id ; 
    float *results_dis ; 
    unsigned *result_k_id ; 
    unsigned *task_id_mapper ; 
    unsigned *assembled_graph ;


    auto ms = chrono::high_resolution_clock::now() ;
    cudaMalloc((void**) &results_id , qnum * L * sizeof(unsigned)) ;
    cudaMalloc((void**) &results_dis , qnum * L * sizeof(unsigned)) ;
    cudaMalloc((void**) &result_k_id , qnum * K * sizeof(unsigned)) ;
    cudaMalloc((void**) &task_id_mapper , qnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &assembled_graph , 32 * 1000000 * sizeof(unsigned)) ;

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
    uniform_int_distribution<> intd(0 ,1000000 - 1) ;

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

    auto assemble_edges_s = chrono::high_resolution_clock::now() ;
    assemble_edges<<<1000000 , 32>>>(graph_infos[0], degree , graph_infos[2] , graph_infos[1] ,
        global_edges , edge_num_contain, 2 , pnum , idx_mapper[0] ,
        idx_mapper[1] , idx_mapper[2] , pid_list , assembled_graph) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    auto assemble_edges_e = chrono::high_resolution_clock::now() ;
    float assemble_edges_cost = chrono::duration<float>(assemble_edges_e - assemble_edges_s).count() ;

    dim3 grid_s(qnum , 1 , 1) , block_s(32 ,6 , 1) ;
    unsigned byteSize =
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float) + (DIM/4) * sizeof(half4) + alignof(half4) ;
    cout << "byteSize : " << (byteSize * 1.0 / 1024) << "KB" << endl ;
    auto s = chrono::high_resolution_clock::now() ;

    // full_graph_search_postfiltering<<<grid_s , block_s , byteSize>>>(graph_infos[0], data_, query, 
    //     results_id, results_dis, ent_pts , degree , L , dim , graph_infos[2] ,graph_infos[1] ,
    //     l_bound , r_bound , attrs , attr_dim , global_edges , edge_num_contain,
    //     idx_mapper[0] ,idx_mapper[1] ,idx_mapper[2] , pnum , K, pid_list, result_k_id , task_id_mapper) ;
    full_graph_search_postfiltering_V2<<<grid_s , block_s , byteSize>>>(assembled_graph, data_, query, 
        results_id, results_dis, ent_pts , 32 , L , dim , 
        l_bound , r_bound , attrs , attr_dim , K, result_k_id , task_id_mapper) ;
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

    // s = chrono::high_resolution_clock::now() ;
    // unsigned* symbol_address , *symbol_address_dcnt , *symbol_address_pmcnt ; 
    // cudaGetSymbolAddress((void**) &symbol_address , jump_num_cnt) ;
    // cudaGetSymbolAddress((void**) &symbol_address_dcnt , dis_cal_cnt) ;
    // cudaGetSymbolAddress((void**) &symbol_address_pmcnt , post_merge_cnt) ;
    // unsigned total_j_num = thrust::reduce(thrust::device , symbol_address , symbol_address + qnum , 0) ;
    // unsigned total_c_num = thrust::reduce(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    // unsigned total_m_num = thrust::reduce(thrust::device , symbol_address_pmcnt , symbol_address_pmcnt + qnum , 0) ;
    // e = chrono::high_resolution_clock::now() ;
    // float jump_cal_cost = chrono::duration<double>(e - s).count() ; 
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "组装边耗时:" << assemble_edges_cost << endl ;
    // cout << "总的跳转次数: " << total_j_num << " , 平均跳转次数: " << (total_j_num * 1.0 / qnum) << endl ;
    // cout << "总的距离计算次数: " << total_c_num << ", 平均距离计算次数: " << (total_c_num * 1.0 / qnum) << endl ;
    // cout << "总的后合并次数: " << total_m_num << ", 平均后合并次数: " << (total_m_num * 1.0 / qnum) << endl ; 
    cout << "结果传递时间: " << result_load_cost << endl ;
    // cout << "计算跳数时间: " << jump_cal_cost << endl ;
    cout << "获取入口点时间: " << find_ep_cost << endl ; 
    cout << "排序时间: " << sort_cost << endl ;
    // thrust::fill(thrust::device , symbol_address_dcnt , symbol_address_dcnt + qnum , 0) ;
    ms = chrono::high_resolution_clock::now() ;
    cudaFree(results_id) ;
    cudaFree(results_dis) ;
    cudaFree(ent_pts) ;
    cudaFree(result_k_id) ;
    cudaFree(assembled_graph) ;
    me = chrono::high_resolution_clock::now() ;
    m_cost += chrono::duration<double>(me - ms).count() ;
    cout << "管理内存耗时: " << m_cost << endl ;
    return results ;
}


} 