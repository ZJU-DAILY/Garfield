#include "production_adapter.cuh"
using namespace std ;



namespace smart::big_dataset_adapter {

    // 8个int8类型, 共占8字节
    struct __align__(8) int8_8 {
        // half2 x, y;
        char4 x , y ;
    };
      
    __device__ __forceinline__ int8_8 BitCast(const uint2& src) noexcept {
        int8_8 dst;
        std::memcpy(&dst, &src, sizeof(int8_8));
        return dst;
    }
      
    // __device__ __forceinline__ int8_8 Load(const int8_t* address) {
    //    uint2 x = __ldg(reinterpret_cast<const uint2*>(address));
    //     return BitCast(x);
    // }
 
    __device__ __forceinline__ char4 Load(const int8_t* address) {
        return __ldg(reinterpret_cast<const char4*>(address)) ;
    }

    __device__ double dis_cal_cost_total[10000] = {0.0};
    template<typename attrType = float>
    __global__ void select_path_contain_attr_filter_V2_without_output_dual_channel_in_turn_with_prefilter_batch_process_compressed_graph_T_dataset_resident(const unsigned batch_size , const unsigned* batch_partition_ids , const unsigned* __restrict__ batch_all_graphs, const int8_t* __restrict__ batch_values, const half* __restrict__ query_data, 
        const float* __restrict__ scales_video_memory , const int* __restrict__ zero_points_video_memory ,
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
        __shared__ float scales[128] ;
        __shared__ int zero_points[128] ;
        
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

        if(total_psize_block == 0) {
            for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
                top_M_Cand_video_memory[bid * TOPM + i] = 0xFFFFFFFFu ;
                top_M_Cand_dis_video_memory[bid * TOPM + i] = INF_DIS ;
            }

            return ;
        } 
       

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
        }
        for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
            q_l_bound[i] = l_bound[bid * attr_dim + i] ;
            q_r_bound[i] = r_bound[bid * attr_dim + i] ;
        }

        for(unsigned i = tid ; i < DIM ; i  += blockDim.x * blockDim.y) {
            scales[i] = scales_video_memory[i] ;
            zero_points[i] = zero_points_video_memory[i] ;
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

        
        if(psize_block > 0 && little_seg_num_block == 0) {
            psize_block = total_psize_block ;
            
            unsigned batch_pid[2] = {batch_pids[plist_index] , (psize_block >= 2 ? batch_pids[plist_index + 1] : batch_pids[plist_index])} ;
            unsigned pid[2] = {batch_partition_ids[batch_pid[0]] , batch_partition_ids[batch_pid[1]]} ;
            // pid[0] = 9 ; 
            const unsigned* graph[2] = {batch_all_graphs + batch_graph_start_index[batch_pid[0]] , batch_all_graphs + batch_graph_start_index[batch_pid[1]]};
            const unsigned DEGREE2 = DEGREE * 2 ;


            // 有两种可能, 没有局部结果的从入口点扩展, 有局部结果的先从局部结果扩展, 若局部结果中点个数不够从入口点补充

            // 将 top_M_Cand的后K位设置成入口节点, 然后将其插入到哈希表中
            for(unsigned i = tid , L2 = DEGREE ; i < DEGREE2; i += blockDim.x * blockDim.y){
                unsigned shift_bits = 30 +  i / L2;
                unsigned mask = (batch_pid[0] == batch_pid[1] ? 0 : 1 << shift_bits) ; 
                // 用反向掩码, 将另一位置1
                top_M_Cand[i] = global_idx[seg_global_idx_start_idx[pid[i / L2]] + ent_pts[i]] | mask ;
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

                half2 val_res ;
                val_res.x = 0.0 ; val_res.y = 0.0 ;
                for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x) {
                    // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                    // half4 val2 = Load(&batch_values[vec_offset + static_cast<size_t>(j * 4)]);
                    // char4 val = Load(&batch_values[vec_offset + static_cast<size_t>(j * 4)]) ;
                    half4 val2 ; 
                    val2.x = __half2(
                        __float2half(scales[j * 4] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4)] - zero_points[j * 4])) ,
                        __float2half(scales[j * 4 + 1] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4 + 1)] - zero_points[j * 4 + 1]))
                    ) ;
                    val2.y = __half2(
                        __float2half(scales[j * 4 + 2] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4 + 2)] - zero_points[j * 4 + 2])) ,
                        __float2half(scales[j * 4 + 3] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4 + 3)] - zero_points[j * 4 + 3]))
                    ) ;
                   
                    val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
                    // val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                    // val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
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
                // start = clock64() ;
            }
            __syncthreads() ;

            bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis , top_M_Cand , DEGREE2) ;
            __syncthreads() ;

            if(tid == 0) {
                cur_visit_p = (psize_block >= 2 ? 2 : 1) ;
                main_channel = 0 ; 
                turn = 0 ; 
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
                            unsigned offset = batch_all_point_num * edges_num_in_global_graph * next_batch_pid + stored_pos * edges_num_in_global_graph ;

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
                                // half4 val2 = Load(&batch_values[vec_offset + static_cast<size_t>(k * 4)]);
                                // char4 val = Load(&batch_values[vec_offset + static_cast<size_t>(k * 4)]) ;
                                half4 val2 ; 
                                val2.x = __half2(
                                    __float2half(scales[k * 4] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4)] - zero_points[k * 4])) ,
                                    __float2half(scales[k * 4 + 1] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 1)] - zero_points[k * 4 + 1]))
                                ) ;
                                val2.y = __half2(
                                    __float2half(scales[k * 4 + 2] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 2)] - zero_points[k * 4 + 2])) ,
                                    __float2half(scales[k * 4 + 3] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 3)] - zero_points[k * 4 + 3]))
                                ) ;
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
                        unsigned mask = (batch_pid[0] == batch_pid[1] ? 0 : (1 << shift_bits)) ; 
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

                    half2 val_res;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                        // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                        // half4 val2 = Load(&batch_values[vec_offset + static_cast<size_t>(j * 4)]);
                        // char4 val = Load(&batch_values[vec_offset + static_cast<size_t>(j * 4)]) ;
                        half4 val2 ; 
                        val2.x = __half2(
                            __float2half(scales[j * 4] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4)] - zero_points[j * 4])) ,
                            __float2half(scales[j * 4 + 1] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4 + 1)] - zero_points[j * 4 + 1]))
                        ) ;
                        val2.y = __half2(
                            __float2half(scales[j * 4 + 2] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4 + 2)] - zero_points[j * 4 + 2])) ,
                            __float2half(scales[j * 4 + 3] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4 + 3)] - zero_points[j * 4 + 3]))
                        ) ;
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
            
            }  
            

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
                }
                __syncthreads() ;
                for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                    // unsigned cell_id = chunk_id + (i >> 5) ; 
                    unsigned cell_id = (i >> 5) ;
                    
                    unsigned id_in_cell = i % 32 ;
                    unsigned bit_mask = (1 << id_in_cell) ;
                    if((bit_map[cell_id] & bit_mask) != 0) {
                        // 获取插入位置
                        
                        unsigned local_cell_id = (i >> 5) ;
                        unsigned put_pos = (local_cell_id == 0 ? 0 : top_prefix[local_cell_id - 1]) + (id_in_cell == 0 ? 0 : prefix_sum[i - 1]) ;
                        indices_arr[put_pos] = i ;
                    }
                }
                __syncthreads() ;
                if(tid == 0) {
                    start = clock64() ;
                }
                __syncthreads() ;
                 
                // 计算距离, 然后用原子操作插入队列中
                for(unsigned i = threadIdx.y ; i < task_num ; i += blockDim.y) {
                    unsigned cand_id = indices_arr[i] + point_batch_start_id ;
                    
                    // 计算距离
                    unsigned g_global_id = global_idx[global_idx_offset + cand_id] ;
                    size_t vec_offset = static_cast<size_t>(g_global_id) * static_cast<size_t>(DIM) ;
                    float dis = 0.0f ;

                    half2 val_res;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                        // half4 val2 = Load(&batch_values[vec_offset + static_cast<size_t>(k * 4)]);
                        // char4 val = Load(&batch_values[vec_offset + static_cast<size_t>(k * 4)]) ;
                        half4 val2 ; 
                        val2.x = __half2(
                            __float2half(scales[k * 4] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4)] - zero_points[k * 4])) ,
                            __float2half(scales[k * 4 + 1] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 1)] - zero_points[k * 4 + 1]))
                        ) ;
                        val2.y = __half2(
                            __float2half(scales[k * 4 + 2] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 2)] - zero_points[k * 4 + 2])) ,
                            __float2half(scales[k * 4 + 3] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 3)] - zero_points[k * 4 + 3]))
                        ) ;
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
                if(tid == 0) {
                    dis_cal_cost += ((double)(clock64() - start) / 1582000) ;
                }
                __syncthreads() ;
               
                if(prefilter_pointer >= 32 || (chunk_id + chunk_batch_size >= chunk_num && prefilter_pointer > 0)) {
                  
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

        /**
        #pragma unroll
        for (unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
            work_mem_unsigned[i] = top_M_Cand_video_memory[bid * TOPM + i] ;
            work_mem_float[i] = top_M_Cand_dis_video_memory[bid * TOPM + i] ;
        }
        
        __syncthreads() ;
        merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float , tid , K , TOPM) ;
        __syncthreads() ;
        **/

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
            dis_cal_cost_total[bid] = dis_cal_cost ;
        }
    }


    template<typename attrType = float>
    __global__ void select_path_contain_attr_filter_V2_without_output_single_channel_with_prefilter_batch_process_compressed_graph_T_dataset_resident(const unsigned batch_size , const unsigned* batch_partition_ids , const unsigned* __restrict__ batch_all_graphs, const int8_t* __restrict__ batch_values, const half* __restrict__ query_data, 
        const float* __restrict__ scales_video_memory , const int* __restrict__ zero_points_video_memory ,
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
        __shared__ float scales[128] ;
        __shared__ int zero_points[128] ;
        
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

        if(total_psize_block == 0) {
            for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
                top_M_Cand_video_memory[bid * TOPM + i] = 0xFFFFFFFFu ;
                top_M_Cand_dis_video_memory[bid * TOPM + i] = INF_DIS ;
            }

            return ;
        } 
       
        // if(bid >= 100)
        //     return ;
        __shared__ unsigned cur_visit_p , main_channel , turn ;
        __shared__ unsigned to_explore ;
        __shared__ unsigned prefilter_pointer ;
        __shared__ long long start ;
        __shared__ double dis_cal_cost ;

        __shared__ unsigned hash_table[HASHLEN];
        // 从全局内存中恢复topm
        for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
            top_M_Cand[i] = 0xFFFFFFFF;
            top_M_Cand_dis[i] = INF_DIS;
        }
        for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
            q_l_bound[i] = l_bound[bid * attr_dim + i] ;
            q_r_bound[i] = r_bound[bid * attr_dim + i] ;
        }

        for(unsigned i = tid ; i < DIM ; i  += blockDim.x * blockDim.y) {
            scales[i] = scales_video_memory[i] ;
            zero_points[i] = zero_points_video_memory[i] ;
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

        
        if(psize_block > 0 && little_seg_num_block == 0) {
            psize_block = total_psize_block ;
            unsigned init_pid = batch_partition_ids[batch_pids[plist_index]] ;
            for(unsigned i = tid; i < DEGREE; i += blockDim.x * blockDim.y) {
                top_M_Cand[TOPM + i] = global_idx[seg_global_idx_start_idx[init_pid] + ent_pts[i]];
                hash_insert(hash_table , top_M_Cand[TOPM + i]) ;
            }
            __syncthreads() ;
            
            // if(tid == 0) {
            //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
            // }
            // __syncthreads(); 

            for(unsigned p_i = 0 ; p_i < psize_block; p_i ++) {
                // 得到图的入口
                unsigned batch_pid = batch_pids[plist_index + p_i] ;
                unsigned pid = batch_partition_ids[batch_pid] ;
                const unsigned* graph = batch_all_graphs + batch_graph_start_index[batch_pid] ;
                // __syncthreads() ;
                #pragma unroll
                for(unsigned i = threadIdx.y; i < DEGREE; i += blockDim.y){
                    // threadIdx.y 0-5 , blockDim.y 即 6
                    if(top_M_Cand[TOPM + i] == 0xFFFFFFFFu)
                        continue ;

                    unsigned pure_id = top_M_Cand[TOPM + i] & 0x7FFFFFFF ;
                    size_t vec_offset = static_cast<size_t>(pure_id) * static_cast<size_t>(DIM) ;
                    half2 val_res;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                        // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                        // half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
                        half4 val2 ; 
                        val2.x = __half2(
                            __float2half(scales[j * 4] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4)] - zero_points[j * 4])) ,
                            __float2half(scales[j * 4 + 1] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4 + 1)] - zero_points[j * 4 + 1]))
                        ) ;
                        val2.y = __half2(
                            __float2half(scales[j * 4 + 2] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4 + 2)] - zero_points[j * 4 + 2])) ,
                            __float2half(scales[j * 4 + 3] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4 + 3)] - zero_points[j * 4 + 3]))
                        ) ;
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
                //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                // }
                // __syncthreads(); 
                // if(tid == 0)
                //     printf("can click here %d\n" , __LINE__) ;
                // __syncthreads() ;
             
                // 直接对整个列表做距离计算
                if(tid < DEGREE)
                    warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
                __syncthreads() ;
                // 此处被挤出去的点也应该合并到recycle_list 上
                // 将两个列表归并到一起
                merge_top_with_recycle_list<attrType>(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
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
                                hash_insert(hash_table , (top_M_Cand[j] & 0x7FFFFFFF)) ; 
                            }
                        }
                    }
                    __syncthreads();
        
                    // if(tid == 0) {
                    //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                    // }
                    // __syncthreads(); 
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
                    // if(tid == 0) {
                    //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                    // }
                    // __syncthreads(); 
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
                        // if(to_explore * DEGREE + j >= 2777778 * 16) {
                            // printf("bid : %d , to explore * degree + j : %d , to explore : %d\n" , bid , to_explore * DEGREE + j , to_explore) ;
                        // }
                        unsigned to_append_local = graph[to_explore * DEGREE + j];
                        // if(seg_global_idx_start_idx[pid] + to_append_local >= 100000000) {
                            // printf("bid : %d ,  pid : %d , seg_global_idx_start_idx[pid] : %d , to_append_local : %d , all : %d\n" ,
                            // bid , pid , seg_global_idx_start_idx[pid] , to_append_local ,  seg_global_idx_start_idx[pid] + to_append_local) ;
                            // printf("bid : %d , j : %d , globla_idx : %d\n" , bid , j , global_idx[seg_global_idx_start_idx[pid] + to_append_local]) ;
                        // }
                        unsigned to_append = global_idx[seg_global_idx_start_idx[pid] + to_append_local] ;
                        // printf("j : %d , go in  \n" , j) ;
                        top_M_Cand_dis[TOPM + j] = (hash_insert(hash_table , to_append) == 0 ? INF_DIS : 0.0) ;
                        top_M_Cand[TOPM + j] = to_append;
                        // printf("j : %d , go out \n" , j ) ;
                    }
                    __syncthreads();
                    // if(tid == 0) {
                    //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                    // }
                    // __syncthreads(); 
                    // calculate distance
                    #pragma unroll
                    for(unsigned j = threadIdx.y; j < DEGREE; j += blockDim.y){
                        // 6个warp, 每个warp负责一个距离计算
                        // if(top_M_Cand_dis[TOPM + j] == INF_DIS)
                        if(top_M_Cand[TOPM + j] == 0xFFFFFFFFu)
                            continue ;
                            unsigned pure_id = top_M_Cand[TOPM + j] & 0x7FFFFFFF ;
                            size_t vec_offset = static_cast<size_t>(pure_id) * static_cast<size_t>(DIM) ;

                            half2 val_res;

                            val_res.x = 0.0; val_res.y = 0.0;
                            for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                                // half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                                half4 val2 ; 
                                val2.x = __half2(
                                    __float2half(scales[k * 4] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4)] - zero_points[k * 4])) ,
                                    __float2half(scales[k * 4 + 1] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 1)] - zero_points[k * 4 + 1]))
                                ) ;
                                val2.y = __half2(
                                    __float2half(scales[k * 4 + 2] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 2)] - zero_points[k * 4 + 2])) ,
                                    __float2half(scales[k * 4 + 3] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 3)] - zero_points[k * 4 + 3]))
                                ) ;
                                val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                                val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                            }
                            #pragma unroll
                            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                            }
        
                            if(laneid == 0){
                                top_M_Cand_dis[TOPM + j] += __half2float(val_res.x) + __half2float(val_res.y) ;
                                atomicAdd(dis_cal_cnt + bid , 1) ;
                            }
                    }
                    __syncthreads();
                    // if(tid == 0) {
                    //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                    // }
                    // __syncthreads(); 
        
                    if(tid < DEGREE)
                        warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
                    __syncthreads() ;
                    if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                        merge_top_with_recycle_list<attrType>(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , q_l_bound , q_r_bound , attrs , attr_dim) ;
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
                // if(tid == 0) {
                //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                // }
                // __syncthreads(); 
        
                // bug产生原因: 在扩展完候选点后将hash表清空了
                // for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y)
                //     top_M_Cand[i] |= 0x80000000 ;
                // for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
                //     hash_table[i] = 0xFFFFFFFF ;
                // }
                // __syncthreads() ;
        
                // 修改一下顺序, 先扩展候选点, 然后再筛除不符合条件的点
                // 从global_edges 中扩展候选点
                if(p_i < psize_block - 1) {
                    // 修改一下扩展候选点的逻辑, 向工作区中输送64个候选点, 计算距离后排序, 然后将前16个点送入top_M_Cand + TOPM 中
                    // unsigned next_pid = pids[plist_index + p_i + 1] ;
                    unsigned next_batch_pid = batch_pids[plist_index + p_i + 1] ;
                    unsigned next_pid = batch_partition_ids[next_batch_pid] ;
                    
                    #pragma unroll
                    for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
                
                        if(top_M_Cand[i % 64] == 0xFFFFFFFF) {
                            // 填充一个随机数
                            work_mem_unsigned[i] = 0xFFFFFFFF ;
                            work_mem_float[i] = INF_DIS ;
                            continue ;
                        }
                        // 用前两个填充, 有重复的部分皆替换为随机点
                        unsigned pure_id = top_M_Cand[i % 64] & 0x7FFFFFFF ;
                        // unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / 64] ;
                        unsigned stored_pos = global_vec_stored_pos[pure_id] ;    
                        // unsigned offset = batch_all_point_num * edges_num_in_global_graph * next_batch_pid + stored_pos * edges_num_in_global_graph ;
                        size_t offset = static_cast<size_t>(batch_all_point_num) * static_cast<size_t>(edges_num_in_global_graph) * static_cast<size_t>(next_batch_pid) +
                            static_cast<size_t>(stored_pos) * static_cast<size_t>(edges_num_in_global_graph) ;
                        unsigned next_local_id = batch_global_graph[offset + static_cast<size_t>(i / 64)] ;
                        unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                        work_mem_unsigned[i] = (hash_insert(hash_table , next_global_id) != 0 ? next_global_id : 0xFFFFFFFF) ;
                        work_mem_float[i] = (work_mem_unsigned[i] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    }
                    
                    /**
                    size_t j = 0 ;
                    for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
                        work_mem_unsigned[i] = 0xFFFFFFFFu ;
                        work_mem_float[i] = INF_DIS ;
                        if(TOPM < (i % 64) || top_M_Cand[i % 64] == 0xFFFFFFFFu)
                            continue ;
                        unsigned pure_id = top_M_Cand[i % 64] & 0x7FFFFFFF ;
                        unsigned stored_pos = global_vec_stored_pos[pure_id] ;    
                        size_t offset = static_cast<size_t>(batch_all_point_num) * static_cast<size_t>(edges_num_in_global_graph) * static_cast<size_t>(next_batch_pid) +
                            static_cast<size_t>(stored_pos) * static_cast<size_t>(edges_num_in_global_graph) ;
                        while(j < edges_num_in_global_graph) {
                            unsigned next_local_id = batch_global_graph[offset + j] ;
                            unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                            j ++ ;
                            if(hash_insert(hash_table , next_local_id) != 0) {
                                work_mem_unsigned[i] = next_global_id ;
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
                    **/
                    __syncthreads() ;
                    // if(tid == 0) {
                    //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                    // }
                    // __syncthreads(); 
                    // 计算这64个点的距离
                    for(unsigned i = threadIdx.y ; i < 128 ; i += blockDim.y) {
                        if(work_mem_unsigned[i] == 0xFFFFFFFFu)
                            continue ;
                        half2 val_res;
                        val_res.x = 0.0; val_res.y = 0.0;
                        unsigned pure_id = work_mem_unsigned[i] & 0x7FFFFFFF ;
                        size_t vec_offset = static_cast<size_t>(pure_id) * static_cast<size_t>(DIM) ;
                        for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                            // half4 val2 = Load(&values[work_mem_unsigned[i] * DIM + k * 4]);
                            half4 val2 ; 
                            val2.x = __half2(
                                __float2half(scales[k * 4] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4)] - zero_points[k * 4])) ,
                                __float2half(scales[k * 4 + 1] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 1)] - zero_points[k * 4 + 1]))
                            ) ;
                            val2.y = __half2(
                                __float2half(scales[k * 4 + 2] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 2)] - zero_points[k * 4 + 2])) ,
                                __float2half(scales[k * 4 + 3] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 3)] - zero_points[k * 4 + 3]))
                            ) ;
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
                    // if(tid == 0) {
                    //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                    // }
                    // __syncthreads(); 
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
                            hash_insert(hash_table , work_mem_unsigned[i]) ;
                    }
                    for(unsigned i = tid ; i < 128 - DEGREE ; i += blockDim.x * blockDim.y)
                        if(work_mem_unsigned[DEGREE + i] != 0xFFFFFFFFu)
                            hash_insert(hash_table , work_mem_unsigned[DEGREE + i]) ;
                    // 将后64 - 16 个点与前表合并
                    // __syncthreads() ;
                    // if(tid == 0) {
                    //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                    // }
                    // __syncthreads(); 
        
                }
                __syncthreads() ;
        
                if(p_i < psize_block - 1) {
                    merge_top_with_recycle_list<attrType>(top_M_Cand, work_mem_unsigned + DEGREE, top_M_Cand_dis, work_mem_float + DEGREE, tid , 128 - DEGREE , TOPM , q_l_bound , q_r_bound , attrs , attr_dim) ;
                    __syncthreads() ;
                    // if(tid < DEGREE)
                    //     warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
                    bitonic_sort_id_by_dis_no_explore(work_mem_float + DEGREE , work_mem_unsigned + DEGREE , 128 - DEGREE) ;
                    __syncthreads() ;
                    if(work_mem_float[DEGREE] != 0xFFFFFFFF) {
                    // 此时top_M_Cand + TOPM中的点一定是满足属性过滤条件的点
                        merge_top(recycle_list_id , work_mem_unsigned + DEGREE , recycle_list_dis , work_mem_float + DEGREE, tid , 128 - DEGREE , recycle_list_size) ;
                        __syncthreads() ;
                    }
                    for(unsigned i = tid ; i < TOPM + DEGREE ; i += blockDim.x * blockDim.y) {
                        if(top_M_Cand[i] != 0xFFFFFFFF)
                            hash_insert(hash_table , top_M_Cand[i] & 0x7FFFFFFF) ;
                    }
                    __syncthreads() ;
                }
                    // merge_top(top_M_Cand , work_mem_unsigned + DEGREE, top_M_Cand_dis , work_mem_float + DEGREE , tid , 128 - DEGREE , TOPM) ;
                    // merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float  , tid , 64 , TOPM) ;
                // __syncthreads() ;

                // if(tid == 0) {
                //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                // }
                // __syncthreads(); 
            }

            // 搜索最后做后过滤
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
                }
                __syncthreads() ;
                for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                    // unsigned cell_id = chunk_id + (i >> 5) ; 
                    unsigned cell_id = (i >> 5) ;
                    
                    unsigned id_in_cell = i % 32 ;
                    unsigned bit_mask = (1 << id_in_cell) ;
                    if((bit_map[cell_id] & bit_mask) != 0) {
                        // 获取插入位置
                        
                        unsigned local_cell_id = (i >> 5) ;
                        unsigned put_pos = (local_cell_id == 0 ? 0 : top_prefix[local_cell_id - 1]) + (id_in_cell == 0 ? 0 : prefix_sum[i - 1]) ;
                        indices_arr[put_pos] = i ;
                    }
                }
                __syncthreads() ;
                if(tid == 0) {
                    start = clock64() ;
                }
                __syncthreads() ;
                 
                // 计算距离, 然后用原子操作插入队列中
                for(unsigned i = threadIdx.y ; i < task_num ; i += blockDim.y) {
                    unsigned cand_id = indices_arr[i] + point_batch_start_id ;
                    
                    // 计算距离
                    unsigned g_global_id = global_idx[global_idx_offset + cand_id] ;
                    size_t vec_offset = static_cast<size_t>(g_global_id) * static_cast<size_t>(DIM) ;
                    float dis = 0.0f ;

                    half2 val_res;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                        // half4 val2 = Load(&batch_values[vec_offset + static_cast<size_t>(k * 4)]);
                        // char4 val = Load(&batch_values[vec_offset + static_cast<size_t>(k * 4)]) ;
                        half4 val2 ; 
                        val2.x = __half2(
                            __float2half(scales[k * 4] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4)] - zero_points[k * 4])) ,
                            __float2half(scales[k * 4 + 1] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 1)] - zero_points[k * 4 + 1]))
                        ) ;
                        val2.y = __half2(
                            __float2half(scales[k * 4 + 2] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 2)] - zero_points[k * 4 + 2])) ,
                            __float2half(scales[k * 4 + 3] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 3)] - zero_points[k * 4 + 3]))
                        ) ;
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
                if(tid == 0) {
                    dis_cal_cost += ((double)(clock64() - start) / 1582000) ;
                }
                __syncthreads() ;
               
                if(prefilter_pointer >= 32 || (chunk_id + chunk_batch_size >= chunk_num && prefilter_pointer > 0)) {
                  
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

        /**
        #pragma unroll
        for (unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
            work_mem_unsigned[i] = top_M_Cand_video_memory[bid * TOPM + i] ;
            work_mem_float[i] = top_M_Cand_dis_video_memory[bid * TOPM + i] ;
        }
        
        __syncthreads() ;
        merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float , tid , K , TOPM) ;
        __syncthreads() ;
        **/

        // 每次执行完毕, 和现有结果做合并(和前k个结果合并即可)
        for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
            top_M_Cand_video_memory[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
            top_M_Cand_dis_video_memory[bid * TOPM + i] = top_M_Cand_dis[i];
        }
        for(unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
            result_k_id[bid * K + i] = (top_M_Cand[i] & 0x7FFFFFFF) ;
            // result_k_dis[bid * K + i] = top_M_Cand_dis[i] ;
        }

        if(tid == 0) {
            // printf("bid : %d , dis cal cost : %f\n" , bid , dis_cal_cost ) ;
            dis_cal_cost_total[bid] = dis_cal_cost ;
        }
    }



    template<typename attrType = float>
    __global__ void select_path_contain_attr_filter_V2_without_output_single_channel_with_prefilter_batch_process_compressed_graph_T_dataset_resident_pure_sort(const unsigned batch_size , const unsigned* batch_partition_ids , const unsigned* __restrict__ batch_all_graphs, const int8_t* __restrict__ batch_values, const half* __restrict__ query_data, 
        const float* __restrict__ scales_video_memory , const int* __restrict__ zero_points_video_memory ,
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
        __shared__ float scales[128] ;
        __shared__ int zero_points[128] ;
        
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

        if(total_psize_block == 0) {
            for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
                top_M_Cand_video_memory[bid * TOPM + i] = 0xFFFFFFFFu ;
                top_M_Cand_dis_video_memory[bid * TOPM + i] = INF_DIS ;
            }

            return ;
        } 
       
        // if(bid >= 100)
        //     return ;
        __shared__ unsigned cur_visit_p , main_channel , turn ;
        __shared__ unsigned to_explore ;
        __shared__ unsigned prefilter_pointer ;
        __shared__ long long start ;
        __shared__ double dis_cal_cost ;

        __shared__ unsigned hash_table[HASHLEN];
        // 从全局内存中恢复topm
        for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
            top_M_Cand[i] = 0xFFFFFFFF;
            top_M_Cand_dis[i] = INF_DIS;
        }
        for(unsigned i = tid ; i < attr_dim ; i += blockDim.x * blockDim.y) {
            q_l_bound[i] = l_bound[bid * attr_dim + i] ;
            q_r_bound[i] = r_bound[bid * attr_dim + i] ;
        }

        for(unsigned i = tid ; i < DIM ; i  += blockDim.x * blockDim.y) {
            scales[i] = scales_video_memory[i] ;
            zero_points[i] = zero_points_video_memory[i] ;
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

        
        if(psize_block > 0 && little_seg_num_block == 0) {
            psize_block = total_psize_block ;
            unsigned init_pid = batch_partition_ids[batch_pids[plist_index]] ;
            for(unsigned i = tid; i < DEGREE; i += blockDim.x * blockDim.y) {
                top_M_Cand[TOPM + i] = global_idx[seg_global_idx_start_idx[init_pid] + ent_pts[i]];
                hash_insert(hash_table , top_M_Cand[TOPM + i]) ;
            }
            __syncthreads() ;
            
            // if(tid == 0) {
            //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
            // }
            // __syncthreads(); 

            for(unsigned p_i = 0 ; p_i < psize_block; p_i ++) {
                // 得到图的入口
                unsigned batch_pid = batch_pids[plist_index + p_i] ;
                unsigned pid = batch_partition_ids[batch_pid] ;
                const unsigned* graph = batch_all_graphs + batch_graph_start_index[batch_pid] ;
                // __syncthreads() ;
                #pragma unroll
                for(unsigned i = threadIdx.y; i < DEGREE; i += blockDim.y){
                    // threadIdx.y 0-5 , blockDim.y 即 6
                    if(top_M_Cand[TOPM + i] == 0xFFFFFFFFu)
                        continue ;

                    unsigned pure_id = top_M_Cand[TOPM + i] & 0x7FFFFFFF ;
                    size_t vec_offset = static_cast<size_t>(pure_id) * static_cast<size_t>(DIM) ;
                    half2 val_res;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                        // warp中的每个子线程负责8个字节 即4个half类型分量的计算
                        // half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
                        half4 val2 ; 
                        val2.x = __half2(
                            __float2half(scales[j * 4] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4)] - zero_points[j * 4])) ,
                            __float2half(scales[j * 4 + 1] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4 + 1)] - zero_points[j * 4 + 1]))
                        ) ;
                        val2.y = __half2(
                            __float2half(scales[j * 4 + 2] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4 + 2)] - zero_points[j * 4 + 2])) ,
                            __float2half(scales[j * 4 + 3] * (float)(batch_values[vec_offset + static_cast<size_t>(j * 4 + 3)] - zero_points[j * 4 + 3]))
                        ) ;
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
                //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                // }
                // __syncthreads(); 
                // if(tid == 0)
                //     printf("can click here %d\n" , __LINE__) ;
                // __syncthreads() ;
             
                // 直接对整个列表做距离计算
                if(tid < DEGREE)
                    warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
                __syncthreads() ;
                merge_top_with_recycle_list<attrType>(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , q_l_bound , q_r_bound , attrs ,attr_dim) ;
                
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
                                hash_insert(hash_table , (top_M_Cand[j] & 0x7FFFFFFF)) ; 
                            }
                        }
                    }
                    __syncthreads();
        
                    // if(tid == 0) {
                    //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                    // }
                    // __syncthreads(); 
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
                    // if(tid == 0) {
                    //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                    // }
                    // __syncthreads(); 
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
                        // if(to_explore * DEGREE + j >= 2777778 * 16) {
                            // printf("bid : %d , to explore * degree + j : %d , to explore : %d\n" , bid , to_explore * DEGREE + j , to_explore) ;
                        // }
                        unsigned to_append_local = graph[to_explore * DEGREE + j];
                        // if(seg_global_idx_start_idx[pid] + to_append_local >= 100000000) {
                            // printf("bid : %d ,  pid : %d , seg_global_idx_start_idx[pid] : %d , to_append_local : %d , all : %d\n" ,
                            // bid , pid , seg_global_idx_start_idx[pid] , to_append_local ,  seg_global_idx_start_idx[pid] + to_append_local) ;
                            // printf("bid : %d , j : %d , globla_idx : %d\n" , bid , j , global_idx[seg_global_idx_start_idx[pid] + to_append_local]) ;
                        // }
                        unsigned to_append = global_idx[seg_global_idx_start_idx[pid] + to_append_local] ;
                        // printf("j : %d , go in  \n" , j) ;
                        top_M_Cand_dis[TOPM + j] = (hash_insert(hash_table , to_append) == 0 ? INF_DIS : 0.0) ;
                        top_M_Cand[TOPM + j] = to_append;
                        // printf("j : %d , go out \n" , j ) ;
                    }
                    __syncthreads();
                    // if(tid == 0) {
                    //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                    // }
                    // __syncthreads(); 
                    // calculate distance
                    #pragma unroll
                    for(unsigned j = threadIdx.y; j < DEGREE; j += blockDim.y){
                        // 6个warp, 每个warp负责一个距离计算
                        // if(top_M_Cand_dis[TOPM + j] == INF_DIS)
                        if(top_M_Cand[TOPM + j] == 0xFFFFFFFFu)
                            continue ;
                            unsigned pure_id = top_M_Cand[TOPM + j] & 0x7FFFFFFF ;
                            size_t vec_offset = static_cast<size_t>(pure_id) * static_cast<size_t>(DIM) ;

                            half2 val_res;

                            val_res.x = 0.0; val_res.y = 0.0;
                            for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                                // half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                                half4 val2 ; 
                                val2.x = __half2(
                                    __float2half(scales[k * 4] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4)] - zero_points[k * 4])) ,
                                    __float2half(scales[k * 4 + 1] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 1)] - zero_points[k * 4 + 1]))
                                ) ;
                                val2.y = __half2(
                                    __float2half(scales[k * 4 + 2] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 2)] - zero_points[k * 4 + 2])) ,
                                    __float2half(scales[k * 4 + 3] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 3)] - zero_points[k * 4 + 3]))
                                ) ;
                                val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                                val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                            }
                            #pragma unroll
                            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                            }
        
                            if(laneid == 0){
                                top_M_Cand_dis[TOPM + j] += __half2float(val_res.x) + __half2float(val_res.y) ;
                                atomicAdd(dis_cal_cnt + bid , 1) ;
                            }
                    }
                    __syncthreads();
                    // if(tid == 0) {
                    //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                    // }
                    // __syncthreads(); 
                    /**
                    if(tid < DEGREE)
                        warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
                    __syncthreads() ;
                    if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                        merge_top_with_recycle_list<attrType>(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid , DEGREE , TOPM , q_l_bound , q_r_bound , attrs , attr_dim) ;
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
                        **/
                    // 先对列表整体排序, 然后对后段列表做过滤, 并将其合并到循环池
                    bitonic_sort_id_by_dis_no_explore(top_M_Cand_dis , top_M_Cand , TOPM + DEGREE) ;
                    __syncthreads() ;
                    for(unsigned j = tid ; j < DEGREE ; j += blockDim.x * blockDim.y) {
                        if(top_M_Cand[TOPM + j] == 0xFFFFFFFF)
                            continue ;
                        bool flag = true ;
                        unsigned pure_id = top_M_Cand[TOPM + j] & 0x7FFFFFFF ;
                        for(unsigned j = 0 ; j < attr_dim && flag ; j ++) {
                            flag = (flag && (attrs[pure_id * attr_dim + j] >= q_l_bound[j]) && (attrs[pure_id * attr_dim + j] <= q_r_bound[j])) ;
                        }
                        if(!flag) {
                            top_M_Cand[TOPM + j] = 0xFFFFFFFF ;
                            top_M_Cand_dis[TOPM + j] = INF_DIS ;
                        }
                    }
                    __syncthreads() ;
                    if(tid < DEGREE)
                        warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
                    __syncthreads() ;
                    if(top_M_Cand[TOPM] != 0xFFFFFFFF) {
                        // 此时top_M_Cand + TOPM中的点一定是满足属性过滤条件的点
                        merge_top(recycle_list_id , &top_M_Cand[TOPM] , recycle_list_dis , &top_M_Cand_dis[TOPM], tid , DEGREE , recycle_list_size) ;
                        __syncthreads() ;
                    }
                    // 问题来了, 当图开始切换时, 当前部分结果的id应该采用局部id还是全局id? 在列表中的所有id皆采用全局id
                    // 对后半段列表sort一次, 然后和recycle pool进行merge
                    // bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM] , &top_M_Cand[TOPM] , DEGREE) ;
        
                }
                __syncthreads();
                // if(tid == 0) {
                //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                // }
                // __syncthreads(); 
        
                // bug产生原因: 在扩展完候选点后将hash表清空了
                // for(unsigned i = tid ; i < TOPM ; i += blockDim.x * blockDim.y)
                //     top_M_Cand[i] |= 0x80000000 ;
                // for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
                //     hash_table[i] = 0xFFFFFFFF ;
                // }
                // __syncthreads() ;
        
                // 修改一下顺序, 先扩展候选点, 然后再筛除不符合条件的点
                // 从global_edges 中扩展候选点
                if(p_i < psize_block - 1) {
                    // 修改一下扩展候选点的逻辑, 向工作区中输送64个候选点, 计算距离后排序, 然后将前16个点送入top_M_Cand + TOPM 中
                    // unsigned next_pid = pids[plist_index + p_i + 1] ;
                    unsigned next_batch_pid = batch_pids[plist_index + p_i + 1] ;
                    unsigned next_pid = batch_partition_ids[next_batch_pid] ;
                    
                    #pragma unroll
                    for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
                
                        if(top_M_Cand[i % 64] == 0xFFFFFFFF) {
                            // 填充一个随机数
                            work_mem_unsigned[i] = 0xFFFFFFFF ;
                            work_mem_float[i] = INF_DIS ;
                            continue ;
                        }
                        // 用前两个填充, 有重复的部分皆替换为随机点
                        unsigned pure_id = top_M_Cand[i % 64] & 0x7FFFFFFF ;
                        // unsigned next_local_id = global_graph[pure_id * pnum * edges_num_in_global_graph + next_pid * edges_num_in_global_graph + i / 64] ;
                        unsigned stored_pos = global_vec_stored_pos[pure_id] ;    
                        // unsigned offset = batch_all_point_num * edges_num_in_global_graph * next_batch_pid + stored_pos * edges_num_in_global_graph ;
                        size_t offset = static_cast<size_t>(batch_all_point_num) * static_cast<size_t>(edges_num_in_global_graph) * static_cast<size_t>(next_batch_pid) +
                            static_cast<size_t>(stored_pos) * static_cast<size_t>(edges_num_in_global_graph) ;
                        unsigned next_local_id = batch_global_graph[offset + static_cast<size_t>(i / 64)] ;
                        unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                        work_mem_unsigned[i] = (hash_insert(hash_table , next_global_id) != 0 ? next_global_id : 0xFFFFFFFF) ;
                        work_mem_float[i] = (work_mem_unsigned[i] != 0xFFFFFFFF ? 0.0f : INF_DIS) ;
                    }
                    
                    /**
                    size_t j = 0 ;
                    for(unsigned i = tid ; i < 128 ; i += blockDim.x * blockDim.y) { 
                        work_mem_unsigned[i] = 0xFFFFFFFFu ;
                        work_mem_float[i] = INF_DIS ;
                        if(TOPM < (i % 64) || top_M_Cand[i % 64] == 0xFFFFFFFFu)
                            continue ;
                        unsigned pure_id = top_M_Cand[i % 64] & 0x7FFFFFFF ;
                        unsigned stored_pos = global_vec_stored_pos[pure_id] ;    
                        size_t offset = static_cast<size_t>(batch_all_point_num) * static_cast<size_t>(edges_num_in_global_graph) * static_cast<size_t>(next_batch_pid) +
                            static_cast<size_t>(stored_pos) * static_cast<size_t>(edges_num_in_global_graph) ;
                        while(j < edges_num_in_global_graph) {
                            unsigned next_local_id = batch_global_graph[offset + j] ;
                            unsigned next_global_id = global_idx[seg_global_idx_start_idx[next_pid] + next_local_id] ;
                            j ++ ;
                            if(hash_insert(hash_table , next_local_id) != 0) {
                                work_mem_unsigned[i] = next_global_id ;
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
                    **/
                    __syncthreads() ;
                    // if(tid == 0) {
                    //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                    // }
                    // __syncthreads(); 
                    // 计算这64个点的距离
                    for(unsigned i = threadIdx.y ; i < 128 ; i += blockDim.y) {
                        if(work_mem_unsigned[i] == 0xFFFFFFFFu)
                            continue ;
                        half2 val_res;
                        val_res.x = 0.0; val_res.y = 0.0;
                        unsigned pure_id = work_mem_unsigned[i] & 0x7FFFFFFF ;
                        size_t vec_offset = static_cast<size_t>(pure_id) * static_cast<size_t>(DIM) ;
                        for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                            // half4 val2 = Load(&values[work_mem_unsigned[i] * DIM + k * 4]);
                            half4 val2 ; 
                            val2.x = __half2(
                                __float2half(scales[k * 4] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4)] - zero_points[k * 4])) ,
                                __float2half(scales[k * 4 + 1] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 1)] - zero_points[k * 4 + 1]))
                            ) ;
                            val2.y = __half2(
                                __float2half(scales[k * 4 + 2] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 2)] - zero_points[k * 4 + 2])) ,
                                __float2half(scales[k * 4 + 3] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 3)] - zero_points[k * 4 + 3]))
                            ) ;
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
                    // if(tid == 0) {
                    //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                    // }
                    // __syncthreads(); 
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
                            hash_insert(hash_table , work_mem_unsigned[i]) ;
                    }
                    for(unsigned i = tid ; i < 128 - DEGREE ; i += blockDim.x * blockDim.y)
                        if(work_mem_unsigned[DEGREE + i] != 0xFFFFFFFFu)
                            hash_insert(hash_table , work_mem_unsigned[DEGREE + i]) ;
                    // 将后64 - 16 个点与前表合并
                    // __syncthreads() ;
                    // if(tid == 0) {
                    //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                    // }
                    // __syncthreads(); 
        
                }
                __syncthreads() ;
        
                if(p_i < psize_block - 1) {
                    merge_top_with_recycle_list<attrType>(top_M_Cand, work_mem_unsigned + DEGREE, top_M_Cand_dis, work_mem_float + DEGREE, tid , 128 - DEGREE , TOPM , q_l_bound , q_r_bound , attrs , attr_dim) ;
                    __syncthreads() ;
                    // if(tid < DEGREE)
                    //     warpSort(&top_M_Cand_dis[TOPM + tid], &top_M_Cand[TOPM + tid], tid , DEGREE);
                    bitonic_sort_id_by_dis_no_explore(work_mem_float + DEGREE , work_mem_unsigned + DEGREE , 128 - DEGREE) ;
                    __syncthreads() ;
                    if(work_mem_float[DEGREE] != 0xFFFFFFFF) {
                    // 此时top_M_Cand + TOPM中的点一定是满足属性过滤条件的点
                        merge_top(recycle_list_id , work_mem_unsigned + DEGREE , recycle_list_dis , work_mem_float + DEGREE, tid , 128 - DEGREE , recycle_list_size) ;
                        __syncthreads() ;
                    }
                    for(unsigned i = tid ; i < TOPM + DEGREE ; i += blockDim.x * blockDim.y) {
                        if(top_M_Cand[i] != 0xFFFFFFFF)
                            hash_insert(hash_table , top_M_Cand[i] & 0x7FFFFFFF) ;
                    }
                    __syncthreads() ;
                }
                    // merge_top(top_M_Cand , work_mem_unsigned + DEGREE, top_M_Cand_dis , work_mem_float + DEGREE , tid , 128 - DEGREE , TOPM) ;
                    // merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float  , tid , 64 , TOPM) ;
                // __syncthreads() ;

                // if(tid == 0) {
                //     printf("bid : %d , can click  : %d\n" , bid , __LINE__) ;
                // }
                // __syncthreads(); 
            }

            // 搜索最后做后过滤
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
                }
                __syncthreads() ;
                for(unsigned i = tid ; i < point_batch_size ; i += blockDim.x * blockDim.y) {
                    // unsigned cell_id = chunk_id + (i >> 5) ; 
                    unsigned cell_id = (i >> 5) ;
                    
                    unsigned id_in_cell = i % 32 ;
                    unsigned bit_mask = (1 << id_in_cell) ;
                    if((bit_map[cell_id] & bit_mask) != 0) {
                        // 获取插入位置
                        
                        unsigned local_cell_id = (i >> 5) ;
                        unsigned put_pos = (local_cell_id == 0 ? 0 : top_prefix[local_cell_id - 1]) + (id_in_cell == 0 ? 0 : prefix_sum[i - 1]) ;
                        indices_arr[put_pos] = i ;
                    }
                }
                __syncthreads() ;
                if(tid == 0) {
                    start = clock64() ;
                }
                __syncthreads() ;
                 
                // 计算距离, 然后用原子操作插入队列中
                for(unsigned i = threadIdx.y ; i < task_num ; i += blockDim.y) {
                    unsigned cand_id = indices_arr[i] + point_batch_start_id ;
                    
                    // 计算距离
                    unsigned g_global_id = global_idx[global_idx_offset + cand_id] ;
                    size_t vec_offset = static_cast<size_t>(g_global_id) * static_cast<size_t>(DIM) ;
                    float dis = 0.0f ;

                    half2 val_res;
                    val_res.x = 0.0; val_res.y = 0.0;
                    for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                        // half4 val2 = Load(&batch_values[vec_offset + static_cast<size_t>(k * 4)]);
                        // char4 val = Load(&batch_values[vec_offset + static_cast<size_t>(k * 4)]) ;
                        half4 val2 ; 
                        val2.x = __half2(
                            __float2half(scales[k * 4] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4)] - zero_points[k * 4])) ,
                            __float2half(scales[k * 4 + 1] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 1)] - zero_points[k * 4 + 1]))
                        ) ;
                        val2.y = __half2(
                            __float2half(scales[k * 4 + 2] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 2)] - zero_points[k * 4 + 2])) ,
                            __float2half(scales[k * 4 + 3] * (float)(batch_values[vec_offset + static_cast<size_t>(k * 4 + 3)] - zero_points[k * 4 + 3]))
                        ) ;
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
                if(tid == 0) {
                    dis_cal_cost += ((double)(clock64() - start) / 1582000) ;
                }
                __syncthreads() ;
               
                if(prefilter_pointer >= 32 || (chunk_id + chunk_batch_size >= chunk_num && prefilter_pointer > 0)) {
                  
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

        /**
        #pragma unroll
        for (unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
            work_mem_unsigned[i] = top_M_Cand_video_memory[bid * TOPM + i] ;
            work_mem_float[i] = top_M_Cand_dis_video_memory[bid * TOPM + i] ;
        }
        
        __syncthreads() ;
        merge_top(top_M_Cand , work_mem_unsigned , top_M_Cand_dis , work_mem_float , tid , K , TOPM) ;
        __syncthreads() ;
        **/

        // 每次执行完毕, 和现有结果做合并(和前k个结果合并即可)
        for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
            top_M_Cand_video_memory[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
            top_M_Cand_dis_video_memory[bid * TOPM + i] = top_M_Cand_dis[i];
        }
        for(unsigned i = tid ; i < K ; i += blockDim.x * blockDim.y) {
            result_k_id[bid * K + i] = (top_M_Cand[i] & 0x7FFFFFFF) ;
            // result_k_dis[bid * K + i] = top_M_Cand_dis[i] ;
        }

        if(tid == 0) {
            // printf("bid : %d , dis cal cost : %f\n" , bid , dis_cal_cost ) ;
            dis_cal_cost_total[bid] = dis_cal_cost ;
        }
    }



    template<typename attrType = float>
    vector<vector<unsigned>> batch_graph_search_gpu_with_prefilter_batch_process_compressed_graph_T_dataset_resident(unsigned batch_num , vector<vector<unsigned>>& batch_info , unsigned degree , const float* data_ori  , const int8_t* data_ , const float* scales , const int* zero_points , array<unsigned*,2>& graph_buffer , array<unsigned*,2>& global_graph_buffer ,
        array<unsigned*,2>& generic_buffer , const float* query_ori , __half* query , unsigned dim , unsigned qnum ,  attrType* l_bound , attrType* r_bound , attrType* attrs , 
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
        
        vector<vector<unsigned>> resort_list_big(qnum) ;

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
                size_t graph_byteSize = static_cast<size_t>(graph_info[1][pid]) * static_cast<size_t>(degree) * sizeof(unsigned) ;
                cudaMemcpyAsync(graph_buffer[turn] + start_pointer[1] , graph_info[0] + graph_info[2][pid] , graph_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
                CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
               
                for(unsigned j = 0 ; j < batch_size ; j ++) {
                    unsigned jpid = batch_info[batch][j] ;
                    // 行开始索引 + 列索引
                    size_t global_graph_offset = static_cast<size_t>(pid) * static_cast<size_t>(cardinality) * static_cast<size_t>(edge_num_contain) + 
                    static_cast<size_t>(seg_partition_point_start_index[jpid]) * static_cast<size_t>(edge_num_contain) ;
                    size_t global_graph_byteSize = static_cast<size_t>(graph_info[1][jpid]) * static_cast<size_t>(edge_num_contain) * sizeof(unsigned) ;
                    cudaMemcpyAsync(global_graph_buffer[turn] + start_pointer[2] , global_edges + global_graph_offset , global_graph_byteSize , cudaMemcpyHostToDevice , stream_non_block) ;
                    start_pointer[2] += static_cast<size_t>(graph_info[1][jpid]) * static_cast<size_t>(edge_num_contain) ;
                }
    
                // 映射所涉及全局id的存储位置
                thrust::transform(
                    thrust::cuda::par.on(stream_non_block) , 
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
                mapper_start += static_cast<size_t>(graph_info[1][pid]) ;
            }
            cout << "内存导入完毕 batch : " << batch <<  endl ;
            unsigned* batch_global_graph_start_index_address_pointer = generic_buffer[turn] ;
            unsigned* batch_graph_size_address_pointer = batch_global_graph_start_index_address_pointer + batch_size ;
            unsigned* batch_partition_ids_address_pointer = batch_graph_size_address_pointer + batch_size ;
           
            cudaMemcpyAsync(batch_global_graph_start_index_address_pointer , batch_graph_start_index_host.data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
            cudaMemcpyAsync(batch_graph_size_address_pointer , batch_graph_size_host.data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
            cudaMemcpyAsync(batch_partition_ids_address_pointer , batch_info[batch].data() , batch_size * sizeof(unsigned) , cudaMemcpyHostToDevice , stream_non_block) ;
          
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

            // 发射核函数, 开始执行
            auto kernal_s = chrono::high_resolution_clock::now() ;
            unsigned byteSize = recycle_list_size * sizeof(unsigned) + recycle_list_size * sizeof(float) + 
            (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float) + 2 * attr_dim * sizeof(float) + (DIM/4) * sizeof(half4) + alignof(half4) ;
            cout << "byteSize : " << (byteSize * 1.0 / 1024) << "KB" << endl ;
         
            select_path_contain_attr_filter_V2_without_output_single_channel_with_prefilter_batch_process_compressed_graph_T_dataset_resident_pure_sort<attrType><<<grid_s , block_s , byteSize>>>(batch_size , batch_partition_ids_address_pointer , graph_buffer[turn], data_ , query, 
                scales , zero_points ,
                results_id , results_dis , ent_pts , DEGREE , TOPM , DIM , batch_global_graph_start_index_address_pointer , batch_graph_size_address_pointer ,
                partition_buffer[turn][0] , partition_buffer[turn][1] , partition_buffer[turn][2] , l_bound , r_bound , attrs , attr_dim , global_graph_buffer[turn] , edge_num_contain ,
                recycle_list_size , idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , pnum , K, result_k_id  , task_id_mapper , partition_infos[3]
                , idx_mapper[3] , mapper_start) ;
            cudaDeviceSynchronize() ;
            CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            auto kernal_e = chrono::high_resolution_clock::now() ;
            // 此处取出所有id, 然后重排
            vector<unsigned> resort_list(qnum * L) ;
            cudaMemcpy(resort_list.data() , results_id , qnum * L * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
            for(int qid = 0 ; qid < qnum ; qid ++) {
                resort_list_big[qid].insert(resort_list_big[qid].end() , resort_list.begin() + qid * L , resort_list.begin() + (qid + 1) * L) ;
            }
            // int t ;
            // cin >> t ;
            
            select_path_kernal_cost += chrono::duration<double>(kernal_e - kernal_s).count() ;
            batch_process_cost.push_back(chrono::duration<double>(kernal_e - kernal_s).count()) ;
    
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
        cudaGetSymbolAddress((void**) &symbol_address_dcct , dis_cal_cost_total) ;
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
    


        // 此处重排 + 返回值
        vector<vector<float>> resort_list_big_float(qnum , vector<float>(L * batch_num)) ;
        auto rerank_s = chrono::high_resolution_clock::now() ;
        #pragma omp parallel for num_threads(10)
        for(int i = 0 ; i < qnum  ; i ++) {
            for(int j = 0 ; j < L * batch_num ; j ++) {
                unsigned vid = resort_list_big[i][j] ;
                if(vid >= 100000000) {
                    resort_list_big_float[i][j] = INF_DIS ;
                    continue ;
                }
                float dis = 0.0 ;
                for(int d = 0 ; d < dim ; d ++) {
                    size_t v_index = static_cast<size_t>(vid) * static_cast<size_t>(dim) + static_cast<size_t>(d) ;
                    dis += (data_ori[v_index] - query_ori[i * dim + d]) * (data_ori[v_index] - query_ori[i * dim + d]) ;
                }
                resort_list_big_float[i][j] = dis ;
            }
            vector<int> args(L * batch_num) ;
            for(int j = 0 ;j  < L * batch_num ; j ++)
                args[j] = j ; 
            sort(args.begin() , args.end() , [&] (int a , int b) {
                return resort_list_big_float[i][a] < resort_list_big_float[i][b] ;
            }) ;
            
            for(int j = 0 ; j < K ; j  ++) {
                results[i][j] = resort_list_big[i][args[j]] ;
            }
        }


        // print_first5_rows(results) ;

        auto rerank_e = chrono::high_resolution_clock::now() ;
        float rerank_cost = chrono::duration<float>(rerank_e - rerank_s).count() ;
        cout << "rerank 耗时: " << rerank_cost << endl ;
        // cout << "result dis : [" ;
        // for(unsigned i = 0 ; i < K ;i  ++) {
        //     cout << results_vec_dis[0][i]  ;
        //     if(i < K - 1)
        //         cout << " ," ;
        // }
        // cout << "]" << endl ;
        return results ;
    }
    
}