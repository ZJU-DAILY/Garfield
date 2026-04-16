#pragma once
#include "gpu_graph.cu"


// 实现, 新的分区识别逻辑, 分区长度不固定, 利用二分查找选界
// 存放分区的右界, 用upper_bound找, l = upper_bound(bound , l_bound) - 1 , r = upper_bound(bound , r_bound) - 1
namespace real_dataset_adapter{ 



    template <typename T>
    __device__ size_t to_upper_bound(const T* sorted_array, size_t array_size, const T& value) {
        size_t left = 0;
        size_t right = array_size;
    
        while (left < right) {
            size_t mid = left + (right - left) / 2;
    
            if (sorted_array[mid] > value) {
                right = mid;  // We want the first element greater than the value
            } else {
                left = mid + 1;  // Continue searching in the right half
            }
        }
    
        return (left == 0 ? left : left - 1);  // This is the index of the first element greater than value
    }

    template <typename T>
    __device__ size_t to_upper_bound_linear_scan(const T* sorted_array, size_t array_size , const T& value) {
        size_t index = 0 ;
        for(; index < array_size && sorted_array[index] <= value ; index ++) ;
        return index - 1 ; 
    }

    template <typename T>
    __device__ size_t to_lower_bound_linear_scan(const T* sorted_array, size_t array_size , const T& value) {
        size_t index = 0 ;
        for(; index < array_size && sorted_array[index] < value ; index ++) ;
        return index ; 
    }

    template<unsigned ATTR_DIM=2 , typename attrType = float>
    __global__ void intersection_area_batch_mark_little_seg(const attrType* l_bound, const attrType* r_bound , const attrType* quantile_bound , const attrType* attr_min , unsigned* attr_grid_size , unsigned attr_dim , 
        // quantile_bound 的形状为(attr_dim , attr_grid_size)
        const unsigned* quantile_dim_offset ,
        unsigned qnum , bool* is_selected , unsigned pnum, bool* is_little_seg , attrType little_seg_threshold) {
        // 每个线程处理一个查询
        unsigned tid = blockDim.x * blockIdx.x + threadIdx.x  , total_num = blockDim.x * gridDim.x ;
        
        for(unsigned i = tid ; i < qnum * pnum ; i += total_num) 
        is_little_seg[i] = is_selected[i] = false ;
        __syncthreads() ;


        unsigned partition_indexes[ATTR_DIM] ;
        unsigned partition_l[ATTR_DIM] ;
        unsigned partition_r[ATTR_DIM] ;
        __shared__ unsigned attr_grid_size_register[ATTR_DIM] ;
        __shared__ attrType* quantile_bound_shared[ATTR_DIM] ;
        __shared__ unsigned* quantile_dim_offset_shared[ATTR_DIM] ;
        __shared__ attrType* attr_min_shared[ATTR_DIM] ;


        for(unsigned i = threadIdx.x ; i < ATTR_DIM ;i  += blockDim.x) {
            attr_grid_size_register[i] = attr_grid_size[i] ;
            quantile_bound_shared[i] = quantile_bound[i] ;
            quantile_dim_offset_shared[i] = quantile_dim_offset[i] ;
            attr_min_shared[i] = attr_min[i] ;
        }
        __syncthreads() ;

        for(unsigned i = tid ; i < qnum ; i += total_num) {
            //得出每个维度的范围, 用设备二分查找
            for(unsigned j = 0 ; j < attr_dim ; j ++) {
                partition_l[j] = to_upper_bound_linear_scan<attrType> (quantile_bound_shared + quantile_dim_offset_shared[j], attr_grid_size_register[j] , l_bound[i * attr_dim + j]) ;
                partition_r[j] = to_upper_bound_linear_scan<attrType> (quantile_bound_shared + quantile_dim_offset_shared[j], attr_grid_size_register[j] , r_bound[i * attr_dim + j]) ;
                // partition_r[j] = (unsigned)((r_bound[i * attr_dim + j] - attr_min[j]) / attr_width[j]) ;
                // if(partition_r[j] >= attr_grid_size[j])
                //     partition_r[j] = attr_grid_size[j] - 1 ;    
                partition_indexes[j] = partition_l[j] ;
            }

            // 枚举attr_dim个范围中的所有组合
            while(true) {
                // printf("tid : %d, attr_grid_size[0] : %d , attr_grid_size[1] : %d , attr_dim : %d\n" , tid , attr_grid_size[0] , attr_grid_size[1] , attr_dim) ;
                unsigned pid = get_partition_id(partition_indexes , attr_grid_size_register , attr_dim) ;
                
                is_selected[i * pnum + pid] = true ; 
                attrType area = 1.0f ;
                // 计算得出当前分区与查询的交叉面积
                for(unsigned j = 0 ; j < attr_dim ;  j++) {
                    // attrType step = (attrType) partition_indexes[j] ;
                    unsigned step = partition_indexes[j] ;
                    // 若在左界的右边或者在
                    // attrType lb = max(attr_min[j] + step * attr_width[j] , l_bound[i * attr_dim + j]) ;
                    // attrType rb = min(attr_min[j] + (step + 1) * attr_width[j] , r_bound[i * attr_dim + j]) ;
                    // attrType lb = max(quantile_bound_shared[quantile_dim_offset_shared[j] + step] , l_bound[i * attr_dim + j]) ;
                    attrType bound_l_seg = (step != 0 ? quantile_bound_shared[quantile_dim_offset_shared[j] + step - 1] : attr_min_shared[j] );
                    attrType bound_r_seg = quantile_bound_shared[quantile_dim_offset_shared[j] + step] ;
                    attrType lb = max(bound_l_seg , l_bound[i * attr_dim + j]) ;
                    attrType rb = min(bound_r_seg , r_bound[i * attr_dim + j]) ;
                    area *= (rb - lb) / (bound_r_seg - bound_l_seg) ;
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
    }

    // 这一版需要左界右界同时保存, 左界用 lower_bound , 右界用 upper_bound 
    template<unsigned ATTR_DIM=2 , typename attrType = float>
    __global__ void intersection_area_batch_mark_little_seg_closed_interval(const attrType* l_bound, const attrType* r_bound , const attrType* quantile_bound_l , const attrType* quantile_bound_r , unsigned* attr_grid_size , unsigned attr_dim , 
        // quantile_bound 的形状为(attr_dim , attr_grid_size)
        const unsigned* quantile_dim_offset ,
        unsigned qnum , bool* is_selected , unsigned pnum, bool* is_little_seg , attrType little_seg_threshold) {
        // 每个线程处理一个查询
        unsigned tid = blockDim.x * blockIdx.x + threadIdx.x  , total_num = blockDim.x * gridDim.x ;
        
        for(unsigned i = tid ; i < qnum * pnum ; i += total_num) 
        is_little_seg[i] = is_selected[i] = false ;
        __syncthreads() ;


        unsigned partition_indexes[ATTR_DIM] ;
        unsigned partition_l[ATTR_DIM] ;
        unsigned partition_r[ATTR_DIM] ;
        __shared__ unsigned attr_grid_size_register[ATTR_DIM] ;
        // __shared__ attrType* quantile_bound_shared_l[ATTR_DIM] ;
        // __shared__ attrType* quantile_bound_shared_r[ATTR_DIM] ;
        __shared__ unsigned quantile_dim_offset_shared[ATTR_DIM] ;
        // __shared__ attrType* attr_min_shared[ATTR_DIM] ;


        for(unsigned i = threadIdx.x ; i < ATTR_DIM ;i  += blockDim.x) {
            attr_grid_size_register[i] = attr_grid_size[i] ;
            // quantile_bound_shared_l[i] = quantile_bound_l[i] ;
            quantile_dim_offset_shared[i] = quantile_dim_offset[i] ;
            // attr_min_shared[i] = attr_min[i] ;
        }
        __syncthreads() ;

        for(unsigned i = tid ; i < qnum ; i += total_num) {
            //得出每个维度的范围, 用设备二分查找
            for(unsigned j = 0 ; j < attr_dim ; j ++) {
                // partition_l[j] = to_upper_bound_linear_scan<attrType> (quantile_bound_shared + quantile_dim_offset_shared[j], attr_grid_size_register[j] , l_bound[i * attr_dim + j]) ;
                partition_l[j] = to_lower_bound_linear_scan<attrType> (quantile_bound_r + quantile_dim_offset_shared[j], attr_grid_size_register[j] , l_bound[i * attr_dim + j]) ;
                // partitoin_r[j] = to_upper_bound_linear_scan<attrType> (quantile_bound_shared + quantile_dim_offset_shared[j], attr_grid_size_register[j] , r_bound[i * attr_dim + j]) ;
                partition_r[j] = to_upper_bound_linear_scan<attrType> (quantile_bound_l + quantile_dim_offset_shared[j], attr_grid_size_register[j] , r_bound[i * attr_dim + j]) ;
                // partition_r[j] = (unsigned)((r_bound[i * attr_dim + j] - attr_min[j]) / attr_width[j]) ;
                // if(partition_r[j] >= attr_grid_size[j])
                //     partition_r[j] = attr_grid_size[j] - 1 ;    
                partition_indexes[j] = partition_l[j] ;
            }

            // 枚举attr_dim个范围中的所有组合
            while(true) {
                // printf("tid : %d, attr_grid_size[0] : %d , attr_grid_size[1] : %d , attr_dim : %d\n" , tid , attr_grid_size[0] , attr_grid_size[1] , attr_dim) ;
                unsigned pid = get_partition_id(partition_indexes , attr_grid_size_register , attr_dim) ;
                
                is_selected[i * pnum + pid] = true ; 
                /**
                attrType area = 1.0f ;
                // 计算得出当前分区与查询的交叉面积
                for(unsigned j = 0 ; j < attr_dim ;  j++) {
                    // attrType step = (attrType) partition_indexes[j] ;
                    unsigned step = partition_indexes[j] ;
                    // 若在左界的右边或者在
                    // attrType lb = max(attr_min[j] + step * attr_width[j] , l_bound[i * attr_dim + j]) ;
                    // attrType rb = min(attr_min[j] + (step + 1) * attr_width[j] , r_bound[i * attr_dim + j]) ;
                    // attrType lb = max(quantile_bound_shared[quantile_dim_offset_shared[j] + step] , l_bound[i * attr_dim + j]) ;
                    // attrType bound_l_seg = (step != 0 ? quantile_bound_l[quantile_dim_offset_shared[j] + step - 1] : attr_min_shared[j] );
                    attrType bound_l_seg = quantile_bound_l[quantile_dim_offset_shared[j] + step] ;
                    // attrType bound_r_seg = quantile_bound_shared[quantile_dim_offset_shared[j] + step] ;
                    attrType bound_r_seg = quantile_bound_r[quantile_dim_offset_shared[j] + step] ;
                    attrType lb = max(bound_l_seg , l_bound[i * attr_dim + j]) ;
                    attrType rb = min(bound_r_seg , r_bound[i * attr_dim + j]) ;
                    area *= (bound_r_seg != bound_l_seg ? (rb - lb) / (bound_r_seg - bound_l_seg) : 0.0) ;
                }

                is_little_seg[i * pnum + pid] = (area <= little_seg_threshold) ;

                **/
                // if(area > 1.0f) {
                    // printf("tid : %d , %f , %f\n" , tid , area , little_seg_threshold) ;
                // }
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
    }


    // 目前只实现逻辑与
    template<unsigned ATTR_DIM = 2, typename T = float>
    array<unsigned*, 5> intersection_area_mark_little_seg(T* l_bound , T* r_bound , T* quantile_bound_l , T* quantile_bound_r , unsigned* quantile_dim_offset, unsigned* attr_grid_size , unsigned attr_dim , unsigned qnum , unsigned pnum , T little_seg_threshold , unsigned OP_TYPE = 0) {
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
            intersection_area_batch_mark_little_seg_closed_interval<ATTR_DIM,T><<<(qnum + 31) / 32 , 32>>>(l_bound, r_bound , quantile_bound_l , quantile_bound_r , attr_grid_size , attr_dim , quantile_dim_offset ,
                 qnum , is_selected , pnum, is_little_seg , little_seg_threshold) ;
        // 逻辑或
        // else if(OP_TYPE == 1)
        //     intersection_area_batch_mark_little_seg_logicalor<ATTR_DIM><<<(qnum + 31) / 32 , 32>>>(l_bound, r_bound , attr_min , attr_width , attr_grid_size , attr_dim , qnum ,  is_selected ,pnum , is_little_seg , little_seg_threshold) ;
        
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

    template<typename attrType = float>
    __global__ void prefiltering_tag_elements(const attrType* __restrict__ attrs , const attrType* __restrict__ l_bound , const attrType* __restrict__ r_bound , 
        const unsigned attr_dim , const unsigned num , attrType threshold , unsigned* seg_num_counts , unsigned* p_counts) {
            extern __shared__ unsigned char shared_mem[] ; 
            attrType* q_l_bound = (attrType*) shared_mem ; 
            attrType* q_r_bound = (attrType*) (q_l_bound + attr_dim) ;
            unsigned tid = threadIdx.x , total_num = blockDim.x , bid = blockIdx.x ;
            for(unsigned i = threadIdx.x ; i < attr_dim ; i += blockDim.x)
                q_l_bound[i] = l_bound[bid * attr_dim + i] , q_r_bound[i] = r_bound[bid * attr_dim + i] ;
            __syncthreads() ;
            // unsigned tid = blockDim.x * blockIdx.x + threadIdx.x , total_num = gridDim.x * blockDim.x ;
            
            unsigned cnt = 0 ; 
            // 每个线程去收集所有flag, 然后规约到一起
            for(unsigned i = tid ;i  < num ;i  += total_num) {
                bool flag = true ;
                for(unsigned j = 0 ; j < attr_dim ; j ++)
                    flag = (flag && (attrs[i * attr_dim + j] >= q_l_bound[j]) && (attrs[i * attr_dim + j] <= q_r_bound[j])) ;
                if(flag) 
                    cnt ++ ;
                // unsigned ballot_res = __ballot_sync(__activemask() , flag) ;
                // if(laneid == 0)
                    // bit_map[(i >> 5)] = ballot_res ; 
                // is_selected[i] = flag ;
            }
            __syncthreads() ;
            for(int offset = 16 ; offset > 0 ; offset >>= 1)  {
                cnt += __shfl_down_sync(0xffffffff , cnt , offset) ;
            }
            
            if(tid == 0) {
                float proportion = (1.0 * cnt) / (1.0 * num) ;
                // if(proportion <= threshold) {
                    // printf("bid : %d , proportion : %f , threshold : %f \n" , bid , proportion , threshold) ;
                // }
                seg_num_counts[bid] = (proportion <= threshold ? p_counts[bid] : 0) ;
            }
    }




    // 目前只实现逻辑与
    template<unsigned ATTR_DIM = 2, typename T = float>
    array<unsigned*, 5> intersection_area_mark_little_seg_V2(T* l_bound , T* r_bound , T* quantile_bound_l , T* quantile_bound_r , unsigned* quantile_dim_offset, unsigned* attr_grid_size , unsigned attr_dim , unsigned qnum , unsigned pnum , unsigned cardinality , T* attrs , T little_seg_threshold , unsigned OP_TYPE = 0) {
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
            intersection_area_batch_mark_little_seg_closed_interval<ATTR_DIM,T><<<(qnum + 31) / 32 , 32>>>(l_bound, r_bound , quantile_bound_l , quantile_bound_r , attr_grid_size , attr_dim , quantile_dim_offset ,
                 qnum , is_selected , pnum, is_little_seg , little_seg_threshold) ;
        // 逻辑或
        // else if(OP_TYPE == 1)
        //     intersection_area_batch_mark_little_seg_logicalor<ATTR_DIM><<<(qnum + 31) / 32 , 32>>>(l_bound, r_bound , attr_min , attr_width , attr_grid_size , attr_dim , qnum ,  is_selected ,pnum , is_little_seg , little_seg_threshold) ;
        
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
        // thrust::reduce_by_key(
        //     thrust::device , 
        //     row_key_begin, row_key_begin + pnum * qnum ,
        //     val_begin2 ,
        //     thrust::make_discard_iterator() ,
        //     little_seg_counts ,
        //     thrust::equal_to<unsigned>() ,
        //     thrust::plus<unsigned>() 
        // ) ;
        // 此处预过滤

        thrust::exclusive_scan(thrust::device , p_counts , p_counts + qnum , p_start_index , 0) ;

        prefiltering_tag_elements<T><<<qnum , 32>>>(attrs , l_bound , r_bound , attr_dim , cardinality , little_seg_threshold , little_seg_counts , p_counts) ;

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



    typedef struct BaseGridInfo {
        vector<int> l_quantiles , r_quantiles , pid_list ;
    } ;
    typedef struct GridInfo {
        int dim ;
        int num ;
        vector<BaseGridInfo> p_infos ;
        vector<int> global_pid ;
        vector<int> grid ;

        void showInfo() {
            for(int i = 0 ; i < dim ; i ++) {
                int grid_size = grid[i] ;
                BaseGridInfo& b = p_infos[i] ;
                cout << "dim" << i << ": " << endl ;
                cout << "l_quantile[" ;
                for(int j = 0 ; j < grid_size ;j  ++) {
                    cout << b.l_quantiles[j] ;
                    if(j < grid_size - 1)
                        cout << "," ;
                }
                cout << "]" << endl ;
                cout << "r_quantile[" ;
                for(int j = 0 ; j < grid_size ;j  ++) {
                    cout << b.r_quantiles[j] ;
                    if(j < grid_size - 1)
                        cout << "," ;
                }
                cout << "]" << endl ;
            }
        }
    } ;
    template<typename T = float>
    struct GridInfoGpu {
        T* l_quantiles , *r_quantiles ;
        unsigned* dim_offset , * grid_size ;
        unsigned* global_pid ;
    } ;


    void load_grid_infos(string filename , GridInfo& g) {
        std::ifstream in(filename, std::ios::binary);
        if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
        int num , dim ;
        in.read(reinterpret_cast<char*>(&num) , sizeof(int)) ;
        in.read(reinterpret_cast<char*>(&dim) , sizeof(int)) ;
        g.num = num , g.dim = dim ; 
        g.grid.resize(dim) ;
        in.read(reinterpret_cast<char*>(g.grid.data()) , sizeof(int) * dim) ;
        g.p_infos.resize(dim) ;
        for(int i = 0 ; i < dim ; i ++) {
            int grid_size = g.grid[i] ;
            g.p_infos[i].l_quantiles.resize(grid_size) ;
            g.p_infos[i].r_quantiles.resize(grid_size) ;
            g.p_infos[i].pid_list.resize(num) ;
            in.read(reinterpret_cast<char*>(g.p_infos[i].l_quantiles.data()) , sizeof(int) * grid_size) ;
            in.read(reinterpret_cast<char*>(g.p_infos[i].r_quantiles.data()) , sizeof(int) * grid_size) ;
            in.read(reinterpret_cast<char*>(g.p_infos[i].pid_list.data()) , sizeof(int) * num) ;
        }
        g.global_pid.resize(num) ;
        in.read(reinterpret_cast<char*>(g.global_pid.data()) , sizeof(int) * num) ;
        in.close() ;
        cout << "文件读入完成 :" << filename << endl ; 
    }

    template<typename T = float>
    void load2GPU(GridInfo& g , GridInfoGpu<T>& g_gpu) {
        vector<unsigned> dim_offset_host(g.dim  , 0) ;
        unsigned len = 0 ;
        for(unsigned i = 0 ; i < g.dim ; i ++) {
            dim_offset_host[i] = len ; 
            len += g.p_infos[i].l_quantiles.size() ;
        }
        cudaMalloc((void**) &g_gpu.l_quantiles , len * sizeof(T)) ;
        cudaMalloc((void**) &g_gpu.r_quantiles , len * sizeof(T)) ;
        cudaMalloc((void**) &g_gpu.dim_offset , g.dim * sizeof(unsigned)) ;
        cudaMalloc((void**) &g_gpu.grid_size , g.dim * sizeof(unsigned)) ;
        cudaMalloc((void**) &g_gpu.global_pid , g.num * sizeof(unsigned)) ;

        // 现在cpu上把所有bound转为float类型
        vector<float> cpu_bound_l(len) , cpu_bound_r(len) ;
        for(unsigned i = 0 ; i < g.dim ; i ++) {
            for(unsigned j = 0 ; j < g.p_infos[i].l_quantiles.size() ; j ++) {
                cpu_bound_l[dim_offset_host[i] + j] = (float) g.p_infos[i].l_quantiles[j] ;
                cpu_bound_r[dim_offset_host[i] + j] = (float) g.p_infos[i].r_quantiles[j] ;
            }
        }

        // 导入gpu
        cudaMemcpy(g_gpu.l_quantiles , cpu_bound_l.data() , len * sizeof(T) , cudaMemcpyHostToDevice) ;
        cudaMemcpy(g_gpu.r_quantiles , cpu_bound_r.data() , len * sizeof(T) , cudaMemcpyHostToDevice) ;
        cudaMemcpy(g_gpu.dim_offset , dim_offset_host.data() , g.dim * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
        cudaMemcpy(g_gpu.grid_size , g.grid.data() , g.dim * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
        cudaMemcpy(g_gpu.global_pid , g.global_pid.data() , g.num * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
        cout << "grid相关信息已载入至GPU" << endl ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    }
    
}