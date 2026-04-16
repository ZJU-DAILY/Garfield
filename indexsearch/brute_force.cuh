#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
// #include <thrust/nth_element.h>
#include <algorithm>
#include "tools.cuh"

using namespace std ;

// 此处实现基于GPU的 range-filter ANNS 其中 range-filter 的属性可以是任意维 predicate_dim



__global__ void condition_prefiltering(bool* is_filter , float* attributes_vec , unsigned attr_dim , float* l_bound , float* r_bound , unsigned points_num) {
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x ;
    unsigned total_num = gridDim.x * blockDim.x ;

    // 实际场景中, 属性的维度通常比较少, 因此每个线程专门处理一个向量即可
    for(unsigned i = tid ; i < points_num ; i += total_num) {   
        bool flag = true ;
        for(unsigned j = 0 ; j < attr_dim ; j ++)
            flag =  flag && (attributes_vec[i * attr_dim + j] <= r_bound[j] && attributes_vec[i * attr_dim + j] >= l_bound[j]) ;
        is_filter[i] = flag ;
    }
}


__global__ void dis_cal(unsigned* cands , unsigned cand_num , __half* data_ , __half* q , unsigned dim , float* ret_dis) {
    // 每个线程组负责一个点的距离计算, 每个线程块包括blockDim.y 个线程组

    // 先把q存入共享内存
    extern __shared__ __half q_vector[] ;
    unsigned tid = threadIdx.y * blockDim.x + threadIdx.x , b_num = blockDim.x * blockDim.y ;
    for(unsigned i = tid ; i < dim ; i += b_num)
        q_vector[i] = q[i] ;
    __syncthreads() ;
    // 计算距离, 然后合并到0号处
    for(unsigned i = blockIdx.x * blockDim.y + threadIdx.y ; i < cand_num ; i += blockDim.y * gridDim.x) {
        __half dis = __float2half(0.0f) ;
        for(unsigned lane_id = threadIdx.x ; lane_id < dim ; lane_id += blockDim.x) {
            dis = __hfma(__hsub(q_vector[lane_id], data_[cands[i] * dim + lane_id]), __hsub(q_vector[lane_id], data_[cands[i] * dim + lane_id]), dis);
            // dis = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
        }
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            // 将warp内所有子线程的计算结果归并到一起
            dis = __hadd(dis, __shfl_down_sync(0xffffffff, dis, lane_mask));
        }
        if(threadIdx.x == 0)
            ret_dis[i] = __half2float(dis) ;
        // __syncwarp() ;
        // printf("res_dis[i] : %f , q_vector[lane_id] : %f , data_[cand[i]*dim] : %f \n" 
        // , ret_dis[i] , __half2float(q_vector[0]) , __half2float(data_[cands[i] * dim])) ;
    }
}

// 每个查询向量与聚类中心做距离计算
__global__ void dis_cal_matrix(__half* data_ , __half* q , unsigned qnum ,unsigned cnum, unsigned dim , float* ret_dis) {
    // 每个线程组负责一个点的距离计算, 每个线程块包括blockDim.y 个线程组
    // 这里每个线程块负责处理L份距离计算

    // 先明确每个线程块负责的q, 以及c的范围(每多少个线程块服务一个q)
    unsigned served_block_num = (gridDim.x + qnum - 1) / qnum ;
    // 注: 线程块的数量一定为qnum的整数倍
    unsigned qid = blockIdx.x / served_block_num ; 
    // 每个线程块需要为qid做的距离计算数量
    unsigned average_discal_num = (cnum + served_block_num - 1) / served_block_num ;
    unsigned s_share = blockIdx.x % served_block_num ; 
    unsigned real_load = (s_share == served_block_num - 1 ? cnum - average_discal_num * (served_block_num - 1) : average_discal_num) ;
    // 先把q存入共享内存
    extern __shared__ __half q_vector[] ;
    unsigned tid = threadIdx.y * blockDim.x + threadIdx.x , b_num = blockDim.x * blockDim.y ;
    for(unsigned i = tid ; i < dim ; i += b_num)
        q_vector[i] = q[qid * dim + i] ;
    __syncthreads() ;
    unsigned load_offset = s_share * average_discal_num ; 
    // 计算距离, 然后合并到0号处
    for(unsigned i = threadIdx.y ; i < real_load ; i += blockDim.y) {
        unsigned cid = load_offset + i ;
        __half dis = __float2half(0.0f) ;
        for(unsigned lane_id = threadIdx.x ; lane_id < dim ; lane_id += blockDim.x) {
            dis = __hfma(__hsub(q_vector[lane_id], data_[cid * dim + lane_id]), __hsub(q_vector[lane_id], data_[cid * dim + lane_id]), dis);
            // dis = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
        }
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            // 将warp内所有子线程的计算结果归并到一起
            dis = __hadd(dis, __shfl_down_sync(0xffffffff, dis, lane_mask));
        }
        if(threadIdx.x == 0)
            ret_dis[qid * cnum + cid] = __half2float(dis) ;
        // __syncwarp() ;
        // printf("res_dis[i] : %f , q_vector[lane_id] : %f , data_[cand[i]*dim] : %f \n" 
        // , ret_dis[i] , __half2float(q_vector[0]) , __half2float(data_[cands[i] * dim])) ;
    }
}

struct __align__(8) half4bf {
    half2 x, y;
};
  
__device__ __forceinline__ half4bf BitCastBf(const float2& src) noexcept {
    half4bf dst;
    std::memcpy(&dst, &src, sizeof(half4bf));
    return dst;
}

__device__ __forceinline__ half4bf LoadBf(const half* address) {
    float2 x = __ldg(reinterpret_cast<const float2*>(address));
    return BitCastBf(x);
}

template<unsigned DIM = 128>
__global__ void dis_cal_test_prefilter(__half* data_ , __half* q , unsigned batch_size , unsigned dim , const unsigned* ids , float* ret_dis) {

    unsigned bid = blockIdx.x ; 
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x ;
    unsigned laneid =  threadIdx.x ;
    __shared__ half4bf tmp_val_sha[DIM / 4] ;
    for(unsigned i = tid; i < (dim / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = LoadBf(&q[bid * DIM + 4 * i]);
	}
    __syncthreads() ;
    #pragma unroll
    for(unsigned i = threadIdx.y; i < batch_size; i += blockDim.y){
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
            half4bf val2 = LoadBf(&data_[ids[bid * batch_size + i] * DIM + j * 4]);
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
            ret_dis[bid * batch_size + i] = __half2float(val_res.x) + __half2float(val_res.y) ;
            // printf("epoch %d/%d ,top_M_Cand_dis[TOPM + %d] : %f\n" , p_i , psize_block , i , top_M_Cand_dis[TOPM + i]) ;
        }
    }
    // __syncthreads();

}




vector<unsigned> brute_force_gpu(__half* q , __half* data_ , unsigned k  , unsigned dim , float* attributes_vec , unsigned attr_dim , float* l_bound , float* r_bound , unsigned points_num) {
    // q, data_ , attributes_vec 为设备数组, 其余为主存数组
    // 先预筛选点
    bool* is_filter ; 
    cudaMalloc((void**) &is_filter , points_num * sizeof(bool)) ;
    float* l_bound_dev , *r_bound_dev ;
    cudaMalloc((void**) &l_bound_dev , attr_dim * sizeof(float)) ;
    cudaMalloc((void**) &r_bound_dev , attr_dim * sizeof(float)) ;
    cudaMemcpy(l_bound_dev , l_bound , attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(r_bound_dev , r_bound , attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    condition_prefiltering<<<(points_num + 31) / 32 , 32>>>(is_filter , attributes_vec ,attr_dim ,l_bound_dev ,r_bound_dev ,points_num) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    // 规约bool数组条件, 形成索引数组
    // thrust::device_vector<bool> flags = {...};
    int N = points_num ; 

    thrust::device_vector<unsigned> indices(N);

    // 生成 0,1,2,...,N-1
    thrust::sequence(thrust::device , indices.begin(), indices.end());

    // 过滤，把 true 的 index 保留下来
    auto end_it = thrust::copy_if(
        thrust::device , 
        indices.begin(), indices.end(),      // input indices
        is_filter,                       // stencil (bool array)
        indices.begin(),                     // output
        thrust::identity<bool>()             // keep only true
    );

    indices.resize(end_it - indices.begin());
    // 此时得到了索引数组
    
    float* ret_dis ;
    cudaMalloc((void**) &ret_dis , indices.size() * sizeof(float)) ;
    dim3 grid_s((indices.size() - 32 * 6 + 1) / (32 * 6), 1, 1);
    dim3 block_s(32, 6, 1); // 6是block中的warp数量，可以调整
    dis_cal<<<grid_s , block_s , dim * sizeof(__half)>>>(indices.data().get() , indices.size() , data_ , q , dim , ret_dis) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    thrust::sort_by_key(thrust::device ,  ret_dis , ret_dis + indices.size() , indices.begin()) ;
    
    vector<unsigned> res(k) ;
    cudaMemcpy(res.data() , indices.data().get() , k * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;

    cudaFree(is_filter) ;
    cudaFree(l_bound_dev) ; 
    cudaFree(r_bound_dev) ;
    cudaFree(ret_dis) ;
    return res ;
}


vector<unsigned> brute_force_gpu(__half* q , __half* data_ , unsigned k  , unsigned dim,  unsigned points_num) {
    
    int N = points_num ; 

    thrust::device_vector<unsigned> indices(N);

    // 生成 0,1,2,...,N-1
    thrust::sequence(thrust::device , indices.begin(), indices.end());
    // cout << "indices.size() :" << indices.size() << endl ;
    // 此时得到了索引数组
    
    float* ret_dis ;
    cudaMalloc((void**) &ret_dis , indices.size() * sizeof(float)) ;
    dim3 grid_s((indices.size() + 5) / 6, 1, 1);
    dim3 block_s(32, 6, 1); // 6是block中的warp数量，可以调整
    dis_cal<<<grid_s , block_s , dim * sizeof(__half)>>>(indices.data().get() , indices.size() , data_ , q , dim , ret_dis) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    thrust::sort_by_key(thrust::device , ret_dis , ret_dis + indices.size() , indices.begin()) ;
    
    vector<unsigned> res(k) ;
    cudaMemcpy(res.data() , indices.data().get() , k * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;


    // float* dis_host = new float[k * 2] ;
    // cudaMemcpy(dis_host,  ret_dis , k * 2 * sizeof(float) , cudaMemcpyDeviceToHost) ;
    // for(unsigned i = 0 ; i < k * 2 ; i ++)
    //     cout << dis_host[i] << "," ;
    // cout << endl ;
    // cudaFree(is_filter) ;
    // cudaFree(l_bound_dev) ; 
    // cudaFree(r_bound_dev) ;
    cudaFree(ret_dis) ;
    return res ;
}

template<typename attrType = float>
__global__ void prefiltering_tag_elements(const attrType* __restrict__ attrs , const attrType* __restrict__ l_bound , const attrType* __restrict__ r_bound , 
    const unsigned attr_dim , const unsigned num , bool* is_selected) {
        extern __shared__ unsigned char shared_mem[] ; 
        attrType* q_l_bound = (attrType*) shared_mem ; 
        attrType* q_r_bound = (attrType*) (q_l_bound + attr_dim) ;
        for(unsigned i = threadIdx.x ; i < attr_dim ; i += blockDim.x)
            q_l_bound[i] = l_bound[i] , q_r_bound[i] = r_bound[i] ;
        __syncthreads() ;
        unsigned tid = blockDim.x * blockIdx.x + threadIdx.x , total_num = gridDim.x * blockDim.x ;
        for(unsigned i = tid ;i  < num ;i  += total_num) {
            bool flag = true ;
            for(unsigned j = 0 ; j < attr_dim ; j ++)
                flag = (flag && (attrs[i * attr_dim + j] >= q_l_bound[j]) && (attrs[i * attr_dim + j] <= q_r_bound[j])) ;
            // unsigned ballot_res = __ballot_sync(__activemask() , flag) ;
            // if(laneid == 0)
                // bit_map[(i >> 5)] = ballot_res ; 
            is_selected[i] = flag ;
        }
}

template<typename attrType = float>
__global__ void prefiltering_tag_elements_logicalor(const attrType* __restrict__ attrs , const attrType* __restrict__ l_bound , const attrType* __restrict__ r_bound , 
    const unsigned attr_dim , const unsigned num , bool* is_selected) {
        extern __shared__ unsigned char shared_mem[] ; 
        attrType* q_l_bound = (attrType*) shared_mem ; 
        attrType* q_r_bound = (attrType*) (q_l_bound + attr_dim) ;
        for(unsigned i = threadIdx.x ; i < attr_dim ; i += blockDim.x)
            q_l_bound[i] = l_bound[i] , q_r_bound[i] = r_bound[i] ;
        __syncthreads() ;
        unsigned tid = blockDim.x * blockIdx.x + threadIdx.x , total_num = gridDim.x * blockDim.x ;
        for(unsigned i = tid ;i  < num ;i  += total_num) {
            bool flag = false ;
            for(unsigned j = 0 ; j < attr_dim ; j ++)
                flag = (flag || ((attrs[i * attr_dim + j] >= q_l_bound[j]) && (attrs[i * attr_dim + j] <= q_r_bound[j]))) ;
            // unsigned ballot_res = __ballot_sync(__activemask() , flag) ;
            // if(laneid == 0)
                // bit_map[(i >> 5)] = ballot_res ; 
            is_selected[i] = flag ;
        }
}

__global__ void dis_cal_reduce(const __half* __restrict__ q , const __half* __restrict__ data_ , const unsigned* __restrict__ task_list , const unsigned task_num, const unsigned DIM , 
    float* result_dis) {
    extern __shared__ __align__(8) half4 q_vec[] ;
    unsigned tid = blockDim.x * threadIdx.y + threadIdx.x ;
    unsigned laneid = threadIdx.x , warpId = blockDim.y * blockIdx.x + threadIdx.y , warpNum = gridDim.x * blockDim.y ;
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		q_vec[i] = Load(&q[4 * i]);
	}
    __syncthreads() ;

    for(unsigned i = warpId ; i < task_num ; i += warpNum) {
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0 ;
        unsigned task_id = task_list[i] ;
        size_t task_vec_offset = static_cast<size_t>(task_id) * static_cast<size_t>(DIM) ;
        #pragma unroll
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            // warp中的每个子线程负责8个字节 即4个half类型分量的计算

            half4 val2 = Load(&data_[task_vec_offset + static_cast<size_t>(j * 4)]);
            val_res = __hfma2(__hsub2(q_vec[j].x, val2.x), __hsub2(q_vec[j].x, val2.x), val_res);
            val_res = __hfma2(__hsub2(q_vec[j].y, val2.y), __hsub2(q_vec[j].y, val2.y), val_res);
        }
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            // 将warp内所有子线程的计算结果归并到一起
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
        }
        if(laneid == 0){
            result_dis[i] = __half2float(val_res.x) + __half2float(val_res.y) ;
            // printf("epoch %d/%d ,top_M_Cand_dis[TOPM + %d] : %f\n" , p_i , psize_block , i , top_M_Cand_dis[TOPM + i]) ;
        }
        __syncwarp() ;
    }

}

template<typename attrType = float>
vector<vector<unsigned>> prefiltering_and_bruteforce(const __half* q , const __half* data_ , const unsigned qnum ,  const unsigned k , const unsigned dim , const unsigned points_num , const attrType* attrs ,
    const attrType* l_bound , const attrType* r_bound , const unsigned attr_dim ,const unsigned OP_TYPE = 0 , const unsigned batch_num = 4) {

    vector<vector<unsigned>> result(qnum , vector<unsigned>()) ;
    vector<vector<float>> result_dis_cpu(qnum , vector<float>()) ;
    // 先按照属性条件过滤节点
    bool* is_selected ;
    unsigned average_num = (points_num + batch_num - 1) / batch_num ; 
    cudaMalloc((void**) &is_selected , average_num * sizeof(bool)) ;
    __half* data_buffer ; 
    size_t data_buffer_size = static_cast<size_t>(average_num) * static_cast<size_t>(dim) * sizeof(__half) ;
    cudaMalloc((void**) &data_buffer , data_buffer_size) ;
    unsigned * task_id_buffer  ;
    // cudaMalloc((void**) &result_buffer , k * sizeof(unsigned)) ;
    cudaMalloc((void**) &task_id_buffer , average_num * sizeof(unsigned)) ;
    float* result_dis ; 
    cudaMalloc((void**) &result_dis , average_num * sizeof(float)) ;
    // cudaMalloc((void**) &prefix , average_num * sizeof(unsigned)) ;
    unsigned p_kernal_byte_size = 2 * attr_dim * sizeof(attrType) ;
    unsigned vec_byte_size = dim * sizeof(__half) ;
    // dim3 grid_s()


    float sort_cost = 0.0 , dis_cal_cost = 0.0 , mem_trans_cost = 0.0 , tag_kernal_cost = 0.0 , merge_cost = 0.0 ;
    // for(unsigned i = 0 ; i < qnum ; i ++) {
    // 解决带宽问题, 先分批, 再循环处理查询
    for(unsigned batch = 0 ; batch < batch_num ; batch ++) {
        // vector<pair<float,unsigned>> result_i ; 
        // vector<unsigned> result_i_id ; 
        // vector<float> result_i_dis ; 
        // result_i_id.reserve(k * 2) ;
        // result_i_dis.reserve(k * 2) ;
        unsigned real_load = (batch < batch_num - 1 ? average_num : points_num - average_num * (batch_num - 1)) ;
        unsigned id_offset = batch * average_num ; 
        size_t workload_byteSize = static_cast<size_t>(real_load) * static_cast<size_t>(dim) * sizeof(__half) ;
        size_t data_offset = static_cast<size_t>(id_offset) * static_cast<size_t>(dim) ;
        auto s = chrono::high_resolution_clock::now() ;
        cudaMemcpy(data_buffer , data_ + data_offset , workload_byteSize , cudaMemcpyHostToDevice) ;
        auto e = chrono::high_resolution_clock::now() ;
        mem_trans_cost += chrono::duration<double>(e - s).count() ;
        // 先分批
        // for(unsigned batch = 0 ; batch < batch_num ; batch ++) {
        for(unsigned i = 0 ; i < qnum ; i ++) {

            cudaMemset(is_selected , 0 , average_num * sizeof(bool)) ;
            s = chrono::high_resolution_clock::now() ;
            if(OP_TYPE == 0)
                prefiltering_tag_elements<attrType><<<(real_load + 255) / 256 , 256 , p_kernal_byte_size>>>(attrs + attr_dim * id_offset , l_bound + i * attr_dim , r_bound + i * attr_dim , attr_dim , real_load , is_selected) ;
            else if(OP_TYPE == 1)
                prefiltering_tag_elements_logicalor<attrType><<<(real_load + 255) / 256 , 256 , p_kernal_byte_size>>>(attrs + attr_dim * id_offset , l_bound + i * attr_dim , r_bound + i * attr_dim , attr_dim , real_load , is_selected) ;
            cudaDeviceSynchronize() ;
            CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            e = chrono::high_resolution_clock::now() ;
            tag_kernal_cost += chrono::duration<double>(e - s).count() ;
            // 规约
            // thrust::exclusive_scan(thrust::device , is_selected , is_selected + real_load , prefix , 0) ;
            auto cit = thrust::make_counting_iterator<unsigned>(0) ;
            auto out_end = thrust::copy_if(
                thrust::device , 
                // prefix , prefix + real_load ,
                cit , cit + real_load , 
                is_selected , 
                task_id_buffer , 
                thrust::identity<bool>() 
            ) ;
            unsigned task_num = out_end - task_id_buffer ;
            
            
            // cout << "batch" << batch << " q" << i << " : task_num - " << task_num << endl ;
            if(task_num == 0)
                continue ;
            // 计算距离
            dim3 grid_s((task_num + 63) / 64 , 1 , 1) , block_s(32 , 6 , 1) ;
            s = chrono::high_resolution_clock::now() ;
            dis_cal_reduce<<<grid_s , block_s , vec_byte_size>>>(q + i * dim , data_buffer , task_id_buffer , task_num, dim ,result_dis) ;
            cudaDeviceSynchronize() ;
            CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            e = chrono::high_resolution_clock::now() ;
            dis_cal_cost += chrono::duration<double>(e - s).count() ;

            // 选择出前k个元素
            // auto zipped_begin = thrust::make_zip_iterator(
            //     thrust::make_tuple(result_dis , task_id_buffer) 
            // ) ;
            unsigned realk = min(k , task_num) ;
            // thrust::nth_element( 
            //     zipped_begin , 
            //     zipped_begin + realk , 
            //     zipped_begin + task_num ,
            //     [] __device__ (auto a , auto b) {
            //         return thrust::get<0>(a) < thrust::get<0>(b) ;
            //     }
            // ) ;
            s = chrono::high_resolution_clock::now() ;
            thrust::sort_by_key(thrust::device , result_dis , result_dis + task_num , task_id_buffer) ;
            e = chrono::high_resolution_clock::now() ;
            sort_cost += chrono::duration<double>(e - s).count() ;
            vector<unsigned> &result_i_id = result[i];
            vector<float> &result_i_dis = result_dis_cpu[i];
            vector<unsigned> tmp_r_id_cur(realk) , tmp_r_id_pre = result_i_id ; 
            vector<float> tmp_r_dis_cur(realk) , tmp_r_dis_pre = result_i_dis ; 
            cudaMemcpy(tmp_r_id_cur.data() , task_id_buffer , realk * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
            cudaMemcpy(tmp_r_dis_cur.data() , result_dis , realk * sizeof(float) , cudaMemcpyDeviceToHost) ;
            result_i_dis.clear() ;
            result_i_id.clear() ;
            unsigned i_index = 0  , j_index = 0 ; 
            s = chrono::high_resolution_clock::now() ;
            while(i_index < tmp_r_id_pre.size() && j_index < tmp_r_id_cur.size() && result_i_id.size() < k) {
                if(tmp_r_dis_pre[i_index] <= tmp_r_dis_cur[j_index]) {
                    result_i_id.push_back(tmp_r_id_pre[i_index]) ;
                    result_i_dis.push_back(tmp_r_dis_pre[i_index]) ;
                    i_index ++ ;
                } else {
                    result_i_id.push_back(tmp_r_id_cur[j_index] + id_offset) ;
                    result_i_dis.push_back(tmp_r_dis_cur[j_index]) ;
                    j_index ++ ;
                }
            }
            if(result_i_id.size() < k) {
                while(i_index < tmp_r_id_pre.size() && result_i_id.size() < k) {
                    result_i_id.push_back(tmp_r_id_pre[i_index]) ;
                    result_i_dis.push_back(tmp_r_dis_pre[i_index]) ;
                    i_index ++ ;
                }
                while(j_index < tmp_r_id_cur.size() && result_i_id.size() < k) {
                    result_i_id.push_back(tmp_r_id_cur[j_index] + id_offset) ;
                    result_i_dis.push_back(tmp_r_dis_cur[j_index]) ;
                    j_index ++ ;
                }
            }  
            e = chrono::high_resolution_clock::now() ;  
            merge_cost += chrono::duration<double>(e - s).count() ;
            // result[i] = result_i_id ; 
            // result_dis_cpu[i] = result_i_dis ;
            
        }
        // result[i] = result_i_id ; 
        // cout << "批次" << batch << ", 处理完成" << endl ;
    }
    cout << "距离计算用时: " << dis_cal_cost << endl ;
    cout << "数据传输用时: " << mem_trans_cost << endl ;
    cout << "排序用时: " << sort_cost << endl ;
    cout << "标记元素用时: " << tag_kernal_cost << endl ;
    cout << "合并用时: " << merge_cost << endl ;


    cudaFree(is_selected) ;
    cudaFree(data_buffer) ;
    // cudaFree(result_buffer) ;
    cudaFree(task_id_buffer) ;
    cudaFree(result_dis) ;
    // cudaFree(prefix) ;

    return result ;
}


template<typename attrType = float>
vector<vector<unsigned>> prefiltering_and_bruteforce(const __half* q , const __half* data_ , const unsigned qnum ,  const unsigned k , const unsigned dim , const unsigned points_num , const attrType* attrs ,
    const attrType* l_bound , const attrType* r_bound , const unsigned attr_dim ,vector<vector<float>>& result_dis_return, const unsigned OP_TYPE = 0, const unsigned batch_num = 4) {

    vector<vector<unsigned>> result(qnum , vector<unsigned>()) ;
    vector<vector<float>> result_dis_cpu(qnum , vector<float>()) ;
    // 先按照属性条件过滤节点
    bool* is_selected ;
    unsigned average_num = (points_num + batch_num - 1) / batch_num ; 
    cudaMalloc((void**) &is_selected , average_num * sizeof(bool)) ;
    __half* data_buffer ; 
    size_t data_buffer_size = static_cast<size_t>(average_num) * static_cast<size_t>(dim) * sizeof(__half) ;
    cudaMalloc((void**) &data_buffer , data_buffer_size) ;
    unsigned * task_id_buffer  ;
    // cudaMalloc((void**) &result_buffer , k * sizeof(unsigned)) ;
    cudaMalloc((void**) &task_id_buffer , average_num * sizeof(unsigned)) ;
    float* result_dis ; 
    cudaMalloc((void**) &result_dis , average_num * sizeof(float)) ;
    // cudaMalloc((void**) &prefix , average_num * sizeof(unsigned)) ;
    unsigned p_kernal_byte_size = 2 * attr_dim * sizeof(attrType) ;
    unsigned vec_byte_size = dim * sizeof(__half) ;
    // dim3 grid_s()


    float sort_cost = 0.0 , dis_cal_cost = 0.0 , mem_trans_cost = 0.0 , tag_kernal_cost = 0.0 , merge_cost = 0.0 ;
    // for(unsigned i = 0 ; i < qnum ; i ++) {
    // 解决带宽问题, 先分批, 再循环处理查询
    for(unsigned batch = 0 ; batch < batch_num ; batch ++) {
        // vector<pair<float,unsigned>> result_i ; 
        // vector<unsigned> result_i_id ; 
        // vector<float> result_i_dis ; 
        // result_i_id.reserve(k * 2) ;
        // result_i_dis.reserve(k * 2) ;
        unsigned real_load = (batch < batch_num - 1 ? average_num : points_num - average_num * (batch_num - 1)) ;
        unsigned id_offset = batch * average_num ; 
        size_t workload_byteSize = static_cast<size_t>(real_load) * static_cast<size_t>(dim) * sizeof(__half) ;
        size_t data_offset = static_cast<size_t>(id_offset) * static_cast<size_t>(dim) ;
        auto s = chrono::high_resolution_clock::now() ;
        cudaMemcpy(data_buffer , data_ + data_offset , workload_byteSize , cudaMemcpyHostToDevice) ;
        auto e = chrono::high_resolution_clock::now() ;
        mem_trans_cost += chrono::duration<double>(e - s).count() ;
        // 先分批
        // for(unsigned batch = 0 ; batch < batch_num ; batch ++) {
        for(unsigned i = 0 ; i < qnum ; i ++) {

            cudaMemset(is_selected , 0 , average_num * sizeof(bool)) ;
            s = chrono::high_resolution_clock::now() ;
            if(OP_TYPE == 0)
                prefiltering_tag_elements<attrType><<<(real_load + 255) / 256 , 256 , p_kernal_byte_size>>>(attrs + attr_dim * id_offset , l_bound + i * attr_dim , r_bound + i * attr_dim , attr_dim , real_load , is_selected) ;
            else if(OP_TYPE == 1)
                prefiltering_tag_elements_logicalor<attrType><<<(real_load + 255) / 256 , 256 , p_kernal_byte_size>>>(attrs + attr_dim * id_offset , l_bound + i * attr_dim , r_bound + i * attr_dim , attr_dim , real_load , is_selected) ;
            cudaDeviceSynchronize() ;
            CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            e = chrono::high_resolution_clock::now() ;
            tag_kernal_cost += chrono::duration<double>(e - s).count() ;
            // 规约
            // thrust::exclusive_scan(thrust::device , is_selected , is_selected + real_load , prefix , 0) ;
            auto cit = thrust::make_counting_iterator<unsigned>(0) ;
            auto out_end = thrust::copy_if(
                thrust::device , 
                // prefix , prefix + real_load ,
                cit , cit + real_load , 
                is_selected , 
                task_id_buffer , 
                thrust::identity<bool>() 
            ) ;
            unsigned task_num = out_end - task_id_buffer ;
            
            
            // cout << "batch" << batch << " q" << i << " : task_num - " << task_num << endl ;
            if(task_num == 0)
                continue ;
            // 计算距离
            dim3 grid_s((task_num + 63) / 64 , 1 , 1) , block_s(32 , 6 , 1) ;
            s = chrono::high_resolution_clock::now() ;
            dis_cal_reduce<<<grid_s , block_s , vec_byte_size>>>(q + i * dim , data_buffer , task_id_buffer , task_num, dim ,result_dis) ;
            cudaDeviceSynchronize() ;
            CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
            e = chrono::high_resolution_clock::now() ;
            dis_cal_cost += chrono::duration<double>(e - s).count() ;

            // 选择出前k个元素
            // auto zipped_begin = thrust::make_zip_iterator(
            //     thrust::make_tuple(result_dis , task_id_buffer) 
            // ) ;
            unsigned realk = min(k , task_num) ;
            // thrust::nth_element( 
            //     zipped_begin , 
            //     zipped_begin + realk , 
            //     zipped_begin + task_num ,
            //     [] __device__ (auto a , auto b) {
            //         return thrust::get<0>(a) < thrust::get<0>(b) ;
            //     }
            // ) ;
            s = chrono::high_resolution_clock::now() ;
            thrust::sort_by_key(thrust::device , result_dis , result_dis + task_num , task_id_buffer) ;
            e = chrono::high_resolution_clock::now() ;
            sort_cost += chrono::duration<double>(e - s).count() ;
            vector<unsigned> &result_i_id = result[i];
            vector<float> &result_i_dis = result_dis_cpu[i];
            vector<unsigned> tmp_r_id_cur(realk) , tmp_r_id_pre = result_i_id ; 
            vector<float> tmp_r_dis_cur(realk) , tmp_r_dis_pre = result_i_dis ; 
            cudaMemcpy(tmp_r_id_cur.data() , task_id_buffer , realk * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
            cudaMemcpy(tmp_r_dis_cur.data() , result_dis , realk * sizeof(float) , cudaMemcpyDeviceToHost) ;
            result_i_dis.clear() ;
            result_i_id.clear() ;
            unsigned i_index = 0  , j_index = 0 ; 
            s = chrono::high_resolution_clock::now() ;
            while(i_index < tmp_r_id_pre.size() && j_index < tmp_r_id_cur.size() && result_i_id.size() < k) {
                if(tmp_r_dis_pre[i_index] <= tmp_r_dis_cur[j_index]) {
                    result_i_id.push_back(tmp_r_id_pre[i_index]) ;
                    result_i_dis.push_back(tmp_r_dis_pre[i_index]) ;
                    i_index ++ ;
                } else {
                    result_i_id.push_back(tmp_r_id_cur[j_index] + id_offset) ;
                    result_i_dis.push_back(tmp_r_dis_cur[j_index]) ;
                    j_index ++ ;
                }
            }
            if(result_i_id.size() < k) {
                while(i_index < tmp_r_id_pre.size() && result_i_id.size() < k) {
                    result_i_id.push_back(tmp_r_id_pre[i_index]) ;
                    result_i_dis.push_back(tmp_r_dis_pre[i_index]) ;
                    i_index ++ ;
                }
                while(j_index < tmp_r_id_cur.size() && result_i_id.size() < k) {
                    result_i_id.push_back(tmp_r_id_cur[j_index] + id_offset) ;
                    result_i_dis.push_back(tmp_r_dis_cur[j_index]) ;
                    j_index ++ ;
                }
            }  
            e = chrono::high_resolution_clock::now() ;  
            merge_cost += chrono::duration<double>(e - s).count() ;
            // result[i] = result_i_id ; 
            // result_dis_cpu[i] = result_i_dis ;
            
        }
        // result[i] = result_i_id ; 
        // cout << "批次" << batch << ", 处理完成" << endl ;
    }
    cout << "距离计算用时: " << dis_cal_cost << endl ;
    cout << "数据传输用时: " << mem_trans_cost << endl ;
    cout << "排序用时: " << sort_cost << endl ;
    cout << "标记元素用时: " << tag_kernal_cost << endl ;
    cout << "合并用时: " << merge_cost << endl ;


    cudaFree(is_selected) ;
    cudaFree(data_buffer) ;
    // cudaFree(result_buffer) ;
    cudaFree(task_id_buffer) ;
    cudaFree(result_dis) ;
    // cudaFree(prefix) ;
    result_dis_return = result_dis_cpu ;
    return result ;
}

// 默认所有操作数组都在显存中的预过滤
template<typename attrType = float>
vector<vector<unsigned>> prefiltering_and_bruteforce_resident(const __half* q , const __half* data_ , const unsigned qnum ,  const unsigned k , const unsigned dim , const unsigned points_num , const attrType* attrs ,
    const attrType* l_bound , const attrType* r_bound , const unsigned attr_dim , const unsigned OP_TYPE = 0) {
    
    vector<vector<unsigned>> result(qnum , vector<unsigned>(k)) ;
    vector<vector<float>> result_dis_cpu(qnum , vector<float>(k)) ;
    // 先按照属性条件过滤节点
    bool* is_selected ;
    // unsigned average_num = (points_num + batch_num - 1) / batch_num ; 
    cudaMalloc((void**) &is_selected , points_num * sizeof(bool)) ;

    unsigned * task_id_buffer  ;
    // cudaMalloc((void**) &result_buffer , k * sizeof(unsigned)) ;
    cudaMalloc((void**) &task_id_buffer , points_num * sizeof(unsigned)) ;
    float* result_dis ; 
    cudaMalloc((void**) &result_dis , points_num * sizeof(float)) ;
    // cudaMalloc((void**) &prefix , average_num * sizeof(unsigned)) ;
    unsigned p_kernal_byte_size = 2 * attr_dim * sizeof(attrType) ;
    unsigned vec_byte_size = dim * sizeof(__half) ;
    // dim3 grid_s()


    float sort_cost = 0.0 , dis_cal_cost = 0.0 , mem_trans_cost = 0.0 , tag_kernal_cost = 0.0 , merge_cost = 0.0 ;
    // for(unsigned i = 0 ; i < qnum ; i ++) {
    // 解决带宽问题, 先分批, 再循环处理查询
    // for(unsigned batch = 0 ; batch < batch_num ; batch ++) {
        // vector<pair<float,unsigned>> result_i ; 
        // vector<unsigned> result_i_id ; 
        // vector<float> result_i_dis ; 
        // result_i_id.reserve(k * 2) ;
        // result_i_dis.reserve(k * 2) ;
    // unsigned real_load = (batch < batch_num - 1 ? average_num : points_num - average_num * (batch_num - 1)) ;
    // unsigned id_offset = batch * average_num ; 
    // size_t workload_byteSize = static_cast<size_t>(real_load) * static_cast<size_t>(dim) * sizeof(__half) ;
    // size_t data_offset = static_cast<size_t>(id_offset) * static_cast<size_t>(dim) ;
    // auto s = chrono::high_resolution_clock::now() ;
    // cudaMemcpy(data_buffer , data_ + data_offset , workload_byteSize , cudaMemcpyHostToDevice) ;
    // auto e = chrono::high_resolution_clock::now() ;
    // mem_trans_cost += chrono::duration<double>(e - s).count() ;
    // 先分批
    // for(unsigned batch = 0 ; batch < batch_num ; batch ++) {
    for(unsigned i = 0 ; i < qnum ; i ++) {

        cudaMemset(is_selected , 0 , points_num * sizeof(bool)) ;
        auto s = chrono::high_resolution_clock::now() ;
        if(OP_TYPE == 0)
            prefiltering_tag_elements<attrType><<<(points_num + 255) / 256 , 256 , p_kernal_byte_size>>>(attrs , l_bound + i * attr_dim , r_bound + i * attr_dim , attr_dim , points_num , is_selected) ;
        else if(OP_TYPE == 1)
            prefiltering_tag_elements_logicalor<attrType><<<(points_num + 255) / 256 , 256 , p_kernal_byte_size>>>(attrs , l_bound + i * attr_dim , r_bound + i * attr_dim , attr_dim , points_num , is_selected) ;
        cudaDeviceSynchronize() ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;

        // cout << i << endl ;
        // int t ;
        // cin >> t ;
        auto e = chrono::high_resolution_clock::now() ;
        tag_kernal_cost += chrono::duration<double>(e - s).count() ;
        // 规约
        // thrust::exclusive_scan(thrust::device , is_selected , is_selected + real_load , prefix , 0) ;
        auto cit = thrust::make_counting_iterator<unsigned>(0) ;
        auto out_end = thrust::copy_if(
            thrust::device , 
            // prefix , prefix + real_load ,
            cit , cit + points_num , 
            is_selected , 
            task_id_buffer , 
            thrust::identity<bool>() 
        ) ;
        unsigned task_num = out_end - task_id_buffer ;
        
        
        // cout << "batch" << batch << " q" << i << " : task_num - " << task_num << endl ;
        if(task_num == 0)
            continue ;
        // 计算距离
        dim3 grid_s((task_num + 11) / 12 , 1 , 1) , block_s(32 , 6 , 1) ;
        s = chrono::high_resolution_clock::now() ;
        dis_cal_reduce<<<grid_s , block_s , vec_byte_size>>>(q + i * dim , data_ , task_id_buffer , task_num, dim ,result_dis) ;
        cudaDeviceSynchronize() ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        e = chrono::high_resolution_clock::now() ;
        dis_cal_cost += chrono::duration<double>(e - s).count() ;

        // 选择出前k个元素
        // auto zipped_begin = thrust::make_zip_iterator(
        //     thrust::make_tuple(result_dis , task_id_buffer) 
        // ) ;
        unsigned realk = min(k , task_num) ;
        // thrust::nth_element( 
        //     zipped_begin , 
        //     zipped_begin + realk , 
        //     zipped_begin + task_num ,
        //     [] __device__ (auto a , auto b) {
        //         return thrust::get<0>(a) < thrust::get<0>(b) ;
        //     }
        // ) ;
        s = chrono::high_resolution_clock::now() ;
        thrust::sort_by_key(thrust::device , result_dis , result_dis + task_num , task_id_buffer) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        e = chrono::high_resolution_clock::now() ;
        sort_cost += chrono::duration<double>(e - s).count() ;
        
        
        cudaMemcpy(result[i].data() , task_id_buffer , realk * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
        cudaMemcpy(result_dis_cpu[i].data() , result_dis , realk * sizeof(float) , cudaMemcpyDeviceToHost) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // result[i] = result_i_id ; 
        // result_dis_cpu[i] = result_i_dis ;
        
    }
        // result[i] = result_i_id ; 
        // cout << "批次" << batch << ", 处理完成" << endl ;/
    // }
    cout << "距离计算用时: " << dis_cal_cost << endl ;
    // cout << "数据传输用时: " << mem_trans_cost << endl ;
    cout << "排序用时: " << sort_cost << endl ;
    cout << "标记元素用时: " << tag_kernal_cost << endl ;
    cout << "合并用时: " << merge_cost << endl ;


    cudaFree(is_selected) ;
    // cudaFree(data_buffer) ;
    // cudaFree(result_buffer) ;
    cudaFree(task_id_buffer) ;
    cudaFree(result_dis) ;
    // cudaFree(prefix) ;
    // result_dis_return = result_dis_cpu ;
    return result ;

}
