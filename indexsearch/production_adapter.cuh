#pragma once
#include "gpu_graph.cu"

using namespace std ;


namespace smart::production_adapter {
    __global__ void integer_production_kernal(float* data_ , const unsigned num , const unsigned dim , int8_t* data_p, float* scales , int* zero_points) {
        float scale_ground = 255.0f ;
        int qmin = -128 , qmax = 127 ;
        unsigned tid = threadIdx.x ;
        unsigned bid = blockIdx.x ;
        // 先求出在当前维度上的最小值 , 每个线程块处理一个维度
        float min_value = 100000.0f , max_value = -100000.0f ;

        if(tid < 32) {
            for(unsigned i = tid ; i < num ; i += blockDim.x) {
                min_value = fminf(min_value , data_[i * dim + bid]) ;
                max_value = fmaxf(max_value , data_[i * dim + bid]) ;
            }
            // __syncthreads() ;
            // 把一个warp内的最大值最小值规约
            for(int offset = 16 ; offset > 0 ; offset /= 2) {
                min_value = fminf(min_value , __shfl_down_sync(0xffffffff , min_value , offset)) ;
                max_value = fmaxf(max_value , __shfl_down_sync(0xffffffff , max_value , offset)) ;
            }
            // __syncthreads() ;
            min_value = __shfl_sync(0xffffffff, min_value , 0) ;
            max_value = __shfl_sync(0xffffffff, max_value , 0) ;
        }
        // 此时所有线程都拿到了最大值和最小值
        float scale = (max_value - min_value) / scale_ground ;
        int zero_point = __float2int_rn(qmin - min_value / scale) ;
        zero_point = max(qmin , min(qmax , zero_point)) ;

        for(unsigned i = tid ; i < num ; i  += blockDim.x) {
            int q = __float2int_rn(data_[i * dim + bid] / scale) + zero_point ;
            q = max(qmin , min(qmax , q)) ;
            data_p[i * dim + bid] = static_cast<int8_t>(q) ; 
        }

        if(tid == 0) {
            scales[bid] = scale ;
            zero_points[bid] = zero_point ;
        }

    }

    // 反量化: scale * (q - zero_point)
    void integer_production(float* data_ , const unsigned num , const unsigned dim, int8_t* data_p) {
        // 假设data_是 device端数组, data_p 也是device端数组
        float* scales ; 
        int* zero_points ;
        cudaMalloc((void**) &scales , dim * sizeof(float)) ;
        cudaMalloc((void**) &zero_points , dim * sizeof(int)) ;
        integer_production_kernal<<<dim, 32>>>(data_ , num , dim , data_p ,scales ,zero_points) ;
        cudaDeviceSynchronize() ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        int8_t* data_p_host = new int8_t[num * dim] ;
        float* data_host = new float[num * dim] ;
        float* scales_host = new float[dim] ;
        int* zero_points_host = new int[dim] ;
        cudaMemcpy(data_host , data_ , num * dim * sizeof(float) , cudaMemcpyDeviceToHost) ;
        cudaMemcpy(data_p_host , data_p , num * dim * sizeof(int8_t) , cudaMemcpyDeviceToHost) ;
        cudaMemcpy(scales_host , scales , dim * sizeof(float) , cudaMemcpyDeviceToHost) ;
        cudaMemcpy(zero_points_host , zero_points , dim * sizeof(int) , cudaMemcpyDeviceToHost) ;
        //打印前5行
        for(int i = 0 ; i < 5 ; i ++) {
            cout << "[" ;
            for(int j = 0 ; j < dim ; j ++) {
                cout << data_host[i * dim + j] ;
                if(j < dim - 1)
                    cout << ", " ;
            }
            cout << "]" << endl ;
            cout << "*[" ;
            for(int j = 0 ; j < dim ; j ++) {
                cout << (int)data_p_host[i * dim + j] ;
                if(j < dim - 1)
                    cout << ", " ;
            }
            cout << "]" << endl ;
        }

        cout << "***********************************************" << endl ;
        // 试试反量化前5行
        for(int i = 0 ; i < 5 ; i ++) {
            cout << "[" ;
            for(int j = 0 ; j < dim ; j ++) {
                cout << data_host[i * dim + j] ;
                if(j < dim - 1)
                    cout << ", " ;
            }
            cout << "]" << endl ;
            cout << "*[" ;
            for(int j = 0 ; j < dim ; j ++) {
                // cout << (int)data_p_host[i * dim + j] ;
                float re_production = ((int) data_p_host[i * dim + j] - zero_points_host[j]) * scales_host[j] ;
                cout << re_production ;
                if(j < dim - 1)
                    cout << ", " ;
            }
            cout << "]" << endl ;
        }
    }




    void load_dataset_return_int8_arr(string filepath ,float* &data_return ,  unsigned &num , unsigned& dim , float alpha, int8_t* &data_int8 , float* &scales_return , int* &zero_points_return) {
        // unsigned num , dim  ;
        float* data_  ;
        load_data(filepath , data_ , num , dim) ;
        cout << "num : " << num << ", dim : " << dim << endl ;
        #pragma omp parallel for num_threads(10)
        for(int i = 0 ; i < num ; i ++) {
            for(int j = 0 ; j < dim ; j ++) {
                size_t index = static_cast<size_t>(i) * static_cast<size_t>(dim) + static_cast<size_t>(j) ;
                data_[index] *= alpha ;
            }
        }

        unsigned average_num = num / 16 ;
        cout << "average_num : " << average_num << endl ;
        
        size_t arr_len = static_cast<size_t>(num) * static_cast<size_t>(dim) ;
        int8_t* data_q = new int8_t[arr_len] ;
        int8_t* data_int8_dev ; 
        size_t total_size = static_cast<size_t>(num) * static_cast<size_t>(dim) * sizeof(int8_t) ;
        cudaMalloc((void**) &data_int8_dev , total_size) ;

        
        float* scales ; 
        int* zero_points ;
        cudaMalloc((void**) &scales , dim * sizeof(float)) ;
        cudaMalloc((void**) &zero_points , dim * sizeof(int)) ;
        cout << total_size << endl ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;

        vector<float> scales_cpu(dim) ;
        vector<int> zero_points_cpu(dim) ;

        float scale_ground = 255.0f ;
        int qmin = -128 , qmax = 127 ;
        
        #pragma omp parallel for num_threads(10)
        for(int i = 0 ; i < dim ; i ++) {
            float min_value = 100000.0f , max_value = -100000.0f ;
            for(unsigned j = 0 ; j < num ; j ++) {
                size_t index = static_cast<size_t>(j) * static_cast<size_t>(dim) + static_cast<size_t>(i) ;
                min_value = min(min_value , data_[index]) ;
                max_value = max(max_value , data_[index]) ;
            }
            float scale = (max_value - min_value) / scale_ground ;
            int zero_point = round(qmin - min_value / scale) ;
            zero_point = max(qmin , min(qmax , zero_point)) ;

            for(unsigned j = 0 ; j < num ; j ++) {
                size_t index = static_cast<size_t>(j) * static_cast<size_t>(dim) + static_cast<size_t>(i) ;
                int q = round(data_[index] / scale) + zero_point ;
                q = max(qmin , min(qmax , q)) ;
                data_q[index] = static_cast<int8_t>(q) ; 
            }
            scales_cpu[i] = scale ;
            zero_points_cpu[i] = zero_point ;
        }
        
        cudaMemcpy(data_int8_dev , data_q , total_size , cudaMemcpyHostToDevice) ;
        cudaMemcpy(scales , scales_cpu.data() , dim * sizeof(float) , cudaMemcpyHostToDevice) ;
        cudaMemcpy(zero_points , zero_points_cpu.data() , total_size , cudaMemcpyHostToDevice) ;

        delete [] data_q ; 
        
        // cudaFree(data_half_dev) ;
        cout << "内存中所有向量量化完成" << endl ;

        data_return = data_ ;
        data_int8 = data_int8_dev ; 
        scales_return = scales ; 
        zero_points_return = zero_points ;
        // return data_half_pined ;
    }


    void show_dataset_int8_lines(float* data_origin , const int8_t* data_ , unsigned num , unsigned dim , float* scales , int* zero_points , int lines) {
        int8_t* data_p_host = new int8_t[lines * dim] ;
        float* data_host = new float[lines * dim] ;
        float* scales_host = new float[dim] ;
        int* zero_points_host = new int[dim] ;
        // cudaMemcpy(data_host , data_ , lines * dim * sizeof(float) , cudaMemcpyDeviceToHost) ;
        memcpy(data_host , data_origin , lines * dim * sizeof(float)) ;
        cudaMemcpy(data_p_host , data_ , lines * dim * sizeof(int8_t) , cudaMemcpyDeviceToHost) ;
        cudaMemcpy(scales_host , scales , dim * sizeof(float) , cudaMemcpyDeviceToHost) ;
        cudaMemcpy(zero_points_host , zero_points , dim * sizeof(int) , cudaMemcpyDeviceToHost) ;
        //打印前5行
        for(int i = 0 ; i < lines ; i ++) {
            cout << "[" ;
            for(int j = 0 ; j < dim ; j ++) {
                cout << data_host[i * dim + j] ;
                if(j < dim - 1)
                    cout << ", " ;
            }
            cout << "]" << endl ;
            cout << "*[" ;
            for(int j = 0 ; j < dim ; j ++) {
                cout << (int)data_p_host[i * dim + j] ;
                if(j < dim - 1)
                    cout << ", " ;
            }
            cout << "]" << endl ;
        }

        cout << "***********************************************" << endl ;
        // 试试反量化前5行
        for(int i = 0 ; i < lines ; i ++) {
            // cout << "[" ;
            // for(int j = 0 ; j < dim ; j ++) {
            //     cout << data_host[i * dim + j] ;
            //     if(j < dim - 1)
            //         cout << ", " ;
            // }
            // cout << "]" << endl ;
            cout << "*[" ;
            for(int j = 0 ; j < dim ; j ++) {
                // cout << (int)data_p_host[i * dim + j] ;
                float re_production = ((int) data_p_host[i * dim + j] - zero_points_host[j]) * scales_host[j] ;
                cout << abs(re_production - data_host[i * dim + j]);
                if(j < dim - 1)
                    cout << ", " ;
            }
            cout << "]" << endl ;
        }

        delete [] data_p_host ;
        delete [] data_host ;
        delete [] scales_host ;
        delete [] zero_points_host ;
    }

    void save_dataset_quantified(string filename , const int8_t* data_ , float* scales , int* zero_points , unsigned num , unsigned dim) {
        ofstream out(filename , ios::binary) ;
        if (!out) {
            std::cerr << "无法打开文件!" << std::endl;
            return ;
        }
        out.write(reinterpret_cast<const char*>(&num) , sizeof(unsigned)) ;
        out.write(reinterpret_cast<const char*>(&dim) , sizeof(unsigned)) ;
        out.write(reinterpret_cast<const char*>(scales) , dim * sizeof(float)) ;
        out.write(reinterpret_cast<const char*>(zero_points) , dim * sizeof(int)) ;
        size_t total_size = static_cast<size_t>(num) * static_cast<size_t>(dim) * sizeof(int8_t) ;
        out.write(reinterpret_cast<const char*>(data_) , total_size) ;

        out.close() ;
    }

    void load_dataset_quantified(
        string filename,
        int8_t*& data_,
        float*& scales,
        int*& zero_points,
        unsigned& num,
        unsigned& dim) {
        ifstream in(filename, ios::binary);
        if (!in) {
            std::cerr << "无法打开文件!" << std::endl;
            return;
        }

        in.read(reinterpret_cast<char*>(&num), sizeof(unsigned));
        in.read(reinterpret_cast<char*>(&dim), sizeof(unsigned));

        scales = new float[dim];
        zero_points = new int[dim];

        in.read(reinterpret_cast<char*>(scales), dim * sizeof(float));
        in.read(reinterpret_cast<char*>(zero_points), dim * sizeof(int));

        size_t total_size = static_cast<size_t>(num) * static_cast<size_t>(dim);
        data_ = new int8_t[total_size];

        in.read(reinterpret_cast<char*>(data_), total_size * sizeof(int8_t));

        in.close();
    }
    
    void load_dataset_return_int8_arr_infile(string filepath , string filepath_q , float* &data_return ,  unsigned &num , unsigned& dim , float alpha, int8_t* &data_int8 , float* &scales_return , int* &zero_points_return) {
        // unsigned num , dim  ;
        float* data_  ;
        load_data(filepath , data_ , num , dim) ;
        cout << "num : " << num << ", dim : " << dim << endl ;
        #pragma omp parallel for num_threads(10)
        for(int i = 0 ; i < num ; i ++) {
            for(int j = 0 ; j < dim ; j ++) {
                size_t index = static_cast<size_t>(i) * static_cast<size_t>(dim) + static_cast<size_t>(j) ;
                data_[index] *= alpha ;
            }
        }

        unsigned average_num = num / 16 ;
        cout << "average_num : " << average_num << endl ;
        int8_t* data_q ;
        float* scales_cpu ;
        int* zero_points_cpu ;

        int8_t* data_int8_dev ; 
        size_t total_size = static_cast<size_t>(num) * static_cast<size_t>(dim) * sizeof(int8_t) ;
        cudaMalloc((void**) &data_int8_dev , total_size) ;

        
        float* scales ; 
        int* zero_points ;
        cudaMalloc((void**) &scales , dim * sizeof(float)) ;
        cudaMalloc((void**) &zero_points , dim * sizeof(int)) ;
        cout << total_size << endl ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


        if(fileExists(filepath_q)) {
            unsigned num_q , dim_q ; 
            load_dataset_quantified(
                filepath_q,
                data_q,
                scales_cpu,
                zero_points_cpu,
                num_q,
                dim_q) ;

            cout << "文件载入完成" << endl ;
            
        } else {
            size_t arr_len = static_cast<size_t>(num) * static_cast<size_t>(dim) ;
            data_q = new int8_t[arr_len] ;
            // vector<float> scales_cpu(dim) ;
            // vector<int> zero_points_cpu(dim) ;
            scales_cpu = new float[dim] ;
            zero_points_cpu = new int[dim] ;
            float scale_ground = 255.0f ;
            int qmin = -128 , qmax = 127 ;
            
            #pragma omp parallel for num_threads(10)
            for(int i = 0 ; i < dim ; i ++) {
                float min_value = 100000.0f , max_value = -100000.0f ;
                for(unsigned j = 0 ; j < num ; j ++) {
                    size_t index = static_cast<size_t>(j) * static_cast<size_t>(dim) + static_cast<size_t>(i) ;
                    min_value = min(min_value , data_[index]) ;
                    max_value = max(max_value , data_[index]) ;
                }
                float scale = (max_value - min_value) / scale_ground ;
                int zero_point = round(qmin - min_value / scale) ;
                zero_point = max(qmin , min(qmax , zero_point)) ;

                for(unsigned j = 0 ; j < num ; j ++) {
                    size_t index = static_cast<size_t>(j) * static_cast<size_t>(dim) + static_cast<size_t>(i) ;
                    int q = round(data_[index] / scale) + zero_point ;
                    q = max(qmin , min(qmax , q)) ;
                    data_q[index] = static_cast<int8_t>(q) ; 
                }
                scales_cpu[i] = scale ;
                zero_points_cpu[i] = zero_point ;
            }
            save_dataset_quantified(filepath_q , data_q , scales_cpu , zero_points_cpu , num , dim) ;
        }
        cudaMemcpy(data_int8_dev , data_q , total_size , cudaMemcpyHostToDevice) ;
        cudaMemcpy(scales , scales_cpu , dim * sizeof(float) , cudaMemcpyHostToDevice) ;
        cudaMemcpy(zero_points , zero_points_cpu , dim * sizeof(int) , cudaMemcpyHostToDevice) ;

        
        
        // cudaFree(data_half_dev) ;
        cout << "内存中所有向量量化完成" << endl ;


        delete [] data_q ; 
        delete [] scales_cpu ;
        delete [] zero_points_cpu ;

        data_return = data_ ;
        data_int8 = data_int8_dev ; 
        scales_return = scales ; 
        zero_points_return = zero_points ;
        // return data_half_pined ;
    }

}