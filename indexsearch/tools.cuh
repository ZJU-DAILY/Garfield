#pragma once 
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <random>
#include <mma.h>
#include <unordered_set>
#include <omp.h>
#include <cstring>
#include <map>
using namespace std ;
using namespace nvcuda ;

// 此处声明谓词, 包含维度, 左端点, 右端点
string CHECK(cudaError_t err , unsigned line_id) {
    if(err != cudaSuccess)
        cout << "error at : " << line_id << "," << cudaGetErrorString(err) << endl ;
    return cudaGetErrorString(err) ;
}

string CHECK(cudaError_t err , unsigned line_id , string filename) {
    if(err != cudaSuccess)
        cout << "error at : " << filename << "-" << line_id << "," << cudaGetErrorString(err) << endl ;
    return cudaGetErrorString(err) ;
}

void showMemInfo(string message) {
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    double free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    double total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << message << ", 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;
}

struct __align__(8) half4 {
    half2 x, y;
};
  
__device__ __forceinline__ half4 BitCast(const float2& src) noexcept {
    half4 dst;
    std::memcpy(&dst, &src, sizeof(half4));
    return dst;
}
  
__device__ __forceinline__ half4 Load(const half* address) {
    float2 x = __ldg(reinterpret_cast<const float2*>(address));
    return BitCast(x);
}

bool fileExists(const string filename) {
    std::ifstream f(filename) ;
    bool flag = f.is_open() ;
    f.close() ;
    return flag ;
}

void load_data(const char* filename, float*& data, unsigned& num,
    unsigned& dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
    }
    in.read((char*)&dim, 4);
    // std::cout<<"data dimension: "<<dim<<std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[(size_t)num * (size_t)dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
}

void load_data(string filename, float*& data, unsigned& num,
    unsigned& dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
    }
    in.read((char*)&dim, 4);
    // std::cout<<"data dimension: "<<dim<<std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[(size_t)num * (size_t)dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
}

void load_data_u8(string filename, float*& data, unsigned& num,
    unsigned& dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
    }
    in.read((char*)&num, sizeof(unsigned));
    in.read((char*)&dim, sizeof(unsigned));
    // std::cout<<"data dimension: "<<dim<<std::endl;
    size_t element_num = static_cast<size_t>(num) * static_cast<size_t>(dim) ;
    uint8_t* data_u8 = new uint8_t[element_num] ;
    in.read((char*)(data_u8), element_num * sizeof(uint8_t));
    
    data = new float[element_num] ;
    for(unsigned i = 0 ; i < num ; i ++) {
        for(unsigned j = 0 ; j < dim ; j ++) {
            size_t index = static_cast<size_t>(i) * static_cast<size_t>(dim) + static_cast<size_t>(j) ;
            data[index] = data_u8[index] ;
        }
    }
    delete [] data_u8 ;

    in.close();
}

void load_data_u8(string filename, uint8_t*& data, unsigned& num,
    unsigned& dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
    }
    in.read((char*)&num, sizeof(unsigned));
    in.read((char*)&dim, sizeof(unsigned));
    // std::cout<<"data dimension: "<<dim<<std::endl;
    size_t element_num = static_cast<size_t>(num) * static_cast<size_t>(dim) ;
    data = new uint8_t[element_num] ;
    in.read((char*)(data), element_num * sizeof(uint8_t));
    

    in.close();
}


void load_result(const char* filename, unsigned* &graph , unsigned& degree , unsigned& num_in_graph){// load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    unsigned k, num ;
    in.read((char*)&num,4);
    in.read((char*)&k , 4) ;
    
    // for(size_t i = 0; i < num; i++){
    //     in.seekg(4,std::ios::cur);
    //     results[i].resize(k);
    //     results[i].reserve(k);
    //     in.read((char*)results[i].data(), k * sizeof(unsigned));
    // }
    graph = new unsigned[num * k] ;
    in.read(reinterpret_cast<char*>(graph) , num * k * sizeof(unsigned)) ;
    in.close();
    degree = k  , num_in_graph = num ; 
    cout << "End" << endl;
}

void save_global_graph(string filename ,unsigned num , unsigned degree , const unsigned* graph) {
    ofstream out(filename , ios::binary) ;
    if (!out) {
        std::cerr << "无法打开文件!" << std::endl;
        return ;
    }
    out.write(reinterpret_cast<const char*>(&num) , sizeof(unsigned)) ;
    out.write(reinterpret_cast<const char*>(&degree) , sizeof(unsigned)) ;
    size_t write_size = static_cast<size_t>(num) * static_cast<size_t>(degree) * sizeof(unsigned) ;
    out.write(reinterpret_cast<const char*>(graph) , write_size) ;
    out.close() ;
    cout << "写入完成" << endl ;
}

void save_global_graph(string filename ,vector<vector<unsigned>>& graph) {
    ofstream out(filename , ios::binary) ;
    if (!out) {
        std::cerr << "无法打开文件!" << std::endl;
        return ;
    }
    unsigned num = graph.size() , degree = graph[0].size() ;
    out.write(reinterpret_cast<const char*>(&num) , sizeof(unsigned)) ;
    out.write(reinterpret_cast<const char*>(&degree) , sizeof(unsigned)) ;
    // size_t write_size = static_cast<size_t>(num) * static_cast<size_t>(degree) * sizeof(unsigned) ;
    size_t write_size = static_cast<size_t>(degree) * sizeof(unsigned) ;
    for(unsigned i = 0 ;i  < num ;i ++) {
        out.write(reinterpret_cast<const char*>(graph[i].data()) , write_size) ;  
    }
    out.close() ;
    cout << "写入完成" << endl ;
}

void load_result(string filename, unsigned* &graph , unsigned& degree , unsigned& num_in_graph){// load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error : "<< filename << std::endl;exit(-1);}
    unsigned k, num ;
    in.read((char*)&num,4);
    in.read((char*)&k , 4) ;
    
    // for(size_t i = 0; i < num; i++){
    //     in.seekg(4,std::ios::cur);
    //     results[i].resize(k);
    //     results[i].reserve(k);
    //     in.read((char*)results[i].data(), k * sizeof(unsigned));
    // }
    graph = new unsigned[num * k] ;
    in.read(reinterpret_cast<char*>(graph) , num * k * sizeof(unsigned)) ;
    in.close();
    degree = k  , num_in_graph = num ; 
    cout << "End" << endl;
}

void load_result_pined(const char* filename, unsigned* &graph , unsigned& degree , unsigned& num_in_graph){// load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    unsigned k, num ;
    in.read((char*)&num,4);
    in.read((char*)&k , 4) ;
    
    // graph = new unsigned[num * k] ;
    size_t byteSize = static_cast<size_t>(k) * static_cast<size_t>(num) * sizeof(unsigned) ;
    cudaMallocHost((void**) &graph , byteSize) ;
    in.read(reinterpret_cast<char*>(graph) , byteSize) ;
    in.close();
    degree = k  , num_in_graph = num ; 
    cout << "End" << endl;
}

void load_result_pined(string filename, unsigned* &graph , unsigned& degree , unsigned& num_in_graph){// load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    unsigned k, num ;
    in.read((char*)&num,4);
    in.read((char*)&k , 4) ;
    
    // graph = new unsigned[num * k] ;
    size_t byteSize = static_cast<size_t>(k) * static_cast<size_t>(num) * sizeof(unsigned) ;
    cudaMallocHost((void**) &graph , byteSize) ;
    in.read(reinterpret_cast<char*>(graph) , byteSize) ;
    in.close();
    degree = k  , num_in_graph = num ; 
    cout << "End" << endl;
}


void load_range(char* filename, std::vector<std::pair<unsigned, unsigned>>& data, unsigned num) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
      std::cout << "open file error" << std::endl;
      exit(-1);
    }
    in.seekg(0, std::ios::beg);
    unsigned tmp_data1, tmp_data2;
    for (size_t i = 0; i < num; i++) {
      in.read((char*)&tmp_data1, 4);
      in.read((char*)&tmp_data2, 4);
      data.push_back(std::pair<unsigned, unsigned>(tmp_data1, tmp_data2));
    }
    in.close();
}

template<typename T = float>
void load_range(string filename , T* l_bound , T* r_bound , const unsigned attr_dim ,const unsigned num) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
      std::cout << "open file error" << std::endl;
      exit(-1);
    }
    T* ranges = new T[attr_dim * num * 2] ;
    in.read(reinterpret_cast<char*> (ranges) , 2 * attr_dim * num * sizeof(T)) ;
    for(unsigned i = 0 ; i < num ;i  ++) {
        for(unsigned j = 0 ; j < attr_dim ; j ++) {
            l_bound[i * attr_dim + j] = ranges[i * attr_dim * 2 + j * 2] ;
            r_bound[i * attr_dim + j] = ranges[i * attr_dim * 2 + j * 2 + 1] ;
        }
    }
    in.close() ;
    cout << "range 文件读取完成" << endl ;
}


__global__ void f2h(float* data, half* data_half, unsigned point_num){
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned DIM = 128 ;
    // data 为向量, data_half为按半精度存储的向量, points_num为数据点个数
    for(unsigned i = tid; i < point_num * DIM; i += blockDim.x * gridDim.x){
        // printf("%d ," , i) ;
        data_half[i] = __float2half(data[i] / 100.0);
    }
    // __syncthreads() ;
    // if(tid == 0) {
    //     printf("finished ! f2h\n") ;
    // }
}

__global__ void f2h(float* data, half* data_half, unsigned point_num , unsigned DIM , float alpha){
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    // unsigned DIM = 128 ;
    // data 为向量, data_half为按半精度存储的向量, points_num为数据点个数
    for(unsigned i = tid; i < point_num * DIM; i += blockDim.x * gridDim.x){
        // printf("%d ," , i) ;
        data_half[i] = __float2half(data[i] * alpha);
    }
    // __syncthreads() ;
    // if(tid == 0) {
    //     printf("finished ! f2h\n") ;
    // }
}

__global__ void f2h(float* data, half* data_half, unsigned point_num , unsigned DIM){
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    // unsigned DIM = 128 ;
    // data 为向量, data_half为按半精度存储的向量, points_num为数据点个数
    for(unsigned i = tid; i < point_num * DIM; i += blockDim.x * gridDim.x){
        // printf("%d ," , i) ;
        data_half[i] = __float2half(data[i]);
    }
    // __syncthreads() ;
    // if(tid == 0) {
    //     printf("finished ! f2h\n") ;
    // }
}


void generate_range_by_selectivity(float p , float min_value , float max_value , vector<pair<unsigned , unsigned>>& ranges , unsigned rnum) {
    mt19937 engine ;
    float tail = (max_value - min_value) * p ;
    uniform_int_distribution<int> dist(min_value , max_value - tail) ;
    ranges.resize(rnum) ;
    for(int i = 0 ;i  < rnum ;i  ++) {
      ranges[i].first = dist(engine) , ranges[i].second = ranges[i].first + tail ;
    }
}

void generate_range_multi_attr(float* l_bound , float* r_bound , unsigned attr_dim , float* attr_min , float* attr_max ,
    unsigned num) {
    mt19937 rng ; 
    vector<float> span(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++)
        span[i] = attr_max[i] -  attr_min[i] ;
    for(unsigned i = 0 ; i < num ; i ++) {
        for(unsigned j = 0 ; j < attr_dim ; j ++) {
            uniform_real_distribution<float> dist(0.0 , span[j]) ;
            float coverage = dist(rng) ;
            uniform_real_distribution<float> start_point(0.0 , span[j] - coverage) ;
            float ep = start_point(rng) ;
            l_bound[i * attr_dim + j] = attr_min[j] + ep ; 
            r_bound[i * attr_dim + j] = attr_min[j] + ep + coverage ;
        }
    }
}

template<typename attrType = float>
void generate_range_multi_attr_with_selectivity(attrType* l_bound , attrType* r_bound , unsigned attr_dim , attrType* attr_min , attrType* attr_max ,
    unsigned num , attrType selectivity) {
    mt19937 rng ; 
    vector<attrType> span(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++)
        span[i] = (attr_max[i] -  attr_min[i]) * selectivity ;
    for(unsigned i = 0 ; i < num ; i ++) {
        for(unsigned j = 0 ; j < attr_dim ; j ++) {
            // uniform_real_distribution<float> dist(0.0 , span[j]) ;
            attrType coverage = span[j] ;
            uniform_real_distribution<attrType> start_point(0.0 , attr_max[j] - coverage) ;
            attrType ep = start_point(rng) ;
            l_bound[i * attr_dim + j] = attr_min[j] + ep ; 
            r_bound[i * attr_dim + j] = attr_min[j] + ep + coverage ;
        }
    }
}

template<typename attrType = float>
float cal_recall_show_partition_multi_attrs(vector<vector<unsigned>>& target_res , vector<vector<unsigned>>& gt ,
    attrType* l_bound , attrType* r_bound , unsigned attr_dim , attrType* attr_width , attrType* attr_min, bool flag = true) {
    float total_recall = 0.0 ;
    
    for(unsigned i = 0 ; i < gt.size() ; i ++) {

        sort(target_res[i].begin() , target_res[i].end()) ;
        sort(gt[i].begin() , gt[i].end()) ;
        vector<unsigned> result ; 
        set_intersection(target_res[i].begin() , target_res[i].end() , gt[i].begin() , gt[i].end() , back_inserter(result)) ;
        
        float recall = (gt[i].size() > 0 ? (result.size() * 1.0) / (gt[i].size() * 1.0) : 1.0f) ; 
        // total_recall += (result.size() * 1.0) / (gt[i].size() * 1.0) ;
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
            // unsigned begin_par = query_range_x[i].first / x_width, end_par = min((unsigned)3, (unsigned)(query_range_x[i].second / x_width));
            // unsigned y_begin_par = query_range_y[i].first / y_width , y_end_par = min((unsigned) 3 ,(unsigned)(query_range_y[i].second / y_width)) ;
            vector<unsigned> begin_par(attr_dim) , end_par(attr_dim) ;
            for(unsigned j = 0 ; j < attr_dim ; j ++)
                begin_par[j] = (unsigned)((l_bound[i * attr_dim + j] - attr_min[j]) / attr_width[j]) , end_par[j] = (unsigned)((r_bound[i * attr_dim + j] - attr_min[j]) / attr_width[j]) ;
            cout << "q" << i << ": recall : " << recall << endl ;
            // cout << "x(" << begin_par << "," << end_par << ") , y(" << y_begin_par << "," << y_end_par << ")" << endl ;
            for(unsigned j = 0 ;j < attr_dim ; j ++) {
                cout << "D" << j <<"(" << begin_par[j] << "," << end_par[j] << "),[" << l_bound[i * attr_dim + j] << "," << r_bound[i * attr_dim + j] << "]" ;
                if(j < attr_dim - 1)
                    cout << ";" ;
                else 
                    cout << endl ; 
            }

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
        cout << "3155 !!! nan come !!!!" << endl ; 
    total_recall /= gt.size() ;
    if(isnan(total_recall))
        cout << "3158!!! nan come !!!!" << endl ;
    cout << "in function cal_recall : " << total_recall << endl ;
    return total_recall ;
}
  
template<typename attrType = float>
float cal_recall_show_partition_multi_attrs_show_attr(vector<vector<unsigned>>& target_res , vector<vector<unsigned>>& gt ,
    attrType* l_bound , attrType* r_bound , attrType* attrs,  unsigned attr_dim , attrType* attr_width , attrType* attr_min, bool flag = true) {
    float total_recall = 0.0 ;
    
    for(unsigned i = 0 ; i < gt.size() ; i ++) {

        sort(target_res[i].begin() , target_res[i].end()) ;
        sort(gt[i].begin() , gt[i].end()) ;
        vector<unsigned> result ; 
        set_intersection(target_res[i].begin() , target_res[i].end() , gt[i].begin() , gt[i].end() , back_inserter(result)) ;
        
        float recall = (gt[i].size() > 0 ? (result.size() * 1.0) / (gt[i].size() * 1.0) : 1.0f) ; 
        // total_recall += (result.size() * 1.0) / (gt[i].size() * 1.0) ;
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
            // unsigned begin_par = query_range_x[i].first / x_width, end_par = min((unsigned)3, (unsigned)(query_range_x[i].second / x_width));
            // unsigned y_begin_par = query_range_y[i].first / y_width , y_end_par = min((unsigned) 3 ,(unsigned)(query_range_y[i].second / y_width)) ;
            vector<unsigned> begin_par(attr_dim) , end_par(attr_dim) ;
            for(unsigned j = 0 ; j < attr_dim ; j ++)
                begin_par[j] = (unsigned)((l_bound[i * attr_dim + j] - attr_min[j]) / attr_width[j]) , end_par[j] = (unsigned)((r_bound[i * attr_dim + j] - attr_min[j]) / attr_width[j]) ;
            cout << "q" << i << ": recall : " << recall << endl ;
            // cout << "x(" << begin_par << "," << end_par << ") , y(" << y_begin_par << "," << y_end_par << ")" << endl ;
            for(unsigned j = 0 ;j < attr_dim ; j ++) {
                cout << "D" << j <<"(" << begin_par[j] << "," << end_par[j] << "),[" << l_bound[i * attr_dim + j] << "," << r_bound[i * attr_dim + j] << "]" ;
                if(j < attr_dim - 1)
                    cout << ";" ;
                else 
                    cout << endl ; 
            }

            cout << "A[" ;
            for(unsigned j = 0 ;j < target_res[i].size() ; j ++) {
                cout << target_res[i][j] << " - (" ;
                unsigned idx = target_res[i][j] ;
                for(unsigned d = 0 ; d < attr_dim ; d ++) {
                    if(idx < 1000000)
                        cout << attrs[idx * attr_dim + d] ;
                    else
                        cout << "#" ;
                    if(d < attr_dim -1)
                        cout << "," ;
                }
                cout << ")" ;
                if(j < target_res[i].size() - 1)
                    cout << " ," ;
            }
            cout << "]" << endl ;
            cout << "B[" ;
            for(unsigned j = 0 ;j < gt[i].size() ; j ++) {
                cout << gt[i][j] << " - (" ;
                unsigned idx = gt[i][j] ;
                for(unsigned d = 0 ; d < attr_dim ; d ++) {
                    if(idx < 1000000)
                        cout << attrs[idx * attr_dim + d] ;
                    else
                        cout << "#" ;
                    if(d < attr_dim -1)
                        cout << "," ;
                }
                cout << ")" ;
                if(j < gt[i].size() - 1)
                    cout << " ," ;
            }
            cout << "]" << endl ;
        }

        
    }
    if(isnan(total_recall))
        cout << "3155 !!! nan come !!!!" << endl ; 
    total_recall /= gt.size() ;
    if(isnan(total_recall))
        cout << "3158!!! nan come !!!!" << endl ;
    cout << "in function cal_recall : " << total_recall << endl ;
    return total_recall ;
}


void load_data_cluster(const char* filename, unsigned*& data, unsigned& num) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
      std::cout << "open file error" << std::endl;
      exit(-1);
    }
    in.read((char*)&num, 4);
    // std::cout<<"data dimension: "<<dim<<std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    // num = (unsigned)(fsize / (dim + 1) / 4);
    data = new unsigned[(size_t)num];
  
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < 1; i++) {
      in.seekg(4, std::ios::cur);
      in.read((char*)(data), num * 4);
    }
    in.close();
  }

  void load_clustering(vector<unsigned>& cluster_id , string cluster_path , float* &cluster_centers , unsigned dim , vector<vector<unsigned>>& num_c_p , 
    float* data_load) {
    // 簇id, 聚类结果存放路径, 聚类中心, 维度, 每个分区分别在每个簇存放了多少点
    // 先加载簇
    unsigned num  ; 
    unsigned* cids ; 
    load_data_cluster(cluster_path.c_str() , cids  , num) ;
    cout << "clustering num :" << num << endl ;
    cluster_id = vector<unsigned>(cids , cids + num) ;
    // cluster_center = new float[dim * num] 
    unsigned type_num = *max_element(cids , cids + num) + 1 ;
  
    // cout << "cluster ids : " << endl ; 
    // for(unsigned i = 0 ; i < num ; i ++)
    //   cout << cids[i] << "," ;
    cluster_centers = new float[dim * type_num] ;
    fill(cluster_centers , cluster_centers + type_num * dim , 0.0) ;
    // 得出聚类中心的坐标, 同步计算出在分区/聚类点的数量
    num_c_p = vector<vector<unsigned>>(16 , vector<unsigned>(type_num , 0)) ;
    vector<unsigned> num_in_cluster(type_num , 0) ;
    for(unsigned i = 0 ; i < num ; i ++) {
      unsigned pid = i / 62500 , cid = cids[i] ;
      // cout << "pid : " << pid << ", cid :" << cid << endl ;
      num_in_cluster[cid] ++ , num_c_p[pid][cid] ++ ;
      // 累加聚类中心
      for(unsigned j = 0 ;j  < dim ; j ++)
        cluster_centers[cid * dim + j] += data_load[i * dim + j] ;
    }
    // cout << 11111111 << endl ;
    // 聚类中心取均值
    for(unsigned i = 0 ; i < type_num ; i ++)
      for(unsigned j = 0 ;j  < dim ; j ++)
        cluster_centers[i * dim + j] /= num_in_cluster[i] ;
  }


  void load_clustering(vector<unsigned>& cluster_id , string cluster_path , float* &cluster_centers , unsigned dim , vector<vector<unsigned>>& num_c_p , 
    float* data_load , unsigned pnum) {
    // 簇id, 聚类结果存放路径, 聚类中心, 维度, 每个分区分别在每个簇存放了多少点
    // 先加载簇
    unsigned num  ; 
    unsigned* cids ; 
    load_data_cluster(cluster_path.c_str() , cids  , num) ;
    cout << "clustering num :" << num << endl ;
    cluster_id = vector<unsigned>(cids , cids + num) ;
    // cluster_center = new float[dim * num] 
    unsigned type_num = *max_element(cids , cids + num) + 1 ;
  
    // cout << "cluster ids : " << endl ; 
    // for(unsigned i = 0 ; i < num ; i ++)
    //   cout << cids[i] << "," ;
    cluster_centers = new float[dim * type_num] ;
    fill(cluster_centers , cluster_centers + type_num * dim , 0.0) ;
    // 得出聚类中心的坐标, 同步计算出在分区/聚类点的数量
    num_c_p = vector<vector<unsigned>>(pnum , vector<unsigned>(type_num , 0)) ;
    vector<unsigned> num_in_cluster(type_num , 0) ;

    unsigned avg_num = (num + pnum - 1) / pnum ; 
    for(unsigned i = 0 ; i < num ; i ++) {
      unsigned pid = i / avg_num , cid = cids[i] ;
      // cout << "pid : " << pid << ", cid :" << cid << endl ;
      num_in_cluster[cid] ++ , num_c_p[pid][cid] ++ ;
      // 累加聚类中心
      for(unsigned j = 0 ;j  < dim ; j ++)
        cluster_centers[cid * dim + j] += data_load[i * dim + j] ;
    }
    // cout << 11111111 << endl ;
    // 聚类中心取均值
    for(unsigned i = 0 ; i < type_num ; i ++)
      for(unsigned j = 0 ;j  < dim ; j ++)
        cluster_centers[i * dim + j] /= num_in_cluster[i] ;
  }

unsigned get_partition_id_host(unsigned* partition_indexes , unsigned* attr_grid_size , unsigned attr_dim) {
    unsigned pid = 0 ;
    unsigned factor = 1 ;
    for(int i = attr_dim - 1 ; i >= 0 ; i --) {
        pid += partition_indexes[i] * factor ;
        factor *= attr_grid_size[i] ;
    }
    return pid ; 
}




template<typename ATTR_TYPE = float>
void re_generate(ATTR_TYPE* attrs , unsigned attr_dim, unsigned num , ATTR_TYPE* l_bound , ATTR_TYPE* r_bound , unsigned* attr_width) {
// 先确定每个分区有多少点
    unsigned grid_size = 1 ; 
    for(unsigned i = 0 ; i < attr_dim ; i ++)
        grid_size *= attr_width[i] ;
    unsigned average_num = num / grid_size ; 
    vector<ATTR_TYPE> axis_range(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++)
        axis_range[i] = (r_bound[i] - l_bound[i]) / attr_width[i] ;
    mt19937 rng ; 
    vector<uniform_real_distribution<ATTR_TYPE>> urds(attr_dim) ; 
    vector<unsigned> pid_index(attr_dim , 0) ;
    // for(unsigned i = 0 ; i < attr_dim ; i ++)
    //     attr_width[i] = l_bound[i] ;
    while(true) {
        unsigned pid = get_partition_id_host(pid_index.data() , attr_width , attr_dim) ;
        // is_selected[pid] = true ; 
        for(unsigned i = 0 ; i < attr_dim ; i ++)
            urds[i] = uniform_real_distribution<ATTR_TYPE>(l_bound[i] + pid_index[i] * axis_range[i] ,l_bound[i] +  (pid_index[i] + 1) * axis_range[i]) ;
        // #pragma omp parallel for num_threads(10)
        for(unsigned i = pid * average_num ; i < (pid + 1) * average_num ;i  ++) {
            for(unsigned j = 0 ; j < attr_dim ; j ++)
                attrs[i * attr_dim + j] = urds[j](rng) ;
        }

        int pos = 0 ; 
        while(pos < attr_dim) {
            pid_index[pos] ++ ;
            if(pid_index[pos] <= attr_width[pos] - 1)
                break ; 
            else {
                pid_index[pos] = 0 ;
                pos ++ ;
            }
        }
        if(pos >= attr_dim)
            break ; 
    }
}

template<typename ATTR_TYPE = float>
void re_generate(ATTR_TYPE* attrs , unsigned attr_dim, unsigned num , ATTR_TYPE* l_bound , ATTR_TYPE* r_bound , unsigned* attr_width, const unsigned* seg_scheme , const unsigned* seg_start_index ,  const unsigned pnum) {
// 先确定每个分区有多少点
    unsigned grid_size = 1 ; 
    for(unsigned i = 0 ; i < attr_dim ; i ++)
        grid_size *= attr_width[i] ;
    // unsigned average_num = num / grid_size ; 
    vector<ATTR_TYPE> axis_range(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++)
        axis_range[i] = (r_bound[i] - l_bound[i]) / attr_width[i] ;
    mt19937 rng ; 
    vector<uniform_real_distribution<ATTR_TYPE>> urds(attr_dim) ; 
    vector<unsigned> pid_index(attr_dim , 0) ;
    // for(unsigned i = 0 ; i < attr_dim ; i ++)
    //     attr_width[i] = l_bound[i] ;
    // cout << "开始循环" << endl ;
    while(true) {
        unsigned pid = get_partition_id_host(pid_index.data() , attr_width , attr_dim) ;
        // is_selected[pid] = true ; 
        for(unsigned i = 0 ; i < attr_dim ; i ++)
            urds[i] = uniform_real_distribution<ATTR_TYPE>(l_bound[i] + pid_index[i] * axis_range[i] ,l_bound[i] +  (pid_index[i] + 1) * axis_range[i]) ;
        // #pragma omp parallel for num_threads(10)
        // for(unsigned i = pid * average_num ; i < (pid + 1) * average_num ;i  ++) {
        //     for(unsigned j = 0 ; j < attr_dim ; j ++)
        //         attrs[i * attr_dim + j] = urds[j](rng) ;
        // }
        // cout << "pid:" << pid << ", seg_start_index : " << seg_start_index[pid] << ", seg_scheme : " << seg_scheme[pid] << endl ;
        for(unsigned i = seg_start_index[pid] ; i < seg_start_index[pid] + seg_scheme[pid] ; i ++) {
            // if(i >= 100000000) {
            //     cout << "error : " << i << endl ;
            // }
            for(unsigned j = 0 ; j < attr_dim ; j ++)
                attrs[i * attr_dim + j] = urds[j](rng) ;
        }

        int pos = 0 ; 
        while(pos < attr_dim) {
            pid_index[pos] ++ ;
            if(pid_index[pos] <= attr_width[pos] - 1)
                break ; 
            else {
                pid_index[pos] = 0 ;
                pos ++ ;
            }
        }
        if(pos >= attr_dim)
            break ; 
    }
}

template<typename attrType = float>
void load_attrs(string filename, attrType* &attrs , unsigned &attr_dim, unsigned &num) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
    }
    in.read((char*)&num, 4);
    in.read((char*)&attr_dim , 4) ;
    // std::cout<<"data dimension: "<<dim<<std::endl;
    // in.seekg(0, std::ios::end);
    // std::ios::pos_type ss = in.tellg();
    size_t fsize = static_cast<size_t>(num) * static_cast<size_t>(attr_dim) * sizeof(attrType) ;
    // num = (unsigned)(fsize / (dim + 1) / 4);
    attrs = new attrType[static_cast<size_t>(num) * static_cast<size_t>(attr_dim)];

    in.read(reinterpret_cast<char*>(attrs) , fsize) ;
    in.close();
    cout << "属性读取完成" << endl ;
}

template<typename attrType = float>
void load_attrs(string filename, attrType* &attrs) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
    }
    unsigned num , attr_dim ;
    in.read((char*)&num, sizeof(unsigned));
    in.read((char*)&attr_dim , sizeof(unsigned)) ;
    // std::cout<<"data dimension: "<<dim<<std::endl;
    // in.seekg(0, std::ios::end);
    // std::ios::pos_type ss = in.tellg();
    size_t fsize = static_cast<size_t>(num) * static_cast<size_t>(attr_dim) * sizeof(attrType) ;
    // num = (unsigned)(fsize / (dim + 1) / 4);
    attrs = new attrType[static_cast<size_t>(num) * static_cast<size_t>(attr_dim)];

    in.read(reinterpret_cast<char*>(attrs) , fsize) ;
    in.close();
    cout << "属性读取完成" << endl ;
}

template<typename attrType = float>
void save_attrs(string filename ,unsigned num , unsigned attr_dim, const attrType* attrs) {
    ofstream out(filename , ios::binary) ;
    if (!out) {
        std::cerr << "无法打开文件!" << std::endl;
        return ;
    }
    out.write(reinterpret_cast<const char*>(&num) , sizeof(unsigned)) ;
    out.write(reinterpret_cast<const char*>(&attr_dim) , sizeof(unsigned)) ;
    size_t write_size = static_cast<size_t>(num) * static_cast<size_t>(attr_dim) * sizeof(attrType) ;
    out.write(reinterpret_cast<const char*>(attrs) , write_size) ;
    out.close() ;
    cout << "写入完成" << endl ;
}

void save_ground_truth(string filename , unsigned qnum , unsigned k, vector<vector<unsigned>>& gt) {
    ofstream out(filename , ios::binary) ;
    if (!out) {
        std::cerr << "无法打开文件!" << std::endl;
        return ;
    }
    out.write(reinterpret_cast<const char*>(&qnum) , sizeof(unsigned)) ;
    out.write(reinterpret_cast<const char*>(&k) , sizeof(unsigned)) ;
    for(unsigned i = 0 ; i < qnum ; i ++) {
        out.write(reinterpret_cast<const char*>(gt[i].data()) , k * sizeof(unsigned)) ;
    }
    out.close() ;
    cout << "ground truth 保存完成" << endl ;
}

void load_ground_truth(string filename , vector<vector<unsigned>>& gt) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
    }
    unsigned qnum , k ; 
    in.read((char*)&qnum, sizeof(unsigned));
    in.read((char*)&k , sizeof(unsigned)) ;
    gt = vector<vector<unsigned>>(qnum , vector<unsigned>(k)) ;
    for(unsigned i = 0 ; i < qnum ; i ++) {
        in.read((char*)gt[i].data() , k * sizeof(unsigned)) ;
    }
    in.close() ;
    cout << "ground truth 载入完成" << endl ;
}

void save_query_ids(string filename , unsigned qnum , vector<unsigned>& qids) {
    ofstream out(filename , ios::binary) ;
    if (!out) {
        std::cerr << "无法打开文件!" << std::endl;
        return ;
    }
    out.write(reinterpret_cast<const char*>(&qnum) , sizeof(unsigned)) ;
    out.write(reinterpret_cast<const char*>(qids.data()) , qnum * sizeof(unsigned)) ;
    out.close() ;
    cout << "query ids 保存完成" << endl ;
}

void load_query_ids(string filename , unsigned& qnum , vector<unsigned>& qids) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
    }
    in.read((char*)&qnum, sizeof(unsigned));
    qids.resize(qnum) ;
    in.read((char*)qids.data() , qnum * sizeof(unsigned)) ;
    in.close() ;
    cout << "query ids 载入完成" << endl ;
}


array<unsigned* , 3> get_idx_mapper(const unsigned N , const unsigned graph_num , const unsigned* graph_size, const unsigned* pid_list) {
    // global_idx : 代表offset + local_idx号向量的存放位置
    // local_idx : 代表当前存放位置向量的局部id
    // offset : 代表每张图的图开始逻辑id
    // 真实存放位置 : global_id , 逻辑位置 : offset[pid] + local_idx , 局部id : local_idx[真实存放位置]
    unsigned* global_idx , * local_idx , * offset ;
    // unsigned N = 1000000 ;
    cudaMalloc((void**) &global_idx , N * sizeof(unsigned)) ;
    cudaMalloc((void**) &local_idx , N * sizeof(unsigned)) ;
    cudaMalloc((void**) &offset , graph_num * sizeof(unsigned)) ;
    // 先申请三个host数组, 填充完毕后传到GPU上
    unsigned* global_idx_host = new unsigned[N] ;
    unsigned* local_idx_host = new unsigned[N] ;
    unsigned* offset_host = new unsigned[graph_num] ;
    unsigned* local_idx_index = new unsigned[graph_num] ;
    
    offset_host[0] = 0 ;
    for(int i = 1 ; i < graph_num ; i ++) {
        offset_host[i] = offset_host[i - 1] + graph_size[i - 1] ;
    }
    for(int i = 0 ; i < graph_num ; i ++)
        local_idx_index[i] = 0 ;
    for(int i = 0 ;i  < N ;i  ++) {
        unsigned pid = pid_list[i] ;
        global_idx_host[offset_host[pid] + local_idx_index[pid]] = i ;
        local_idx_host[i] = local_idx_index[pid] ;
        local_idx_index[pid] ++ ;
    }
    cudaMemcpy(global_idx , global_idx_host , N * sizeof(unsigned) ,cudaMemcpyHostToDevice) ;
    cudaMemcpy(local_idx , local_idx_host , N * sizeof(unsigned) ,cudaMemcpyHostToDevice) ;
    cudaMemcpy(offset , offset_host , graph_num * sizeof(unsigned) ,cudaMemcpyHostToDevice) ;

    cout << "graph size [" ;
    for(int i = 0 ; i < graph_num ; i ++) {
        cout << graph_size[i] ;
        if(i < graph_num - 1)
            cout << "," ;
    }
    cout << "]" << endl ;

    cout << "local idx index [" ;
    for(int i = 0 ; i < graph_num ; i ++) {
        cout << local_idx_index[i] ;
        if(i < graph_num - 1)
            cout << "," ;
    }
    cout << "]" << endl ;

    delete [] global_idx_host ;
    delete [] local_idx_host ;
    delete [] offset_host ;
    delete [] local_idx_index ;
    return {global_idx , local_idx , offset} ;
}

array<unsigned*,5> compress_global_graph_extreme(array<unsigned*,3>& graph_infos, const unsigned degree ,  const unsigned* global_graph_T , const unsigned pnum , const unsigned points_num , const unsigned num_per_seg , 
    const unsigned* seg_partition_point_start_index) {
    // 返回 : 压缩矩阵, 每行起始索引, 每行元素个数, 中心点存放位置
    vector<unsigned> spokesman(points_num) ;
    vector<unsigned> center_num_p(pnum , 0) ;
    for(unsigned i = 0 ; i < points_num ; i ++)
        spokesman[i] = i ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        unsigned offset = seg_partition_point_start_index[i] ;
        unsigned num = graph_infos[1][i] ;
        for(unsigned j = 0 ; j < num ; j ++) {
            // unsigned idx = offset + i ; 
            if(spokesman[offset + j] != offset + j)
                continue ;
            for(unsigned k = 0 ; k < degree ; k ++) {
                unsigned idx = graph_infos[0][graph_infos[2][i] + j * degree + k] + offset ;
                if(spokesman[idx] == idx && idx > (offset + j))
                    spokesman[idx] = offset + j ;
            }
        }
    }
    
    for(unsigned i = 0 ; i < pnum ; i ++) {
        unsigned offset = seg_partition_point_start_index[i] ;
        unsigned num = graph_infos[1][i] ;
        for(unsigned j = 0 ; j < num ; j ++) {
            unsigned idx = offset + j ; 
            if(spokesman[idx] == idx)
                center_num_p[i] ++ ;
        }
    }



    cout << "center num " << pnum << " [" ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cout << center_num_p[i] ;
        if(i < pnum - 1)
            cout << " ," ;
    }
    cout << "]" << endl ; 
    // 压缩
    unsigned total_center_num = accumulate(center_num_p.begin() , center_num_p.end() , 0) ;
    unsigned* compressed_graph , *col_offset , *center_num , *center_pos , *batch_stored_pos ; 
    size_t compressed_graph_size = static_cast<size_t>(total_center_num) * static_cast<size_t>(num_per_seg) * static_cast<size_t>(pnum) * sizeof(unsigned) ;
    cudaMallocHost((void**) &compressed_graph , compressed_graph_size) ;
    cudaMallocHost((void**) &center_num , pnum * sizeof(unsigned)) ;
    cudaMallocHost((void**) &col_offset , pnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &center_pos , points_num * sizeof(unsigned)) ;
    cudaMalloc((void**) &batch_stored_pos , total_center_num * sizeof(unsigned)) ;
    
    memcpy(center_num , center_num_p.data() , pnum * sizeof(unsigned)) ;

    unordered_map<unsigned , unsigned> um ; 

    // 写入的图需为列主序
    vector<unsigned> center_pos_host(points_num) ;
    unsigned cur = 0 , graph_pointer = 0 ;  
    // 将中心点对应的id压缩到前面
    for(unsigned i = 0 ; i < points_num ; i ++) {
        if(!um.count(spokesman[i])) {
            // spokesman[i] = cur ++ ;
            um[spokesman[i]] = cur ++ ;
        }
        center_pos_host[i] = um[spokesman[i]] ;
    }
    cout << "cur : " << cur <<" , total_center_num : " << total_center_num << " , um.size() : " << um.size() << endl ; 
    // 取出中心点id和对应存放位置, 按照列主序填充图
    size_t row_length = total_center_num * num_per_seg ;
    for(auto it = um.begin() ; it != um.end() ; it ++) {
        unsigned idx = it->first , pos = it->second ; 
        // if(idx >= 1e8 || pos >= total_center_num)
        //     cout << "idx : " << idx << ", pos : " << pos << endl ;
        for(unsigned i = 0 ; i < pnum ; i ++) {
            // cout << "cg index : " << i * row_length + pos * num_per_seg << ", gb index : " << i * points_num * num_per_seg + idx * num_per_seg << endl ;
            // unsigned cg_index = i * row_length + pos * num_per_seg , gb_index = i * points_num * num_per_seg + idx * num_per_seg ;
            // if(cg_index >= total_center_num * num_per_seg * pnum || gb_index >= 32 * 100000000)
            //     cout << "cg index : " << i * row_length + pos * num_per_seg << ", gb index : " << i * points_num * num_per_seg + idx * num_per_seg << endl ;
            // 填充每一列
            size_t cg_index = static_cast<size_t>(i) * static_cast<size_t>(row_length) + static_cast<size_t>(pos) * static_cast<size_t>(num_per_seg) ;
            size_t gb_index = static_cast<size_t>(i) * static_cast<size_t>(points_num) * static_cast<size_t>(num_per_seg) + static_cast<size_t>(idx) * static_cast<size_t>(num_per_seg) ;
            compressed_graph[cg_index] = global_graph_T[gb_index] ;
            compressed_graph[cg_index + 1] = global_graph_T[gb_index + 1] ;
        }
    }
    cout << "填充完毕" << endl ;
    col_offset[0] = 0 ; 
    for(unsigned i = 1 ; i < pnum ;i  ++) 
        col_offset[i] = col_offset[i - 1] + center_num_p[i - 1] ;
    cudaMemcpy(center_pos , center_pos_host.data() , points_num * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    // 压缩后的列主序图, 每个分区的列数量, 每一列的开始偏移, 每个点对应的中心点存放位置
    return {compressed_graph , center_num , col_offset , center_pos , batch_stored_pos} ;
}   


// 注: 后两个数组为设备内存
void save_global_graph_extreme(string filename , array<unsigned*,5>& global_graph , unsigned pnum , unsigned points_num , unsigned num_per_seg) {
    ofstream out(filename , ios::binary) ;
    if (!out) {
        std::cerr << "无法打开文件!" << std::endl;
        return ;
    }
    unsigned* center_num = global_graph[1] , *col_offset = global_graph[2] , *center_pos = global_graph[3] , * compressed_graph = global_graph[0] ;

    out.write(reinterpret_cast<const char*>(&points_num) , sizeof(unsigned)) ;
    out.write(reinterpret_cast<const char*>(&pnum) , sizeof(unsigned)) ;
    out.write(reinterpret_cast<const char*>(&num_per_seg) , sizeof(unsigned)) ;
    out.write(reinterpret_cast<const char*>(center_num) , pnum * sizeof(unsigned)) ;
    out.write(reinterpret_cast<const char*>(col_offset) , pnum *sizeof(unsigned)) ;

    unsigned* center_pos_host = new unsigned[points_num] ;
    cudaMemcpy(center_pos_host , center_pos , points_num * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    out.write(reinterpret_cast<const char*>(center_pos_host) , points_num *sizeof(unsigned)) ;
    delete [] center_pos_host ;
    unsigned total_center_num = accumulate(center_num , center_num + pnum , 0) ;
    size_t bytesSize = static_cast<size_t>(total_center_num) * static_cast<size_t>(num_per_seg) * static_cast<size_t>(pnum) * sizeof(unsigned) ;
    out.write(reinterpret_cast<const char*>(compressed_graph) , bytesSize) ;
    out.close() ;
    cout << "极度压缩图保存完成" << endl ;
}

void load_global_graph_extreme(string filename , array<unsigned*,5>& global_graph) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
    }
    unsigned points_num , pnum , num_per_seg ;
    in.read((char*) &points_num , sizeof(unsigned)) ;
    in.read((char*) &pnum , sizeof(unsigned)) ;
    in.read((char*) &num_per_seg , sizeof(unsigned)) ;
    cudaMallocHost((void**) &global_graph[1] , pnum * sizeof(unsigned)) ;
    cudaMallocHost((void**) &global_graph[2] , pnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &global_graph[3] , points_num * sizeof(unsigned)) ;
    unsigned* center_pos_host = new unsigned[points_num] ;

    cudaMalloc((void**) &global_graph[4] , points_num * sizeof(unsigned)) ;
    in.read((char*) global_graph[1] , pnum * sizeof(unsigned)) ;
    in.read((char*) global_graph[2] , pnum * sizeof(unsigned)) ;
    in.read((char*) center_pos_host , points_num * sizeof(unsigned)) ;
    cudaMemcpy(global_graph[3] , center_pos_host , points_num * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    delete [] center_pos_host ;

    unsigned total_center_num = accumulate(global_graph[1] , global_graph[1] + pnum , 0) ;
    size_t bytesSize = static_cast<size_t>(total_center_num) * static_cast<size_t>(num_per_seg) * static_cast<size_t>(pnum) * sizeof(unsigned) ;
    cudaMallocHost((void**) &global_graph[0] , bytesSize) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    in.read((char*) global_graph[0] , bytesSize) ;
    in.close() ;
    cout << "极度压缩图读取完成" << endl ;
}


array<unsigned*,5> compress_global_graph_extreme2(array<unsigned*,3>& graph_infos, const unsigned degree ,  const unsigned* global_graph_T , const unsigned pnum , const unsigned points_num , const unsigned num_per_seg , 
    const unsigned* seg_partition_point_start_index) {
    // 返回 : 压缩矩阵, 每行起始索引, 每行元素个数, 中心点存放位置
    vector<unsigned> spokesman(points_num) ;
    vector<unsigned> center_num_p(pnum , 0) ;
    for(unsigned i = 0 ; i < points_num ; i ++)
        spokesman[i] = i ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        unsigned offset = seg_partition_point_start_index[i] ;
        unsigned num = graph_infos[1][i] ;
        for(unsigned j = 0 ; j < num ; j ++) {
            // unsigned idx = offset + i ; 
            if(spokesman[offset + j] != offset + j)
                continue ;
            for(unsigned k = 0 ; k < degree ; k ++) {
                unsigned idx = graph_infos[0][graph_infos[2][i] + j * degree + k] + offset ;
                if(spokesman[idx] == idx && idx > (offset + j))
                    spokesman[idx] = offset + j ;
            }
        }
    }
    
    for(unsigned i = 0 ; i < pnum ; i ++) {
        unsigned offset = seg_partition_point_start_index[i] ;
        unsigned num = graph_infos[1][i] ;
        for(unsigned j = 0 ; j < num ; j ++) {
            unsigned idx = offset + j ; 
            if(spokesman[idx] == idx)
                center_num_p[i] ++ ;
        }
    }



    cout << "center num " << pnum << " [" ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cout << center_num_p[i] ;
        if(i < pnum - 1)
            cout << " ," ;
    }
    cout << "]" << endl ; 
    // 压缩
    unsigned total_center_num = accumulate(center_num_p.begin() , center_num_p.end() , 0) ;
    unsigned* compressed_graph , *col_offset , *center_num , *center_pos , *batch_stored_pos ; 
    size_t compressed_graph_size = static_cast<size_t>(total_center_num) * static_cast<size_t>(num_per_seg) * static_cast<size_t>(pnum) * sizeof(unsigned) ;
    cudaMallocHost((void**) &compressed_graph , compressed_graph_size) ;
    cudaMallocHost((void**) &center_num , pnum * sizeof(unsigned)) ;
    cudaMallocHost((void**) &col_offset , pnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &center_pos , points_num * sizeof(unsigned)) ;
    cudaMalloc((void**) &batch_stored_pos , total_center_num * sizeof(unsigned)) ;
    
    memcpy(center_num , center_num_p.data() , pnum * sizeof(unsigned)) ;

    unordered_map<unsigned , unsigned> um ; 

    // 写入的图需为列主序
    vector<unsigned> center_pos_host(points_num) ;
    unsigned cur = 0 , graph_pointer = 0 ;  
    // 将中心点对应的id压缩到前面
    for(unsigned i = 0 ; i < points_num ; i ++) {
        if(!um.count(spokesman[i])) {
            // spokesman[i] = cur ++ ;
            um[spokesman[i]] = cur ++ ;
        }
        center_pos_host[i] = um[spokesman[i]] ;
    }
    cout << "cur : " << cur <<" , total_center_num : " << total_center_num << " , um.size() : " << um.size() << endl ; 
    // 取出中心点id和对应存放位置, 按照列主序填充图
    size_t row_length = total_center_num * num_per_seg ;
    for(auto it = um.begin() ; it != um.end() ; it ++) {
        unsigned idx = it->first , pos = it->second ; 
        // if(idx >= 1e8 || pos >= total_center_num)
        //     cout << "idx : " << idx << ", pos : " << pos << endl ;
        for(unsigned i = 0 ; i < pnum ; i ++) {
            // cout << "cg index : " << i * row_length + pos * num_per_seg << ", gb index : " << i * points_num * num_per_seg + idx * num_per_seg << endl ;
            // unsigned cg_index = i * row_length + pos * num_per_seg , gb_index = i * points_num * num_per_seg + idx * num_per_seg ;
            // if(cg_index >= total_center_num * num_per_seg * pnum || gb_index >= 32 * 100000000)
            //     cout << "cg index : " << i * row_length + pos * num_per_seg << ", gb index : " << i * points_num * num_per_seg + idx * num_per_seg << endl ;
            // 填充每一列
            size_t cg_index = static_cast<size_t>(i) * static_cast<size_t>(row_length) + static_cast<size_t>(pos) * static_cast<size_t>(num_per_seg) ;
            size_t gb_index = static_cast<size_t>(i) * static_cast<size_t>(points_num) * static_cast<size_t>(num_per_seg) + static_cast<size_t>(idx) * static_cast<size_t>(num_per_seg) ;
            compressed_graph[cg_index] = global_graph_T[gb_index] ;
            compressed_graph[cg_index + 1] = global_graph_T[gb_index + 1] ;
        }
    }
    cout << "填充完毕" << endl ;
    col_offset[0] = 0 ; 
    for(unsigned i = 1 ; i < pnum ;i  ++) 
        col_offset[i] = col_offset[i - 1] + center_num_p[i - 1] ;
    cudaMemcpy(center_pos , center_pos_host.data() , points_num * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    // 压缩后的列主序图, 每个分区的列数量, 每一列的开始偏移, 每个点对应的中心点存放位置
    return {compressed_graph , center_num , col_offset , center_pos , batch_stored_pos} ;
}   

  inline void gemm_GT_rowmajor_f16_to_f32(
    cublasHandle_t handle,
    const __half* P_half, // (Np x 128) row-major
    const __half* Q_half, // (Nq x 128) row-major
    float* GT_f32,          // (Np x Nq) column-major
    unsigned Np ,
    unsigned Nq , 
    unsigned d 
) {
    float alpha = 1.0f;
    float beta  = 0.0f;

    // GT (Np x Nq) = P_rm (Np x d) * Q_rm^T (d x Nq)
    // cublas (col-major view) does: GT = op(A)*op(B) with A=P, B=Q
    // op(A)=T, op(B)=N
    cublasStatus_t st = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,   // P^T * Q
        Np, Nq, d,                  // m=Np, n=Nq, k=128
        &alpha,
        P_half, CUDA_R_16F, d,      // lda = d (row-major stride)
        Q_half, CUDA_R_16F, d,      // ldb = d
        &beta,
        GT_f32, CUDA_R_32F, Np,     // ldc = m = Np
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    // 建议加 assert(st==CUBLAS_STATUS_SUCCESS)
}

__global__ void row_norm2_f16_rowmajor(const __half* X, int rows, int d, float* out) {
    int r = blockIdx.x;
    if (r >= rows) return;

    float sum = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float v = __half2float(X[r * d + j]); // row-major
        sum += v * v;
    }

    __shared__ float buf[256];
    buf[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[r] = buf[0];
}

__global__ void fuse_dist2_from_GT(
    const float* q_norm, const float* p_norm,
    const float* GT_colmajor, // (Np x Nq) col-major
    int Nq, int Np,
    float* D2_rowmajor        // (Nq x Np) row-major
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; // p index [0, Np)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // q index [0, Nq)
    if (i >= Nq || k >= Np) return;

    float g = GT_colmajor[k + i * Np]; // GT(k,i) = G(i,k)

    float v = q_norm[i] + p_norm[k] - 2.0f * g;
    if (v < 0.f) v = 0.f; // 防止浮点误差导致的负零附近

    D2_rowmajor[i * Np + k] = v;
}

__global__ void gather_topk_element_from_per_group(const unsigned group_size , const unsigned group_num , const unsigned element_k , unsigned* input , unsigned* output) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x ;
    #pragma unroll
    for(unsigned i = 0 ; i < element_k ; i ++)
        output[element_k * tid + i] = input[group_size * tid + i] ;
}

__device__ __forceinline__ void topk_insert(float* topk , unsigned* topv , float x , unsigned vx , unsigned K) {
    if(x >= topk[K - 1])
        return ;
    int pos = K - 1 ; 
    topk[pos] = x ;
    topv[pos] = vx ;
    
    #pragma unroll
    for(int i = K - 1 ; i > 0 ; i --) {
        if(topk[i] >= topk[i - 1])
            break ;
        float tk = topk[i] ;
        unsigned tv = topv[i] ;
        topk[i] = topk[i - 1] ;
        topv[i] = topv[i - 1] ;
        topk[i - 1] = tk ;
        topv[i - 1] = tv ; 
    }
}

// template<float INFINITY_DIS = 10000.0>
template<unsigned BLOCK_THREADS = 256> 
__global__ void select_topk_parallel(const unsigned group_size , const unsigned group_num , float* __restrict__ dis_arr , unsigned* __restrict__ ids_arr ,
    const unsigned K , unsigned* topk_arr , const float INFINITY_DIS) {
        unsigned tid = blockDim.x * threadIdx.y + threadIdx.x ;
        // unsigned BLOCK_THREADS = blockDim.x * blockDim.y ;
        float topk[16];
        unsigned topv[16];
        #pragma unroll 
        for (int i = 0; i < K; ++i) {
            topk[i] = INFINITY_DIS;
            topv[i] = 0u;
        }
        
        for (int j = threadIdx.x; j < group_size; j += BLOCK_THREADS) {
            float x = dis_arr[blockIdx.x * group_size + j];
            unsigned vx = ids_arr[blockIdx.x * group_size + j];
            topk_insert(topk, topv, x, vx , K);
        }

        __shared__ float sk[BLOCK_THREADS * 16] ;
        __shared__ unsigned sv[BLOCK_THREADS * 16] ;

        #pragma unroll
        for (int i = 0 ; i < K ; i ++) {
            sk[tid * K + i] = topk[i] ;
            sv[tid * K + i] = topv[i] ;
        }
        __syncthreads() ;

        if (tid == 0) {
            float bestk[16];
            unsigned bestv[16];
            #pragma unroll
            for (int i = 0; i < K; ++i) {
                bestk[i] = INFINITY_DIS;
                bestv[i] = 0u;
            }
    
            // 扫候选集合
            for (int idx = 0; idx < BLOCK_THREADS * K; ++idx) {
                float x = sk[idx];
                unsigned vx = sv[idx];
                topk_insert(bestk, bestv, x, vx , K );
            }
    
            // 输出前 k 个（升序）
            int out_base = blockIdx.x * K;  // 如果你希望 out 固定按 MAX_K 存，也可以改成 g*MAX_K
            for (int i = 0; i < K; ++i) {
                // topk_arr[out_base + i] = bestk[i];
                topk_arr[out_base + i] = bestv[i];
            }
        }
}


#include <cfloat>
template<int KMAX = 16, int BLOCK_THREADS = 256>
__global__ void bitonic_topk_only_per_group_1000(
    const float* __restrict__ keys_in,
    const unsigned* __restrict__ vals_in,
    // float* __restrict__ keys_out,
    unsigned* __restrict__ vals_out,
    int num_groups,
    int k,             // 运行时需要的 k（1..KMAX）
    int out_stride     // 建议固定为 KMAX，方便复用输出 buffer
){
    constexpr int N   = 1000;
    constexpr int NP2 = 1024; // 2^10

    int g = blockIdx.x;
    if (g >= num_groups) return;

    if (k < 1) k = 1;
    if (k > KMAX) k = KMAX;

    extern __shared__ unsigned char smem[];
    float*    sk = reinterpret_cast<float*>(smem);
    unsigned* sv = reinterpret_cast<unsigned*>(sk + NP2);

    // load + padding
    int base = g * N;
    for (int idx = threadIdx.x; idx < NP2; idx += BLOCK_THREADS) {
        if (idx < N) {
            sk[idx] = keys_in[base + idx];
            sv[idx] = vals_in[base + idx];
        } else {
            sk[idx] = FLT_MAX; // padding: +INF
            sv[idx] = 0u;
        }
    }
    __syncthreads();

    // Bitonic network, but ONLY do compare-swap if (idx < KMAX || ixj < KMAX)
    // 这样能保证前 KMAX 个最终正确且有序
    for (int k_ = 2; k_ <= NP2; k_ <<= 1) {
        for (int j = k_ >> 1; j > 0; j >>= 1) {

            for (int idx = threadIdx.x; idx < NP2; idx += BLOCK_THREADS) {
                int ixj = idx ^ j;
                if (ixj > idx) {
                    // 只关心会影响前 KMAX 的比较
                    if (idx < KMAX || ixj < KMAX) {

                        bool ascending = ((idx & k_) == 0); // bitonic 方向
                        float a = sk[idx];
                        float b = sk[ixj];

                        bool do_swap = ascending ? (a > b) : (a < b);
                        if (do_swap) {
                            sk[idx] = b; sk[ixj] = a;

                            unsigned va = sv[idx];
                            unsigned vb = sv[ixj];
                            sv[idx] = vb; sv[ixj] = va;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // 输出前 k 个（升序）
    if (threadIdx.x == 0) {
        int out_base = g * out_stride;
        #pragma unroll
        for (int i = 0; i < KMAX; ++i) {
            if (i < k) {
                // keys_out[out_base + i] = sk[i];
                vals_out[out_base + i] = sv[i];
            }
        }
    }
}


template<unsigned K = 4 ,unsigned MAX_WARP_NUM = 32>
__global__ void select_topk_by_reduce_min(const unsigned group_size , const unsigned group_num ,const float* __restrict__ dis_arr ,const unsigned* __restrict__ ids_arr ,
    unsigned* topk_arr) {
    // 每轮reduce一个后把其从列表中去掉
    // 由于整个列表要扫描K次, 需要将数据导入共享内存

    __shared__ unsigned topk_idx[16] ;
    // __shared__ float topk_dis[16] ;
    __shared__ float dis[1000] ;
    __shared__ unsigned middle_arr_ids[MAX_WARP_NUM] ;
    __shared__ float middle_arr_dis[MAX_WARP_NUM] ;
    unsigned tid = blockDim.x * threadIdx.y + threadIdx.x , thread_num = blockDim.x * blockDim.y ;
    #pragma unroll
    for(unsigned i = tid ; i < group_size ; i += thread_num) 
        dis[i] = dis_arr[blockIdx.x * group_size + i] ;
    __syncthreads() ;
    
    unsigned lane_id = threadIdx.x ;
    // 开始reduce
    #pragma unroll
    for(unsigned i = 0 ; i < K ; i ++) {
        // 每个线程先拿到自己的局部最小值
        float min_dis = 10000.0 ;
        unsigned min_index = -1 ; 
        #pragma unroll
        for(unsigned j = tid ; j < group_size ; j += thread_num) {
            if(min_dis > dis[j]) {
                min_dis = dis[j] ;
                min_index = j ;
            }
        }
        // middle_arr_dis[tid] = min_dis ; 
        // middle_arr_ids[tid] = min_index ; 
        __syncthreads() ;
        // 每个线程组先reduce出自己的最小值
        #pragma unroll
        for(int offset = 16 ; offset > 0 ; offset >>= 1) {
            float next_dis = __shfl_down_sync(0xffffffff , min_dis , offset) ;
            unsigned next_idx = __shfl_down_sync(0xffffffff , min_index , offset) ;
            if(next_dis < min_dis) {
                min_dis = next_dis ;
                min_index = next_idx ; 
            }
        }
        if(threadIdx.x == 0) {
            middle_arr_ids[threadIdx.y] = min_index ;
            middle_arr_dis[threadIdx.y] = min_dis ;
        }
        __syncthreads() ;
        if(threadIdx.y == 0) {
            // unsigned mask = (1 << blockDim.y) - 1 ;
            
            // if(threadIdx.x < blockDim.y) {
            //     min_dis = middle_arr_dis[threadIdx.x] ;
            //     min_index = middle_arr_ids[threadIdx.x] ;
            // }
            // __syncwarp() ;
            bool participate = (threadIdx.x < blockDim.y);

            float min_dis = 1e10f;
            unsigned min_index = UINT_MAX;
            if (participate) {
                min_dis = middle_arr_dis[threadIdx.x];
                min_index = middle_arr_ids[threadIdx.x];
            }
        
            unsigned mask = __ballot_sync(0xffffffffu, participate);

            #pragma unroll
            for(int offset = 16 ; offset > 0 ; offset >>= 1) {
                float next_dis = __shfl_down_sync(mask , min_dis , offset) ;
                unsigned next_idx = __shfl_down_sync(mask , min_index , offset) ;
                if(lane_id + offset < blockDim.y) {
                    if(next_dis < min_dis) {
                        min_dis = next_dis ;
                        min_index = next_idx ;
                    }
                }
            }
            __syncwarp() ;
            // 此时规约出了唯一的最小值, 将其放入topk中
            if(threadIdx.x == 0) {
                topk_idx[i] = min_index ;  
                dis[min_index] = 10000.0f ;
            }
        }
        __syncthreads() ;
    }
    
    #pragma unroll
    for(unsigned i = tid ; i < K ; i  ++)
        topk_arr[blockIdx.x * K  + i] = ids_arr[blockIdx.x * group_size + topk_idx[i]] ;
    
}

template<unsigned MAX_WARP_NUM = 16>
__global__ void dis_cal_tensor_core(const __half* __restrict__ query , const __half* __restrict__ cluster_centers, const unsigned qnum , 
    const unsigned cnum , const unsigned dim ,const float* q_norm , const float* c_norm , float* __restrict__ ret_dis) {
    // 一个warp做一个向量的距离计算 16 * 16 (10000 + 15) / 16
    // 每个线程块做 (10000 + 15) / 16 的距离计算
    unsigned tid = blockDim.x * threadIdx.y  + threadIdx.x ;
    unsigned warpId = threadIdx.y , global_warp_id = blockIdx.x * blockDim.y + threadIdx.y ; 
    unsigned laneId = threadIdx.x ;
    // 此处占用 MAX_WARP_NUM KB
    __shared__ float inner_product_dis[MAX_WARP_NUM][16][16] ;
    // 每个warp做16个查询与16个聚类中心点的矩阵乘法
    // 以16为一组, 算出查询的tile号
    // unsigned workload = (qnum + 15) / 16 ;
    unsigned global_tile_id = blockIdx.x ;
    // unsigned aligned_length = (cnum + 15) / 16 * 16 ;
    unsigned cnum_with_pad = (cnum + 15) / 16 * 16 ;
    // 处理的向量编号offset = global_tile_id * 16 * dim

    unsigned row_offset = global_tile_id * 16 * dim ;
    wmma::fragment<wmma::matrix_a , 16 , 16 , 16 , __half , wmma::row_major> a ;
    wmma::fragment<wmma::matrix_b , 16 , 16 , 16 , __half , wmma::col_major> b ;
    wmma::fragment<wmma::accumulator , 16 , 16 , 16 , float> c ;
    wmma::fill_fragment(c , 0.0f) ;
    // const __half* A_tile = query + global_tile_id * 16 * dim ; 
    // 对每组聚类中心做遍历
    #pragma unroll
    for(unsigned c_tile_id = warpId ; c_tile_id < (cnum + 15) / 16 ; c_tile_id += blockDim.y) {
        unsigned col_offset = c_tile_id * 16 * dim ; 
        #pragma unroll
        for(unsigned i = 0 ; i < dim ; i += 16) {
            const __half* A_tile = query + row_offset + i ; 
            // 第几条向量的第几维 : 第c_tile_id * 16 条向量的第i维 即 i * cnum_with_pad + c_tile_id * 16
            const __half* B_tile = cluster_centers + col_offset + i ; 
            // const __half* B_tile = cluster_centers + i * cnum_with_pad + c_tile_id * 16 ;
            wmma::load_matrix_sync(a , A_tile , dim) ;
            wmma::load_matrix_sync(b , B_tile , dim) ;
            wmma::mma_sync(c , a , b, c) ;
        }

        // float* C_tile = ret_dis + global_tile_id * 16 * cnum + c_tile_id * 16 ; 
        // wmma::store_matrix_sync(C_tile , c , cnum, wmma::mem_row_major) ;

        // 先把部分计算结果写入共享内存, 和norm结合得到 欧氏距离
        float* C_tile = & (inner_product_dis[warpId][0][0]) ;
        wmma::store_matrix_sync(C_tile , c , 16 , wmma::mem_row_major) ;
        wmma::fill_fragment(c , 0.0f) ;
        // __syncwarp() ;
        
        #pragma unroll
        for(unsigned i = threadIdx.x ; i < 16 * 16 ; i += blockDim.x) {
            unsigned row = i / 16 , col = i % 16 ;
            unsigned qid = global_tile_id * 16 + row ;
            unsigned cid = c_tile_id * 16 + col ;
            if(qid < qnum && cid < cnum) { 
                inner_product_dis[warpId][row][col] = q_norm[qid] + c_norm[cid] - ((2.0f) * inner_product_dis[warpId][row][col]) ;
                ret_dis[qid * cnum + cid] = inner_product_dis[warpId][row][col] ;
            }    
        }
        // for(unsigned i = 0 ; i < 16 ; i ++) {
        //     #pragma unroll
        //     for(unsigned j = 0 ; j < 16 ; j ++) {
        //         inner_product_dis[warpId][i][j] = q_norm[global_tile_id * 16 + i] + c_norm[c_tile_id * 16 + j] - 2.0f * inner_product_dis[warpId][i][j] ;
        //     }
        // }
        __syncwarp() ;
    }
}

__global__ void find_norm(const __half* __restrict__ vectors , float* target , const unsigned dim , const unsigned pad_num) {
    // 每个warp做一个norm
    unsigned warpId = threadIdx.y , global_warp_id = blockIdx.x * blockDim.y + threadIdx.y ;
    unsigned total_warp_num = blockDim.y * gridDim.x ;
    #pragma unroll 
    for(unsigned i = global_warp_id ; i < pad_num ;i += total_warp_num) {
        float norm = 0.0 ;

        #pragma unroll
        for(unsigned j = threadIdx.x ; j < dim ; j += blockDim.x) {
            float v = __half2float(vectors[i * dim + j] ) ;
            // norm += __half2float(vectors[i * dim + j]) * __half2float(vectors[i * dim + j]) ;
            norm += v * v ;
        }
        __syncwarp() ;

        #pragma unroll
        for(unsigned offset = 16 ; offset > 0 ; offset >>= 1) {
            norm += __shfl_down_sync(0xFFFFFFFF , norm , offset) ;
        }
        if(threadIdx.x == 0) {
            target[i] = norm ; 
        }
        __syncwarp() ;
    }   
}

__global__ void matrix_transpose(const __half* __restrict__ matrix , __half* target , const unsigned row_dim , const unsigned col_dim) {
    
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x ;
    unsigned total_num = blockDim.x * gridDim.x ;
    #pragma unroll
    for(unsigned i = tid ; i < row_dim * col_dim ; i += total_num) {
        unsigned row = i / col_dim , col = i % col_dim ;
        target[col * row_dim + row] = matrix[i] ;
    }
}

__global__ void fill_partition_info0(unsigned batch_size , const unsigned* batch_partition_ids , unsigned* partition_info0 ,const unsigned* psize_list ,const unsigned* plist_start ,
     const bool* is_selected , const unsigned qnum , const unsigned pnum) {
    // if(batch_size == 0)
        // return ;
    
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if(tid >= qnum)
        return ;
    
    unsigned qid = tid , psize = psize_list[qid] , pstart = plist_start[qid] ;
    if(psize == 0)
        return ; 
    unsigned j = 0 ; 
    for(unsigned i = 0 ; i < batch_size ;i  ++) {
        if(is_selected[qid * pnum + batch_partition_ids[i]]) {
            partition_info0[pstart + j] = i ;
            j ++ ;
        }
    }

    // __syncthreads() ;
    // if(tid == 1) {
    //     printf("batch size : %d\n" , batch_size ) ;
    //     printf("batch partition ids : [") ;
    //     for(unsigned i = 0 ; i < batch_size ; i ++) {
    //         printf("%d ," , batch_partition_ids[i]) ;
    //     }
    //     printf("]\n") ;
    //     printf("psize list : [") ;
    //     for(unsigned i = 0 ; i < qnum ; i ++) {
    //         printf("%d ," , psize_list[i]) ;
    //     }
    //     printf("]\n") ;
    //     printf("plist start : [") ;
    //     for(unsigned i = 0 ; i < qnum ;i ++) {
    //         printf("%d ," , plist_start[i]) ;
    //     }
    //     printf("]\n") ;
    //     printf("partition info : [") ;
    //     for(unsigned i = 0 ; i < plist_start[qnum - 1] + psize_list[qnum - 1 ] ;i ++) {
    //         printf("%d ," , partition_info0[i]) ;
    //     }
    //     printf("]\n") ;
    // }

}

__global__ void fill_partition_info03(unsigned batch_size , const unsigned* batch_partition_ids , unsigned* partition_info0 , const unsigned* psize_list ,const unsigned* plist_start , unsigned* little_seg_num ,
    const bool* is_selected , const unsigned qnum , const unsigned pnum, const bool* is_little_seg) {
   // if(batch_size == 0)
       // return ;
   
   unsigned tid = blockIdx.x * blockDim.x + threadIdx.x ;
   if(tid >= qnum)
       return ;
   
   unsigned qid = tid , psize = psize_list[qid] , pstart = plist_start[qid] ;
   if(psize == 0)
       return ; 
   unsigned j = 0 ; 
   unsigned cnt = 0 ;
   for(unsigned i = 0 ; i < batch_size ;i  ++) {
       if(is_selected[qid * pnum + batch_partition_ids[i]]) {
            partition_info0[pstart + j] = i ;
            j ++ ;
            if(is_little_seg[qid * pnum + batch_partition_ids[i]])
                cnt ++ ;
       }
   }
   little_seg_num[qid] = (cnt == psize) ;

}


__global__ void fill_partition_info_check_global_idx(unsigned batch_size , const unsigned* batch_partition_ids , unsigned* partition_info0 ,const unsigned* psize_list ,const unsigned* plist_start ,
    const bool* is_selected , const unsigned qnum , const unsigned pnum , const unsigned* global_idx) {
   // if(batch_size == 0)
       // return ;
   
   unsigned tid = blockIdx.x * blockDim.x + threadIdx.x ;
   if(tid >= qnum)
       return ;
   
   unsigned qid = tid , psize = psize_list[qid] , pstart = plist_start[qid] ;
   if(psize == 0)
       return ; 
   unsigned j = 0 ; 
   for(unsigned i = 0 ; i < batch_size ;i  ++) {
       if(is_selected[qid * pnum + batch_partition_ids[i]]) {
           partition_info0[pstart + j] = i ;
           j ++ ;
       }
   }

   if(tid == 0) {
       for(int i = 0 ; i  < 100000000 ; i ++)
               if(global_idx[i] != i) {
                   printf("error : %d , global_idx[%d]\n" , global_idx[i] , i ) ;
               }
   }
}



__half* load_dataset_return_pined_pointer(string filepath , unsigned &num , unsigned& dim , float alpha) {
    // unsigned num , dim  ;
    float* data_  ;
    load_data(filepath , data_ , num , dim) ;
    cout << "num : " << num << ", dim : " << dim << endl ;
    unsigned average_num = num / 16 ;
    cout << "average_num : " << average_num << endl ;
    __half* data_half_pined , *data_half_dev ;
    float* data_dev ;
    cudaMalloc((void**) &data_half_dev , average_num * dim * sizeof(__half)) ;
    cudaMalloc((void**) &data_dev , average_num * dim * sizeof(float)) ;
    size_t total_size = static_cast<size_t>(num) * static_cast<size_t>(dim) * sizeof(__half) ;
    cudaMallocHost((void**) &data_half_pined , total_size) ;
    cout << "指针大小 : " << sizeof(data_half_pined) << "bytes" << endl ;
    cout << num * dim * sizeof(__half) << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    unsigned chunk_size = 62500 ;
    unsigned chunk_num = num / chunk_size ;
    cout << "chunk_size = " << chunk_size << " ,chunk_num = " << chunk_num << endl ;
    for(unsigned i = 0 ; i < chunk_num ; i ++) {
        size_t load_size = static_cast<size_t>(chunk_size) * static_cast<size_t>(dim) * static_cast<size_t>(i) ;
        cudaMemcpy(data_dev , data_ + load_size , chunk_size * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
        // cout << i << endl ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        f2h<<<chunk_size, 256>>>(data_dev, data_half_dev, chunk_size ,dim , alpha); // 先将数据转换成fp16
        cudaDeviceSynchronize() ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        size_t offset = static_cast<size_t>(chunk_size) * static_cast<size_t>(dim) * static_cast<size_t>(i) ;
        cudaMemcpy(data_half_pined + offset , data_half_dev , chunk_size * dim * sizeof(__half) , cudaMemcpyDeviceToHost) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        // if(i > 200) {
        //     int t ;
        //     cin >> t ;
        // }
    }
    delete [] data_ ;
    // 此时内存中的所有向量已经转化为__half类型
    cudaFree(data_dev) ;
    cudaFree(data_half_dev) ;
    cout << "内存中所有向量转换为半精度类型" << endl ;

    return data_half_pined ;
}


template<typename T>
void print_first5_rows(const vector<vector<T>>& mat) {
    int rows = min((size_t)5, mat.size());

    for (int i = 0; i < rows; ++i) {
        cout << "row " << i << " : ";

        for (size_t j = 0; j < mat[i].size(); ++j) {
            cout << mat[i][j];
            if (j != mat[i].size() - 1)
                cout << ", ";
        }

        cout << endl;
    }
}