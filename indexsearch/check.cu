#include "gpu_graph.cu"
#include <random>
#include <dlfcn.h>
#include <stdio.h>

void load_result(char* filename, std::vector<std::vector<unsigned>>& results){// load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    unsigned k;
    in.read((char*)&k,4);
    in.seekg(0,std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    size_t num = (unsigned)(fsize / (k+1) / 4);

    cout << fsize << "," << num << endl;
    in.seekg(0,std::ios::beg);
    results.resize(num);
    for(size_t i = 0; i < num; i++){
        in.seekg(4,std::ios::cur);
        results[i].resize(k);
        results[i].reserve(k);
        in.read((char*)results[i].data(), k * sizeof(unsigned));
    }
    in.close();
    cout << "End" << endl;
}

vector<float> generate_2d_attribute_uniform(float* &attrs , unsigned points_num , unsigned l , unsigned r) {
    // 共l * r 个分区 , 每隔 points_num / (l * r) 为一个分区
    vector<vector<vector<unsigned>>> partitions(l , vector<vector<unsigned>>(r , vector<unsigned>())) ;
    attrs = new float[points_num * 2] ;
    cout << "points_num : " << points_num << endl ;
    // 第一维以id为属性, 第二位随机
    for(int i = 0 ; i < points_num ;i ++)
      attrs[i] = i * 1.0 ;
    random_shuffle(attrs , attrs + points_num) ;
    for(int i = points_num - 1 ; i >= 0 ; i --) {
      attrs[i * 2] = i * 1.0 , attrs[i * 2 + 1] = attrs[i] ;
    }
    // 属性生成完成, 分配到各个分区中
    float x_min = 0.0 , x_max = (points_num) * 1.0 , y_min = 0.0 , y_max = (points_num) * 1.0 ;
    float x_width = (x_max - x_min) / (l * 1.0) , y_width = (y_max - y_min) / (r * 1.0) ;
    // float x_width = 62500.0 , y_width = 62500.0 ;
  
    // 按照 [) 左开右闭的方式, 将所有点分配到分区中
    for(unsigned i = 0 ; i < points_num ; i ++) {
      unsigned x_index =  (unsigned)((attrs[i * 2] - x_min) / x_width) , y_index = (unsigned)((attrs[i* 2 + 1] - y_min) / y_width) ;
      if(x_index >= l) x_index -- ;
      if(y_index >= r) y_index -- ;
      if(x_index >= l || y_index >=r ) {
        cout << "error : " << x_index << ", " << y_index << endl ;
      }
  
      partitions[x_index][y_index].push_back(i) ;
    }
    cout << 111111111<< endl ;
    vector<unsigned> redundant_list ; 
    unsigned average_num = points_num / l / r ;
    for(unsigned i = 0 ; i < l ; i  ++)
      for(unsigned j = 0 ; j < r ; j ++) {
        while(partitions[i][j].size() > average_num) {
          // 若该分区内点的个数大于平均值, 则从末尾取出一些点
          redundant_list.push_back(partitions[i][j].back()) ;
          partitions[i][j].pop_back() ;
        }
    }
    cout << 2222222 << endl ;
    // 再次遍历列表, 若有分区点个数不足平均值, 则填充, 并修改相应属性值
    mt19937 engine ; 
    unsigned pointer = 0 ; 
    for(unsigned i = 0 ; i < l ; i  ++)
      for(unsigned j = 0 ; j < r ; j ++) {
        vector<unsigned> &p = partitions[i][j] ;
        if(p.size() < average_num) {
          
          uniform_real_distribution<float> x_pos(x_min + i * x_width , x_min + (i + 1) * x_width) ;
          uniform_real_distribution<float> y_pos(y_min + j * y_width , y_min + (j + 1) * y_width) ;
          while(p.size() < average_num) {
            unsigned cand = redundant_list[pointer ++] ;
            attrs[cand * 2] = x_pos(engine) , attrs[cand * 2 + 1] = y_pos(engine) ;
            p.push_back(cand) ;
          }
        }
    }
  
    // 分区已平均化完成, 按照顺序将属性重新排列
    // 第(i , j)个分区的属性范围为 [(i * r + j) * average_num , (i * r + j + 1) * average_num)
    cout << "points_num : " << points_num << endl ;
    float* tmp_attr = new float[2 * points_num] ;
    cout << "points_num : " << points_num << endl ;
    pointer = 0 ; 
    for(unsigned i = 0 ; i < l ; i  ++)
      for(unsigned j = 0 ; j < r ; j ++) {
        vector<unsigned> &p = partitions[i][j] ;
        for(unsigned k = 0 ; k < p.size() ; k ++) {
          unsigned cur_point = p[k] ;
          tmp_attr[pointer * 2] = attrs[cur_point * 2] , tmp_attr[pointer * 2 + 1] = attrs[cur_point * 2 + 1] ;
          pointer ++ ;
        }
    }
    cout << "pointer :" << pointer << endl ;
    delete [] attrs ;
    attrs = tmp_attr ; 
  
    vector<float> f = {
      x_min , x_max , y_min , y_max , x_width , y_width
    } ;
    return f; 
}



// 计算topk个聚类中心, 然后看分区和这些聚类中心的交集数量, 并按照数量排序
vector<unsigned> generate_access_order_by_cluster(vector<unsigned>& cand_pools , float* query , float* cluster_centers , unsigned dim , vector<vector<unsigned>>& num_c_p , unsigned rK) {
    unsigned type_num = num_c_p[0].size() ;
    // cout << "type_num : " << type_num << endl ;
    vector<float> dis_to_cc(type_num , 0.0) ;
    
    // #pragma omp parallel for num_threads(10)
    for(unsigned i = 0 ; i < type_num ; i ++) {
      float dis = 0.0 ;
      for(unsigned j = 0 ; j < dim ; j ++)
        dis += (query[j] - cluster_centers[i * dim + j])  * (query[j] - cluster_centers[i * dim + j]) ;
      dis_to_cc[i] = sqrt(dis) ;
    }
    vector<unsigned> index_arr(type_num) ;
    for(unsigned i = 0 ; i < index_arr.size() ; i ++)
      index_arr[i] = i ;
    sort(index_arr.begin() , index_arr.end() , [&dis_to_cc](unsigned x , unsigned y)-> bool {
      return dis_to_cc[x] < dis_to_cc[y] ;
    });
  
    // cout << "index arr : " << endl ;
    // for(auto it = index_arr.begin() ; it != index_arr.end() ; it ++)
    //   cout << *it << "," ;
    // cout << " ajapohioaej" << endl ;
    // 对候选数组中的元素排序
    vector<unsigned> cand_count(cand_pools.size() , 0) ;
    for(unsigned i = 0 ; i < cand_pools.size() ; i ++) {
      for(unsigned j = 0 ; j < rK ; j  ++) {
        // cout << "cand_pools[i] : " << cand_pools[i] << ", index_arr[j] : "<< index_arr[j] << endl ;
        cand_count[i] += num_c_p[cand_pools[i]][index_arr[j]] ; 
        }
    }
    // cout << " euituetet" << endl ;
    vector<unsigned> index_arr_cand_cnt(cand_pools.size()) ;
    for(unsigned i = 0 ; i < cand_pools.size() ; i ++)
      index_arr_cand_cnt[i] =  i ; 
    sort(index_arr_cand_cnt.begin() , index_arr_cand_cnt.end() , [&cand_count] (unsigned x , unsigned y)->bool{
      return cand_count[x] > cand_count[y] ;
    }) ;
    vector<unsigned> res(cand_pools.size());
    for(unsigned i = 0 ; i < index_arr_cand_cnt.size() ; i ++)
      res[i] = cand_pools[index_arr_cand_cnt[i]] ;
    // for(unsigned i = 0 ; i < index_arr_cand_cnt.size() ;i  ++)
    //     cout << cand_count[index_arr_cand_cnt[i]] << "," ;
    
    return res ; 
  }


void check_intersect_area() {
    cout << "sizeof(half) = " << sizeof(half) << endl ;

    // 先载入数据集和图索引
    unsigned num , dim  ;
    float* data_  ;
    load_data("/home/yhr/workspace/cuda_clustering/data/sift_base.fvecs" , data_ , num , dim) ;
    cout << "num : " << num << ", dim : " << dim << endl ;
    float* query_data_load ;
    unsigned query_num , query_dim ;
    load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_range_query.fvecs" , query_data_load , query_num , query_dim) ;
    // query_num = 100; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;

    // TOPM = 20 , DIM = dim ; 

    float* data_dev ;
    half *data_half_dev ;
    float* query_data_dev ;
    half *query_data_half_dev ;
    cudaMalloc((void**)&data_dev , num * dim * sizeof(float)) ;
    cudaMalloc((void**) &data_half_dev , num * dim * sizeof(half)) ;
    cudaMalloc((void**) &query_data_dev , query_num * query_dim   * sizeof(float)) ;
    cudaMalloc((void**) &query_data_half_dev , query_num * query_dim * sizeof(half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(data_dev , data_ , num * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(query_data_dev , query_data_load , query_num * query_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    unsigned points_num = 62500 ; 
	// 调用实例
	f2h<<<points_num, 256>>>(data_dev, data_half_dev, points_num); // 先将数据转换成fp16
    // cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    f2h<<<query_num, 256>>>(query_data_dev, query_data_half_dev, query_num); // 将查询转换成fp16
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, query_num) ;
    query_range_x[0] = query_range_x[1] ;
    unsigned attr_dim =  2 ;

    float* l_bound , * r_bound , *l_bound_dev , *r_bound_dev , *attr_min , *attr_width ;
    unsigned *attr_grid_size ;
    l_bound = new float[query_num * attr_dim] , r_bound = new float[query_num * attr_dim] ;
    for(unsigned i = 0 ; i < query_num ; i ++)
        for(unsigned j = 0 ;j  < attr_dim ; j ++)
            l_bound[i * attr_dim + j] = (float) query_range_x[i].first , r_bound[i * attr_dim + j] = (float) query_range_x[i].second ; 
    cudaMalloc((void**) &l_bound_dev , attr_dim * query_num * sizeof(float)) ;
    cudaMalloc((void**) &r_bound_dev , attr_dim * query_num * sizeof(float)) ;
    cudaMalloc((void**) &attr_min , attr_dim * sizeof(float)) ;
    cudaMalloc((void**) &attr_width , attr_dim * sizeof(float)) ;
    cudaMalloc((void**) &attr_grid_size , attr_dim * sizeof(unsigned)) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    cudaMemcpy(l_bound_dev , l_bound , query_num * attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(r_bound_dev , r_bound , query_num * attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    thrust::fill(thrust::device , attr_min , attr_min + attr_dim , 0.0) ;
    thrust::fill(thrust::device , attr_width , attr_width + attr_dim , 250000.0) ;
    thrust::fill(thrust::device , attr_grid_size , attr_grid_size + attr_dim , 4) ;
    unsigned qnum = query_num ;
    unsigned pnum = thrust::reduce(
        thrust::device , 
        attr_grid_size , attr_grid_size + attr_dim ,
        1 , thrust::multiplies<unsigned>() 
    ) ;
    array<unsigned* , 3> partition_infos = intersection_area(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum) ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;

    // 以下为顺序执行代码, 用于检验结果
    vector<float> attr_min_host(attr_dim) , attr_width_host(attr_dim) ;
    vector<unsigned> attr_grid_size_host(attr_dim) ;
    thrust::copy(
         thrust::device_pointer_cast(attr_min),
         thrust::device_pointer_cast(attr_min) + attr_dim ,
          attr_min_host.data()) ;
    cout << "程序执行至line:" <<__LINE__ << endl ;
    thrust::copy(
        thrust::device_pointer_cast(attr_width) ,
        thrust::device_pointer_cast(attr_width) + attr_dim ,
         attr_width_host.data()) ;
    cout << "程序执行至line:" <<__LINE__ << endl ;
    thrust::copy(
        thrust::device_pointer_cast(attr_grid_size) ,
        thrust::device_pointer_cast(attr_grid_size) + attr_dim ,
         attr_grid_size_host.data()) ;
    cout << "程序执行至line:" <<__LINE__ << endl ;
    vector<vector<unsigned>> access_order ;
    for(unsigned i = 0 ; i < query_num ; i ++) {
        vector<unsigned> cur_acor ; 
        vector<unsigned> min_indexes(attr_dim) ;
        vector<unsigned> max_indexes(attr_dim) ;
        vector<unsigned> pids(attr_dim) ;
        // cout << "[" ;
        for(unsigned j = 0 ; j < attr_dim ; j ++) {
            min_indexes[j] = (unsigned)((l_bound[i * attr_dim + j] - attr_min_host[j]) / attr_width_host[j]) ;
            max_indexes[j] = (unsigned)((r_bound[i * attr_dim + j] - attr_min_host[j]) / attr_width_host[j]) ;
            if(max_indexes[j] >= attr_grid_size_host[j])
                max_indexes[j] = attr_grid_size_host[j] - 1 ;    
            // partition_indexes[j] = partition_l[j] ;
            pids[j] = min_indexes[j] ;

            // cout << "l :" << l_bound[i * attr_dim + j] << ", r :" << r_bound[i * attr_dim + j] <<
            // ", attr_min : " << attr_min_host[j] << ", attr_width :" << attr_width_host[j] << "-" ;

            // cout << "(" << min_indexes[j] << "," << max_indexes[j] << ")" ;
            // if(j < attr_dim - 1) 
            //     cout << ", " ;
        }
        // cout << "]" << endl ;

        while(true) {
            unsigned pid = get_partition_id_host(pids.data() , attr_grid_size_host.data() , attr_dim) ;
            // is_selected[pid] = true ; 
            cur_acor.push_back(pid) ;
            int pos = 0 ; 
            while(pos < attr_dim) {
                pids[pos] ++ ;
                if(pids[pos] <= max_indexes[pos])
                    break ; 
                else {
                    pids[pos] = min_indexes[pos] ;
                    pos ++ ;
                }
            }
            if(pos >= attr_dim)
                break ; 
        }
        access_order.push_back(cur_acor) ;
    }
    cout << "程序执行至line:" << __LINE__ << endl ;
    cout << "access_order.size() : " << access_order.size() << ", access_order[0].size() : " << access_order[0].size() << endl ;
    int acor_cnt = 0 ; 
    for(unsigned i = 0 ; i < access_order.size() ; i ++)
        acor_cnt += access_order[i].size() ;
    cout << "all cnt : " << acor_cnt << endl ;
    vector<unsigned> forehead20(acor_cnt) ;
    thrust::copy(thrust::device_pointer_cast(partition_infos[0]) , thrust::device_pointer_cast(partition_infos[0]) + acor_cnt , forehead20.data()) ;
    cout << "程序执行至line:" << __LINE__ << endl ;
    unsigned offset = 0 , iidx = 0 ;
    unsigned corr = 0 , num_corr = 0 ;
    for(unsigned i = 0 ; i < acor_cnt ; i ++) {
        corr += (forehead20[i] - access_order[iidx][offset]) * (forehead20[i] - access_order[iidx][offset]) ;
        cout << forehead20[i] << ":" << access_order[iidx][offset] << endl ;
        offset ++ ;
        if(offset >= access_order[iidx].size())
            offset = 0 ,  iidx ++ ;
    }
    cout << "--------------------------------------" << endl ;

    vector<unsigned> second_arr(query_num) ;
    thrust::copy(
        thrust::device_pointer_cast(partition_infos[1]) ,
        thrust::device_pointer_cast(partition_infos[1]) + query_num ,
        second_arr.data()
    ) ;
    for(unsigned i = 0 ; i < query_num ;i ++)
        num_corr += (second_arr[i] - access_order[i].size()) * (second_arr[i] - access_order[i].size()) ;

    cout << "方差:" << corr << endl ;
    cout << "数量方差:" << num_corr << endl ;

    // 需要: 聚类中心, 还有聚类中心的数量统计
    vector<unsigned> cluster_id ;
    float* cluster_centers;
    vector<vector<unsigned>> num_c_p ;
    load_clustering(cluster_id , "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/clustering1000.bin" , cluster_centers , dim , num_c_p , data_) ;
    float* cluster_centers_dev ;
    unsigned* cluster_p_info ;
    __half* cluster_centers_half_dev ;
    unsigned cnum = num_c_p[0].size() ;
    cout << "聚类个数:" << cnum << endl ;
    cudaMalloc((void**) &cluster_centers_dev , cnum * dim * sizeof(float)) ;
    cudaMalloc((void**) &cluster_p_info, cnum * pnum * sizeof(unsigned)) ;
    cudaMalloc((void**) &cluster_centers_half_dev , cnum * dim * sizeof(__half)) ;
    cudaMemcpy(cluster_centers_dev , cluster_centers , cnum * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    f2h<<<cnum, 256>>>(cluster_centers_dev, cluster_centers_half_dev, cnum);
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    
    // unsigned num_c_p_offset = 0 ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cudaMemcpy(cluster_p_info + i * cnum , num_c_p[i].data() , cnum * sizeof(unsigned) , cudaMemcpyHostToDevice ) ;
    }

    // get_partitions_access_order(partition_infos , pnum , cnum , qnum , cluster_p_info , 4 , cluster_centers_half_dev , query_data_half_dev , dim)  ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    
    vector<vector<unsigned>> access_order_by_cluster_host(qnum) ;
    for(unsigned i = 0 ; i < qnum ; i ++)
        access_order_by_cluster_host[i] = generate_access_order_by_cluster(access_order[i] , query_data_load + i * dim,  cluster_centers , dim , num_c_p , 4)  ;

    thrust::copy(thrust::device_pointer_cast(partition_infos[0]) , thrust::device_pointer_cast(partition_infos[0]) + acor_cnt , forehead20.data()) ;
    offset = 0 , iidx = 0  , corr = 0 ;
    int error_cnt = 0;
    for(unsigned i = 0 ; i < acor_cnt ; i ++) {
        // corr += (forehead20[i] - access_order_by_cluster_host[iidx][offset]) * (forehead20[i] - access_order_by_cluster_host[iidx][offset]) ;
        corr += abs((int)(forehead20[i] - access_order_by_cluster_host[iidx][offset])) ;
        if(forehead20[i] != access_order_by_cluster_host[iidx][offset]) {
            cout << "gpu-" << forehead20[i] << ":" << "cpu-" << access_order_by_cluster_host[iidx][offset] << endl ;
            error_cnt ++ ;
        }
        offset ++ ;
        if(offset >= access_order[iidx].size())
            offset = 0 ,  iidx ++ ;
    }

    cout << "-------------------------------" << endl ;
    cout << "new-绝对差:" << corr << endl ;
    cout << "error-cnt : " << error_cnt << endl ;
    cout << "-------------------------------" << endl ;
    cout << "acor_cnt : " << acor_cnt << endl ;
    cout << "gpu list size : " << thrust::reduce(thrust::device , partition_infos[1] , partition_infos[1] + query_num , 0) << endl ;
    cout << "检验完毕!" << endl ;
}

array<unsigned* , 3> get_graph_info() {
    
    unsigned * all_graphs , * graph_size , *graph_start_index ;
    cudaMalloc((void**) & all_graphs , 16 * 62500 * 16 * sizeof(unsigned)) ;
    cudaMalloc((void**) & graph_size , 16 * sizeof(unsigned)) ;
    cudaMalloc((void**) & graph_start_index , 16 * sizeof(unsigned)) ;
    vector<unsigned> graph_size_host(16 , 0) ;
    vector<unsigned> graph_start_index_host(16 , 0) ;
    for(unsigned i = 0 ; i < 16 ; i ++) {
        unsigned * graph ;
        unsigned degree , num_in_graph ;
        string id = to_string(i) ;
        string filename = ("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/sift" + id + ".cagra") ;
        load_result(filename.c_str() , graph , degree , num_in_graph) ;
        cout << i << "-grapH degree:" << degree << ", num : " << num_in_graph << endl ; 
        graph_size_host[i] = num_in_graph ; 
        if(i < 15)
            graph_start_index_host[i + 1] = graph_start_index_host[i] + degree * num_in_graph ;
        cudaMemcpy(all_graphs + graph_start_index_host[i] , graph , degree * num_in_graph * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
        delete [] graph ;
    }
    cudaMemcpy(graph_size , graph_size_host.data() , 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(graph_start_index , graph_start_index_host.data() , 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    return {all_graphs , graph_size , graph_start_index} ;
}

array<unsigned* , 3> get_graph_info(string prefix , string suffix , const unsigned graph_num , const unsigned unified_degree, const unsigned points_num) {
    
    unsigned * all_graphs , * graph_size , *graph_start_index ;
    // cudaMalloc((void**) & all_graphs , 16 * 62500 * 16 * sizeof(unsigned)) ;
    // all_graphs = new unsigned[16 * 62500 * 16 * sizeof(unsigned)] ;
    // cudaMalloc((void**) & graph_size , 16 * sizeof(unsigned)) ;
    // graph_size = new unsigned[16 * sizeof(unsigned)] ;
    // cudaMalloc((void**) & graph_start_index , 16 * sizeof(unsigned)) ;
    // graph_start_index = new unsigned[16 * sizeof(unsigned)] ;
    // vector<unsigned> graph_size_host(16 , 0) ;
    // vector<unsigned> graph_start_index_host(16 , 0) ;
    // size_t byteSize = static_cast<size_t>(graph_num) * static_cast<size_t>(avg_graph_size) * static_cast<size_t>(unified_degree) * sizeof(unsigned) ;
    size_t byteSize = static_cast<size_t>(unified_degree) * static_cast<size_t>(points_num) * sizeof(unsigned) ;
    cudaMalloc(&all_graphs , byteSize) ; 
    cudaMalloc(&graph_size , graph_num * sizeof(unsigned)) ;
    cudaMalloc(&graph_start_index , graph_num * sizeof(unsigned)) ;
    vector<unsigned> graph_size_host(graph_num , 0) ;
    vector<unsigned> graph_start_index_host(graph_num , 0) ;
    graph_start_index_host[0] = 0 ; 
    for(unsigned i = 0 ; i < graph_num ; i ++) {
        unsigned * graph ;
        unsigned degree , num_in_graph ;
        string id = to_string(i) ;
        string filename = (prefix + id + suffix) ;
        // unsigned offset = (i == 0 ? 0 : graph_start_index[i]) ;
        unsigned offset = graph_start_index_host[i] ;
        // cout << "图" << i << " load result" << endl ;
        load_result(filename.c_str() , graph , degree , num_in_graph) ;

        // if(i == 17) {
        //     cout << "[" ;
        //     for(unsigned j = 0 ; j < 100 ;j ++) {
        //         for(unsigned t = 0 ; t < 16 ; t ++) {
        //             cout << graph[j * 16 + t] << " ," ;
        //         }
        //         cout << endl;
        //     }
        //     cout << "]" << endl ;
        // }
        // cout << "[" ;
        // for(unsigned j = 0 ; j < 100 ;j ++) {
        //     cout << "[" ;
        //     for(unsigned t = 0 ; t < 16 ; t ++) {
        //         cout << graph[j * 16 + t] << " ," ;
        //     }
        //     cout <<  "]" << endl;
        // }
        // cout << "]" << endl ;
        // if(i == 0) {
        //     cout << "graph 0 [" ;
        //     for(int j = 0 ;j  < 16 ; j ++) {
        //         cout << "[" ;
        //         for(int k = 0 ; k < 16 ; k ++) {
        //             cout << graph[j * 16 + k] ;
        //             if(k < 15)
        //                 cout << "," ;
        //         }
        //         cout << "]" << endl ;
        //     }
        //     cout << "]" << endl ;
        // }

        cout << i << "-grapH degree:" << degree << ", num : " << num_in_graph << endl ; 
        graph_size_host[i] = num_in_graph ; 
        if(i < graph_num - 1)
            graph_start_index_host[i + 1] = graph_start_index_host[i] + degree * num_in_graph ;
        // cudaMemcpy(all_graphs + graph_start_index_host[i] , graph , degree * num_in_graph * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
        size_t load_size = static_cast<size_t>(degree) * static_cast<size_t>(num_in_graph) * sizeof(unsigned) ;
        // memcpy(all_graphs + offset , graph , load_size) ;
        // cout << "图" << i << " gpu 复制" << endl ;
        cudaMemcpy(all_graphs + offset , graph , load_size , cudaMemcpyHostToDevice) ;
        CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
        cout << "图" << i << "传输完成" << endl ;
        delete [] graph ;
    }
    cudaMemcpy(graph_size , graph_size_host.data() , graph_num * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(graph_start_index , graph_start_index_host.data() , graph_num * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    return {all_graphs , graph_size , graph_start_index} ;
}

array<unsigned* , 3> get_graph_info_host() {
    
    unsigned * all_graphs , * graph_size , *graph_start_index ;
    // cudaMalloc((void**) & all_graphs , 16 * 62500 * 16 * sizeof(unsigned)) ;
    // all_graphs = new unsigned[16 * 62500 * 16 * sizeof(unsigned)] ;
    // cudaMalloc((void**) & graph_size , 16 * sizeof(unsigned)) ;
    // graph_size = new unsigned[16 * sizeof(unsigned)] ;
    // cudaMalloc((void**) & graph_start_index , 16 * sizeof(unsigned)) ;
    // graph_start_index = new unsigned[16 * sizeof(unsigned)] ;
    // vector<unsigned> graph_size_host(16 , 0) ;
    // vector<unsigned> graph_start_index_host(16 , 0) ;
    cudaMallocHost(&all_graphs , 16 * 62500 * 16 * sizeof(unsigned)) ; 
    cudaMallocHost(&graph_size , 16 * sizeof(unsigned)) ;
    cudaMallocHost(&graph_start_index , 16 * sizeof(unsigned)) ;

    for(unsigned i = 0 ; i < 16 ; i ++) {
        unsigned * graph ;
        unsigned degree , num_in_graph ;
        string id = to_string(i) ;
        string filename = ("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/sift" + id + ".cagra") ;
        unsigned offset = (i == 0 ? 0 : graph_start_index[i]) ;
        load_result(filename.c_str() , graph , degree , num_in_graph) ;
        cout << i << "-grapH degree:" << degree << ", num : " << num_in_graph << endl ; 
        graph_size[i] = num_in_graph ; 
        if(i < 15)
            graph_start_index[i + 1] = graph_start_index[i] + degree * num_in_graph ;
        // cudaMemcpy(all_graphs + graph_start_index_host[i] , graph , degree * num_in_graph * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
        memcpy(all_graphs + offset , graph , degree * num_in_graph * sizeof(unsigned)) ;
        delete [] graph ;
    }
    // cudaMemcpy(graph_size , graph_size_host.data() , 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    // cudaMemcpy(graph_start_index , graph_start_index_host.data() , 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    return {all_graphs , graph_size , graph_start_index} ;
}

array<unsigned* , 3> get_graph_info_host_avg_size(string prefix , string suffix , const unsigned graph_num , const unsigned avg_graph_size , const unsigned unified_degree) {
    
    unsigned * all_graphs , * graph_size , *graph_start_index ;
    // cudaMalloc((void**) & all_graphs , 16 * 62500 * 16 * sizeof(unsigned)) ;
    // all_graphs = new unsigned[16 * 62500 * 16 * sizeof(unsigned)] ;
    // cudaMalloc((void**) & graph_size , 16 * sizeof(unsigned)) ;
    // graph_size = new unsigned[16 * sizeof(unsigned)] ;
    // cudaMalloc((void**) & graph_start_index , 16 * sizeof(unsigned)) ;
    // graph_start_index = new unsigned[16 * sizeof(unsigned)] ;
    // vector<unsigned> graph_size_host(16 , 0) ;
    // vector<unsigned> graph_start_index_host(16 , 0) ;
    size_t byteSize = static_cast<size_t>(graph_num) * static_cast<size_t>(avg_graph_size) * static_cast<size_t>(unified_degree) * sizeof(unsigned) ;
    cudaMallocHost(&all_graphs , byteSize) ; 
    cudaMallocHost(&graph_size , graph_num * sizeof(unsigned)) ;
    cudaMallocHost(&graph_start_index , graph_num * sizeof(unsigned)) ;
    graph_start_index[0] = 0 ; 
    for(unsigned i = 0 ; i < graph_num ; i ++) {
        unsigned * graph ;
        unsigned degree , num_in_graph ;
        string id = to_string(i) ;
        string filename = (prefix + id + suffix) ;
        // unsigned offset = (i == 0 ? 0 : graph_start_index[i]) ;
        unsigned offset = graph_start_index[i] ;
        load_result(filename.c_str() , graph , degree , num_in_graph) ;

        // if(i == 4) {
        //     for(unsigned j = 0 ; j < 100 ;j ++) {
        //         for(unsigned t = 0 ; t < 16 ; t ++) {
        //             cout << graph[j * 16 + t] << " ," ;
        //         }
        //         cout << endl;
        //     }
        
        // }

        cout << i << "-grapH degree:" << degree << ", num : " << num_in_graph << endl ; 
        graph_size[i] = num_in_graph ; 
        if(i < graph_num - 1)
            graph_start_index[i + 1] = graph_start_index[i] + degree * num_in_graph ;
        // cudaMemcpy(all_graphs + graph_start_index_host[i] , graph , degree * num_in_graph * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
        size_t load_size = static_cast<size_t>(degree) * static_cast<size_t>(num_in_graph) * sizeof(unsigned) ;
        memcpy(all_graphs + offset , graph , load_size) ;
        cout << "图" << i << "传输完成" << endl ;
        delete [] graph ;
    }
    // cudaMemcpy(graph_size , graph_size_host.data() , 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    // cudaMemcpy(graph_start_index , graph_start_index_host.data() , 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    return {all_graphs , graph_size , graph_start_index} ;
}


array<unsigned* , 3> get_graph_info_host(string prefix , string suffix , const unsigned graph_num , const unsigned unified_degree, const unsigned points_num) {
    
    unsigned * all_graphs , * graph_size , *graph_start_index ;
    // cudaMalloc((void**) & all_graphs , 16 * 62500 * 16 * sizeof(unsigned)) ;
    // all_graphs = new unsigned[16 * 62500 * 16 * sizeof(unsigned)] ;
    // cudaMalloc((void**) & graph_size , 16 * sizeof(unsigned)) ;
    // graph_size = new unsigned[16 * sizeof(unsigned)] ;
    // cudaMalloc((void**) & graph_start_index , 16 * sizeof(unsigned)) ;
    // graph_start_index = new unsigned[16 * sizeof(unsigned)] ;
    // vector<unsigned> graph_size_host(16 , 0) ;
    // vector<unsigned> graph_start_index_host(16 , 0) ;
    // size_t byteSize = static_cast<size_t>(graph_num) * static_cast<size_t>(avg_graph_size) * static_cast<size_t>(unified_degree) * sizeof(unsigned) ;
    size_t byteSize = static_cast<size_t>(unified_degree) * static_cast<size_t>(points_num) * sizeof(unsigned) ;
    cudaMallocHost(&all_graphs , byteSize) ; 
    cudaMallocHost(&graph_size , graph_num * sizeof(unsigned)) ;
    cudaMallocHost(&graph_start_index , graph_num * sizeof(unsigned)) ;
    graph_start_index[0] = 0 ; 
    for(unsigned i = 0 ; i < graph_num ; i ++) {
        unsigned * graph ;
        unsigned degree , num_in_graph ;
        string id = to_string(i) ;
        string filename = (prefix + id + suffix) ;
        // unsigned offset = (i == 0 ? 0 : graph_start_index[i]) ;
        unsigned offset = graph_start_index[i] ;
        load_result(filename.c_str() , graph , degree , num_in_graph) ;

        // if(i == 4) {
            // cout << "[" ;
            // for(unsigned j = 0 ; j < 100 ;j ++) {
            //     for(unsigned t = 0 ; t < 16 ; t ++) {
            //         cout << graph[j * 16 + t] << " ," ;
            //     }
            //     cout << endl;
            // }
            // cout << "]" << endl ;
        // }
    

        cout << i << "-grapH degree:" << degree << ", num : " << num_in_graph << endl ; 
        graph_size[i] = num_in_graph ; 
        if(i < graph_num - 1)
            graph_start_index[i + 1] = graph_start_index[i] + degree * num_in_graph ;
        // cudaMemcpy(all_graphs + graph_start_index_host[i] , graph , degree * num_in_graph * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
        size_t load_size = static_cast<size_t>(degree) * static_cast<size_t>(num_in_graph) * sizeof(unsigned) ;
        memcpy(all_graphs + offset , graph , load_size) ;
        cout << "图" << i << "传输完成" << endl ;
        delete [] graph ;
    }
    // cudaMemcpy(graph_size , graph_size_host.data() , 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    // cudaMemcpy(graph_start_index , graph_start_index_host.data() , 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    return {all_graphs , graph_size , graph_start_index} ;
}

array<unsigned* , 3> get_idx_mapper() {
    unsigned* global_idx , * local_idx , * offset ;
    unsigned N = 1000000 ;
    cudaMalloc((void**) &global_idx , N * sizeof(unsigned)) ;
    cudaMalloc((void**) &local_idx , N * sizeof(unsigned)) ;
    cudaMalloc((void**) &offset , 16 * sizeof(unsigned)) ;
    auto cit = thrust::make_counting_iterator<unsigned>(0) ;
    thrust::transform(
        thrust::device , 
        cit , cit + 16 ,
        offset , 
        [] __device__ __host__ (unsigned i) {
            return i * 62500 ;
        }
    ) ;
    thrust::transform(
        thrust::device , 
        cit , cit + N ,
        local_idx ,
        [] __device__ __host__ (unsigned i) {
            return i % 62500 ;
        }
    ) ;
    thrust::sequence(thrust::device , global_idx , global_idx + N) ;
    return {global_idx , local_idx , offset} ;
}

array<unsigned* , 3> get_idx_mapper(const unsigned N , const unsigned graph_num , const unsigned graph_size) {
    unsigned* global_idx , * local_idx , * offset ;
    // unsigned N = 1000000 ;
    cudaMalloc((void**) &global_idx , N * sizeof(unsigned)) ;
    cudaMalloc((void**) &local_idx , N * sizeof(unsigned)) ;
    cudaMalloc((void**) &offset , graph_num * sizeof(unsigned)) ;
    auto cit = thrust::make_counting_iterator<unsigned>(0) ;
    thrust::transform(
        thrust::device , 
        cit , cit + graph_num ,
        offset , 
        [graph_size] __device__ __host__ (unsigned i) {
            return i * graph_size ;
        }
    ) ;
    thrust::transform(
        thrust::device , 
        cit , cit + N ,
        local_idx ,
        [graph_size] __device__ __host__ (unsigned i) {
            return i % graph_size ;
        }
    ) ;
    thrust::sequence(thrust::device , global_idx , global_idx + N) ;
    return {global_idx , local_idx , offset} ;
}

array<unsigned* , 3> get_idx_mapper(const unsigned N , const unsigned graph_num , array<unsigned*,3>& graph_info) {
    unsigned* global_idx , * local_idx , * offset ;
    // unsigned N = 1000000 ;
    cudaMalloc((void**) &global_idx , N * sizeof(unsigned)) ;
    cudaMalloc((void**) &local_idx , N * sizeof(unsigned)) ;
    cudaMalloc((void**) &offset , graph_num * sizeof(unsigned)) ;
    auto cit = thrust::make_counting_iterator<unsigned>(0) ;
    // thrust::transform(
    //     thrust::device , 
    //     cit , cit + graph_num ,
    //     offset , 
    //     [graph_size] __device__ __host__ (unsigned i) {
    //         return i * graph_size ;
    //     }
    // ) ;
    vector<unsigned> offset_host(graph_num) ;
    offset_host[0] = 0 ;
    for(unsigned i = 1 ; i < graph_num ; i ++)
        offset_host[i] = offset_host[i - 1] + graph_info[1][i - 1] ;
    cudaMemcpy(offset , offset_host.data() , graph_num * sizeof(unsigned) , cudaMemcpyHostToDevice) ; 

    cout << "seg_global_idx_start_idx : [" ;
    for(int i = 0 ; i < graph_num ;i  ++) {
        cout << offset_host[i] ;
        if(i < graph_num - 1)
            cout << "," ;
    }
    cout << "]" << endl ;
    // thrust::transform(
    //     thrust::device , 
    //     cit , cit + N ,
    //     local_idx ,
    //     [graph_size] __device__ __host__ (unsigned i) {
    //         return i % graph_size ;
    //     }
    // ) ;
    // cout << "offset host[" ;
    // for(unsigned i = 0 ;i < graph_num ; i ++) {
    //     cout << offset_host[i] ;
    //     if(i < graph_num - 1)
    //         cout << " ," ;
    // }
    // cout << "]" << endl ;
    for(unsigned i = 0 ; i < graph_num ; i ++) {
        // thrust::transform(
        //     thrust::device , 
        //     cit , cit + graph_info[1][i] , 
        //     local_idx + offset_host[i] ,
        // )
        thrust::sequence(thrust::device , local_idx + offset_host[i] , local_idx + offset_host[i] + graph_info[1][i]) ;
    }
    thrust::sequence(thrust::device , global_idx , global_idx + N) ;
    return {global_idx , local_idx , offset} ;
}


float* generate_attrs() {
    unsigned N = 1000000 ;
    float* attrs ;
    generate_2d_attribute_uniform(attrs , N , 4 , 4) ;
    return attrs ;
}

__device__ __forceinline__
unsigned int fast_rand(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

__global__ void sort_test() {
    __shared__ unsigned ids[200] ;
    __shared__ float dis[200] ;

    __shared__ long long start ;
    __shared__ double cost ;
    __shared__ unsigned error_cnt ;
    // __shared__ unsigned flag ; 
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if(tid == 0) {
        cost = 0.0 ;
        error_cnt = 0 ;
        start = clock64() ;
    }
    __syncthreads() ;
    // unsigned error_cnt = 0 ; 
    for(unsigned i = 0 ; i < 1000 ; i ++)  {
        for(unsigned j = tid ; j < 200 ; j += blockDim.x) {
            // unsigned rand_number = fast_rand(tid) % 100;
            ids[j] = j ;
            dis[j] = 200.0 - j * 1.0 ;
        }
        __syncthreads() ;
        bitonic_sort_id_by_dis_no_explore(dis, ids, 200) ;
        __syncthreads() ;
        
        // 检查排序结果是否正确
        if(tid == 0) {
            bool flag = false ; 
            for(unsigned j = 0 ; j < 200 ; j ++) {
                printf("%f" , dis[j]) ;
                if(j < 199)
                    printf(" ") ;
                else 
                    printf("\n") ;
                if(j && dis[j] < dis[j - 1]) {
                    // error_cnt ++ ;
                    flag = true ;
                }

            }
            if(flag)
                error_cnt ++ ;
        }
        __syncthreads() ;
    }
    __syncthreads() ;
    if(tid == 0) {
        cost += ((double)(clock64() - start) / 1582000); ;
        printf("%fms\n" , cost) ;
        printf("排序错误次数: %d\n" , error_cnt) ;
    }
} 
void print_sym() {
    Dl_info info;
    if (dladdr((void*)cudaGetLastError, &info)) {
      printf("cudaGetLastError from: %s\n", info.dli_fname);
    } else {
      printf("dladdr failed\n");
    }
  }

void check_process1(unsigned attr_dim , float* attr_min , float* attr_width , unsigned* attr_grid_size , float* l_bound , float* r_bound , array<unsigned* , 3>& partition_infos
 , unsigned query_num) {
    vector<float> attr_min_host(attr_dim) , attr_width_host(attr_dim) ;
    vector<unsigned> attr_grid_size_host(attr_dim) ;
    thrust::copy(
         thrust::device_pointer_cast(attr_min),
         thrust::device_pointer_cast(attr_min) + attr_dim ,
          attr_min_host.data()) ;
    cout << "程序执行至line:" <<__LINE__ << endl ;
    thrust::copy(
        thrust::device_pointer_cast(attr_width) ,
        thrust::device_pointer_cast(attr_width) + attr_dim ,
         attr_width_host.data()) ;
    cout << "程序执行至line:" <<__LINE__ << endl ;
    thrust::copy(
        thrust::device_pointer_cast(attr_grid_size) ,
        thrust::device_pointer_cast(attr_grid_size) + attr_dim ,
         attr_grid_size_host.data()) ;
    cout << "程序执行至line:" <<__LINE__ << endl ;
    vector<vector<unsigned>> access_order ;
    for(unsigned i = 0 ; i < query_num ; i ++) {
        vector<unsigned> cur_acor ; 
        vector<unsigned> min_indexes(attr_dim) ;
        vector<unsigned> max_indexes(attr_dim) ;
        vector<unsigned> pids(attr_dim) ;
        // cout << "[" ;
        for(unsigned j = 0 ; j < attr_dim ; j ++) {
            min_indexes[j] = (unsigned)((l_bound[i * attr_dim + j] - attr_min_host[j]) / attr_width_host[j]) ;
            max_indexes[j] = (unsigned)((r_bound[i * attr_dim + j] - attr_min_host[j]) / attr_width_host[j]) ;
            if(max_indexes[j] >= attr_grid_size_host[j])
                max_indexes[j] = attr_grid_size_host[j] - 1 ;    
            // partition_indexes[j] = partition_l[j] ;
            pids[j] = min_indexes[j] ;

            // cout << "l :" << l_bound[i * attr_dim + j] << ", r :" << r_bound[i * attr_dim + j] <<
            // ", attr_min : " << attr_min_host[j] << ", attr_width :" << attr_width_host[j] << "-" ;

            // cout << "(" << min_indexes[j] << "," << max_indexes[j] << ")" ;
            // if(j < attr_dim - 1) 
            //     cout << ", " ;
        }
        // cout << "]" << endl ;

        while(true) {
            unsigned pid = get_partition_id_host(pids.data() , attr_grid_size_host.data() , attr_dim) ;
            // is_selected[pid] = true ; 
            cur_acor.push_back(pid) ;
            int pos = 0 ; 
            while(pos < attr_dim) {
                pids[pos] ++ ;
                if(pids[pos] <= max_indexes[pos])
                    break ; 
                else {
                    pids[pos] = min_indexes[pos] ;
                    pos ++ ;
                }
            }
            if(pos >= attr_dim)
                break ; 
        }
        access_order.push_back(cur_acor) ;
    }
    cout << "程序执行至line:" << __LINE__ << endl ;
    cout << "access_order.size() : " << access_order.size() << ", access_order[0].size() : " << access_order[0].size() << endl ;
    int acor_cnt = 0 ; 
    for(unsigned i = 0 ; i < access_order.size() ; i ++)
        acor_cnt += access_order[i].size() ;
    cout << "all cnt : " << acor_cnt << endl ;
    vector<unsigned> forehead20(acor_cnt) ;
    thrust::copy(thrust::device_pointer_cast(partition_infos[0]) , thrust::device_pointer_cast(partition_infos[0]) + acor_cnt , forehead20.data()) ;
    cout << "程序执行至line:" << __LINE__ << endl ;
    unsigned offset = 0 , iidx = 0 ;
    unsigned corr = 0 , num_corr = 0 ;
    for(unsigned i = 0 ; i < acor_cnt ; i ++) {
        // corr += (forehead20[i] - access_order[iidx][offset]) * (forehead20[i] - access_order[iidx][offset]) ;
        corr += (forehead20[i] != access_order[iidx][offset]) ;
        cout << forehead20[i] << ":" << access_order[iidx][offset] << endl ;
        offset ++ ;
        if(offset >= access_order[iidx].size())
            offset = 0 ,  iidx ++ ;
    }
    cout << "--------------------------------------" << endl ;

    vector<unsigned> second_arr(query_num) ;
    thrust::copy(
        thrust::device_pointer_cast(partition_infos[1]) ,
        thrust::device_pointer_cast(partition_infos[1]) + query_num ,
        second_arr.data()
    ) ;
    for(unsigned i = 0 ; i < query_num ;i ++)
        // num_corr += (second_arr[i] - access_order[i].size()) * (second_arr[i] - access_order[i].size()) ;
        num_corr += (second_arr[i] != access_order[i].size()) ;

    cout << "方差:" << corr << endl ;
    cout << "数量方差:" << num_corr << endl ;

    vector<unsigned> offsets_host(query_num) ; 
    thrust::copy(
        thrust::device_pointer_cast(partition_infos[2]) ,
        thrust::device_pointer_cast(partition_infos[2] + query_num) ,
        offsets_host.data() 
    ) ;
    for(unsigned i = 0 ; i < query_num ; i ++)
        cout << offsets_host[i] << ", " ;
    cout << endl ;
}


void check_process2(unsigned attr_dim , float* attr_min , float* attr_width , unsigned* attr_grid_size , float* l_bound , float* r_bound , array<unsigned* , 5>& partition_infos
    , unsigned query_num , float* query_data_load , unsigned dim , float* data_) {
       vector<float> attr_min_host(attr_dim) , attr_width_host(attr_dim) ;
       vector<unsigned> attr_grid_size_host(attr_dim) ;
       thrust::copy(
            thrust::device_pointer_cast(attr_min),
            thrust::device_pointer_cast(attr_min) + attr_dim ,
             attr_min_host.data()) ;
       cout << "程序执行至line:" <<__LINE__ << endl ;
       thrust::copy(
           thrust::device_pointer_cast(attr_width) ,
           thrust::device_pointer_cast(attr_width) + attr_dim ,
            attr_width_host.data()) ;
       cout << "程序执行至line:" <<__LINE__ << endl ;
       thrust::copy(
           thrust::device_pointer_cast(attr_grid_size) ,
           thrust::device_pointer_cast(attr_grid_size) + attr_dim ,
            attr_grid_size_host.data()) ;
       cout << "程序执行至line:" <<__LINE__ << endl ;
       vector<vector<unsigned>> access_order ;
       for(unsigned i = 0 ; i < query_num ; i ++) {
           vector<unsigned> cur_acor ; 
           vector<unsigned> min_indexes(attr_dim) ;
           vector<unsigned> max_indexes(attr_dim) ;
           vector<unsigned> pids(attr_dim) ;
           // cout << "[" ;
           for(unsigned j = 0 ; j < attr_dim ; j ++) {
               min_indexes[j] = (unsigned)((l_bound[i * attr_dim + j] - attr_min_host[j]) / attr_width_host[j]) ;
               max_indexes[j] = (unsigned)((r_bound[i * attr_dim + j] - attr_min_host[j]) / attr_width_host[j]) ;
               if(max_indexes[j] >= attr_grid_size_host[j])
                   max_indexes[j] = attr_grid_size_host[j] - 1 ;    
               // partition_indexes[j] = partition_l[j] ;
               pids[j] = min_indexes[j] ;
   
               // cout << "l :" << l_bound[i * attr_dim + j] << ", r :" << r_bound[i * attr_dim + j] <<
               // ", attr_min : " << attr_min_host[j] << ", attr_width :" << attr_width_host[j] << "-" ;
   
               // cout << "(" << min_indexes[j] << "," << max_indexes[j] << ")" ;
               // if(j < attr_dim - 1) 
               //     cout << ", " ;
           }
           // cout << "]" << endl ;
   
           while(true) {
               unsigned pid = get_partition_id_host(pids.data() , attr_grid_size_host.data() , attr_dim) ;
               // is_selected[pid] = true ; 
               cur_acor.push_back(pid) ;
               int pos = 0 ; 
               while(pos < attr_dim) {
                   pids[pos] ++ ;
                   if(pids[pos] <= max_indexes[pos])
                       break ; 
                   else {
                       pids[pos] = min_indexes[pos] ;
                       pos ++ ;
                   }
               }
               if(pos >= attr_dim)
                   break ; 
           }
           access_order.push_back(cur_acor) ;
       }
       // 对分区排序
        vector<unsigned> cluster_id ;
        float* cluster_centers;
        vector<vector<unsigned>> num_c_p ;
        load_clustering(cluster_id , "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/clustering1000.bin" , cluster_centers , dim , num_c_p , data_) ;
        vector<vector<unsigned>> access_order_by_cluster_host(query_num) ;
        for(unsigned i = 0 ; i < query_num ; i ++)
            access_order_by_cluster_host[i] = generate_access_order_by_cluster(access_order[i] , query_data_load + i * dim,  cluster_centers , dim , num_c_p , 4)  ;
        access_order = access_order_by_cluster_host ;

       cout << "程序执行至line:" << __LINE__ << endl ;
       cout << "access_order.size() : " << access_order.size() << ", access_order[0].size() : " << access_order[0].size() << endl ;
       int acor_cnt = 0 ; 
       for(unsigned i = 0 ; i < access_order.size() ; i ++)
           acor_cnt += access_order[i].size() ;
       cout << "all cnt : " << acor_cnt << endl ;
       vector<unsigned> forehead20(acor_cnt) ;
       thrust::copy(thrust::device_pointer_cast(partition_infos[0]) , thrust::device_pointer_cast(partition_infos[0]) + acor_cnt , forehead20.data()) ;
       cout << "程序执行至line:" << __LINE__ << endl ;
   
       vector<unsigned> second_arr(query_num) ;
       thrust::copy(
           thrust::device_pointer_cast(partition_infos[1]) ,
           thrust::device_pointer_cast(partition_infos[1]) + query_num ,
           second_arr.data()
       ) ;
   
       vector<unsigned> offsets_host(query_num) ; 
       thrust::copy(
           thrust::device_pointer_cast(partition_infos[2]) ,
           thrust::device_pointer_cast(partition_infos[2] + query_num) ,
           offsets_host.data() 
       ) ;

       vector<unsigned> little_seg_counts_host(query_num) ;
       thrust::copy(
        thrust::device_pointer_cast(partition_infos[3]) ,
        thrust::device_pointer_cast(partition_infos[3]) + query_num ,
        little_seg_counts_host.data() 
       ) ;

       bool * is_little_seg_host = new bool[acor_cnt] ;
       thrust::copy(
        thrust::device_pointer_cast(partition_infos[4]) ,
        thrust::device_pointer_cast(partition_infos[4]) + acor_cnt ,
        is_little_seg_host 
       ) ;

    //    for(int i = 0 ; i < acor_cnt ; i ++)
    //     cout << is_little_seg_host[i] << ", " ;
       int ssss ;
       cin >> ssss;
       // 验证是否正确
       for(unsigned i = 0 ;i < query_num ;i ++) {
            int lscnt = little_seg_counts_host[i] ;
            int ofst = offsets_host[i] ;
            int len = second_arr[i] ;
            
            cout << "q" << i << "[" ;
            for(unsigned j = 0 ; j < access_order[i].size() ; j ++) {
                cout << access_order[i][j] ;
                if(j < access_order[i].size() - 1)
                    cout << ", " ;
            }
            cout << "]" << endl ;

            cout << "q" << i << "----" << lscnt << "[" ;
            for(unsigned j = 0 ; j < len ; j ++) {
                float area = 1.0f ;
                int idd = (int) forehead20[ofst + j] ;
                // cout << idd << "----" ;
            // 计算得出当前分区与查询的交叉面积
                for(unsigned k = 0 ; k < attr_dim ;  k++) {
                    float step = (float) (k == 0 ? idd / 4 : idd % 4) ;
                    // 若在左界的右边或者在
                    float lb = max(attr_min_host[k] + step * attr_width_host[k] , l_bound[i * attr_dim + k]) ;
                    float rb = min(attr_min_host[k] + (step + 1) * attr_width_host[k] , r_bound[i * attr_dim + k]) ;
                    area *= (rb - lb) / attr_width_host[k] ;
                    // cout << "l : " << l_bound[i * attr_dim + k] << ", r : " << r_bound[i * attr_dim + k] << "," ;
                    // cout << "lb: " << lb << ", rb : " << rb  << ", attr_width : " << attr_width_host[k] << " , " ;
                }
                // cout << area ;
                cout << idd ;
                if(len - j - 1 == lscnt)
                    cout << "; " ;
                else if(j < len - 1)
                    cout << ", " ;
            }
            cout << "]" << endl ;
       }
       int t ;
       cin >> t; 

}



float rc_recall1 , rc_recall2 , rc_cost1 , rc_cost2 ;

float selectivity ;
void check_all_process(unsigned L = 80) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    array<unsigned*,3> graph_infos = get_graph_info() ;
    array<unsigned*,3> idx_mapper = get_idx_mapper() ;
    vector<vector<unsigned>> global_edges ;
    load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges);
    unsigned *global_graph ;
    cudaMalloc((void**) &global_graph , 1000000 * 16 * 16 * sizeof(unsigned)) ;

    for(unsigned i = 0 ; i < 1000000 ; i ++) {
        cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    }
    float *attrs_dev , *attrs ;
    attrs = generate_attrs() ;
    cudaMalloc((void**) &attrs_dev , 2 * 1000000 * sizeof(float)) ;
    cudaMemcpy(attrs_dev , attrs , 2 * 1000000 * sizeof(float) , cudaMemcpyHostToDevice) ;

    cout << "sizeof(half) = " << sizeof(half) << endl ;

    // 先载入数据集和图索引
    unsigned num , dim  ;
    float* data_  ;
    load_data("/home/yhr/workspace/cuda_clustering/data/sift_base.fvecs" , data_ , num , dim) ;
    cout << "num : " << num << ", dim : " << dim << endl ;
    float* query_data_load ;
    unsigned query_num , query_dim ;
    load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_range_query.fvecs" , query_data_load , query_num , query_dim) ;
    // query_num = 500; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;
    // query_num *= 10 ;
    float* tmp_query_data_load = new float[10 *query_num * query_dim] ;
    for(int i = 0 ; i < 10 ; i ++)
        memcpy(tmp_query_data_load + i * query_num * query_dim  , query_data_load , query_num * query_dim * sizeof(float)) ;
    query_num *= 10 ;
    delete [] query_data_load ;
    query_data_load = tmp_query_data_load ;
    // for(unsigned i = 0 ; i )
    // TOPM = 20 , DIM = dim ; 
    // ************** 若要修改查询数量, 请修改如下变量 **********************
    // query_num = 1000 ;
    // for(int i = 0 ; i < dim ; i ++)
    //     query_data_load[i] = query_data_load[613 * dim + i] ;
    float* data_dev ;
    half *data_half_dev ;
    float* query_data_dev ;
    half *query_data_half_dev ;
    cudaMalloc((void**)&data_dev , num * dim * sizeof(float)) ;
    cudaMalloc((void**) &data_half_dev , num * dim * sizeof(half)) ;
    cudaMalloc((void**) &query_data_dev , query_num * query_dim   * sizeof(float)) ;
    unsigned query_num_with_pad = (query_num + 15) / 16 * 16 ;
    cout << "query_num_with_pad : " << query_num_with_pad << endl ;
    cudaMalloc((void**) &query_data_half_dev , query_num_with_pad * query_dim * sizeof(half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(data_dev , data_ , num * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(query_data_dev , query_data_load , query_num * query_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num ; 
	// 调用实例
	f2h<<<points_num, 256>>>(data_dev, data_half_dev, points_num); // 先将数据转换成fp16
    // cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    f2h<<<query_num, 256>>>(query_data_dev, query_data_half_dev, query_num); // 将查询转换成fp16
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    if(query_num != query_num_with_pad)
        thrust::fill(thrust::device , query_data_half_dev + query_num * query_dim , query_data_half_dev + query_num_with_pad * query_dim , __float2half(0.0f)) ;

    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    if(selectivity == 0.0) {
        load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        query_range_x.reserve(query_num) ;
        for(unsigned i = 1 ; i < 10 ; i ++)
            query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        query_range_y = query_range_x ; 

        // query_range_y[0] = query_range_x[0] = {490000 , 510000} ;
    
        // query_num = 1; 
        // query_range_x[0].first = 250000 , query_range_x[0].second = 499999 ;
        // query_range_y[0].first = 250000 , query_range_y[0].second = 499999 ;
        // for(unsigned i = 0 ; i < query_num ;i ++) {
        //     query_range_x[i] = query_range_x.front() ;
        //     query_range_y[i] = query_range_y.front() ;
        // }
    } else {
        cout << "selectivity : " << selectivity << endl ;
        generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_x , query_num) ;
        generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_y , query_num) ;
    }
    // query_range_x[0].first = 450000 , query_range_x[0].second = 850000 ;
    // cout << "query range x 0 : " << query_range_x[0].first << "," << query_range_x[0].second << endl ;
    

    // for(unsigned i = 1 ; i < query_num ; i ++)
    //     query_range_x[i] = query_range_x[0] ;

    // query_range_x[0] = query_range_x[1] ;
    // for(unsigned i = 0 ; i < query_num ;i ++) {
    //     cout << "query range"<< i << ":" << query_range_x[i].first << "," << query_range_x[i].second << endl ;

    // }
    // for(unsigned i = 1 ; i < 10 ; i ++)
    //     query_range_x[i] = query_range_x[0] ;

    unsigned attr_dim =  2 ;

    float* l_bound , * r_bound , *l_bound_dev , *r_bound_dev , *attr_min , *attr_width ;
    unsigned *attr_grid_size ;
    l_bound = new float[query_num * attr_dim] , r_bound = new float[query_num * attr_dim] ;
    for(unsigned i = 0 ; i < query_num ; i ++) {
        // for(unsigned j = 0 ;j  < attr_dim ; j ++)
            l_bound[i * attr_dim] = (float) query_range_x[i].first , r_bound[i * attr_dim] = (float) query_range_x[i].second ; 
            l_bound[i * attr_dim + 1] = (float) query_range_y[i].first , r_bound[i * attr_dim + 1] = (float) query_range_y[i].second ;
    }
    cudaMalloc((void**) &l_bound_dev , attr_dim * query_num * sizeof(float)) ;
    cudaMalloc((void**) &r_bound_dev , attr_dim * query_num * sizeof(float)) ;
    cudaMalloc((void**) &attr_min , attr_dim * sizeof(float)) ;
    cudaMalloc((void**) &attr_width , attr_dim * sizeof(float)) ;
    cudaMalloc((void**) &attr_grid_size , attr_dim * sizeof(unsigned)) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    cudaMemcpy(l_bound_dev , l_bound , query_num * attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(r_bound_dev , r_bound , query_num * attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    thrust::fill(thrust::device , attr_min , attr_min + attr_dim , 0.0) ;
    thrust::fill(thrust::device , attr_width , attr_width + attr_dim , 250000.0) ;
    thrust::fill(thrust::device , attr_grid_size , attr_grid_size + attr_dim , 4) ;
    unsigned qnum = query_num ;
    unsigned pnum = thrust::reduce(
        thrust::device , 
        attr_grid_size , attr_grid_size + attr_dim ,
        1 , thrust::multiplies<unsigned>() 
    ) ;
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , 0.05) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    vector<unsigned> cluster_id ;
    float* cluster_centers;
    vector<vector<unsigned>> num_c_p ;
    load_clustering(cluster_id , "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/clustering1000.bin" , cluster_centers , dim , num_c_p , data_) ;
    float* cluster_centers_dev ;
    unsigned* cluster_p_info ;
    __half* cluster_centers_half_dev ;
    unsigned cnum = num_c_p[0].size() ;
    cout << "聚类个数:" << cnum << endl ;
    cudaMalloc((void**) &cluster_centers_dev , cnum * dim * sizeof(float)) ;
    cudaMalloc((void**) &cluster_p_info, cnum * pnum * sizeof(unsigned)) ;
    unsigned cnum_with_pad = (cnum + 15) / 16 * 16 ;
    cout << "cnum_with_pad : " << cnum_with_pad << endl ;
    cudaMalloc((void**) &cluster_centers_half_dev , cnum_with_pad * dim * sizeof(__half)) ;
    cudaMemcpy(cluster_centers_dev , cluster_centers , cnum * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    f2h<<<cnum, 256>>>(cluster_centers_dev, cluster_centers_half_dev, cnum);
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    if(cnum != cnum_with_pad)
        thrust::fill(thrust::device , cluster_centers_half_dev + cnum * dim , cluster_centers_half_dev + cnum_with_pad * dim , __float2half(0.0f)) ;
    // 在此处计算得到q_norm 和 c_norm 
    // ***************************************************
    float *q_norm , *c_norm ;
    cudaMalloc((void**) &q_norm , query_num_with_pad * sizeof(float)) ;
    cudaMalloc((void**) &c_norm , cnum_with_pad * sizeof(float)) ;
    find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    find_norm<<<(cnum_with_pad + 3) / 4 , dim3(32 , 4 , 1)>>>(cluster_centers_half_dev , c_norm , dim , cnum_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__ ) ;

    // vector<unsigned> info1_host(qnum) , info2_host(qnum) ;
    // cudaMemcpy(info1_host.data() , partition_infos[1] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // cudaMemcpy(info2_host.data() , partition_infos[2] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // for(unsigned i = 0 ; i < qnum ;i  ++) {
    //     cout << i << ": size-" << info1_host[i] << ", offset-" << info2_host[i] << endl ; 
    // }

    // 检查norm的计算是否正确
    /**
        vector<float> q_norm_host(query_num_with_pad) , c_norm_host(cnum_with_pad) ;
        cudaMemcpy(q_norm_host.data() , q_norm , query_num_with_pad * sizeof(float) , cudaMemcpyDeviceToHost) ;
        cudaMemcpy(c_norm_host.data() , c_norm , cnum_with_pad * sizeof(float) , cudaMemcpyDeviceToHost) ;
        vector<float> q_norm_gt(query_num_with_pad) , c_norm_gt(cnum_with_pad) ;
        for(unsigned i = 0 ; i < query_num ; i++) {
            float dis = 0.0 ;
            for(unsigned d = 0; d < dim ;  d++)
                dis += query_data_load[i * dim + d] * query_data_load[i * dim + d] ;
            q_norm_gt[i] = dis / 10000.0f ;
        }
        for(unsigned i = 0 ; i < cnum ; i ++) {
            float dis = 0.0 ;
            for(unsigned d = 0 ; d < dim ;  d++)
                dis += (cluster_centers[i * dim + d] / 100.0f )* (cluster_centers[i * dim + d] / 100.0f );
            c_norm_gt[i] = dis ;
        }
        float abs_difference = 0.0f ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // printf("%f : %f, " , q_norm_host[i] , q_norm_gt[i]) ;
            abs_difference += abs(q_norm_host[i] - q_norm_gt[i]) ;
            printf("%f, " , abs(q_norm_host[i] - q_norm_gt[i])) ;
        }
        for(unsigned i = 0 ; i < cnum ; i ++)  {
            // printf("%f : %f, " , c_norm_host[i] , c_norm_gt[i]) ; 
            abs_difference += abs(c_norm_host[i] - c_norm_gt[i]) ;
            printf("%f, " , abs(c_norm_host[i] - c_norm_gt[i])) ; 
        }
        cout << "误差:" << abs_difference << endl ;
    **/
    // ***************************************************


    // ******************************
    // 此处把 聚类中心矩阵做转置
    // ******************************
    // __half* cluster_centers_half_dev_T ;
    // cudaMalloc((void**) &cluster_centers_half_dev_T , cnum_with_pad * dim *  sizeof(__half)) ;
    // matrix_transpose<<<cnum_with_pad , 32>>>(cluster_centers_half_dev , cluster_centers_half_dev_T , cnum_with_pad, dim) ;
    // cudaDeviceSynchronize() ;
    // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // ******************************

    unsigned num_c_p_offset = 0 ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cudaMemcpy(cluster_p_info + i * cnum , num_c_p[i].data() , cnum * sizeof(unsigned) , cudaMemcpyHostToDevice ) ;
    }

    cout << partition_infos[0] << "," <<  partition_infos[1] << "," <<  partition_infos[2] << endl ;
    // cnum = 16 , qnum = 16 ;
    s = chrono::high_resolution_clock::now() ;
    get_partitions_access_order_V2(partition_infos , pnum , cnum , qnum , cluster_p_info , 4 , cluster_centers_half_dev , query_data_half_dev , dim , query_data_load , cluster_centers , q_norm , c_norm , num)  ;
    e = chrono::high_resolution_clock::now() ;
    cost2 = chrono::duration<double>(e - s).count() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // check_process2(attr_dim , attr_min ,attr_width ,attr_grid_size ,l_bound ,r_bound ,partition_infos , query_num , query_data_load , dim , data_) ;
 
    s = chrono::high_resolution_clock::now() ;
    vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
        attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
        global_graph , 16 , pnum , L) ;
    e = chrono::high_resolution_clock::now() ;
    cost3 = chrono::duration<double>(e - s).count() ;
    cout << "finish search" << endl ;

    // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
    vector<vector<unsigned>> ground_truth(qnum) ;
    
    #pragma omp parallel for num_threads(10)
    for(unsigned i = 0 ; i < qnum ;i  ++)
        ground_truth[i] = naive_brute_force_with_filter(query_data_load + i * dim , data_ , dim , num , 10 ,
            l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim) ;
    float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
    cout << "recall : " << total_recall << endl  ;

    cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
    cout << "获取分区列表用时:" << cost1 << endl ;
    cout << "对分区列表排序用时:" << cost2 << endl ;
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "搜索用时:" << cost3 << endl ;

    rc_recall1 = total_recall ;
    rc_cost1 = select_path_kernal_cost ;

    cudaFree(global_graph) ;
    cudaFree(attrs_dev) ;
    cudaFree(data_dev) ;
    cudaFree(data_half_dev) ;
    cudaFree(query_data_dev) ;
    cudaFree(query_data_half_dev) ;
    cudaFree(l_bound_dev) ;
    cudaFree(r_bound_dev) ;
    cudaFree(attr_min) ;
    cudaFree(attr_width) ;
    cudaFree(attr_grid_size) ;
    cudaFree(cluster_centers_dev) ;
    cudaFree(cluster_p_info) ;
    cudaFree(cluster_centers_half_dev) ;
    cudaFree(q_norm) ;
    cudaFree(c_norm) ;
}

void check_all_process_version0(unsigned L = 80) {
    // L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    array<unsigned*,3> graph_infos = get_graph_info() ;
    array<unsigned*,3> idx_mapper = get_idx_mapper() ;
    vector<vector<unsigned>> global_edges ;
    load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges);
    unsigned *global_graph ;
    cudaMalloc((void**) &global_graph , 1000000 * 16 * 16 * sizeof(unsigned)) ;
    for(unsigned i = 0 ; i < 1000000 ; i ++) {
        cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    }
    float *attrs_dev , *attrs ;
    attrs = generate_attrs() ;
    cudaMalloc((void**) &attrs_dev , 2 * 1000000 * sizeof(float)) ;
    cudaMemcpy(attrs_dev , attrs , 2 * 1000000 * sizeof(float) , cudaMemcpyHostToDevice) ;

    cout << "sizeof(half) = " << sizeof(half) << endl ;

    // 先载入数据集和图索引
    unsigned num , dim  ;
    float* data_  ;
    load_data("/home/yhr/workspace/cuda_clustering/data/sift_base.fvecs" , data_ , num , dim) ;
    cout << "num : " << num << ", dim : " << dim << endl ;
    float* query_data_load ;
    unsigned query_num , query_dim ;
    load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_range_query.fvecs" , query_data_load , query_num , query_dim) ;
    // query_num = 500; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;
    // query_num *= 10 ;
    float* tmp_query_data_load = new float[10 *query_num * query_dim] ;
    for(int i = 0 ; i < 10 ; i ++)
        memcpy(tmp_query_data_load + i * query_num * query_dim  , query_data_load , query_num * query_dim * sizeof(float)) ;
    query_num *= 10 ;
    delete [] query_data_load ;
    query_data_load = tmp_query_data_load ;
    // for(unsigned i = 0 ; i )
    // TOPM = 20 , DIM = dim ; 
    // ************** 若要修改查询数量, 请修改如下变量 **********************
    // query_num = 100 ;
    // for(int i = 0 ; i < dim ; i ++)
    //     query_data_load[i] = query_data_load[613 * dim + i] ;

    float* data_dev ;
    half *data_half_dev ;
    float* query_data_dev ;
    half *query_data_half_dev ;
    cudaMalloc((void**)&data_dev , num * dim * sizeof(float)) ;
    cudaMalloc((void**) &data_half_dev , num * dim * sizeof(half)) ;
    cudaMalloc((void**) &query_data_dev , query_num * query_dim   * sizeof(float)) ;
    unsigned query_num_with_pad = (query_num + 15) / 16 * 16 ;
    cout << "query_num_with_pad : " << query_num_with_pad << endl ;
    cudaMalloc((void**) &query_data_half_dev , query_num_with_pad * query_dim * sizeof(half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(data_dev , data_ , num * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(query_data_dev , query_data_load , query_num * query_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num ; 
	// 调用实例
	f2h<<<points_num, 256>>>(data_dev, data_half_dev, points_num); // 先将数据转换成fp16
    // cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    f2h<<<query_num, 256>>>(query_data_dev, query_data_half_dev, query_num); // 将查询转换成fp16
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    if(query_num != query_num_with_pad)
        thrust::fill(thrust::device , query_data_half_dev + query_num * query_dim , query_data_half_dev + query_num_with_pad * query_dim , __float2half(0.0f)) ;

    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    if(selectivity == 0.0) {
        load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        query_range_x.reserve(query_num) ;
        for(unsigned i = 1 ; i < 10 ; i ++)
            query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        query_range_y = query_range_x ; 

        // query_range_y[0] = query_range_x[0] = {490000 , 510000} ;
        
        // query_num = 1; 
        // query_range_x[0].first = 250000 , query_range_x[0].second = 499999 ;
        // query_range_y[0].first = 250000 , query_range_y[0].second = 499999 ;
        // for(unsigned i = 0 ; i < query_num ;i ++) {
        //     query_range_x[i] = query_range_x.front() ;
        //     query_range_y[i] = query_range_y.front() ;
        // }
    } else {
        cout << "selectivity : " << selectivity << endl ;
        generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_x , query_num) ;
        generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_y , query_num) ;
    }
    // query_range_x[0].first = 450000 , query_range_x[0].second = 850000 ;
    // cout << "query range x 0 : " << query_range_x[0].first << "," << query_range_x[0].second << endl ;
    

    // for(unsigned i = 1 ; i < query_num ; i ++)
    //     query_range_x[i] = query_range_x[0] ;

    // query_range_x[0] = query_range_x[1] ;
    // for(unsigned i = 0 ; i < query_num ;i ++) {
    //     cout << "query range"<< i << ":" << query_range_x[i].first << "," << query_range_x[i].second << endl ;

    // }
    // for(unsigned i = 1 ; i < 10 ; i ++)
    //     query_range_x[i] = query_range_x[0] ;

    unsigned attr_dim =  2 ;

    float* l_bound , * r_bound , *l_bound_dev , *r_bound_dev , *attr_min , *attr_width ;
    unsigned *attr_grid_size ;
    l_bound = new float[query_num * attr_dim] , r_bound = new float[query_num * attr_dim] ;
    for(unsigned i = 0 ; i < query_num ; i ++) {
        // for(unsigned j = 0 ;j  < attr_dim ; j ++)
            l_bound[i * attr_dim] = (float) query_range_x[i].first , r_bound[i * attr_dim] = (float) query_range_x[i].second ; 
            l_bound[i * attr_dim + 1] = (float) query_range_y[i].first , r_bound[i * attr_dim + 1] = (float) query_range_y[i].second ;
    }
    cudaMalloc((void**) &l_bound_dev , attr_dim * query_num * sizeof(float)) ;
    cudaMalloc((void**) &r_bound_dev , attr_dim * query_num * sizeof(float)) ;
    cudaMalloc((void**) &attr_min , attr_dim * sizeof(float)) ;
    cudaMalloc((void**) &attr_width , attr_dim * sizeof(float)) ;
    cudaMalloc((void**) &attr_grid_size , attr_dim * sizeof(unsigned)) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    cudaMemcpy(l_bound_dev , l_bound , query_num * attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(r_bound_dev , r_bound , query_num * attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    thrust::fill(thrust::device , attr_min , attr_min + attr_dim , 0.0) ;
    thrust::fill(thrust::device , attr_width , attr_width + attr_dim , 250000.0) ;
    thrust::fill(thrust::device , attr_grid_size , attr_grid_size + attr_dim , 4) ;
    unsigned qnum = query_num ;
    unsigned pnum = thrust::reduce(
        thrust::device , 
        attr_grid_size , attr_grid_size + attr_dim ,
        1 , thrust::multiplies<unsigned>() 
    ) ;
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 3> partition_infos = intersection_area(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    vector<unsigned> cluster_id ;
    float* cluster_centers;
    vector<vector<unsigned>> num_c_p ;
    load_clustering(cluster_id , "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/clustering1000.bin" , cluster_centers , dim , num_c_p , data_) ;
    float* cluster_centers_dev ;
    unsigned* cluster_p_info ;
    __half* cluster_centers_half_dev ;
    unsigned cnum = num_c_p[0].size() ;
    cout << "聚类个数:" << cnum << endl ;
    cudaMalloc((void**) &cluster_centers_dev , cnum * dim * sizeof(float)) ;
    cudaMalloc((void**) &cluster_p_info, cnum * pnum * sizeof(unsigned)) ;
    unsigned cnum_with_pad = (cnum + 15) / 16 * 16 ;
    cout << "cnum_with_pad : " << cnum_with_pad << endl ;
    cudaMalloc((void**) &cluster_centers_half_dev , cnum_with_pad * dim * sizeof(__half)) ;
    cudaMemcpy(cluster_centers_dev , cluster_centers , cnum * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    f2h<<<cnum, 256>>>(cluster_centers_dev, cluster_centers_half_dev, cnum);
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    if(cnum != cnum_with_pad)
        thrust::fill(thrust::device , cluster_centers_half_dev + cnum * dim , cluster_centers_half_dev + cnum_with_pad * dim , __float2half(0.0f)) ;
    // 在此处计算得到q_norm 和 c_norm 
    // ***************************************************
    float *q_norm , *c_norm ;
    cudaMalloc((void**) &q_norm , query_num_with_pad * sizeof(float)) ;
    cudaMalloc((void**) &c_norm , cnum_with_pad * sizeof(float)) ;
    find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    find_norm<<<(cnum_with_pad + 3) / 4 , dim3(32 , 4 , 1)>>>(cluster_centers_half_dev , c_norm , dim , cnum_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__ ) ;
    // vector<unsigned> info1_host(qnum) , info2_host(qnum) ;
    // cudaMemcpy(info1_host.data() , partition_infos[1] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // cudaMemcpy(info2_host.data() , partition_infos[2] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // for(unsigned i = 0 ; i < qnum ;i  ++) {
    //     cout << i << ": size-" << info1_host[i] << ", offset-" << info2_host[i] << endl ; 
    // }

    // 检查norm的计算是否正确
    /**
        vector<float> q_norm_host(query_num_with_pad) , c_norm_host(cnum_with_pad) ;
        cudaMemcpy(q_norm_host.data() , q_norm , query_num_with_pad * sizeof(float) , cudaMemcpyDeviceToHost) ;
        cudaMemcpy(c_norm_host.data() , c_norm , cnum_with_pad * sizeof(float) , cudaMemcpyDeviceToHost) ;
        vector<float> q_norm_gt(query_num_with_pad) , c_norm_gt(cnum_with_pad) ;
        for(unsigned i = 0 ; i < query_num ; i++) {
            float dis = 0.0 ;
            for(unsigned d = 0; d < dim ;  d++)
                dis += query_data_load[i * dim + d] * query_data_load[i * dim + d] ;
            q_norm_gt[i] = dis / 10000.0f ;
        }
        for(unsigned i = 0 ; i < cnum ; i ++) {
            float dis = 0.0 ;
            for(unsigned d = 0 ; d < dim ;  d++)
                dis += (cluster_centers[i * dim + d] / 100.0f )* (cluster_centers[i * dim + d] / 100.0f );
            c_norm_gt[i] = dis ;
        }
        float abs_difference = 0.0f ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // printf("%f : %f, " , q_norm_host[i] , q_norm_gt[i]) ;
            abs_difference += abs(q_norm_host[i] - q_norm_gt[i]) ;
            printf("%f, " , abs(q_norm_host[i] - q_norm_gt[i])) ;
        }
        for(unsigned i = 0 ; i < cnum ; i ++)  {
            // printf("%f : %f, " , c_norm_host[i] , c_norm_gt[i]) ; 
            abs_difference += abs(c_norm_host[i] - c_norm_gt[i]) ;
            printf("%f, " , abs(c_norm_host[i] - c_norm_gt[i])) ; 
        }
        cout << "误差:" << abs_difference << endl ;
    **/
    // ***************************************************


    // ******************************
    // 此处把 聚类中心矩阵做转置
    // ******************************
    // __half* cluster_centers_half_dev_T ;
    // cudaMalloc((void**) &cluster_centers_half_dev_T , cnum_with_pad * dim *  sizeof(__half)) ;
    // matrix_transpose<<<cnum_with_pad , 32>>>(cluster_centers_half_dev , cluster_centers_half_dev_T , cnum_with_pad, dim) ;
    // cudaDeviceSynchronize() ;
    // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // ******************************

    unsigned num_c_p_offset = 0 ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cudaMemcpy(cluster_p_info + i * cnum , num_c_p[i].data() , cnum * sizeof(unsigned) , cudaMemcpyHostToDevice ) ;
    }

    cout << partition_infos[0] << "," <<  partition_infos[1] << "," <<  partition_infos[2] << endl ;
    // cnum = 16 , qnum = 16 ;
    s = chrono::high_resolution_clock::now() ;
    get_partitions_access_order(partition_infos , pnum , cnum , qnum , cluster_p_info , 4 , cluster_centers_half_dev , query_data_half_dev , dim , query_data_load , cluster_centers , q_norm , c_norm)  ;
    e = chrono::high_resolution_clock::now() ;
    cost2 = chrono::duration<double>(e - s).count() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // check_process2(attr_dim , attr_min ,attr_width ,attr_grid_size ,l_bound ,r_bound ,partition_infos , query_num , query_data_load , dim , data_) ;
 
    s = chrono::high_resolution_clock::now() ;
    vector<vector<unsigned>> result = batch_graph_search_gpu(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
        attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
        global_graph , 16 , pnum , L) ;
    e = chrono::high_resolution_clock::now() ;
    cost3 = chrono::duration<double>(e - s).count() ;
    cout << "finish search" << endl ;

    // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
    vector<vector<unsigned>> ground_truth(qnum) ;
    
    #pragma omp parallel for num_threads(10)
    for(unsigned i = 0 ; i < qnum ;i  ++)
        ground_truth[i] = naive_brute_force_with_filter(query_data_load + i * dim , data_ , dim , num , 10 ,
            l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim) ;
    float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
    cout << "recall : " << total_recall << endl  ;

    cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
    cout << "获取分区列表用时:" << cost1 << endl ;
    cout << "对分区列表排序用时:" << cost2 << endl ;
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "搜索用时:" << cost3 << endl ;

    rc_recall1 = total_recall ;
    rc_cost1 =  select_path_kernal_cost ;

    cudaFree(global_graph) ;
    cudaFree(attrs_dev) ;
    cudaFree(data_dev) ;
    cudaFree(data_half_dev) ;
    cudaFree(query_data_dev) ;
    cudaFree(query_data_half_dev) ;
    cudaFree(l_bound_dev) ;
    cudaFree(r_bound_dev) ;
    cudaFree(attr_min) ;
    cudaFree(attr_width) ;
    cudaFree(attr_grid_size) ;
    cudaFree(cluster_centers_dev) ;
    cudaFree(cluster_p_info) ;
    cudaFree(cluster_centers_half_dev) ;
    cudaFree(q_norm) ;
    cudaFree(c_norm) ;
}

void multi_attr_test(unsigned L) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    array<unsigned*,3> graph_infos = get_graph_info() ;
    array<unsigned*,3> idx_mapper = get_idx_mapper() ;
    vector<vector<unsigned>> global_edges ;
    load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges);
    unsigned *global_graph ;
    cudaMalloc((void**) &global_graph , 1000000 * 16 * 16 * sizeof(unsigned)) ;

    for(unsigned i = 0 ; i < 1000000 ; i ++) {
        cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    }
    float *attrs_dev , *attrs ;
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    constexpr unsigned attr_dim =  2 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<float> grid_min(attr_dim , 0.0f) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<float> grid_max(attr_dim , 1000000.0f) ;
    vector<unsigned> grid_size_host = {4 , 4} ;
    vector<float> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++)
        attr_width_host[i] = (grid_max[i] - grid_min[i]) / grid_size_host[i] ;
    attrs = new float[attr_dim * 1000000] ;
    re_generate(attrs , attr_dim , 1000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    cudaMalloc((void**) &attrs_dev , attr_dim * 1000000 * sizeof(float)) ;
    cudaMemcpy(attrs_dev , attrs , attr_dim * 1000000 * sizeof(float) , cudaMemcpyHostToDevice) ;

    cout << "sizeof(half) = " << sizeof(half) << endl ;

    // 先载入数据集和图索引
    unsigned num , dim  ;
    float* data_  ;
    load_data("/home/yhr/workspace/cuda_clustering/data/sift_base.fvecs" , data_ , num , dim) ;
    cout << "num : " << num << ", dim : " << dim << endl ;
    float* query_data_load ;
    unsigned query_num , query_dim ;
    load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift1M/sift_range_query.fvecs" , query_data_load , query_num , query_dim) ;
    // query_num = 500; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;
    // query_num *= 10 ;
    float* tmp_query_data_load = new float[10 *query_num * query_dim] ;
    for(int i = 0 ; i < 10 ; i ++)
        memcpy(tmp_query_data_load + i * query_num * query_dim  , query_data_load , query_num * query_dim * sizeof(float)) ;
    query_num *= 10 ;
    delete [] query_data_load ;
    query_data_load = tmp_query_data_load ;
    // for(unsigned i = 0 ; i )
    // TOPM = 20 , DIM = dim ; 
    // ************** 若要修改查询数量, 请修改如下变量 **********************
    query_num = 1000 ;
    // for(int i = 0 ; i < dim ; i ++)
    //     query_data_load[i] = query_data_load[613 * dim + i] ;
    float* data_dev ;
    half *data_half_dev ;
    float* query_data_dev ;
    half *query_data_half_dev ;
    cudaMalloc((void**)&data_dev , num * dim * sizeof(float)) ;
    cudaMalloc((void**) &data_half_dev , num * dim * sizeof(half)) ;
    cudaMalloc((void**) &query_data_dev , query_num * query_dim   * sizeof(float)) ;
    unsigned query_num_with_pad = (query_num + 15) / 16 * 16 ;
    cout << "query_num_with_pad : " << query_num_with_pad << endl ;
    cudaMalloc((void**) &query_data_half_dev , query_num_with_pad * query_dim * sizeof(half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(data_dev , data_ , num * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(query_data_dev , query_data_load , query_num * query_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num ; 
    // 调用实例
    f2h<<<points_num, 256>>>(data_dev, data_half_dev, points_num); // 先将数据转换成fp16
    // cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    f2h<<<query_num, 256>>>(query_data_dev, query_data_half_dev, query_num); // 将查询转换成fp16
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    if(query_num != query_num_with_pad)
        thrust::fill(thrust::device , query_data_half_dev + query_num * query_dim , query_data_half_dev + query_num_with_pad * query_dim , __float2half(0.0f)) ;

    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    vector<float> l_bound_host(query_num * attr_dim) , r_bound_host(query_num * attr_dim) ;
    if(selectivity == 0.0) {
        load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift1M/sift_query_range.bin", query_range_x, 1000) ;
        query_range_x.reserve(query_num) ;
        for(unsigned i = 1 ; i < 10 ; i ++)
            query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        query_range_y = query_range_x ;
        // query_range_x[0] = {0 , 249999} ;
        // query_range_y[0] = {0 , 249999} ;
        // query_range_x[1] = {0 , 499999} , query_range_y[1] = {0 , 249999} ;
        // query_range_x[2] = {0 , 749999} , query_range_y[2] = {0 , 249999} ;
        // query_range_x[3] = {0 , 499999} , query_range_y[3] = {0 , 499999} ;
        // for(unsigned i = 4 ; i < query_num ; i ++)
        //     query_range_x[i] = query_range_x[i % 4] , query_range_y[i] = query_range_y[i % 4] ;
        // query_range_y = query_range_x ; 
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ;j < attr_dim ; j ++)
                l_bound_host[i * attr_dim] = query_range_x[i].first , r_bound_host[i * attr_dim] = query_range_x[i].second ;
                l_bound_host[i * attr_dim + 1] = query_range_y[i].first , r_bound_host[i * attr_dim + 1] = query_range_y[i].second ;
        }
        // 负载量 = 4
        // query_range_y[0] = query_range_x[0] = {0 , 499999} ;
    
        // query_num = 1; 
        // query_range_x[0].first = 250000 , query_range_x[0].second = 499999 ;
        // query_range_y[0].first = 250000 , query_range_y[0].second = 499999 ;
        // for(unsigned i = 0 ; i < query_num ;i ++) {
        //     query_range_x[i] = query_range_x.front() ;
        //     query_range_y[i] = query_range_y.front() ;
        // }
    } else {
        cout << "selectivity : " << selectivity << endl ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_x , query_num) ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_y , query_num) ;
        generate_range_multi_attr_with_selectivity(l_bound_host.data() , r_bound_host.data() , attr_dim , grid_min.data() , grid_max.data() , query_num , selectivity) ;
        cout << "随机生成属性, 完成" << endl ;
    }
    // query_range_x[0].first = 450000 , query_range_x[0].second = 850000 ;
    // cout << "query range x 0 : " << query_range_x[0].first << "," << query_range_x[0].second << endl ;
    

    // for(unsigned i = 1 ; i < query_num ; i ++)
    //     query_range_x[i] = query_range_x[0] ;

    // query_range_x[0] = query_range_x[1] ;
    // for(unsigned i = 0 ; i < query_num ;i ++) {
    //     cout << "query range"<< i << ":" << query_range_x[i].first << "," << query_range_x[i].second << endl ;

    // }
    // for(unsigned i = 1 ; i < 10 ; i ++)
    //     query_range_x[i] = query_range_x[0] ;

    // unsigned attr_dim =  2 ;

    float* l_bound , * r_bound , *l_bound_dev , *r_bound_dev , *attr_min , *attr_width ;
    unsigned *attr_grid_size ;
    // l_bound = new float[query_num * attr_dim] , r_bound = new float[query_num * attr_dim] ;
    l_bound = l_bound_host.data() , r_bound = r_bound_host.data() ;
    // for(unsigned i = 0 ; i < query_num ; i ++) {
    //     // for(unsigned j = 0 ;j  < attr_dim ; j ++)
    //         l_bound[i * attr_dim] = (float) query_range_x[i].first , r_bound[i * attr_dim] = (float) query_range_x[i].second ; 
    //         l_bound[i * attr_dim + 1] = (float) query_range_y[i].first , r_bound[i * attr_dim + 1] = (float) query_range_y[i].second ;
    // }
    cudaMalloc((void**) &l_bound_dev , attr_dim * query_num * sizeof(float)) ;
    cudaMalloc((void**) &r_bound_dev , attr_dim * query_num * sizeof(float)) ;
    cudaMalloc((void**) &attr_min , attr_dim * sizeof(float)) ;
    cudaMalloc((void**) &attr_width , attr_dim * sizeof(float)) ;
    cudaMalloc((void**) &attr_grid_size , attr_dim * sizeof(unsigned)) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    cudaMemcpy(l_bound_dev , l_bound , query_num * attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(r_bound_dev , r_bound , query_num * attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_min , attr_min + attr_dim , 0.0) ;
    cudaMemcpy(attr_min , grid_min.data() , attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_width , attr_width + attr_dim , 250000.0) ;
    cudaMemcpy(attr_width , attr_width_host.data() , attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_grid_size , attr_grid_size + attr_dim , 4) ;
    cudaMemcpy(attr_grid_size , grid_size_host.data() , attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    unsigned qnum = query_num ;
    unsigned pnum = thrust::reduce(
        thrust::device , 
        attr_grid_size , attr_grid_size + attr_dim ,
        1 , thrust::multiplies<unsigned>() 
    ) ;
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg<attr_dim>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , 0.05) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    vector<unsigned> cluster_id ;
    float* cluster_centers;
    vector<vector<unsigned>> num_c_p ;
    load_clustering(cluster_id , "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift1M/clustering1000.bin" , cluster_centers , dim , num_c_p , data_) ;
    float* cluster_centers_dev ;
    unsigned* cluster_p_info ;
    __half* cluster_centers_half_dev ;
    unsigned cnum = num_c_p[0].size() ;
    cout << "聚类个数:" << cnum << endl ;
    cudaMalloc((void**) &cluster_centers_dev , cnum * dim * sizeof(float)) ;
    cudaMalloc((void**) &cluster_p_info, cnum * pnum * sizeof(unsigned)) ;
    unsigned cnum_with_pad = (cnum + 15) / 16 * 16 ;
    cout << "cnum_with_pad : " << cnum_with_pad << endl ;
    cudaMalloc((void**) &cluster_centers_half_dev , cnum_with_pad * dim * sizeof(__half)) ;
    cudaMemcpy(cluster_centers_dev , cluster_centers , cnum * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    f2h<<<cnum, 256>>>(cluster_centers_dev, cluster_centers_half_dev, cnum);
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    if(cnum != cnum_with_pad)
        thrust::fill(thrust::device , cluster_centers_half_dev + cnum * dim , cluster_centers_half_dev + cnum_with_pad * dim , __float2half(0.0f)) ;
    // 在此处计算得到q_norm 和 c_norm 
    // ***************************************************
    float *q_norm , *c_norm ;
    cudaMalloc((void**) &q_norm , query_num_with_pad * sizeof(float)) ;
    cudaMalloc((void**) &c_norm , cnum_with_pad * sizeof(float)) ;
    find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    find_norm<<<(cnum_with_pad + 3) / 4 , dim3(32 , 4 , 1)>>>(cluster_centers_half_dev , c_norm , dim , cnum_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__ ) ;

    // vector<unsigned> info1_host(qnum) , info2_host(qnum) ;
    // cudaMemcpy(info1_host.data() , partition_infos[1] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // cudaMemcpy(info2_host.data() , partition_infos[2] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // for(unsigned i = 0 ; i < qnum ;i  ++) {
    //     cout << i << ": size-" << info1_host[i] << ", offset-" << info2_host[i] << endl ; 
    // }

    // 检查norm的计算是否正确
    /**
        vector<float> q_norm_host(query_num_with_pad) , c_norm_host(cnum_with_pad) ;
        cudaMemcpy(q_norm_host.data() , q_norm , query_num_with_pad * sizeof(float) , cudaMemcpyDeviceToHost) ;
        cudaMemcpy(c_norm_host.data() , c_norm , cnum_with_pad * sizeof(float) , cudaMemcpyDeviceToHost) ;
        vector<float> q_norm_gt(query_num_with_pad) , c_norm_gt(cnum_with_pad) ;
        for(unsigned i = 0 ; i < query_num ; i++) {
            float dis = 0.0 ;
            for(unsigned d = 0; d < dim ;  d++)
                dis += query_data_load[i * dim + d] * query_data_load[i * dim + d] ;
            q_norm_gt[i] = dis / 10000.0f ;
        }
        for(unsigned i = 0 ; i < cnum ; i ++) {
            float dis = 0.0 ;
            for(unsigned d = 0 ; d < dim ;  d++)
                dis += (cluster_centers[i * dim + d] / 100.0f )* (cluster_centers[i * dim + d] / 100.0f );
            c_norm_gt[i] = dis ;
        }
        float abs_difference = 0.0f ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // printf("%f : %f, " , q_norm_host[i] , q_norm_gt[i]) ;
            abs_difference += abs(q_norm_host[i] - q_norm_gt[i]) ;
            printf("%f, " , abs(q_norm_host[i] - q_norm_gt[i])) ;
        }
        for(unsigned i = 0 ; i < cnum ; i ++)  {
            // printf("%f : %f, " , c_norm_host[i] , c_norm_gt[i]) ; 
            abs_difference += abs(c_norm_host[i] - c_norm_gt[i]) ;
            printf("%f, " , abs(c_norm_host[i] - c_norm_gt[i])) ; 
        }
        cout << "误差:" << abs_difference << endl ;
    **/
    // ***************************************************


    // ******************************
    // 此处把 聚类中心矩阵做转置
    // ******************************
    // __half* cluster_centers_half_dev_T ;
    // cudaMalloc((void**) &cluster_centers_half_dev_T , cnum_with_pad * dim *  sizeof(__half)) ;
    // matrix_transpose<<<cnum_with_pad , 32>>>(cluster_centers_half_dev , cluster_centers_half_dev_T , cnum_with_pad, dim) ;
    // cudaDeviceSynchronize() ;
    // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // ******************************

    unsigned num_c_p_offset = 0 ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cudaMemcpy(cluster_p_info + i * cnum , num_c_p[i].data() , cnum * sizeof(unsigned) , cudaMemcpyHostToDevice ) ;
    }

    cout << partition_infos[0] << "," <<  partition_infos[1] << "," <<  partition_infos[2] << endl ;
    // cnum = 16 , qnum = 16 ;
    s = chrono::high_resolution_clock::now() ;
    get_partitions_access_order_V2(partition_infos , pnum , cnum , qnum , cluster_p_info , 4 , cluster_centers_half_dev , query_data_half_dev , dim , query_data_load , cluster_centers , q_norm , c_norm , num)  ;
    e = chrono::high_resolution_clock::now() ;
    cost2 = chrono::duration<double>(e - s).count() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // check_process2(attr_dim , attr_min ,attr_width ,attr_grid_size ,l_bound ,r_bound ,partition_infos , query_num , query_data_load , dim , data_) ;
    
    s = chrono::high_resolution_clock::now() ;
    // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
    //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
    //     global_graph , 16 , pnum , L) ;
    vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
        attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
        global_graph , 16 , pnum , L) ;
    e = chrono::high_resolution_clock::now() ;
    cost3 = chrono::duration<double>(e - s).count() ;
    cout << "finish search" << endl ;

    // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
    vector<vector<unsigned>> ground_truth(qnum) ;
    
    #pragma omp parallel for num_threads(10)
    for(unsigned i = 0 ; i < qnum ;i  ++)
        ground_truth[i] = naive_brute_force_with_filter(query_data_load + i * dim , data_ , dim , num , 10 ,
            l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim) ;
    // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
    float total_recall = cal_recall_show_partition_multi_attrs(result , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;
    cout << "recall : " << total_recall << endl  ;

    cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
    cout << "获取分区列表用时:" << cost1 << endl ;
    cout << "对分区列表排序用时:" << cost2 << endl ;
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "搜索用时:" << cost3 << endl ;

    rc_recall1 = total_recall ;
    rc_cost1 = select_path_kernal_cost ;

    cudaFree(global_graph) ;
    cudaFree(attrs_dev) ;
    cudaFree(data_dev) ;
    cudaFree(data_half_dev) ;
    cudaFree(query_data_dev) ;
    cudaFree(query_data_half_dev) ;
    cudaFree(l_bound_dev) ;
    cudaFree(r_bound_dev) ;
    cudaFree(attr_min) ;
    cudaFree(attr_width) ;
    cudaFree(attr_grid_size) ;
    cudaFree(cluster_centers_dev) ;
    cudaFree(cluster_p_info) ;
    cudaFree(cluster_centers_half_dev) ;
    cudaFree(q_norm) ;
    cudaFree(c_norm) ;
}


void multi_attr_test_batch_process(unsigned L = 80) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    array<unsigned*,3> graph_infos = get_graph_info_host() ;
    array<unsigned*,3> idx_mapper = get_idx_mapper() ;
    vector<vector<unsigned>> global_edges ;
    load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges) ;
    unsigned *global_graph ;
    cudaMalloc((void**) &global_graph , 1000000 * 16 * 16 * sizeof(unsigned)) ;

    for(unsigned i = 0 ; i < 1000000 ; i ++) {
        cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    }
    float *attrs_dev , *attrs ;
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    constexpr unsigned attr_dim =  2 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<float> grid_min(attr_dim , 0.0f) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<float> grid_max(attr_dim , 1000000.0f) ;
    vector<unsigned> grid_size_host = {4 , 4} ;
    vector<float> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++)
        attr_width_host[i] = (grid_max[i] - grid_min[i]) / grid_size_host[i] ;
    attrs = new float[attr_dim * 1000000] ;
    re_generate(attrs , attr_dim , 1000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    cudaMalloc((void**) &attrs_dev , attr_dim * 1000000 * sizeof(float)) ;
    cudaMemcpy(attrs_dev , attrs , attr_dim * 1000000 * sizeof(float) , cudaMemcpyHostToDevice) ;

    cout << "sizeof(half) = " << sizeof(half) << endl ;

    // 先载入数据集和图索引
    unsigned num , dim  ;
    float* data_  , *data_pined ;
    load_data("/home/yhr/workspace/cuda_clustering/data/sift_base.fvecs" , data_ , num , dim) ;
    cout << "num : " << num << ", dim : " << dim << endl ;
    cudaMallocHost(&data_pined , num * dim * sizeof(float)) ;
    memcpy(data_pined , data_ , num * dim * sizeof(float)) ;
    delete [] data_ ;
    data_ = data_pined ; 

    float* query_data_load ;
    unsigned query_num , query_dim ;
    load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_range_query.fvecs" , query_data_load , query_num , query_dim) ;
    // query_num = 500; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;
    // query_num *= 10 ;
    float* tmp_query_data_load = new float[10 *query_num * query_dim] ;
    for(int i = 0 ; i < 10 ; i ++)
        memcpy(tmp_query_data_load + i * query_num * query_dim  , query_data_load , query_num * query_dim * sizeof(float)) ;
    query_num *= 10 ;
    delete [] query_data_load ;
    query_data_load = tmp_query_data_load ;
    // for(unsigned i = 0 ; i )
    // TOPM = 20 , DIM = dim ; 
    // ************** 若要修改查询数量, 请修改如下变量 **********************
    query_num = 1000 ;
    // for(int i = 0 ; i < dim ; i ++)
    //     query_data_load[i] = query_data_load[613 * dim + i] ;
    float* data_dev ;
    half *data_half_dev ;
    float* query_data_dev ;
    half *query_data_half_dev ;
    cudaMalloc((void**)&data_dev , num * dim * sizeof(float)) ;
    cudaMalloc((void**) &data_half_dev , num * dim * sizeof(half)) ;
    cudaMalloc((void**) &query_data_dev , query_num * query_dim   * sizeof(float)) ;
    unsigned query_num_with_pad = (query_num + 15) / 16 * 16 ;
    cout << "query_num_with_pad : " << query_num_with_pad << endl ;
    cudaMalloc((void**) &query_data_half_dev , query_num_with_pad * query_dim * sizeof(half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(data_dev , data_ , num * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(query_data_dev , query_data_load , query_num * query_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num ; 
    // 调用实例
    f2h<<<points_num, 256>>>(data_dev, data_half_dev, points_num); // 先将数据转换成fp16
    // cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    f2h<<<query_num, 256>>>(query_data_dev, query_data_half_dev, query_num); // 将查询转换成fp16
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    if(query_num != query_num_with_pad)
        thrust::fill(thrust::device , query_data_half_dev + query_num * query_dim , query_data_half_dev + query_num_with_pad * query_dim , __float2half(0.0f)) ;

    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    vector<float> l_bound_host(query_num * attr_dim) , r_bound_host(query_num * attr_dim) ;
    if(selectivity == 0.0) {
        cout << "sel = " << selectivity << endl ;
        load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        query_range_x.reserve(query_num) ;
        for(unsigned i = 1 ; i < 10 ; i ++)
            query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        // query_range_y = query_range_x ; 

        // query_range_y[0] = {245000 , 255000} , query_range_x[0] = {245000 , 255000} ;
        // for(unsigned i = 1 ; i < query_num ; i ++)
        //     query_range_y[i] = query_range_y[0] , query_range_x[i] = query_range_x[0] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            for(unsigned j = 0 ;j < attr_dim ; j ++)
                l_bound_host[i * attr_dim + j] = (float)query_range_x[i].first , r_bound_host[i * attr_dim + j] = (float)query_range_x[i].second ;
        }

        cout << "range 1 (" << query_range_x[1].first << "," << query_range_x[1].second << ")" << endl ;
        cout << "range 2 (" << query_range_x[2].first << "," << query_range_x[2].second << ")" << endl ;
        cout << "range 5 (" << query_range_x[5].first << "," << query_range_x[5].second << ")" << endl ;
        // query_range_y[0] = query_range_x[0] = {490000 , 510000} ;
    
        // query_num = 1; 
        // query_range_x[0].first = 250000 , query_range_x[0].second = 499999 ;
        // query_range_y[0].first = 250000 , query_range_y[0].second = 499999 ;
        // for(unsigned i = 0 ; i < query_num ;i ++) {
        //     query_range_x[i] = query_range_x.front() ;
        //     query_range_y[i] = query_range_y.front() ;
        // }
    } else {
        cout << "selectivity : " << selectivity << endl ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_x , query_num) ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_y , query_num) ;
        generate_range_multi_attr_with_selectivity(l_bound_host.data() , r_bound_host.data() , attr_dim , grid_min.data() , grid_max.data() , query_num , selectivity) ;
        cout << "随机生成属性, 完成" << endl ;
    }
    // query_range_x[0].first = 450000 , query_range_x[0].second = 850000 ;
    // cout << "query range x 0 : " << query_range_x[0].first << "," << query_range_x[0].second << endl ;
    

    // for(unsigned i = 1 ; i < query_num ; i ++)
    //     query_range_x[i] = query_range_x[0] ;

    // query_range_x[0] = query_range_x[1] ;
    // for(unsigned i = 0 ; i < query_num ;i ++) {
    //     cout << "query range"<< i << ":" << query_range_x[i].first << "," << query_range_x[i].second << endl ;

    // }
    // for(unsigned i = 1 ; i < 10 ; i ++)
    //     query_range_x[i] = query_range_x[0] ;

    // unsigned attr_dim =  2 ;

    float* l_bound , * r_bound , *l_bound_dev , *r_bound_dev , *attr_min , *attr_width ;
    unsigned *attr_grid_size ;
    // l_bound = new float[query_num * attr_dim] , r_bound = new float[query_num * attr_dim] ;
    l_bound = l_bound_host.data() , r_bound = r_bound_host.data() ;
    // for(unsigned i = 0 ; i < query_num ; i ++) {
    //     // for(unsigned j = 0 ;j  < attr_dim ; j ++)
    //         l_bound[i * attr_dim] = (float) query_range_x[i].first , r_bound[i * attr_dim] = (float) query_range_x[i].second ; 
    //         l_bound[i * attr_dim + 1] = (float) query_range_y[i].first , r_bound[i * attr_dim + 1] = (float) query_range_y[i].second ;
    // }
    cudaMalloc((void**) &l_bound_dev , attr_dim * query_num * sizeof(float)) ;
    cudaMalloc((void**) &r_bound_dev , attr_dim * query_num * sizeof(float)) ;
    cudaMalloc((void**) &attr_min , attr_dim * sizeof(float)) ;
    cudaMalloc((void**) &attr_width , attr_dim * sizeof(float)) ;
    cudaMalloc((void**) &attr_grid_size , attr_dim * sizeof(unsigned)) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    cudaMemcpy(l_bound_dev , l_bound , query_num * attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(r_bound_dev , r_bound , query_num * attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_min , attr_min + attr_dim , 0.0) ;
    cudaMemcpy(attr_min , grid_min.data() , attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_width , attr_width + attr_dim , 250000.0) ;
    cudaMemcpy(attr_width , attr_width_host.data() , attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_grid_size , attr_grid_size + attr_dim , 4) ;
    cudaMemcpy(attr_grid_size , grid_size_host.data() , attr_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    unsigned qnum = query_num ;
    unsigned pnum = thrust::reduce(
        thrust::device , 
        attr_grid_size , attr_grid_size + attr_dim ,
        1 , thrust::multiplies<unsigned>() 
    ) ;
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    bool *partition_is_selected ; 
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg_batch_process<attr_dim>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , 0.05f , partition_is_selected) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    vector<unsigned> cluster_id ;
    float* cluster_centers;
    vector<vector<unsigned>> num_c_p ;
    load_clustering(cluster_id , "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/clustering1000.bin" , cluster_centers , dim , num_c_p , data_) ;
    float* cluster_centers_dev ;
    unsigned* cluster_p_info ;
    __half* cluster_centers_half_dev ;
    unsigned cnum = num_c_p[0].size() ;
    cout << "聚类个数:" << cnum << endl ;
    cudaMalloc((void**) &cluster_centers_dev , cnum * dim * sizeof(float)) ;
    cudaMalloc((void**) &cluster_p_info, cnum * pnum * sizeof(unsigned)) ;
    unsigned cnum_with_pad = (cnum + 15) / 16 * 16 ;
    cout << "cnum_with_pad : " << cnum_with_pad << endl ;
    cudaMalloc((void**) &cluster_centers_half_dev , cnum_with_pad * dim * sizeof(__half)) ;
    cudaMemcpy(cluster_centers_dev , cluster_centers , cnum * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    f2h<<<cnum, 256>>>(cluster_centers_dev, cluster_centers_half_dev, cnum);
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    if(cnum != cnum_with_pad)
        thrust::fill(thrust::device , cluster_centers_half_dev + cnum * dim , cluster_centers_half_dev + cnum_with_pad * dim , __float2half(0.0f)) ;
    // 在此处计算得到q_norm 和 c_norm 
    // ***************************************************
    float *q_norm , *c_norm ;
    cudaMalloc((void**) &q_norm , query_num_with_pad * sizeof(float)) ;
    cudaMalloc((void**) &c_norm , cnum_with_pad * sizeof(float)) ;
    find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    find_norm<<<(cnum_with_pad + 3) / 4 , dim3(32 , 4 , 1)>>>(cluster_centers_half_dev , c_norm , dim , cnum_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__ ) ;

    // vector<unsigned> info1_host(qnum) , info2_host(qnum) ;
    // cudaMemcpy(info1_host.data() , partition_infos[1] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // cudaMemcpy(info2_host.data() , partition_infos[2] , qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // for(unsigned i = 0 ; i < qnum ;i  ++) {
    //     cout << i << ": size-" << info1_host[i] << ", offset-" << info2_host[i] << endl ; 
    // }

    // 检查norm的计算是否正确
    /**
        vector<float> q_norm_host(query_num_with_pad) , c_norm_host(cnum_with_pad) ;
        cudaMemcpy(q_norm_host.data() , q_norm , query_num_with_pad * sizeof(float) , cudaMemcpyDeviceToHost) ;
        cudaMemcpy(c_norm_host.data() , c_norm , cnum_with_pad * sizeof(float) , cudaMemcpyDeviceToHost) ;
        vector<float> q_norm_gt(query_num_with_pad) , c_norm_gt(cnum_with_pad) ;
        for(unsigned i = 0 ; i < query_num ; i++) {
            float dis = 0.0 ;
            for(unsigned d = 0; d < dim ;  d++)
                dis += query_data_load[i * dim + d] * query_data_load[i * dim + d] ;
            q_norm_gt[i] = dis / 10000.0f ;
        }
        for(unsigned i = 0 ; i < cnum ; i ++) {
            float dis = 0.0 ;
            for(unsigned d = 0 ; d < dim ;  d++)
                dis += (cluster_centers[i * dim + d] / 100.0f )* (cluster_centers[i * dim + d] / 100.0f );
            c_norm_gt[i] = dis ;
        }
        float abs_difference = 0.0f ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // printf("%f : %f, " , q_norm_host[i] , q_norm_gt[i]) ;
            abs_difference += abs(q_norm_host[i] - q_norm_gt[i]) ;
            printf("%f, " , abs(q_norm_host[i] - q_norm_gt[i])) ;
        }
        for(unsigned i = 0 ; i < cnum ; i ++)  {
            // printf("%f : %f, " , c_norm_host[i] , c_norm_gt[i]) ; 
            abs_difference += abs(c_norm_host[i] - c_norm_gt[i]) ;
            printf("%f, " , abs(c_norm_host[i] - c_norm_gt[i])) ; 
        }
        cout << "误差:" << abs_difference << endl ;
    **/
    // ***************************************************


    // ******************************
    // 此处把 聚类中心矩阵做转置
    // ******************************
    // __half* cluster_centers_half_dev_T ;
    // cudaMalloc((void**) &cluster_centers_half_dev_T , cnum_with_pad * dim *  sizeof(__half)) ;
    // matrix_transpose<<<cnum_with_pad , 32>>>(cluster_centers_half_dev , cluster_centers_half_dev_T , cnum_with_pad, dim) ;
    // cudaDeviceSynchronize() ;
    // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // ******************************

    unsigned num_c_p_offset = 0 ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cudaMemcpy(cluster_p_info + i * cnum , num_c_p[i].data() , cnum * sizeof(unsigned) , cudaMemcpyHostToDevice ) ;
    }

    cout << partition_infos[0] << "," <<  partition_infos[1] << "," <<  partition_infos[2] << endl ;
    // cnum = 16 , qnum = 16 ;
    s = chrono::high_resolution_clock::now() ;
    get_partitions_access_order_V2(partition_infos , pnum , cnum , qnum , cluster_p_info , 4 , cluster_centers_half_dev , query_data_half_dev , dim , query_data_load , cluster_centers , q_norm , c_norm , num)  ;
    e = chrono::high_resolution_clock::now() ;
    cost2 = chrono::duration<double>(e - s).count() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // check_process2(attr_dim , attr_min ,attr_width ,attr_grid_size ,l_bound ,r_bound ,partition_infos , query_num , query_data_load , dim , data_) ;
    

    // ************************************* 以下声明一些要用到的数据结构, 以及调用函数 -----------------------------------
    // vector<vector<unsigned>> batch_info(4 , vector<unsigned>(4)) ;
    // for(int i =0 ; i < 16 ; i ++)
    //     batch_info[i / 4][i % 4] = i ; 
    // vector<vector<unsigned>> batch_info = {
    //     {10 , 6 , 5 , 3} ,
    //     {9 , 0 , 1 , 7} ,
    //     {4 , 2 , 8 , 11} ,
    //     {12 , 13 , 14 , 15} 
    // } ;
    // vector<vector<unsigned>> batch_info = {
    //     {10 , 6 , 9 , 2} ,
    //     {5 , 0 , 1 , 4} ,
    //     {8 , 3 , 7 , 11} ,
    //     {12 , 13 , 14 , 15} 
    // } ;
    // vector<vector<unsigned>> batch_info = {
    //     {0, 2, 6, 3},
    //     {1, 4, 5, 7}, 
    //     {8, 9, 10, 11},
    //     {12, 13, 14, 15}
    // } ;
    vector<vector<unsigned>> batch_info = {
        {13, 0, 1, 4}, 
        {5, 6, 9, 7}, 
        {15, 10, 11, 14},
        {2, 8, 3, 12}
    } ;
    // random_shuffle(batch_info.begin() , batch_info.end()) ;
    __half *data_half_host , *data_half_dev2 ;
    cudaMallocHost((void**) &data_half_host , num * dim * sizeof(__half)) ;
    cudaMemcpy(data_half_host , data_half_dev , num * dim * sizeof(__half) , cudaMemcpyDeviceToHost) ;
    cudaMalloc((void**) &data_half_dev2 , 4 * dim * 62500 * sizeof(__half)) ;
    array<__half*,2> vec_buffer = {data_half_dev , data_half_dev2} ;
    unsigned* graph_buffer0 , *graph_buffer1 ; 
    cudaMalloc((void**) &graph_buffer0 , 4 * 16 * 62500 *  sizeof(unsigned)) ;
    cudaMalloc((void**) &graph_buffer1 , 4 * 16 * 62500 * sizeof(unsigned)) ;
    array<unsigned*,2> graph_buffer = {graph_buffer0 , graph_buffer1} ;
    unsigned* global_graph_buffer1 ; 
    cudaMalloc((void**) &global_graph_buffer1 , 4 * 16 * 16 * 62500 * sizeof(unsigned)) ;
    array<unsigned*,2> global_graph_buffer = {global_graph , global_graph_buffer1} ;
    unsigned* generic_buffer0 , *generic_buffer1 ; 
    cudaMalloc((void**) &generic_buffer0 , num * sizeof(unsigned)) ;
    cudaMalloc((void**) &generic_buffer1 , num * sizeof(unsigned)) ;
    array<unsigned*,2> generic_buffer = {generic_buffer0 , generic_buffer1} ;
    unsigned* stored_pos_global_idx ;
    cudaMalloc((void**) &stored_pos_global_idx , num * sizeof(unsigned)) ;
    array<unsigned*,4> new_idx_mapper = {idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , stored_pos_global_idx} ;
    vector<unsigned> seg_partition_point_start_index(16) ;
    seg_partition_point_start_index[0] = 0 ;
    for(int i = 1; i < 16 ; i ++)
        seg_partition_point_start_index[i] = seg_partition_point_start_index[i - 1] + graph_infos[1][i - 1] ;
    // unsigned* global_graph_host = new unsigned[num * 16 * 16 * sizeof(unsigned)] ;
    unsigned *global_graph_host ; 
    cudaMallocHost(&global_graph_host , num * 16 * 16 * sizeof(unsigned)) ;
    cudaMemcpy(global_graph_host , global_graph , num * 16 * 16 * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;


    cout << "准备工作完成, 开始查找 :" << endl ;
    s = chrono::high_resolution_clock::now() ;
    // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
    //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
    //     global_graph , 16 , pnum , L) ;
    

    vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter_batch_process(4 , batch_info , 16 , data_half_host , vec_buffer, graph_buffer , global_graph_buffer ,
        generic_buffer , query_data_half_dev , dim , qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        attr_dim  , graph_infos , new_idx_mapper , seg_partition_point_start_index  , partition_infos , L , 10 , 
        global_graph_host , 16 , pnum , partition_is_selected , L , 16) ;
    e = chrono::high_resolution_clock::now() ;
    cost3 = chrono::duration<double>(e - s).count() ;
    cout << "finish search" << endl ;

    // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
    vector<vector<unsigned>> ground_truth(qnum) ;
    vector<vector<float>> gt_dis(qnum) ;
    #pragma omp parallel for num_threads(10)
    for(unsigned i = 0 ; i < qnum ;i  ++) {
        vector<float> gt_d ; 
        ground_truth[i] = naive_brute_force_with_filter_store_dis(query_data_load + i * dim , data_ , dim , num , 10 ,
            l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim , gt_d) ;
        gt_dis[i] = gt_d ; 
    }

    // reverse(ground_truth[0].begin() , ground_truth[0].end()) ;
    // reverse(gt_dis[0].begin() , gt_dis[0].end()) ;
    // cout << "gt id :[" ;
    // for(unsigned i = 0 ; i < ground_truth[0].size() ;i  ++) {
    //     cout << ground_truth[0][i] ;
    //     if(i < ground_truth[0].size() - 1)
    //         cout << " ," ;
    // }
    // cout << "]" << endl ;
    // cout << "gt dis :[" ;
    // for(unsigned i = 0 ; i < gt_dis[0].size() ;i  ++) {
    //     cout << gt_dis[0][i] ;
    //     if(i < gt_dis[0].size() - 1)
    //         cout << " ," ;
    // }
    // cout << "]" << endl ;
    // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
    float total_recall = cal_recall_show_partition_multi_attrs(result , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , false) ;

    
    cout << "recall : " << total_recall << endl  ;

    cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
    cout << "获取分区列表用时:" << cost1 << endl ;
    cout << "对分区列表排序用时:" << cost2 << endl ;
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "核函数 + 数据运输、准备工作:" << other_kernal_cost << endl ;
    cout << "数据传输时间: " << mem_trans_cost << endl ;
    cout << "搜索用时:" << cost3 << endl ;
    cout << "batch cost : [" ;
    for(unsigned i = 0 ; i < batch_process_cost.size() ; i ++) {
        cout << batch_process_cost[i] ;
        if(i < batch_process_cost.size() - 1)
            cout << " ," ;
    }
    cout << "]" << endl ; 

    rc_recall1 = total_recall ;
    rc_cost1 = select_path_kernal_cost ;

    cudaFree(global_graph) ;
    cudaFree(attrs_dev) ;
    cudaFree(data_dev) ;
    cudaFree(data_half_dev) ;
    cudaFree(query_data_dev) ;
    cudaFree(query_data_half_dev) ;
    cudaFree(l_bound_dev) ;
    cudaFree(r_bound_dev) ;
    cudaFree(attr_min) ;
    cudaFree(attr_width) ;
    cudaFree(attr_grid_size) ;
    cudaFree(cluster_centers_dev) ;
    cudaFree(cluster_p_info) ;
    cudaFree(cluster_centers_half_dev) ;
    cudaFree(q_norm) ;
    cudaFree(c_norm) ;
    cudaFree(partition_is_selected) ;
}


void multi_attr_test_batch_process_large_dataset(unsigned L = 80) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    unsigned average_num = 6250000 ;
    array<unsigned*,3> graph_infos = get_graph_info_host_avg_size("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/deep100M/deep100M" 
    , ".cagra32" , 16 , 6250000 , 32) ;
    unsigned degree = 32 , global_graph_edge_per_p = 2 ; 
    array<unsigned*,3> idx_mapper = get_idx_mapper(100000000 , 16 , 6250000) ;
    // vector<vector<unsigned>> global_edges ;
    // load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges) ;
    unsigned *global_graph ;
    unsigned global_graph_degree , global_graph_num_in_graph ;
    // cudaMalloc((void**) &global_graph , 1000000 * 16 * 16 * sizeof(unsigned)) ;
    load_result_pined("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/deep100M/global_graph.graph" ,
     global_graph , global_graph_degree , global_graph_num_in_graph) ;
    // for(unsigned i = 0 ; i < 1000000 ; i ++) {
    //     cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    // }
    using attrType = double ;
    attrType *attrs_dev , *attrs ;
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    constexpr unsigned attr_dim =  2 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<attrType> grid_min(attr_dim , 0.0) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<attrType> grid_max(attr_dim , 1e8) ;
    vector<unsigned> grid_size_host = {4 , 4} ;
    vector<attrType> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++) {
        attr_width_host[i] = (grid_max[i] - grid_min[i]) / grid_size_host[i] ;
        cout << "attr_width_host[i] : " << attr_width_host[i] << "," ;
    }
    // attrs = new attrType[attr_dim * 100000000] ;
    // re_generate<attrType>(attrs , attr_dim , 100000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    // save_attrs("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/attrs.bin" ,100000000 , attr_dim, attrs) ;
    load_attrs<attrType>("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/attrs.bin", attrs) ;
    size_t attrs_dev_byteSize = static_cast<size_t>(attr_dim) * 100000000 * sizeof(attrType) ;
    cudaMalloc((void**) &attrs_dev , attrs_dev_byteSize) ;
    cudaMemcpy(attrs_dev , attrs , attrs_dev_byteSize , cudaMemcpyHostToDevice) ;

    cout << "sizeof(half) = " << sizeof(half) << endl ;

    // 先载入数据集和图索引
    unsigned num , dim  ;
    __half* data_   ;
    cout << "开始载入数据集" << endl ;
    data_ = load_dataset_return_pined_pointer("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs" ,
     num , dim , 10.0) ;

    __half* query_data_load ;
    unsigned query_num = 10000 , query_dim = dim ;
    vector<unsigned> qids(query_num * 10) ;
    /**
    unordered_set<unsigned> qids_contains ;
    mt19937 rng ;
    uniform_int_distribution<> intd(0 , num - 1) ;
    unsigned qids_ofst = 0 ;
    while(qids_ofst < query_num * 10) {
        unsigned t = intd(rng) ;
        if(!qids_contains.count(t)) {
            qids_contains.insert(t) ;
            qids[qids_ofst ++] = t ;
        }
    }
    random_shuffle(qids.begin() , qids.end()) ;
    **/
    // save_query_ids("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/qids.bin" , query_num , qids) ;
    load_query_ids("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/qids.bin" , query_num , qids) ;
    query_num = 1000; 
    // qids[0] = 0 ;

    query_data_load = new __half[query_num * query_dim] ;
    for(unsigned i = 0 ; i < query_num ; i ++) {
        // for(unsigned j = 0 ; j < query_dim ; j ++)
        //     query_data_load[i * query_dim + j] = data_[qids[i] * query_dim + j] ;
        size_t ofst = static_cast<size_t>(qids[i]) * static_cast<size_t>(query_dim) ;
        memcpy(query_data_load + i * query_dim , data_ + ofst , query_dim * sizeof(__half)) ;
    }
    // query_num = 500; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;

    half *query_data_half_dev ;
    
    cudaMalloc((void**) &query_data_half_dev , query_num * query_dim * sizeof(__half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(query_data_half_dev , query_data_load , query_num * query_dim * sizeof(__half) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num ; 
    
    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    vector<attrType> l_bound_host(query_num * attr_dim) , r_bound_host(query_num * attr_dim) ;
    if(selectivity == 0.0) {
        cout << "sel = " << selectivity << endl ;
        load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        query_range_x.reserve(query_num) ;
        for(unsigned i = 1 ; i < 10 ; i ++)
            query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        query_range_y = query_range_x ; 

        query_range_x[0] = {25000000 , 49999999} ;
        query_range_y[0] = {0 , 49999999} ;

        for(unsigned i = 1 ; i < query_num ;i  ++)
            // query_range_x[i].first *= 100 , query_range_x[i].second *= 100 ;
            query_range_x[i] = query_range_x[0] , query_range_y[i] = query_range_y[0] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ;j < attr_dim ; j ++)
            l_bound_host[i * attr_dim] = (attrType)query_range_x[i].first , r_bound_host[i * attr_dim] = (attrType)query_range_x[i].second ;
            l_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].first , r_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].second ;
        }

        cout << "range 0 (" << query_range_x[0].first << "," << query_range_x[0].second << ")" << endl ;
        cout << "range 1 (" << query_range_x[1].first << "," << query_range_x[1].second << ")" << endl ;
        cout << "range 2 (" << query_range_x[2].first << "," << query_range_x[2].second << ")" << endl ;
        cout << "range 5 (" << query_range_x[5].first << "," << query_range_x[5].second << ")" << endl ;
        // query_range_y[0] = query_range_x[0] = {490000 , 510000} ;
    
        // query_num = 1; 
        // query_range_x[0].first = 250000 , query_range_x[0].second = 499999 ;
        // query_range_y[0].first = 250000 , query_range_y[0].second = 499999 ;
        // for(unsigned i = 0 ; i < query_num ;i ++) {
        //     query_range_x[i] = query_range_x.front() ;
        //     query_range_y[i] = query_range_y.front() ;
        // }
    } else {
        cout << "selectivity : " << selectivity << endl ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_x , query_num) ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_y , query_num) ;
        generate_range_multi_attr_with_selectivity<attrType>(l_bound_host.data() , r_bound_host.data() , attr_dim , grid_min.data() , grid_max.data() , query_num , selectivity) ;
        cout << "随机生成属性, 完成" << endl ;
    }


    attrType* l_bound , * r_bound , *l_bound_dev , *r_bound_dev , *attr_min , *attr_width ;
    unsigned *attr_grid_size ;
    // l_bound = new float[query_num * attr_dim] , r_bound = new float[query_num * attr_dim] ;
    l_bound = l_bound_host.data() , r_bound = r_bound_host.data() ;
    // for(unsigned i = 0 ; i < query_num ; i ++) {
    //     // for(unsigned j = 0 ;j  < attr_dim ; j ++)
    //         l_bound[i * attr_dim] = (float) query_range_x[i].first , r_bound[i * attr_dim] = (float) query_range_x[i].second ; 
    //         l_bound[i * attr_dim + 1] = (float) query_range_y[i].first , r_bound[i * attr_dim + 1] = (float) query_range_y[i].second ;
    // }
    cudaMalloc((void**) &l_bound_dev , attr_dim * query_num * sizeof(attrType)) ;
    cudaMalloc((void**) &r_bound_dev , attr_dim * query_num * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_min , attr_dim * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_width , attr_dim * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_grid_size , attr_dim * sizeof(unsigned)) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    cudaMemcpy(l_bound_dev , l_bound , query_num * attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(r_bound_dev , r_bound , query_num * attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_min , attr_min + attr_dim , 0.0) ;
    cudaMemcpy(attr_min , grid_min.data() , attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_width , attr_width + attr_dim , 250000.0) ;
    cudaMemcpy(attr_width , attr_width_host.data() , attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_grid_size , attr_grid_size + attr_dim , 4) ;
    cudaMemcpy(attr_grid_size , grid_size_host.data() , attr_dim * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    unsigned qnum = query_num ;
    unsigned pnum = thrust::reduce(
        thrust::device , 
        attr_grid_size , attr_grid_size + attr_dim ,
        1 , thrust::multiplies<unsigned>() 
    ) ;
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    bool *partition_is_selected ; 
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg_batch_process<attr_dim , attrType>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , 0.05 , partition_is_selected) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    // vector<unsigned> cluster_id ;
    // float* cluster_centers;
    // vector<vector<unsigned>> num_c_p ;
    // load_clustering(cluster_id , "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/clustering1000.bin" , cluster_centers , dim , num_c_p , data_) ;
    // float* cluster_centers_dev ;
    // unsigned* cluster_p_info ;
    // __half* cluster_centers_half_dev ;
    // unsigned cnum = num_c_p[0].size() ;
    // cout << "聚类个数:" << cnum << endl ;
    // cudaMalloc((void**) &cluster_centers_dev , cnum * dim * sizeof(float)) ;
    // cudaMalloc((void**) &cluster_p_info, cnum * pnum * sizeof(unsigned)) ;
    // unsigned cnum_with_pad = (cnum + 15) / 16 * 16 ;
    // cout << "cnum_with_pad : " << cnum_with_pad << endl ;
    // cudaMalloc((void**) &cluster_centers_half_dev , cnum_with_pad * dim * sizeof(__half)) ;
    // cudaMemcpy(cluster_centers_dev , cluster_centers , cnum * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    // f2h<<<cnum, 256>>>(cluster_centers_dev, cluster_centers_half_dev, cnum);
    // cudaDeviceSynchronize() ;
    // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // if(cnum != cnum_with_pad)
    //     thrust::fill(thrust::device , cluster_centers_half_dev + cnum * dim , cluster_centers_half_dev + cnum_with_pad * dim , __float2half(0.0f)) ;
    // // 在此处计算得到q_norm 和 c_norm 
    // // ***************************************************
    // float *q_norm , *c_norm ;
    // cudaMalloc((void**) &q_norm , query_num_with_pad * sizeof(float)) ;
    // cudaMalloc((void**) &c_norm , cnum_with_pad * sizeof(float)) ;
    // find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    // find_norm<<<(cnum_with_pad + 3) / 4 , dim3(32 , 4 , 1)>>>(cluster_centers_half_dev , c_norm , dim , cnum_with_pad) ;
    // cudaDeviceSynchronize() ;
    // CHECK(cudaGetLastError() , __LINE__ , __FILE__ ) ;

   

    // unsigned num_c_p_offset = 0 ; 
    // for(unsigned i = 0 ; i < pnum ; i ++) {
    //     cudaMemcpy(cluster_p_info + i * cnum , num_c_p[i].data() , cnum * sizeof(unsigned) , cudaMemcpyHostToDevice ) ;
    // }

    // cout << partition_infos[0] << "," <<  partition_infos[1] << "," <<  partition_infos[2] << endl ;
    // // cnum = 16 , qnum = 16 ;
    // s = chrono::high_resolution_clock::now() ;
    // get_partitions_access_order_V2(partition_infos , pnum , cnum , qnum , cluster_p_info , 4 , cluster_centers_half_dev , query_data_half_dev , dim , query_data_load , cluster_centers , q_norm , c_norm , num)  ;
    // e = chrono::high_resolution_clock::now() ;
    // cost2 = chrono::duration<double>(e - s).count() ;
    // CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // // check_process2(attr_dim , attr_min ,attr_width ,attr_grid_size ,l_bound ,r_bound ,partition_infos , query_num , query_data_load , dim , data_) ;
  

    // ************************************* 以下声明一些要用到的数据结构, 以及调用函数 -----------------------------------
    // vector<vector<unsigned>> batch_info(4 , vector<unsigned>(4)) ;
    // for(int i =0 ; i < 16 ; i ++)
    //     batch_info[i / 4][i % 4] = i ; 
    vector<vector<unsigned>> batch_info = {
        {0 , 1 , 2} ,
        {3 , 4 , 5} ,
        {6 , 7 , 8} ,
        {9 , 10 , 11} ,
        {12,  13 , 14} ,
        {15} 
    } ;
    // vector<vector<unsigned>> batch_info = {
    //     {10 , 6 , 9 , 2} ,
    //     {5 , 0 , 1 , 4} ,
    //     {8 , 3 , 7 , 11} ,
    //     {12 , 13 , 14 , 15} 
    // } ;
    // vector<vector<unsigned>> batch_info = {
    //     {0, 2, 6, 3},
    //     {1, 4, 5, 7}, 
    //     {8, 9, 10, 11},
    //     {12, 13, 14, 15}
    // } ;
    size_t free_bytes, total_bytes;

    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    double free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    double total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "准备工作完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;

    // vector<vector<unsigned>> batch_info = {
    //     {15, 10, 11, 14},
    //     {5, 6, 9, 7}, 
    //     {13, 0, 1, 4}, 
    //     {2, 8, 3, 12}
    // } ;
    // vector<vector<unsigned>> batch_info = {
    //     {15, 10 , 11},
    //     {5, 6, 9}, 
    //     {13, 0, 1}, 
    //     {2, 8, 3,},
    //     {14 , 7 , 4},
    //     {12}
    // } ;
    // vector<vector<unsigned>> batch_info = {
    //     {14, 10, 15},
    //     {7, 11, 13},
    //     {9, 5, 6},
    //     {4, 0, 1},
    //     {2, 12, 8},
    //     {3}
    // } ;
    // vector<vector<unsigned>> batch_info(16 , vector<unsigned>(1)) ;
    // for(unsigned i =0 ; i < 16 ; i ++)
    //     batch_info[i][0] = i ; 
    const unsigned batch_size = 3 ;
    // random_shuffle(batch_info.begin() , batch_info.end()) ;
    __half *data_half_buffer1 , *data_half_buffer2 ;
    size_t data_half_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(average_num) * static_cast<size_t>(dim) * sizeof(__half) ;
    cudaMalloc((void**) &data_half_buffer1 , data_half_buffer_size) ;
    cudaMalloc((void**) &data_half_buffer2 , data_half_buffer_size) ;
    array<__half*,2> vec_buffer = {data_half_buffer1 , data_half_buffer2} ;
    unsigned* graph_buffer0 , *graph_buffer1 ; 
    size_t graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(degree) * static_cast<size_t>(average_num) * sizeof(unsigned) ;
    cudaMalloc((void**) &graph_buffer0 , graph_buffer_size) ;
    cudaMalloc((void**) &graph_buffer1 , graph_buffer_size) ;
    array<unsigned*,2> graph_buffer = {graph_buffer0 , graph_buffer1} ;
    unsigned* global_graph_buffer0 , *global_graph_buffer1 ;
    // 此处超出内存限制
    size_t global_graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(global_graph_degree) * static_cast<size_t>(average_num) * sizeof(unsigned) ;
    cudaMalloc((void**) &global_graph_buffer0 , global_graph_buffer_size) ;
    cudaMalloc((void**) &global_graph_buffer1 , global_graph_buffer_size) ;
    array<unsigned*,2> global_graph_buffer = {global_graph_buffer0 , global_graph_buffer1} ;
    unsigned* generic_buffer0 , *generic_buffer1 ; 
    cudaMalloc((void**) &generic_buffer0 , 10 * 1024 * 1024 * sizeof(unsigned)) ;
    cudaMalloc((void**) &generic_buffer1 , 10 * 1024 * 1024 * sizeof(unsigned)) ;
    array<unsigned*,2> generic_buffer = {generic_buffer0 , generic_buffer1} ;
    unsigned* stored_pos_global_idx ;
    cudaMalloc((void**) &stored_pos_global_idx , num * sizeof(unsigned)) ;
    array<unsigned*,4> new_idx_mapper = {idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , stored_pos_global_idx} ;
    vector<unsigned> seg_partition_point_start_index(16) ;
    seg_partition_point_start_index[0] = 0 ;
    for(int i = 1; i < 16 ; i ++)
        seg_partition_point_start_index[i] = seg_partition_point_start_index[i - 1] + graph_infos[1][i - 1] ;
    // unsigned* global_graph_host = new unsigned[num * 16 * 16 * sizeof(unsigned)] ;
    // unsigned *global_graph_host ; 
    // cudaMallocHost(&global_graph_host , num * 16 * 16 * sizeof(unsigned)) ;
    // cudaMemcpy(global_graph_host , global_graph , num * 16 * 16 * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cudaError_t err2 = cudaMemGetInfo(&free_bytes, &total_bytes) ;
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "缓冲区申请完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;
    cout << "准备工作完成, 开始查找 :" << endl ;
    s = chrono::high_resolution_clock::now() ;
    // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
    //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
    //     global_graph , 16 , pnum , L) ;
    

    vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter_batch_process<attrType>(batch_info.size() , batch_info , degree , data_ , vec_buffer, graph_buffer , global_graph_buffer ,
        generic_buffer , query_data_half_dev , dim , qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        attr_dim  , graph_infos , new_idx_mapper , seg_partition_point_start_index  , partition_infos , L , 10 , 
        global_graph , 2 , pnum , partition_is_selected , L , 16) ;
    e = chrono::high_resolution_clock::now() ;
    cost3 = chrono::duration<double>(e - s).count() ;
    cout << "finish search" << endl ;

    cudaFree(data_half_buffer1) ;
    cudaFree(data_half_buffer2) ;
    cudaFree(graph_buffer0) ;
    cudaFree(graph_buffer1) ;
    cudaFree(global_graph_buffer0) ;
    cudaFree(global_graph_buffer1) ;
    cudaFree(generic_buffer0) ;
    cudaFree(generic_buffer1) ;
    cudaFree(stored_pos_global_idx) ;

    cout << "gpu 暴力:" << endl ;
    auto gpu_bf_s = chrono::high_resolution_clock::now() ;
    vector<vector<unsigned>> gpu_bf_result ;
    // gpu_bf_result = prefiltering_and_bruteforce<attrType>(query_data_half_dev , data_ , qnum ,  10 , dim , num , attrs_dev,
    //    l_bound_dev , r_bound_dev , attr_dim) ;
    auto gpu_bf_e = chrono::high_resolution_clock::now() ;
    float gpu_bf_cost = chrono::duration<double>(gpu_bf_e - gpu_bf_s).count() ;
    cout << "gpu 暴力 结束: " << gpu_bf_cost << endl ;

    // save_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt", query_num , 10, gpu_bf_result) ;
    // load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt" , gpu_bf_result) ;
    load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery_single_p1.gt" , gpu_bf_result) ;
    cudaFreeHost(data_) ;
    cudaFreeHost(global_graph) ;
    cudaFreeHost(graph_infos[0]) ;
    cudaFreeHost(graph_infos[1]) ;
    cudaFreeHost(graph_infos[2]) ;
    /**
        float* data_float , * query_data_load_float ;
        cout << "开始暴力搜索, 载入数据--------------------" << endl ;
        load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs" , data_float , num , dim) ;
        cout << "num : " << num << ", dim : " << dim << endl ;
        query_data_load_float = new float[query_num * query_dim] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ; j < query_dim ; j ++)
            // query_data_load_float[i * query_dim + j] = data_float[qids[i] * query_dim + j] ;
            size_t ofst = static_cast<size_t>(qids[i]) * static_cast<size_t>(query_dim) ;
            memcpy(query_data_load_float + i * query_dim , data_float + ofst , query_dim * sizeof(float)) ;
        }
        // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
        vector<vector<unsigned>> ground_truth(qnum) ;
        vector<vector<float>> gt_dis(qnum) ;
        auto cpu_bf_s = chrono::high_resolution_clock::now() ;
        #pragma omp parallel for num_threads(10)
        for(unsigned i = 0 ; i < qnum ;i  ++) {
            vector<float> gt_d ; 
            ground_truth[i] = naive_brute_force_with_filter_store_dis<attrType>(query_data_load_float + i * dim , data_float , dim , num , 10 ,
                l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim , gt_d) ;
            gt_dis[i] = gt_d ; 
        }
        auto cpu_bf_e = chrono::high_resolution_clock::now() ;
        float cpu_bf_cost = chrono::duration<double>(cpu_bf_e - cpu_bf_s).count() ;
        reverse(ground_truth[0].begin() , ground_truth[0].end()) ;
        reverse(gt_dis[0].begin() , gt_dis[0].end()) ;
        cout << "gt id :[" ;
        for(unsigned i = 0 ; i < ground_truth[0].size() ;i  ++) {
            cout << ground_truth[0][i] ;
            if(i < ground_truth[0].size() - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;
        cout << "gt dis :[" ;
        for(unsigned i = 0 ; i < gt_dis[0].size() ;i  ++) {
            cout << gt_dis[0][i] ;
            if(i < gt_dis[0].size() - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;
        // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
        float total_recall = cal_recall_show_partition_multi_attrs<attrType>(result , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;

        
        cout << "recall : " << total_recall << endl  ;
    **/
    cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
    cout << "获取分区列表用时:" << cost1 << endl ;
    cout << "对分区列表排序用时:" << cost2 << endl ;
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "核函数 + 数据运输、准备工作:" << other_kernal_cost << endl ;
    cout << "数据传输时间: " << mem_trans_cost << endl ;
    cout << "搜索用时:" << cost3 << endl ;
    cout << "batch cost : [" ;
    for(unsigned i = 0 ; i < batch_process_cost.size() ; i ++) {
        cout << batch_process_cost[i] ;
        if(i < batch_process_cost.size() - 1)
            cout << " ," ;
    }
    cout << "]" << endl ; 
    cout << "batch mem load cost : [" ;
    for(unsigned i = 0 ; i < batch_data_trans_cost.size() ; i ++) {
        cout << batch_data_trans_cost[i] ;
        if(i < batch_data_trans_cost.size() - 1)
            cout << " ," ;
    }
    cout << "]" << endl ;
    cout << "gpu brute force 时间: " << gpu_bf_cost << endl ;
    // cout << "cpu brute force 时间: " << cpu_bf_cost << endl ;
    float total_recall = cal_recall_show_partition_multi_attrs<attrType>(result , gpu_bf_result , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;
    cout << " recall : " << total_recall << endl ;
    // cout << "qids[" ;
    // for(unsigned i = 0 ; i < query_num ;i ++) {
    //     cout << qids[i] ; 
    //     if(i < query_num - 1)
    //         cout << " ," ;
    // }
    // cout << "]" << endl ;

    rc_recall1 = total_recall ;
    rc_cost1 = select_path_kernal_cost ;

    cudaFree(global_graph) ;
    cudaFree(attrs_dev) ;
    // cudaFree(data_dev) ;
    // cudaFree(data_half_dev) ;
    // cudaFree(query_data_dev) ;
    cudaFree(query_data_half_dev) ;
    cudaFree(l_bound_dev) ;
    cudaFree(r_bound_dev) ;
    cudaFree(attr_min) ;
    cudaFree(attr_width) ;
    cudaFree(attr_grid_size) ;
    // cudaFree(cluster_centers_dev) ;
    // cudaFree(cluster_p_info) ;
    // cudaFree(cluster_centers_half_dev) ;
    // cudaFree(q_norm) ;
    // cudaFree(c_norm) ;
    cudaFree(partition_is_selected) ;
    for(unsigned i = 0  ; i < 5 ; i ++)
        cudaFree(partition_infos[i]) ;
    for(unsigned i = 0 ; i < 3 ; i ++) 
        cudaFree(idx_mapper[i]) ;
    
}


void multi_attr_test_batch_process_large_dataset_compressed_graph_T(unsigned L = 80) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    unsigned average_num = 6250000 ;
    array<unsigned*,3> graph_infos = get_graph_info_host_avg_size("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/deep100M/deep100M" 
    , ".cagra16" , 16 , 6250000 , 16) ;
    // unsigned offset4 = 4 * average_num * 16 ;
    // for(unsigned i = 0 ; i < 100 ; i ++) {
    //     for(unsigned j = 0 ; j < 16 ; j ++) {
    //         cout << graph_infos[0][offset4 + i * 16 + j] << " ," ;
    //     }
    //     cout << endl;
    // }


    unsigned degree = 16 , global_graph_edge_per_p = 2 ; 
    array<unsigned*,3> idx_mapper = get_idx_mapper(100000000 , 16 , 6250000) ;
    // vector<vector<unsigned>> global_edges ;
    // load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges) ;
    unsigned *global_graph ;
    unsigned global_graph_degree , global_graph_num_in_graph ;
    // cudaMalloc((void**) &global_graph , 1000000 * 16 * 16 * sizeof(unsigned)) ;
    load_result_pined("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/global_graph_T.graphT" ,
     global_graph , global_graph_degree , global_graph_num_in_graph) ;
    cout << "global graph : degree-" << global_graph_degree << ", num-" << global_graph_num_in_graph << endl ;
    // for(unsigned i = 0 ; i < 1000000 ; i ++) {
    //     cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    // }
    using attrType = double ;
    attrType *attrs_dev , *attrs ;
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    constexpr unsigned attr_dim =  2 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<attrType> grid_min(attr_dim , 0.0) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<attrType> grid_max(attr_dim , 1e8) ;
    vector<unsigned> grid_size_host = {4 , 4} ;
    vector<attrType> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++) {
        attr_width_host[i] = (grid_max[i] - grid_min[i]) / grid_size_host[i] ;
        cout << "attr_width_host[i] : " << attr_width_host[i] << "," ;
    }
    // attrs = new attrType[attr_dim * 100000000] ;
    // re_generate<attrType>(attrs , attr_dim , 100000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    // save_attrs("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/attrs.bin" ,100000000 , attr_dim, attrs) ;
    load_attrs<attrType>("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/attrs.bin", attrs) ;
    size_t attrs_dev_byteSize = static_cast<size_t>(attr_dim) * 100000000 * sizeof(attrType) ;
    cudaMalloc((void**) &attrs_dev , attrs_dev_byteSize) ;
    cudaMemcpy(attrs_dev , attrs , attrs_dev_byteSize , cudaMemcpyHostToDevice) ;

    cout << "sizeof(half) = " << sizeof(half) << endl ;

    // 先载入数据集和图索引
    unsigned num , dim  ;
    __half* data_   ;
    cout << "开始载入数据集" << endl ;
    data_ = load_dataset_return_pined_pointer("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs" ,
     num , dim , 10.0) ;

    __half* query_data_load ;
    unsigned query_num = 1 , query_dim = dim ;
    vector<unsigned> qids(query_num * 10) ;
    /**
    unordered_set<unsigned> qids_contains ;
    mt19937 rng ;
    uniform_int_distribution<> intd(0 , num - 1) ;
    unsigned qids_ofst = 0 ;
    while(qids_ofst < query_num * 10) {
        unsigned t = intd(rng) ;
        if(!qids_contains.count(t)) {
            qids_contains.insert(t) ;
            qids[qids_ofst ++] = t ;
        }
    }
    random_shuffle(qids.begin() , qids.end()) ;
    **/
    // save_query_ids("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/qids.bin" , query_num , qids) ;
    load_query_ids("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/qids.bin" , query_num , qids) ;
    // query_num = 1000 ; 
    // qids[0] = 0 ;

    query_data_load = new __half[query_num * query_dim] ;
    for(unsigned i = 0 ; i < query_num ; i ++) {
        // for(unsigned j = 0 ; j < query_dim ; j ++)
        //     query_data_load[i * query_dim + j] = data_[qids[i] * query_dim + j] ;
        size_t ofst = static_cast<size_t>(qids[i]) * static_cast<size_t>(query_dim) ;
        memcpy(query_data_load + i * query_dim , data_ + ofst , query_dim * sizeof(__half)) ;
    }
    // query_num = 1 ; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;

    half *query_data_half_dev ;
    
    cudaMalloc((void**) &query_data_half_dev , query_num * query_dim * sizeof(__half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(query_data_half_dev , query_data_load , query_num * query_dim * sizeof(__half) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num ; 
    
    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    vector<attrType> l_bound_host(query_num * attr_dim) , r_bound_host(query_num * attr_dim) ;
    if(selectivity == 0.0) {
        cout << "sel = " << selectivity << endl ;
        load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        query_range_x.reserve(query_num) ;
        for(unsigned i = 1 ; i < 10 ; i ++)
            query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        query_range_y = query_range_x ; 

        // query_range_y[0] = {245000 , 255000} , query_range_x[0] = {245000 , 255000} ;
        // for(unsigned i = 1 ; i < query_num ; i ++)
        //     query_range_y[i] = query_range_y[0] , query_range_x[i] = query_range_x[0] ;
        // query_range_x[0] = {25000000 , 49999999} ;
        // query_range_y[0] = {0 , 49999999} ;

        for(unsigned i = 1 ; i < query_num ;i  ++) {
            query_range_x[i].first *= 100 , query_range_x[i].second *= 100 ; 
            query_range_y[i].first *= 100 , query_range_y[i].second *= 100 ;
        }

            // query_range_x[i] = query_range_x[0] , query_range_y[i] = query_range_y[0] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ;j < attr_dim ; j ++)
            l_bound_host[i * attr_dim] = (attrType)query_range_x[i].first , r_bound_host[i * attr_dim] = (attrType)query_range_x[i].second ;
            l_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].first , r_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].second ;
        }

        cout << "range 0 (" << query_range_x[0].first << "," << query_range_x[0].second << ")" << endl ;
        cout << "range 1 (" << query_range_x[1].first << "," << query_range_x[1].second << ")" << endl ;
        cout << "range 2 (" << query_range_x[2].first << "," << query_range_x[2].second << ")" << endl ;
        cout << "range 5 (" << query_range_x[5].first << "," << query_range_x[5].second << ")" << endl ;
        // query_range_y[0] = query_range_x[0] = {490000 , 510000} ;
    
        // query_num = 1; 
        // query_range_x[0].first = 250000 , query_range_x[0].second = 499999 ;
        // query_range_y[0].first = 250000 , query_range_y[0].second = 499999 ;
        // for(unsigned i = 0 ; i < query_num ;i ++) {
        //     query_range_x[i] = query_range_x.front() ;
        //     query_range_y[i] = query_range_y.front() ;
        // }
    } else {
        cout << "selectivity : " << selectivity << endl ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_x , query_num) ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_y , query_num) ;
        generate_range_multi_attr_with_selectivity<attrType>(l_bound_host.data() , r_bound_host.data() , attr_dim , grid_min.data() , grid_max.data() , query_num , selectivity) ;
        cout << "随机生成属性, 完成" << endl ;
    }


    attrType* l_bound , * r_bound , *l_bound_dev , *r_bound_dev , *attr_min , *attr_width ;
    unsigned *attr_grid_size ;
    // l_bound = new float[query_num * attr_dim] , r_bound = new float[query_num * attr_dim] ;
    l_bound = l_bound_host.data() , r_bound = r_bound_host.data() ;

    cudaMalloc((void**) &l_bound_dev , attr_dim * query_num * sizeof(attrType)) ;
    cudaMalloc((void**) &r_bound_dev , attr_dim * query_num * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_min , attr_dim * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_width , attr_dim * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_grid_size , attr_dim * sizeof(unsigned)) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    cudaMemcpy(l_bound_dev , l_bound , query_num * attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(r_bound_dev , r_bound , query_num * attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_min , attr_min + attr_dim , 0.0) ;
    cudaMemcpy(attr_min , grid_min.data() , attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_width , attr_width + attr_dim , 250000.0) ;
    cudaMemcpy(attr_width , attr_width_host.data() , attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_grid_size , attr_grid_size + attr_dim , 4) ;
    cudaMemcpy(attr_grid_size , grid_size_host.data() , attr_dim * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    unsigned qnum = query_num ;
    unsigned pnum = thrust::reduce(
        thrust::device , 
        attr_grid_size , attr_grid_size + attr_dim ,
        1 , thrust::multiplies<unsigned>() 
    ) ;
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    bool *partition_is_selected ; 
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg_batch_process<attr_dim , attrType>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , 0.0 , partition_is_selected) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    // ************************************* 以下声明一些要用到的数据结构, 以及调用函数 -----------------------------------
    // vector<vector<unsigned>> batch_info(4 , vector<unsigned>(4)) ;
    // for(int i =0 ; i < 16 ; i ++)
    //     batch_info[i / 4][i % 4] = i ; 
    // vector<vector<unsigned>> batch_info = {
        // {0 , 1 , 2} ,
        // {3 , 4 , 5} 
        // {6 , 7 , 8} ,
        // {9 , 10 , 11} ,
        // {12,  13 , 14} ,
        // {15} 
    // } ;

    size_t free_bytes, total_bytes;

    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    double free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    double total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "准备工作完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;


    vector<vector<unsigned>> batch_info = {
        {14, 10, 15},
        {7, 11, 13},
        {9, 5, 6},
        {4, 0, 1},
        {2, 12, 8},
        {3}
    } ;
    // vector<vector<unsigned>> batch_info(16 , vector<unsigned>(1)) ;
    // for(unsigned i =0 ; i < 16 ; i ++)
    //     batch_info[i][0] = i ; 
    const unsigned batch_size = 3 ;
    // random_shuffle(batch_info.begin() , batch_info.end()) ;
    __half *data_half_buffer1 , *data_half_buffer2 ;
    size_t data_half_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(average_num) * static_cast<size_t>(dim) * sizeof(__half) ;
    cudaMalloc((void**) &data_half_buffer1 , data_half_buffer_size) ;
    cudaMalloc((void**) &data_half_buffer2 , data_half_buffer_size) ;
    array<__half*,2> vec_buffer = {data_half_buffer1 , data_half_buffer2} ;
    unsigned* graph_buffer0 , *graph_buffer1 ; 
    size_t graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(degree) * static_cast<size_t>(average_num) * sizeof(unsigned) ;
    cudaMalloc((void**) &graph_buffer0 , graph_buffer_size) ;
    cudaMalloc((void**) &graph_buffer1 , graph_buffer_size) ;
    array<unsigned*,2> graph_buffer = {graph_buffer0 , graph_buffer1} ;
    unsigned* global_graph_buffer0 , *global_graph_buffer1 ;
    // 此处超出内存限制
    // size_t global_graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(global_graph_degree) * static_cast<size_t>(average_num) * sizeof(unsigned) ;
    size_t global_graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<const size_t>(batch_size) * static_cast<size_t>(global_graph_edge_per_p) * static_cast<size_t>(average_num) * sizeof(unsigned) ;
    cudaMalloc((void**) &global_graph_buffer0 , global_graph_buffer_size) ;
    cudaMalloc((void**) &global_graph_buffer1 , global_graph_buffer_size) ;
    array<unsigned*,2> global_graph_buffer = {global_graph_buffer0 , global_graph_buffer1} ;
    unsigned* generic_buffer0 , *generic_buffer1 ; 
    cudaMalloc((void**) &generic_buffer0 , 10 * 1024 * 1024 * sizeof(unsigned)) ;
    cudaMalloc((void**) &generic_buffer1 , 10 * 1024 * 1024 * sizeof(unsigned)) ;
    array<unsigned*,2> generic_buffer = {generic_buffer0 , generic_buffer1} ;
    unsigned* stored_pos_global_idx ;
    cudaMalloc((void**) &stored_pos_global_idx , num * sizeof(unsigned)) ;
    array<unsigned*,4> new_idx_mapper = {idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , stored_pos_global_idx} ;
    vector<unsigned> seg_partition_point_start_index(16) ;
    seg_partition_point_start_index[0] = 0 ;
    for(int i = 1; i < 16 ; i ++)
        seg_partition_point_start_index[i] = seg_partition_point_start_index[i - 1] + graph_infos[1][i - 1] ;

    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cudaError_t err2 = cudaMemGetInfo(&free_bytes, &total_bytes) ;
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "缓冲区申请完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;
    cout << "准备工作完成, 开始查找 :" << endl ;
    s = chrono::high_resolution_clock::now() ;
    // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
    //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
    //     global_graph , 16 , pnum , L) ;
    

    vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter_batch_process_compressed_graph_T<attrType>(batch_info.size() , batch_info , degree , data_ , vec_buffer, graph_buffer , global_graph_buffer ,
        generic_buffer , query_data_half_dev , dim , qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        attr_dim  , graph_infos , new_idx_mapper , seg_partition_point_start_index  , partition_infos , L , 10 , 
        global_graph , global_graph_edge_per_p , pnum , partition_is_selected , 10 , num , 16) ;
    e = chrono::high_resolution_clock::now() ;
    cost3 = chrono::duration<double>(e - s).count() ;
    cout << "finish search" << endl ;

    cudaFree(data_half_buffer1) ;
    cudaFree(data_half_buffer2) ;
    cudaFree(graph_buffer0) ;
    cudaFree(graph_buffer1) ;
    cudaFree(global_graph_buffer0) ;
    cudaFree(global_graph_buffer1) ;
    cudaFree(generic_buffer0) ;
    cudaFree(generic_buffer1) ;
    cudaFree(stored_pos_global_idx) ;

    cout << "gpu 暴力:" << endl ;
    auto gpu_bf_s = chrono::high_resolution_clock::now() ;
    vector<vector<unsigned>> gpu_bf_result ;
    gpu_bf_result = prefiltering_and_bruteforce<attrType>(query_data_half_dev , data_ , qnum ,  10 , dim , num , attrs_dev,
       l_bound_dev , r_bound_dev , attr_dim) ;
    auto gpu_bf_e = chrono::high_resolution_clock::now() ;
    float gpu_bf_cost = chrono::duration<double>(gpu_bf_e - gpu_bf_s).count() ;
    cout << "gpu 暴力 结束: " << gpu_bf_cost << endl ;

    // save_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt", query_num , 10, gpu_bf_result) ;
    // load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt" , gpu_bf_result) ;
    // load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt" , gpu_bf_result) ;
    cudaFreeHost(data_) ;
    cudaFreeHost(global_graph) ;
    cudaFreeHost(graph_infos[0]) ;
    cudaFreeHost(graph_infos[1]) ;
    cudaFreeHost(graph_infos[2]) ;
    /**
        float* data_float , * query_data_load_float ;
        cout << "开始暴力搜索, 载入数据--------------------" << endl ;
        load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs" , data_float , num , dim) ;
        cout << "num : " << num << ", dim : " << dim << endl ;
        query_data_load_float = new float[query_num * query_dim] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ; j < query_dim ; j ++)
            // query_data_load_float[i * query_dim + j] = data_float[qids[i] * query_dim + j] ;
            size_t ofst = static_cast<size_t>(qids[i]) * static_cast<size_t>(query_dim) ;
            memcpy(query_data_load_float + i * query_dim , data_float + ofst , query_dim * sizeof(float)) ;
        }
        // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
        vector<vector<unsigned>> ground_truth(qnum) ;
        vector<vector<float>> gt_dis(qnum) ;
        auto cpu_bf_s = chrono::high_resolution_clock::now() ;
        #pragma omp parallel for num_threads(10)
        for(unsigned i = 0 ; i < qnum ;i  ++) {
            vector<float> gt_d ; 
            ground_truth[i] = naive_brute_force_with_filter_store_dis<attrType>(query_data_load_float + i * dim , data_float , dim , num , 10 ,
                l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim , gt_d) ;
            gt_dis[i] = gt_d ; 
        }
        auto cpu_bf_e = chrono::high_resolution_clock::now() ;
        float cpu_bf_cost = chrono::duration<double>(cpu_bf_e - cpu_bf_s).count() ;
        reverse(ground_truth[0].begin() , ground_truth[0].end()) ;
        reverse(gt_dis[0].begin() , gt_dis[0].end()) ;
        cout << "gt id :[" ;
        for(unsigned i = 0 ; i < ground_truth[0].size() ;i  ++) {
            cout << ground_truth[0][i] ;
            if(i < ground_truth[0].size() - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;
        cout << "gt dis :[" ;
        for(unsigned i = 0 ; i < gt_dis[0].size() ;i  ++) {
            cout << gt_dis[0][i] ;
            if(i < gt_dis[0].size() - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;
        // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
        float total_recall = cal_recall_show_partition_multi_attrs<attrType>(result , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;

        
        cout << "recall : " << total_recall << endl  ;
    **/
    cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
    cout << "获取分区列表用时:" << cost1 << endl ;
    cout << "对分区列表排序用时:" << cost2 << endl ;
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "核函数 + 数据运输、准备工作:" << other_kernal_cost << endl ;
    cout << "数据传输时间: " << mem_trans_cost << endl ;
    cout << "搜索用时:" << cost3 << endl ;
    cout << "batch cost : [" ;
    for(unsigned i = 0 ; i < batch_process_cost.size() ; i ++) {
        cout << batch_process_cost[i] ;
        if(i < batch_process_cost.size() - 1)
            cout << " ," ;
    }
    cout << "]" << endl ; 
    cout << "batch mem load cost : [" ;
    for(unsigned i = 0 ; i < batch_data_trans_cost.size() ; i ++) {
        cout << batch_data_trans_cost[i] ;
        if(i < batch_data_trans_cost.size() - 1)
            cout << " ," ;
    }
    cout << "]" << endl ;
    cout << "gpu brute force 时间: " << gpu_bf_cost << endl ;
    gpu_bf_result.resize(qnum) ;
    // cout << "cpu brute force 时间: " << cpu_bf_cost << endl ;
    float total_recall = cal_recall_show_partition_multi_attrs<attrType>(result , gpu_bf_result , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;
    cout << " recall : " << total_recall << endl ;


    rc_recall1 = total_recall ;
    rc_cost1 = select_path_kernal_cost ;

    cudaFree(global_graph) ;
    cudaFree(attrs_dev) ;
    cudaFree(query_data_half_dev) ;
    cudaFree(l_bound_dev) ;
    cudaFree(r_bound_dev) ;
    cudaFree(attr_min) ;
    cudaFree(attr_width) ;
    cudaFree(attr_grid_size) ;
    cudaFree(partition_is_selected) ;
    for(unsigned i = 0  ; i < 5 ; i ++)
        cudaFree(partition_infos[i]) ;
    for(unsigned i = 0 ; i < 3 ; i ++) 
        cudaFree(idx_mapper[i]) ;
    
}

void multi_attr_test_batch_process_large_dataset_compressed_graph_T36(unsigned L = 80) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    showMemInfo("算法开始") ;
    unsigned average_num = 6250000 ;
    array<unsigned*,3> graph_infos = get_graph_info_host("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/deep100M/36/deep100M" 
    , ".cagra16" , 36 , 100000000 , 16) ;
    showMemInfo("图索引载入完成") ;
    // unsigned offset4 = 4 * average_num * 16 ;
    // for(unsigned i = 0 ; i < 100 ; i ++) {
    //     for(unsigned j = 0 ; j < 16 ; j ++) {
    //         cout << graph_infos[0][offset4 + i * 16 + j] << " ," ;
    //     }
    //     cout << endl;
    // }
    unsigned preset_num = 1e8 , preset_pnum = 36 ;

    unsigned degree = 16 , global_graph_edge_per_p = 2 ; 
    array<unsigned*,3> idx_mapper = get_idx_mapper(100000000 , 36 , graph_infos) ;
    showMemInfo("id映射器分配完成") ;

    // 每个分区对应向量id起始位置
    vector<unsigned> seg_partition_point_start_index(preset_pnum) ;
    seg_partition_point_start_index[0] = 0 ;
    for(int i = 1; i < preset_pnum ; i ++)
        seg_partition_point_start_index[i] = seg_partition_point_start_index[i - 1] + graph_infos[1][i - 1] ;

    // vector<vector<unsigned>> global_edges ;
    // load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges) ;
    unsigned *global_graph ;
    unsigned global_graph_degree , global_graph_num_in_graph ;
    // cudaMalloc((void**) &global_graph , 1000000 * 16 * 16 * sizeof(unsigned)) ;
    load_result_pined("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/deep100M/global_graph_T36.graphT" ,
     global_graph , global_graph_degree , global_graph_num_in_graph) ;
    cout << "global graph : degree-" << global_graph_degree << ", num-" << global_graph_num_in_graph << endl ;
    showMemInfo("全局图载入完成") ;
    // for(unsigned i = 0 ; i < 1000000 ; i ++) {
    //     cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    // }
    using attrType = double ;
    attrType *attrs_dev , *attrs ;
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    constexpr unsigned attr_dim =  2 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<attrType> grid_min(attr_dim , 0.0) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<attrType> grid_max(attr_dim , 1e8) ;
    vector<unsigned> grid_size_host = {6 , 6} ;
    vector<attrType> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++) {
        attr_width_host[i] = (grid_max[i] - grid_min[i]) / grid_size_host[i] ;
        cout << "attr_width_host[i] : " << attr_width_host[i] << "," ;
    }
    attrs = new attrType[attr_dim * 100000000] ;
    // re_generate<attrType>(attrs , attr_dim , 100000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    re_generate<attrType>(attrs , attr_dim , preset_num , grid_min.data() , grid_max.data() , grid_size_host.data() , graph_infos[1] , seg_partition_point_start_index.data() , preset_pnum) ;
    // save_attrs("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/attrs.bin" ,100000000 , attr_dim, attrs) ;
    // load_attrs<attrType>("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/attrs.bin", attrs) ;
    size_t attrs_dev_byteSize = static_cast<size_t>(attr_dim) * 100000000 * sizeof(attrType) ;
    cudaMalloc((void**) &attrs_dev , attrs_dev_byteSize) ;
    cudaMemcpy(attrs_dev , attrs , attrs_dev_byteSize , cudaMemcpyHostToDevice) ;
    delete [] attrs ;
    cout << "sizeof(half) = " << sizeof(half) << endl ;

    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    double free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    double total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "属性分配完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;


    // 先载入数据集和图索引
    unsigned num , dim  ;
    __half* data_   ;
    cout << "开始载入数据集" << endl ;
    data_ = load_dataset_return_pined_pointer("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs" ,
     num , dim , 10.0) ;

    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "数据转换完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;


    __half* query_data_load ;
    unsigned query_num = 1 , query_dim = dim ;
    vector<unsigned> qids(query_num * 10) ;
    /**
    unordered_set<unsigned> qids_contains ;
    mt19937 rng ;
    uniform_int_distribution<> intd(0 , num - 1) ;
    unsigned qids_ofst = 0 ;
    while(qids_ofst < query_num * 10) {
        unsigned t = intd(rng) ;
        if(!qids_contains.count(t)) {
            qids_contains.insert(t) ;
            qids[qids_ofst ++] = t ;
        }
    }
    random_shuffle(qids.begin() , qids.end()) ;
    **/
    // save_query_ids("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/qids.bin" , query_num , qids) ;
    load_query_ids("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/qids.bin" , query_num , qids) ;
    // query_num = 10000 ; 
    // qids[0] = 0 ;

    query_data_load = new __half[query_num * query_dim] ;
    for(unsigned i = 0 ; i < query_num ; i ++) {
        // for(unsigned j = 0 ; j < query_dim ; j ++)
        //     query_data_load[i * query_dim + j] = data_[qids[i] * query_dim + j] ;
        size_t ofst = static_cast<size_t>(qids[i]) * static_cast<size_t>(query_dim) ;
        memcpy(query_data_load + i * query_dim , data_ + ofst , query_dim * sizeof(__half)) ;
    }
    // query_num = 1 ; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;

    half *query_data_half_dev ;
    
    cudaMalloc((void**) &query_data_half_dev , query_num * query_dim * sizeof(__half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(query_data_half_dev , query_data_load , query_num * query_dim * sizeof(__half) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num ; 
    
    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    vector<attrType> l_bound_host(query_num * attr_dim) , r_bound_host(query_num * attr_dim) ;
    if(selectivity == 0.0) {
        cout << "sel = " << selectivity << endl ;
        load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        query_range_x.reserve(query_num) ;
        for(unsigned i = 1 ; i < 10 ; i ++)
            query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        query_range_y = query_range_x ; 

        // query_range_x[0] = {0 ,16666600 } , query_range_y[0] = {0 , 16660000 * 3} ;
        // for(unsigned i = 1 ; i < query_num ; i ++)
        //     query_range_y[i] = query_range_y[0] , query_range_x[i] = query_range_x[0] ;
        // query_range_x[0] = {25000000 , 49999999} ;
        // query_range_y[0] = {0 , 49999999} ;

        for(unsigned i = 1 ; i < query_num ;i  ++) {
            query_range_x[i].first *= 100 , query_range_x[i].second *= 100 ; 
            query_range_y[i].first *= 100 , query_range_y[i].second *= 100 ;
        }

            // query_range_x[i] = query_range_x[0] , query_range_y[i] = query_range_y[0] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ;j < attr_dim ; j ++)
            l_bound_host[i * attr_dim] = (attrType)query_range_x[i].first , r_bound_host[i * attr_dim] = (attrType)query_range_x[i].second ;
            l_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].first , r_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].second ;
        }

        cout << "range 0 (" << query_range_x[0].first << "," << query_range_x[0].second << ")" << endl ;
        cout << "range 1 (" << query_range_x[1].first << "," << query_range_x[1].second << ")" << endl ;
        cout << "range 2 (" << query_range_x[2].first << "," << query_range_x[2].second << ")" << endl ;
        cout << "range 5 (" << query_range_x[5].first << "," << query_range_x[5].second << ")" << endl ;
        // query_range_y[0] = query_range_x[0] = {490000 , 510000} ;
    
        // query_num = 1; 
        // query_range_x[0].first = 250000 , query_range_x[0].second = 499999 ;
        // query_range_y[0].first = 250000 , query_range_y[0].second = 499999 ;
        // for(unsigned i = 0 ; i < query_num ;i ++) {
        //     query_range_x[i] = query_range_x.front() ;
        //     query_range_y[i] = query_range_y.front() ;
        // }
    } else {
        cout << "selectivity : " << selectivity << endl ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_x , query_num) ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_y , query_num) ;
        generate_range_multi_attr_with_selectivity<attrType>(l_bound_host.data() , r_bound_host.data() , attr_dim , grid_min.data() , grid_max.data() , query_num , selectivity) ;
        cout << "随机生成属性, 完成" << endl ;
    }


    attrType* l_bound , * r_bound , *l_bound_dev , *r_bound_dev , *attr_min , *attr_width ;
    unsigned *attr_grid_size ;
    // l_bound = new float[query_num * attr_dim] , r_bound = new float[query_num * attr_dim] ;
    l_bound = l_bound_host.data() , r_bound = r_bound_host.data() ;

    cudaMalloc((void**) &l_bound_dev , attr_dim * query_num * sizeof(attrType)) ;
    cudaMalloc((void**) &r_bound_dev , attr_dim * query_num * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_min , attr_dim * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_width , attr_dim * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_grid_size , attr_dim * sizeof(unsigned)) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    cudaMemcpy(l_bound_dev , l_bound , query_num * attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(r_bound_dev , r_bound , query_num * attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_min , attr_min + attr_dim , 0.0) ;
    cudaMemcpy(attr_min , grid_min.data() , attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_width , attr_width + attr_dim , 250000.0) ;
    cudaMemcpy(attr_width , attr_width_host.data() , attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_grid_size , attr_grid_size + attr_dim , 4) ;
    cudaMemcpy(attr_grid_size , grid_size_host.data() , attr_dim * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    unsigned qnum = query_num ;
    unsigned pnum = thrust::reduce(
        thrust::device , 
        attr_grid_size , attr_grid_size + attr_dim ,
        1 , thrust::multiplies<unsigned>() 
    ) ;
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    // size_t free_bytes, total_bytes;

    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "属性, 查询范围分配完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;

    bool *partition_is_selected ; 
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg_batch_process<attr_dim , attrType>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , 0.05 , partition_is_selected) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    // ************************************* 以下声明一些要用到的数据结构, 以及调用函数 -----------------------------------
    // vector<vector<unsigned>> batch_info(4 , vector<unsigned>(4)) ;
    // for(int i =0 ; i < 16 ; i ++)
    //     batch_info[i / 4][i % 4] = i ; 
    // vector<vector<unsigned>> batch_info = {
        // {0 , 1 , 2} ,
        // {3 , 4 , 5} 
        // {6 , 7 , 8} ,
        // {9 , 10 , 11} ,
        // {12,  13 , 14} ,
        // {15} 
    // } ;



    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "准备工作完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;


    // vector<vector<unsigned>> batch_info = {
    //     {14, 10, 15},
    //     {7, 11, 13},
    //     {9, 5, 6},
    //     {4, 0, 1},
    //     {2, 12, 8},
    //     {3}
    // } ;

    // vector<vector<unsigned>> batch_info = {
    //     {0 , 1 , 2} ,
    //     {3 , 4 , 5} ,
    //     {6 , 7 , 8} ,
    //     {9 , 10 , 11} ,
    //     {12 , 13 , 14} ,
    //     {15 , 16 , 17} ,
    //     {18 , 19 , 20} ,
    //     {21 , 22 , 23} ,
    //     {24 , 25 , 26} ,
    //     {27 , 28 , 29} ,
    //     {30 , 31 , 32} ,
    //     {33 , 34 , 35} ,
    // } ;
    // vector<vector<unsigned>> batch_info(4 , vector<unsigned>(9)) ;
    // for(unsigned i = 0 ; i < pnum ;i ++)
    //     batch_info[i / 9][i % 9] = i ; 
    // vector<vector<unsigned>> batch_info =  {
    //     {0 , 1 , 2 , 3 , 4 , 5 , 6 , 7} ,
    //     {8 , 9 , 10 , 11 , 12 , 13 , 14 , 15} ,
    //     {16 ,  17 , 18 , 19 , 20 , 21 , 22 , 23 } ,
    //     {24 , 25 , 26 , 27 , 28 , 29 , 30, 31 } , 
    //     {32 , 33 , 34 , 35}
    // } ;
    // vector<vector<unsigned>> batch_info(9 , vector<unsigned>(4)) ;
    // for(unsigned i = 0 ; i < pnum ; i ++)
    //     batch_info[i / 4][i % 4] = i ; 

    vector<vector<unsigned>> batch_info = {
        {29, 35, 34, 28},
        {21, 27, 22, 16},
        {17, 32, 33, 23},
        {20, 14, 15, 26}, 
        {7, 8, 13, 0}, 
        {25, 19, 9, 10}, 
        {11, 5, 31, 30}, 
        {2, 1, 6, 12}, 
        {24, 4, 18, 3}
    } ;

    // vector<vector<unsigned>> batch_info(16 , vector<unsigned>(1)) ;
    // for(unsigned i =0 ; i < 16 ; i ++)
    //     batch_info[i][0] = i ; 
    unsigned batch_size = 0 ;
    for(unsigned i = 0 ; i < batch_info.size() ; i ++)
        batch_size = max(batch_size , (unsigned) batch_info[i].size()) ;
    unsigned reserve_points_num = *max_element(graph_infos[1] , graph_infos[1] + pnum) ;
    // random_shuffle(batch_info.begin() , batch_info.end()) ;
    __half *data_half_buffer1 , *data_half_buffer2 ;
    size_t data_half_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(reserve_points_num) * static_cast<size_t>(dim) * sizeof(__half) ;
    cudaMalloc((void**) &data_half_buffer1 , data_half_buffer_size) ;
    cudaMalloc((void**) &data_half_buffer2 , data_half_buffer_size) ;
    array<__half*,2> vec_buffer = {data_half_buffer1 , data_half_buffer2} ;


    unsigned* graph_buffer0 , *graph_buffer1 ; 
    size_t graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(degree) * static_cast<size_t>(reserve_points_num) * sizeof(unsigned) ;
    cudaMalloc((void**) &graph_buffer0 , graph_buffer_size) ;
    cudaMalloc((void**) &graph_buffer1 , graph_buffer_size) ;
    array<unsigned*,2> graph_buffer = {graph_buffer0 , graph_buffer1} ;
    unsigned* global_graph_buffer0 , *global_graph_buffer1 ;
    // 此处超出内存限制
    // size_t global_graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(global_graph_degree) * static_cast<size_t>(average_num) * sizeof(unsigned) ;
    size_t global_graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<const size_t>(batch_size) * static_cast<size_t>(global_graph_edge_per_p) * static_cast<size_t>(reserve_points_num) * sizeof(unsigned) ;
    cudaMalloc((void**) &global_graph_buffer0 , global_graph_buffer_size) ;
    cudaMalloc((void**) &global_graph_buffer1 , global_graph_buffer_size) ;
    array<unsigned*,2> global_graph_buffer = {global_graph_buffer0 , global_graph_buffer1} ;
    unsigned* generic_buffer0 , *generic_buffer1 ; 
    cudaMalloc((void**) &generic_buffer0 , 10 * 1024 * 1024 * sizeof(unsigned)) ;
    cudaMalloc((void**) &generic_buffer1 , 10 * 1024 * 1024 * sizeof(unsigned)) ;
    array<unsigned*,2> generic_buffer = {generic_buffer0 , generic_buffer1} ;
    unsigned* stored_pos_global_idx ;
    cudaMalloc((void**) &stored_pos_global_idx , num * sizeof(unsigned)) ;
    array<unsigned*,4> new_idx_mapper = {idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , stored_pos_global_idx} ;


    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cudaError_t err2 = cudaMemGetInfo(&free_bytes, &total_bytes) ;
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "缓冲区申请完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;
    cout << "准备工作完成, 开始查找 :" << endl ;
    s = chrono::high_resolution_clock::now() ;
    // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
    //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
    //     global_graph , 16 , pnum , L) ;
    

    vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter_batch_process_compressed_graph_T<attrType>(batch_info.size() , batch_info , degree , data_ , vec_buffer, graph_buffer , global_graph_buffer ,
        generic_buffer , query_data_half_dev , dim , qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        attr_dim  , graph_infos , new_idx_mapper , seg_partition_point_start_index  , partition_infos , L , 10 , 
        global_graph , global_graph_edge_per_p , pnum , partition_is_selected , 10 , num , 16) ;
    e = chrono::high_resolution_clock::now() ;
    cost3 = chrono::duration<double>(e - s).count() ;
    cout << "finish search" << endl ;

    cudaFree(data_half_buffer1) ;
    cudaFree(data_half_buffer2) ;
    cudaFree(graph_buffer0) ;
    cudaFree(graph_buffer1) ;
    cudaFree(global_graph_buffer0) ;
    cudaFree(global_graph_buffer1) ;
    cudaFree(generic_buffer0) ;
    cudaFree(generic_buffer1) ;
    cudaFree(stored_pos_global_idx) ;

    cout << "gpu 暴力:" << endl ;
    auto gpu_bf_s = chrono::high_resolution_clock::now() ;
    vector<vector<unsigned>> gpu_bf_result ;
    gpu_bf_result = prefiltering_and_bruteforce<attrType>(query_data_half_dev , data_ , qnum ,  10 , dim , num , attrs_dev,
       l_bound_dev , r_bound_dev , attr_dim) ;
    auto gpu_bf_e = chrono::high_resolution_clock::now() ;
    float gpu_bf_cost = chrono::duration<double>(gpu_bf_e - gpu_bf_s).count() ;
    cout << "gpu 暴力 结束: " << gpu_bf_cost << endl ;

    // save_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt", query_num , 10, gpu_bf_result) ;
    // load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt" , gpu_bf_result) ;
    // load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt" , gpu_bf_result) ;
    cudaFreeHost(data_) ;
    cudaFreeHost(global_graph) ;
    cudaFreeHost(graph_infos[0]) ;
    cudaFreeHost(graph_infos[1]) ;
    cudaFreeHost(graph_infos[2]) ;
    /**
        float* data_float , * query_data_load_float ;
        cout << "开始暴力搜索, 载入数据--------------------" << endl ;
        load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs" , data_float , num , dim) ;
        cout << "num : " << num << ", dim : " << dim << endl ;
        query_data_load_float = new float[query_num * query_dim] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ; j < query_dim ; j ++)
            // query_data_load_float[i * query_dim + j] = data_float[qids[i] * query_dim + j] ;
            size_t ofst = static_cast<size_t>(qids[i]) * static_cast<size_t>(query_dim) ;
            memcpy(query_data_load_float + i * query_dim , data_float + ofst , query_dim * sizeof(float)) ;
        }
        // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
        vector<vector<unsigned>> ground_truth(qnum) ;
        vector<vector<float>> gt_dis(qnum) ;
        auto cpu_bf_s = chrono::high_resolution_clock::now() ;
        #pragma omp parallel for num_threads(10)
        for(unsigned i = 0 ; i < qnum ;i  ++) {
            vector<float> gt_d ; 
            ground_truth[i] = naive_brute_force_with_filter_store_dis<attrType>(query_data_load_float + i * dim , data_float , dim , num , 10 ,
                l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim , gt_d) ;
            gt_dis[i] = gt_d ; 
        }
        auto cpu_bf_e = chrono::high_resolution_clock::now() ;
        float cpu_bf_cost = chrono::duration<double>(cpu_bf_e - cpu_bf_s).count() ;
        reverse(ground_truth[0].begin() , ground_truth[0].end()) ;
        reverse(gt_dis[0].begin() , gt_dis[0].end()) ;
        cout << "gt id :[" ;
        for(unsigned i = 0 ; i < ground_truth[0].size() ;i  ++) {
            cout << ground_truth[0][i] ;
            if(i < ground_truth[0].size() - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;
        cout << "gt dis :[" ;
        for(unsigned i = 0 ; i < gt_dis[0].size() ;i  ++) {
            cout << gt_dis[0][i] ;
            if(i < gt_dis[0].size() - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;
        // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
        float total_recall = cal_recall_show_partition_multi_attrs<attrType>(result , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;

        
        cout << "recall : " << total_recall << endl  ;
    **/
    cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
    cout << "获取分区列表用时:" << cost1 << endl ;
    cout << "对分区列表排序用时:" << cost2 << endl ;
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "核函数 + 数据运输、准备工作:" << other_kernal_cost << endl ;
    cout << "数据传输时间: " << mem_trans_cost << endl ;
    cout << "搜索用时:" << cost3 << endl ;
    cout << "batch cost : [" ;
    for(unsigned i = 0 ; i < batch_process_cost.size() ; i ++) {
        cout << batch_process_cost[i] ;
        if(i < batch_process_cost.size() - 1)
            cout << " ," ;
    }
    cout << "]" << endl ; 
    cout << "batch mem load cost : [" ;
    for(unsigned i = 0 ; i < batch_data_trans_cost.size() ; i ++) {
        cout << batch_data_trans_cost[i] ;
        if(i < batch_data_trans_cost.size() - 1)
            cout << " ," ;
    }
    cout << "]" << endl ;
    cout << "gpu brute force 时间: " << gpu_bf_cost << endl ;
    gpu_bf_result.resize(qnum) ;
    // cout << "cpu brute force 时间: " << cpu_bf_cost << endl ;
    float total_recall = cal_recall_show_partition_multi_attrs<attrType>(result , gpu_bf_result , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , false) ;
    cout << " recall : " << total_recall << endl ;


    rc_recall1 = total_recall ;
    rc_cost1 = select_path_kernal_cost ;

    cudaFree(global_graph) ;
    cudaFree(attrs_dev) ;
    cudaFree(query_data_half_dev) ;
    cudaFree(l_bound_dev) ;
    cudaFree(r_bound_dev) ;
    cudaFree(attr_min) ;
    cudaFree(attr_width) ;
    cudaFree(attr_grid_size) ;
    cudaFree(partition_is_selected) ;
    for(unsigned i = 0  ; i < 5 ; i ++)
        cudaFree(partition_infos[i]) ;
    for(unsigned i = 0 ; i < 3 ; i ++) 
        cudaFree(idx_mapper[i]) ;
    
}

void multi_attr_test_batch_process_large_dataset_compressed_graph_T36_dataset_resident(unsigned L = 80) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    showMemInfo("算法开始") ;
    unsigned average_num = 6250000 ;
    array<unsigned*,3> graph_infos = get_graph_info_host("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/deep100M/36/deep100M" 
    , ".cagra16" , 36 , 100000000 , 16) ;
    showMemInfo("图索引载入完成") ;
    // unsigned offset4 = 4 * average_num * 16 ;
    // for(unsigned i = 0 ; i < 100 ; i ++) {
    //     for(unsigned j = 0 ; j < 16 ; j ++) {
    //         cout << graph_infos[0][offset4 + i * 16 + j] << " ," ;
    //     }
    //     cout << endl;
    // }
    unsigned preset_num = 1e8 , preset_pnum = 36 ;

    unsigned degree = 16 , global_graph_edge_per_p = 2 ; 
    array<unsigned*,3> idx_mapper = get_idx_mapper(100000000 , 36 , graph_infos) ;
    showMemInfo("id映射器分配完成") ;

    // 每个分区对应向量id起始位置
    vector<unsigned> seg_partition_point_start_index(preset_pnum) ;
    seg_partition_point_start_index[0] = 0 ;
    for(int i = 1; i < preset_pnum ; i ++)
        seg_partition_point_start_index[i] = seg_partition_point_start_index[i - 1] + graph_infos[1][i - 1] ;

    // vector<vector<unsigned>> global_edges ;
    // load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges) ;
    unsigned *global_graph ;
    unsigned global_graph_degree , global_graph_num_in_graph ;
    // cudaMalloc((void**) &global_graph , 1000000 * 16 * 16 * sizeof(unsigned)) ;
    load_result_pined("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/deep100M/global_graph_T36.graphT" ,
     global_graph , global_graph_degree , global_graph_num_in_graph) ;
    cout << "global graph : degree-" << global_graph_degree << ", num-" << global_graph_num_in_graph << endl ;
    showMemInfo("全局图载入完成") ;
    // for(unsigned i = 0 ; i < 1000000 ; i ++) {
    //     cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    // }
    using attrType = double ;
    attrType *attrs_dev , *attrs ;
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    constexpr unsigned attr_dim =  2 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<attrType> grid_min(attr_dim , 0.0) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<attrType> grid_max(attr_dim , 1e8) ;
    vector<unsigned> grid_size_host = {6 , 6} ;
    vector<attrType> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++) {
        attr_width_host[i] = (grid_max[i] - grid_min[i]) / grid_size_host[i] ;
        cout << "attr_width_host[i] : " << attr_width_host[i] << "," ;
    }
    attrs = new attrType[attr_dim * 100000000] ;
    // re_generate<attrType>(attrs , attr_dim , 100000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    re_generate<attrType>(attrs , attr_dim , preset_num , grid_min.data() , grid_max.data() , grid_size_host.data() , graph_infos[1] , seg_partition_point_start_index.data() , preset_pnum) ;
    // save_attrs("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/attrs.bin" ,100000000 , attr_dim, attrs) ;
    // load_attrs<attrType>("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/attrs.bin", attrs) ;
    size_t attrs_dev_byteSize = static_cast<size_t>(attr_dim) * 100000000 * sizeof(attrType) ;
    cudaMalloc((void**) &attrs_dev , attrs_dev_byteSize) ;
    cudaMemcpy(attrs_dev , attrs , attrs_dev_byteSize , cudaMemcpyHostToDevice) ;
    delete [] attrs ;
    cout << "sizeof(half) = " << sizeof(half) << endl ;

    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    double free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    double total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "属性分配完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;


    // 先载入数据集和图索引
    unsigned num , dim  ;
    __half* data_   ;
    cout << "开始载入数据集" << endl ;
    data_ = load_dataset_return_pined_pointer("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs" ,
     num , dim , 10.0) ;
     cout << "数据集载入完成: num:" << num << ", dim :" << dim << endl ;

    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "数据转换完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;


    __half* query_data_load ;
    unsigned query_num = 1 , query_dim = dim ;
    vector<unsigned> qids(query_num * 10) ;
    /**
    unordered_set<unsigned> qids_contains ;
    mt19937 rng ;
    uniform_int_distribution<> intd(0 , num - 1) ;
    unsigned qids_ofst = 0 ;
    while(qids_ofst < query_num * 10) {
        unsigned t = intd(rng) ;
        if(!qids_contains.count(t)) {
            qids_contains.insert(t) ;
            qids[qids_ofst ++] = t ;
        }
    }
    random_shuffle(qids.begin() , qids.end()) ;
    **/
    // save_query_ids("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/qids.bin" , query_num , qids) ;
    load_query_ids("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/qids.bin" , query_num , qids) ;
    // query_num = 1000 ; 
    // qids[0] = 0 ;

    query_data_load = new __half[query_num * query_dim] ;
    for(unsigned i = 0 ; i < query_num ; i ++) {
        // for(unsigned j = 0 ; j < query_dim ; j ++)
        //     query_data_load[i * query_dim + j] = data_[qids[i] * query_dim + j] ;
        size_t ofst = static_cast<size_t>(qids[i]) * static_cast<size_t>(query_dim) ;
        memcpy(query_data_load + i * query_dim , data_ + ofst , query_dim * sizeof(__half)) ;
    }
    // query_num = 1000 ; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;

    half *query_data_half_dev ;
    
    cudaMalloc((void**) &query_data_half_dev , query_num * query_dim * sizeof(__half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(query_data_half_dev , query_data_load , query_num * query_dim * sizeof(__half) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num ; 
    
    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    vector<attrType> l_bound_host(query_num * attr_dim) , r_bound_host(query_num * attr_dim) ;
    if(selectivity == 0.0) {
        cout << "sel = " << selectivity << endl ;
        load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift1M/sift_query_range.bin", query_range_x, 1000) ;
        query_range_x.reserve(query_num) ;
        for(unsigned i = 1 ; i < 10 ; i ++)
            query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        query_range_y = query_range_x ; 

        query_range_x[0] = {0 ,16666600 } , query_range_y[0] = {0 , 16666600 * 4} ;
        // for(unsigned i = 1 ; i < query_num ; i ++)
        //     query_range_y[i] = query_range_y[0] , query_range_x[i] = query_range_x[0] ;
        // query_range_x[0] = {0 , 24999999} ;
        // query_range_y[0] = {0 , 24999999} ;

        for(unsigned i = 1 ; i < query_num ;i  ++) {
            // query_range_x[i].first *= 100 , query_range_x[i].second *= 100 ; 
            // query_range_y[i].first *= 100 , query_range_y[i].second *= 100 ;
            query_range_x[i] = query_range_x[0] , query_range_y[i] = query_range_y[0] ;
        }

            // query_range_x[i] = query_range_x[0] , query_range_y[i] = query_range_y[0] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ;j < attr_dim ; j ++)
            l_bound_host[i * attr_dim] = (attrType)query_range_x[i].first , r_bound_host[i * attr_dim] = (attrType)query_range_x[i].second ;
            l_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].first , r_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].second ;
        }

        cout << "range 0 (" << query_range_x[0].first << "," << query_range_x[0].second << ")" << endl ;
        cout << "range 1 (" << query_range_x[1].first << "," << query_range_x[1].second << ")" << endl ;
        cout << "range 2 (" << query_range_x[2].first << "," << query_range_x[2].second << ")" << endl ;
        cout << "range 5 (" << query_range_x[5].first << "," << query_range_x[5].second << ")" << endl ;
        // query_range_y[0] = query_range_x[0] = {490000 , 510000} ;
    
        // query_num = 1; 
        // query_range_x[0].first = 250000 , query_range_x[0].second = 499999 ;
        // query_range_y[0].first = 250000 , query_range_y[0].second = 499999 ;
        // for(unsigned i = 0 ; i < query_num ;i ++) {
        //     query_range_x[i] = query_range_x.front() ;
        //     query_range_y[i] = query_range_y.front() ;
        // }
    } else {
        cout << "selectivity : " << selectivity << endl ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_x , query_num) ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_y , query_num) ;
        generate_range_multi_attr_with_selectivity<attrType>(l_bound_host.data() , r_bound_host.data() , attr_dim , grid_min.data() , grid_max.data() , query_num , selectivity) ;
        cout << "随机生成属性, 完成" << endl ;
    }


    attrType* l_bound , * r_bound , *l_bound_dev , *r_bound_dev , *attr_min , *attr_width ;
    unsigned *attr_grid_size ;
    // l_bound = new float[query_num * attr_dim] , r_bound = new float[query_num * attr_dim] ;
    l_bound = l_bound_host.data() , r_bound = r_bound_host.data() ;

    cudaMalloc((void**) &l_bound_dev , attr_dim * query_num * sizeof(attrType)) ;
    cudaMalloc((void**) &r_bound_dev , attr_dim * query_num * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_min , attr_dim * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_width , attr_dim * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_grid_size , attr_dim * sizeof(unsigned)) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    cudaMemcpy(l_bound_dev , l_bound , query_num * attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(r_bound_dev , r_bound , query_num * attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_min , attr_min + attr_dim , 0.0) ;
    cudaMemcpy(attr_min , grid_min.data() , attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_width , attr_width + attr_dim , 250000.0) ;
    cudaMemcpy(attr_width , attr_width_host.data() , attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_grid_size , attr_grid_size + attr_dim , 4) ;
    cudaMemcpy(attr_grid_size , grid_size_host.data() , attr_dim * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    unsigned qnum = query_num ;
    unsigned pnum = thrust::reduce(
        thrust::device , 
        attr_grid_size , attr_grid_size + attr_dim ,
        1 , thrust::multiplies<unsigned>() 
    ) ;
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    // size_t free_bytes, total_bytes;

    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "属性, 查询范围分配完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;

    bool *partition_is_selected ; 
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg_batch_process<attr_dim , attrType>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , 0.0 , partition_is_selected) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    // ************************************* 以下声明一些要用到的数据结构, 以及调用函数 -----------------------------------
    // vector<vector<unsigned>> batch_info(4 , vector<unsigned>(4)) ;
    // for(int i =0 ; i < 16 ; i ++)
    //     batch_info[i / 4][i % 4] = i ; 
    // vector<vector<unsigned>> batch_info = {
        // {0 , 1 , 2} ,
        // {3 , 4 , 5} 
        // {6 , 7 , 8} ,
        // {9 , 10 , 11} ,
        // {12,  13 , 14} ,
        // {15} 
    // } ;



    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "准备工作完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;


    // vector<vector<unsigned>> batch_info = {
    //     {14, 10, 15},
    //     {7, 11, 13},
    //     {9, 5, 6},
    //     {4, 0, 1},
    //     {2, 12, 8},
    //     {3}
    // } ;

    // vector<vector<unsigned>> batch_info = {
    //     {0 , 1 , 2} ,
    //     {3 , 4 , 5} ,
    //     {6 , 7 , 8} ,
    //     {9 , 10 , 11} ,
    //     {12 , 13 , 14} ,
    //     {15 , 16 , 17} ,
    //     {18 , 19 , 20} ,
    //     {21 , 22 , 23} ,
    //     {24 , 25 , 26} ,
    //     {27 , 28 , 29} ,
    //     {30 , 31 , 32} ,
    //     {33 , 34 , 35} ,
    // } ;
    // vector<vector<unsigned>> batch_info(4 , vector<unsigned>(9)) ;
    // for(unsigned i = 0 ; i < pnum ;i ++)
    //     batch_info[i / 9][i % 9] = i ; 
    // vector<vector<unsigned>> batch_info =  {
    //     {0 , 1 , 2 , 3 , 4 , 5 , 6 , 7} ,
    //     {8 , 9 , 10 , 11 , 12 , 13 , 14 , 15} ,
    //     {16 ,  17 , 18 , 19 , 20 , 21 , 22 , 23 } ,
    //     {24 , 25 , 26 , 27 , 28 , 29 , 30, 31 } , 
    //     {32 , 33 , 34 , 35}
    // } ;
    // vector<vector<unsigned>> batch_info(9 , vector<unsigned>(4)) ;
    // for(unsigned i = 0 ; i < pnum ; i ++)
    //     batch_info[i / 4][i % 4] = i ; 

    vector<vector<unsigned>> batch_info = {
        {0 , 1 , 2 , 3}
    } ;

    // vector<vector<unsigned>> batch_info = {
    //     {29, 35, 34, 28},
    //     {21, 27, 22, 16},
    //     {17, 32, 33, 23},
    //     {20, 14, 15, 26}, 
    //     {7, 8, 13, 0}, 
    //     {25, 19, 9, 10}, 
    //     {11, 5, 31, 30}, 
    //     {2, 1, 6, 12}, 
    //     {24, 4, 18, 3}
    // } ;

    // vector<vector<unsigned>> batch_info(16 , vector<unsigned>(1)) ;
    // for(unsigned i =0 ; i < 16 ; i ++)
    //     batch_info[i][0] = i ; 
    unsigned batch_size = 0 ;
    for(unsigned i = 0 ; i < batch_info.size() ; i ++)
        batch_size = max(batch_size , (unsigned) batch_info[i].size()) ;
    unsigned reserve_points_num = *max_element(graph_infos[1] , graph_infos[1] + pnum) ;
    // random_shuffle(batch_info.begin() , batch_info.end()) ;
    __half *data_half_buffer1 , *data_half_buffer2 ;
    // size_t data_half_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(reserve_points_num) * static_cast<size_t>(dim) * sizeof(__half) ;
    size_t data_half_buffer_size = static_cast<const size_t>(num) * static_cast<const size_t>(dim) * sizeof(__half) ;
    cudaMalloc((void**) &data_half_buffer1 , data_half_buffer_size) ;
    cudaMemcpy(data_half_buffer1 , data_ , data_half_buffer_size , cudaMemcpyHostToDevice) ;
    // cudaMalloc((void**) &data_half_buffer2 , data_half_buffer_size) ;
    array<__half*,2> vec_buffer = {data_half_buffer1 , data_half_buffer1} ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;

    unsigned* graph_buffer0 , *graph_buffer1 ; 
    size_t graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(degree) * static_cast<size_t>(reserve_points_num) * sizeof(unsigned) ;
    cudaMalloc((void**) &graph_buffer0 , graph_buffer_size) ;
    cudaMalloc((void**) &graph_buffer1 , graph_buffer_size) ;
    array<unsigned*,2> graph_buffer = {graph_buffer0 , graph_buffer1} ;
    unsigned* global_graph_buffer0 , *global_graph_buffer1 ;
    // 此处超出内存限制
    // size_t global_graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(global_graph_degree) * static_cast<size_t>(average_num) * sizeof(unsigned) ;
    size_t global_graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<const size_t>(batch_size) * static_cast<size_t>(global_graph_edge_per_p) * static_cast<size_t>(reserve_points_num) * sizeof(unsigned) ;
    cudaMalloc((void**) &global_graph_buffer0 , global_graph_buffer_size) ;
    cudaMalloc((void**) &global_graph_buffer1 , global_graph_buffer_size) ;
    array<unsigned*,2> global_graph_buffer = {global_graph_buffer0 , global_graph_buffer1} ;
    unsigned* generic_buffer0 , *generic_buffer1 ; 
    cudaMalloc((void**) &generic_buffer0 , 10 * 1024 * 1024 * sizeof(unsigned)) ;
    cudaMalloc((void**) &generic_buffer1 , 10 * 1024 * 1024 * sizeof(unsigned)) ;
    array<unsigned*,2> generic_buffer = {generic_buffer0 , generic_buffer1} ;
    unsigned* stored_pos_global_idx ;
    cudaMalloc((void**) &stored_pos_global_idx , num * sizeof(unsigned)) ;
    array<unsigned*,4> new_idx_mapper = {idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , stored_pos_global_idx} ;


    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cudaError_t err2 = cudaMemGetInfo(&free_bytes, &total_bytes) ;
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "缓冲区申请完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;
    cout << "准备工作完成, 开始查找 :" << endl ;
    s = chrono::high_resolution_clock::now() ;
    // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
    //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
    //     global_graph , 16 , pnum , L) ;
    

    vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter_batch_process_compressed_graph_T_dataset_resident<attrType>(batch_info.size() , batch_info , degree , data_ , vec_buffer, graph_buffer , global_graph_buffer ,
        generic_buffer , query_data_half_dev , dim , qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        attr_dim  , graph_infos , new_idx_mapper , seg_partition_point_start_index  , partition_infos , L , 10 , 
        global_graph , global_graph_edge_per_p , pnum , partition_is_selected , 10 , num , 16) ;
    e = chrono::high_resolution_clock::now() ;
    cost3 = chrono::duration<double>(e - s).count() ;
    cout << "finish search" << endl ;

    cudaFree(data_half_buffer1) ;
    // (data_half_buffer2) ;
    cudaFree(graph_buffer0) ;
    cudaFree(graph_buffer1) ;
    cudaFree(global_graph_buffer0) ;
    cudaFree(global_graph_buffer1) ;
    cudaFree(generic_buffer0) ;
    cudaFree(generic_buffer1) ;
    cudaFree(stored_pos_global_idx) ;

    cout << "gpu 暴力:" << endl ;
    auto gpu_bf_s = chrono::high_resolution_clock::now() ;
    vector<vector<unsigned>> gpu_bf_result ;
    vector<vector<float>> gpu_bf_result_dis ; 
    // gpu_bf_result = prefiltering_and_bruteforce<attrType>(query_data_half_dev , data_ , qnum ,  10 , dim , num , attrs_dev,
    //    l_bound_dev , r_bound_dev , attr_dim , gpu_bf_result_dis) ;
    auto gpu_bf_e = chrono::high_resolution_clock::now() ;
    float gpu_bf_cost = chrono::duration<double>(gpu_bf_e - gpu_bf_s).count() ;
    cout << "gpu 暴力 结束: " << gpu_bf_cost << endl ;

    // for(unsigned i = 0 ; i < 10  ;i ++)
    //     cout << gpu_bf_result_dis[0][i] << " ," ;

    // save_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/100Kquery.gt", query_num , 10, gpu_bf_result) ;
    load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/100Kquery.gt" , gpu_bf_result) ;
    // load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt" , gpu_bf_result) ;
    cudaFreeHost(data_) ;
    cudaFreeHost(global_graph) ;
    cudaFreeHost(graph_infos[0]) ;
    cudaFreeHost(graph_infos[1]) ;
    cudaFreeHost(graph_infos[2]) ;
    /**
        float* data_float , * query_data_load_float ;
        cout << "开始暴力搜索, 载入数据--------------------" << endl ;
        load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs" , data_float , num , dim) ;
        cout << "num : " << num << ", dim : " << dim << endl ;
        query_data_load_float = new float[query_num * query_dim] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ; j < query_dim ; j ++)
            // query_data_load_float[i * query_dim + j] = data_float[qids[i] * query_dim + j] ;
            size_t ofst = static_cast<size_t>(qids[i]) * static_cast<size_t>(query_dim) ;
            memcpy(query_data_load_float + i * query_dim , data_float + ofst , query_dim * sizeof(float)) ;
        }
        // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
        vector<vector<unsigned>> ground_truth(qnum) ;
        vector<vector<float>> gt_dis(qnum) ;
        auto cpu_bf_s = chrono::high_resolution_clock::now() ;
        #pragma omp parallel for num_threads(10)
        for(unsigned i = 0 ; i < qnum ;i  ++) {
            vector<float> gt_d ; 
            ground_truth[i] = naive_brute_force_with_filter_store_dis<attrType>(query_data_load_float + i * dim , data_float , dim , num , 10 ,
                l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim , gt_d) ;
            gt_dis[i] = gt_d ; 
        }
        auto cpu_bf_e = chrono::high_resolution_clock::now() ;
        float cpu_bf_cost = chrono::duration<double>(cpu_bf_e - cpu_bf_s).count() ;
        reverse(ground_truth[0].begin() , ground_truth[0].end()) ;
        reverse(gt_dis[0].begin() , gt_dis[0].end()) ;
        cout << "gt id :[" ;
        for(unsigned i = 0 ; i < ground_truth[0].size() ;i  ++) {
            cout << ground_truth[0][i] ;
            if(i < ground_truth[0].size() - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;
        cout << "gt dis :[" ;
        for(unsigned i = 0 ; i < gt_dis[0].size() ;i  ++) {
            cout << gt_dis[0][i] ;
            if(i < gt_dis[0].size() - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;
        // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
        float total_recall = cal_recall_show_partition_multi_attrs<attrType>(result , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;

        
        cout << "recall : " << total_recall << endl  ;
    **/
    cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
    cout << "获取分区列表用时:" << cost1 << endl ;
    cout << "对分区列表排序用时:" << cost2 << endl ;
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "核函数 + 数据运输、准备工作:" << other_kernal_cost << endl ;
    cout << "数据传输时间: " << mem_trans_cost << endl ;
    cout << "搜索用时:" << cost3 << endl ;
    cout << "batch cost : [" ;
    for(unsigned i = 0 ; i < batch_process_cost.size() ; i ++) {
        cout << batch_process_cost[i] ;
        if(i < batch_process_cost.size() - 1)
            cout << " ," ;
    }
    cout << "]" << endl ; 
    cout << "batch mem load cost : [" ;
    for(unsigned i = 0 ; i < batch_data_trans_cost.size() ; i ++) {
        cout << batch_data_trans_cost[i] ;
        if(i < batch_data_trans_cost.size() - 1)
            cout << " ," ;
    }
    cout << "]" << endl ;
    cout << "gpu brute force 时间: " << gpu_bf_cost << endl ;
    gpu_bf_result.resize(qnum) ;
    // cout << "cpu brute force 时间: " << cpu_bf_cost << endl ;
    float total_recall = cal_recall_show_partition_multi_attrs<attrType>(result , gpu_bf_result , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , false) ;
    cout << " recall : " << total_recall << endl ;


    rc_recall1 = total_recall ;
    rc_cost1 = select_path_kernal_cost ;

    cudaFree(global_graph) ;
    cudaFree(attrs_dev) ;
    cudaFree(query_data_half_dev) ;
    cudaFree(l_bound_dev) ;
    cudaFree(r_bound_dev) ;
    cudaFree(attr_min) ;
    cudaFree(attr_width) ;
    cudaFree(attr_grid_size) ;
    cudaFree(partition_is_selected) ;
    for(unsigned i = 0  ; i < 5 ; i ++)
        cudaFree(partition_infos[i]) ;
    for(unsigned i = 0 ; i < 3 ; i ++) 
        cudaFree(idx_mapper[i]) ;
    
}




void multi_attr_test_batch_process_large_dataset_extreme_compressed_graph(unsigned L = 80) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    unsigned preset_pnum = 36 ;
    unsigned average_num = 6250000 ;
    array<unsigned*,3> graph_infos = get_graph_info_host_avg_size("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/deep100M/deep100M" 
    , ".cagra16" , 16 , 6250000 , 16) ;
    // unsigned offset4 = 4 * average_num * 16 ;
    // for(unsigned i = 0 ; i < 100 ; i ++) {
    //     for(unsigned j = 0 ; j < 16 ; j ++) {
    //         cout << graph_infos[0][offset4 + i * 16 + j] << " ," ;
    //     }
    //     cout << endl;
    // }


    unsigned degree = 16 , global_graph_edge_per_p = 2 ; 
    array<unsigned*,3> idx_mapper = get_idx_mapper(100000000 , 16 , 6250000) ;
    // vector<vector<unsigned>> global_edges ;
    // load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges) ;
    unsigned *global_graph ;
    unsigned global_graph_degree , global_graph_num_in_graph ;
    // cudaMalloc((void**) &global_graph , 1000000 * 16 * 16 * sizeof(unsigned)) ;
    load_result_pined("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/global_graph_T.graphT" ,
     global_graph , global_graph_degree , global_graph_num_in_graph) ;
    cout << "global graph : degree-" << global_graph_degree << ", num-" << global_graph_num_in_graph << endl ;



    // 压缩全局图
    
    // for(unsigned i = 0 ; i < 1000000 ; i ++) {
    //     cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    // }
    using attrType = double ;
    attrType *attrs_dev , *attrs ;
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    constexpr unsigned attr_dim =  2 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<attrType> grid_min(attr_dim , 0.0) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<attrType> grid_max(attr_dim , 1e8) ;
    vector<unsigned> grid_size_host = {4 , 4} ;
    vector<attrType> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++) {
        attr_width_host[i] = (grid_max[i] - grid_min[i]) / grid_size_host[i] ;
        cout << "attr_width_host[i] : " << attr_width_host[i] << "," ;
    }
    // attrs = new attrType[attr_dim * 100000000] ;
    // re_generate<attrType>(attrs , attr_dim , 100000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    // save_attrs("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/attrs.bin" ,100000000 , attr_dim, attrs) ;
    load_attrs<attrType>("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/attrs.bin", attrs) ;
    size_t attrs_dev_byteSize = static_cast<size_t>(attr_dim) * 100000000 * sizeof(attrType) ;
    cudaMalloc((void**) &attrs_dev , attrs_dev_byteSize) ;
    cudaMemcpy(attrs_dev , attrs , attrs_dev_byteSize , cudaMemcpyHostToDevice) ;

    cout << "sizeof(half) = " << sizeof(half) << endl ;

    // 先载入数据集和图索引
    unsigned num , dim  ;
    __half* data_   ;
    cout << "开始载入数据集" << endl ;
    data_ = load_dataset_return_pined_pointer("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs" ,
     num , dim , 10.0) ;

    __half* query_data_load ;
    unsigned query_num = 1 , query_dim = dim ;
    vector<unsigned> qids(query_num * 10) ;
    /**
    unordered_set<unsigned> qids_contains ;
    mt19937 rng ;
    uniform_int_distribution<> intd(0 , num - 1) ;
    unsigned qids_ofst = 0 ;
    while(qids_ofst < query_num * 10) {
        unsigned t = intd(rng) ;
        if(!qids_contains.count(t)) {
            qids_contains.insert(t) ;
            qids[qids_ofst ++] = t ;
        }
    }
    random_shuffle(qids.begin() , qids.end()) ;
    **/
    // save_query_ids("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/qids.bin" , query_num , qids) ;
    load_query_ids("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/qids.bin" , query_num , qids) ;
    query_num = 1000 ; 
    // qids[0] = 0 ;

    query_data_load = new __half[query_num * query_dim] ;
    for(unsigned i = 0 ; i < query_num ; i ++) {
        // for(unsigned j = 0 ; j < query_dim ; j ++)
        //     query_data_load[i * query_dim + j] = data_[qids[i] * query_dim + j] ;
        size_t ofst = static_cast<size_t>(qids[i]) * static_cast<size_t>(query_dim) ;
        memcpy(query_data_load + i * query_dim , data_ + ofst , query_dim * sizeof(__half)) ;
    }
    // query_num = 1 ; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;

    half *query_data_half_dev ;
    
    cudaMalloc((void**) &query_data_half_dev , query_num * query_dim * sizeof(__half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(query_data_half_dev , query_data_load , query_num * query_dim * sizeof(__half) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num ; 
    
    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    vector<attrType> l_bound_host(query_num * attr_dim) , r_bound_host(query_num * attr_dim) ;
    if(selectivity == 0.0) {
        cout << "sel = " << selectivity << endl ;
        load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        query_range_x.reserve(query_num) ;
        for(unsigned i = 1 ; i < 10 ; i ++)
            query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        query_range_y = query_range_x ; 

        // query_range_y[0] = {245000 , 255000} , query_range_x[0] = {245000 , 255000} ;
        // for(unsigned i = 1 ; i < query_num ; i ++)
        //     query_range_y[i] = query_range_y[0] , query_range_x[i] = query_range_x[0] ;
        // query_range_x[0] = {25000000 , 49999999} ;
        // query_range_y[0] = {0 , 49999999} ;

        for(unsigned i = 1 ; i < query_num ;i  ++) {
            query_range_x[i].first *= 100 , query_range_x[i].second *= 100 ; 
            query_range_y[i].first *= 100 , query_range_y[i].second *= 100 ;
        }

            // query_range_x[i] = query_range_x[0] , query_range_y[i] = query_range_y[0] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ;j < attr_dim ; j ++)
            l_bound_host[i * attr_dim] = (attrType)query_range_x[i].first , r_bound_host[i * attr_dim] = (attrType)query_range_x[i].second ;
            l_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].first , r_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].second ;
        }

        cout << "range 0 (" << query_range_x[0].first << "," << query_range_x[0].second << ")" << endl ;
        cout << "range 1 (" << query_range_x[1].first << "," << query_range_x[1].second << ")" << endl ;
        cout << "range 2 (" << query_range_x[2].first << "," << query_range_x[2].second << ")" << endl ;
        cout << "range 5 (" << query_range_x[5].first << "," << query_range_x[5].second << ")" << endl ;
        // query_range_y[0] = query_range_x[0] = {490000 , 510000} ;
    
        // query_num = 1; 
        // query_range_x[0].first = 250000 , query_range_x[0].second = 499999 ;
        // query_range_y[0].first = 250000 , query_range_y[0].second = 499999 ;
        // for(unsigned i = 0 ; i < query_num ;i ++) {
        //     query_range_x[i] = query_range_x.front() ;
        //     query_range_y[i] = query_range_y.front() ;
        // }
    } else {
        cout << "selectivity : " << selectivity << endl ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_x , query_num) ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_y , query_num) ;
        generate_range_multi_attr_with_selectivity<attrType>(l_bound_host.data() , r_bound_host.data() , attr_dim , grid_min.data() , grid_max.data() , query_num , selectivity) ;
        cout << "随机生成属性, 完成" << endl ;
    }


    attrType* l_bound , * r_bound , *l_bound_dev , *r_bound_dev , *attr_min , *attr_width ;
    unsigned *attr_grid_size ;
    // l_bound = new float[query_num * attr_dim] , r_bound = new float[query_num * attr_dim] ;
    l_bound = l_bound_host.data() , r_bound = r_bound_host.data() ;

    cudaMalloc((void**) &l_bound_dev , attr_dim * query_num * sizeof(attrType)) ;
    cudaMalloc((void**) &r_bound_dev , attr_dim * query_num * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_min , attr_dim * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_width , attr_dim * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_grid_size , attr_dim * sizeof(unsigned)) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    cudaMemcpy(l_bound_dev , l_bound , query_num * attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(r_bound_dev , r_bound , query_num * attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_min , attr_min + attr_dim , 0.0) ;
    cudaMemcpy(attr_min , grid_min.data() , attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_width , attr_width + attr_dim , 250000.0) ;
    cudaMemcpy(attr_width , attr_width_host.data() , attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_grid_size , attr_grid_size + attr_dim , 4) ;
    cudaMemcpy(attr_grid_size , grid_size_host.data() , attr_dim * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    unsigned qnum = query_num ;
    unsigned pnum = thrust::reduce(
        thrust::device , 
        attr_grid_size , attr_grid_size + attr_dim ,
        1 , thrust::multiplies<unsigned>() 
    ) ;
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    bool *partition_is_selected ; 
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg_batch_process<attr_dim , attrType>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , 0.05 , partition_is_selected) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    // ************************************* 以下声明一些要用到的数据结构, 以及调用函数 -----------------------------------
    // vector<vector<unsigned>> batch_info(4 , vector<unsigned>(4)) ;
    // for(int i =0 ; i < 16 ; i ++)
    //     batch_info[i / 4][i % 4] = i ; 
    // vector<vector<unsigned>> batch_info = {
        // {0 , 1 , 2} ,
        // {3 , 4 , 5} 
        // {6 , 7 , 8} ,
        // {9 , 10 , 11} ,
        // {12,  13 , 14} ,
        // {15} 
    // } ;

    size_t free_bytes, total_bytes;

    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    double free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    double total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "准备工作完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;


    // vector<vector<unsigned>> batch_info = {
    //     {14, 10, 15},
    //     {7, 11, 13},
    //     {9, 5, 6},
    //     {4, 0, 1},
    //     {2, 12, 8},
    //     {3}
    // } ;

    vector<vector<unsigned>> batch_info = {
        {14, 10, 15 , 7},
        {11, 13, 9 , 5},
        {6 , 4 , 0 , 1},
        {2, 12, 8 , 3}
    } ;
  
    // 每个分区对应向量id起始位置
    vector<unsigned> seg_partition_point_start_index(16) ;
    seg_partition_point_start_index[0] = 0 ;
    for(int i = 1; i < 16 ; i ++)
        seg_partition_point_start_index[i] = seg_partition_point_start_index[i - 1] + graph_infos[1][i - 1] ;
    
    // unsigned* spokesman_gpu ;
    // cudaMalloc((void**) & spokesman_gpu , num * sizeof(unsigned)) ;
    // cudaMemcpy(spokesman_gpu , spokesman.data() , num * sizeof(unsigned) , cudaMemcpyHostToDevice) ;

    // 压缩全局图
    array<unsigned*,5> compressed_global_graph = compress_global_graph_extreme(graph_infos, degree , global_graph , 16 , num , global_graph_edge_per_p , seg_partition_point_start_index.data()) ;
    cudaFreeHost(global_graph) ;
    unsigned total_spokesman_num = accumulate(compressed_global_graph[1] , compressed_global_graph[1] + 16 , 0) ;
    cout << "全局图压缩完成--------" << endl ; 
    // vector<vector<unsigned>> batch_info(16 , vector<unsigned>(1)) ;
    // for(unsigned i =0 ; i < 16 ; i ++)
    //     batch_info[i][0] = i ; 
    const unsigned batch_size = 4 ;
    // random_shuffle(batch_info.begin() , batch_info.end()) ;
    __half *data_half_buffer1 , *data_half_buffer2 ;
    size_t data_half_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(average_num) * static_cast<size_t>(dim) * sizeof(__half) ;
    cudaMalloc((void**) &data_half_buffer1 , data_half_buffer_size) ;
    cudaMalloc((void**) &data_half_buffer2 , data_half_buffer_size) ;
    array<__half*,2> vec_buffer = {data_half_buffer1 , data_half_buffer2} ;
    unsigned* graph_buffer0 , *graph_buffer1 ; 
    size_t graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(degree) * static_cast<size_t>(average_num) * sizeof(unsigned) ;
    cudaMalloc((void**) &graph_buffer0 , graph_buffer_size) ;
    cudaMalloc((void**) &graph_buffer1 , graph_buffer_size) ;
    array<unsigned*,2> graph_buffer = {graph_buffer0 , graph_buffer1} ;
    unsigned* global_graph_buffer0 , *global_graph_buffer1 ;
    // 此处超出内存限制
    // size_t global_graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(global_graph_degree) * static_cast<size_t>(average_num) * sizeof(unsigned) ;
    // size_t global_graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<const size_t>(batch_size) * static_cast<size_t>(global_graph_edge_per_p) * static_cast<size_t>(average_num) * sizeof(unsigned) ;
    unsigned max_spokesman_num = *max_element(compressed_global_graph[1] , compressed_global_graph[1] + 16) ;
    size_t global_graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(global_graph_edge_per_p) * static_cast<size_t>(max_spokesman_num) * static_cast<size_t>(batch_size) * sizeof(unsigned) ;
    cudaMalloc((void**) &global_graph_buffer0 , global_graph_buffer_size) ;
    cudaMalloc((void**) &global_graph_buffer1 , global_graph_buffer_size) ;
    array<unsigned*,2> global_graph_buffer = {global_graph_buffer0 , global_graph_buffer1} ;
    unsigned* generic_buffer0 , *generic_buffer1 ; 
    cudaMalloc((void**) &generic_buffer0 , 10 * 1024 * 1024 * sizeof(unsigned)) ;
    cudaMalloc((void**) &generic_buffer1 , 10 * 1024 * 1024 * sizeof(unsigned)) ;
    array<unsigned*,2> generic_buffer = {generic_buffer0 , generic_buffer1} ;
    unsigned* stored_pos_global_idx ;
    cudaMalloc((void**) &stored_pos_global_idx , num * sizeof(unsigned)) ;
    array<unsigned*,4> new_idx_mapper = {idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , stored_pos_global_idx} ;


    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cudaError_t err2 = cudaMemGetInfo(&free_bytes, &total_bytes) ;
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "缓冲区申请完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;
    cout << "准备工作完成, 开始查找 :" << endl ;
    s = chrono::high_resolution_clock::now() ;
    // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
    //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
    //     global_graph , 16 , pnum , L) ;
    

    vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter_batch_process_extreme_compressed_graph<attrType>(batch_info.size() , batch_info , degree , data_ , vec_buffer, graph_buffer , global_graph_buffer ,
        generic_buffer , query_data_half_dev , dim , qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        attr_dim  , graph_infos , new_idx_mapper , seg_partition_point_start_index  , partition_infos , L , 10 , 
        compressed_global_graph[0] , global_graph_edge_per_p , pnum , partition_is_selected , 10 , num , compressed_global_graph ,total_spokesman_num , 16) ;
    e = chrono::high_resolution_clock::now() ;
    cost3 = chrono::duration<double>(e - s).count() ;
    cout << "finish search" << endl ;

    cudaFree(data_half_buffer1) ;
    cudaFree(data_half_buffer2) ;
    cudaFree(graph_buffer0) ;
    cudaFree(graph_buffer1) ;
    cudaFree(global_graph_buffer0) ;
    cudaFree(global_graph_buffer1) ;
    cudaFree(generic_buffer0) ;
    cudaFree(generic_buffer1) ;
    cudaFree(stored_pos_global_idx) ;
    // cudaFree(spokesman_gpu) ;

    cout << "gpu 暴力:" << endl ;
    auto gpu_bf_s = chrono::high_resolution_clock::now() ;
    vector<vector<unsigned>> gpu_bf_result ;
    gpu_bf_result = prefiltering_and_bruteforce<attrType>(query_data_half_dev , data_ , qnum ,  10 , dim , num , attrs_dev,
       l_bound_dev , r_bound_dev , attr_dim) ;
    auto gpu_bf_e = chrono::high_resolution_clock::now() ;
    float gpu_bf_cost = chrono::duration<double>(gpu_bf_e - gpu_bf_s).count() ;
    cout << "gpu 暴力 结束: " << gpu_bf_cost << endl ;

    // save_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt", query_num , 10, gpu_bf_result) ;
    // load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt" , gpu_bf_result) ;
    // load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt" , gpu_bf_result) ;
    cudaFreeHost(data_) ;
    cudaFreeHost(global_graph) ;
    cudaFreeHost(graph_infos[0]) ;
    cudaFreeHost(graph_infos[1]) ;
    cudaFreeHost(graph_infos[2]) ;
    /**
        float* data_float , * query_data_load_float ;
        cout << "开始暴力搜索, 载入数据--------------------" << endl ;
        load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs" , data_float , num , dim) ;
        cout << "num : " << num << ", dim : " << dim << endl ;
        query_data_load_float = new float[query_num * query_dim] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ; j < query_dim ; j ++)
            // query_data_load_float[i * query_dim + j] = data_float[qids[i] * query_dim + j] ;
            size_t ofst = static_cast<size_t>(qids[i]) * static_cast<size_t>(query_dim) ;
            memcpy(query_data_load_float + i * query_dim , data_float + ofst , query_dim * sizeof(float)) ;
        }
        // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
        vector<vector<unsigned>> ground_truth(qnum) ;
        vector<vector<float>> gt_dis(qnum) ;
        auto cpu_bf_s = chrono::high_resolution_clock::now() ;
        #pragma omp parallel for num_threads(10)
        for(unsigned i = 0 ; i < qnum ;i  ++) {
            vector<float> gt_d ; 
            ground_truth[i] = naive_brute_force_with_filter_store_dis<attrType>(query_data_load_float + i * dim , data_float , dim , num , 10 ,
                l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim , gt_d) ;
            gt_dis[i] = gt_d ; 
        }
        auto cpu_bf_e = chrono::high_resolution_clock::now() ;
        float cpu_bf_cost = chrono::duration<double>(cpu_bf_e - cpu_bf_s).count() ;
        reverse(ground_truth[0].begin() , ground_truth[0].end()) ;
        reverse(gt_dis[0].begin() , gt_dis[0].end()) ;
        cout << "gt id :[" ;
        for(unsigned i = 0 ; i < ground_truth[0].size() ;i  ++) {
            cout << ground_truth[0][i] ;
            if(i < ground_truth[0].size() - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;
        cout << "gt dis :[" ;
        for(unsigned i = 0 ; i < gt_dis[0].size() ;i  ++) {
            cout << gt_dis[0][i] ;
            if(i < gt_dis[0].size() - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;
        // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
        float total_recall = cal_recall_show_partition_multi_attrs<attrType>(result , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;

        
        cout << "recall : " << total_recall << endl  ;
    **/
    cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
    cout << "获取分区列表用时:" << cost1 << endl ;
    cout << "对分区列表排序用时:" << cost2 << endl ;
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "核函数 + 数据运输、准备工作:" << other_kernal_cost << endl ;
    cout << "数据传输时间: " << mem_trans_cost << endl ;
    cout << "搜索用时:" << cost3 << endl ;
    cout << "batch cost : [" ;
    for(unsigned i = 0 ; i < batch_process_cost.size() ; i ++) {
        cout << batch_process_cost[i] ;
        if(i < batch_process_cost.size() - 1)
            cout << " ," ;
    }
    cout << "]" << endl ; 
    cout << "batch mem load cost : [" ;
    for(unsigned i = 0 ; i < batch_data_trans_cost.size() ; i ++) {
        cout << batch_data_trans_cost[i] ;
        if(i < batch_data_trans_cost.size() - 1)
            cout << " ," ;
    }
    cout << "]" << endl ;
    cout << "gpu brute force 时间: " << gpu_bf_cost << endl ;
    gpu_bf_result.resize(qnum) ;
    // cout << "cpu brute force 时间: " << cpu_bf_cost << endl ;
    float total_recall = cal_recall_show_partition_multi_attrs<attrType>(result , gpu_bf_result , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;
    cout << " recall : " << total_recall << endl ;


    rc_recall1 = total_recall ;
    rc_cost1 = select_path_kernal_cost ;

    cudaFree(global_graph) ;
    cudaFree(attrs_dev) ;
    cudaFree(query_data_half_dev) ;
    cudaFree(l_bound_dev) ;
    cudaFree(r_bound_dev) ;
    cudaFree(attr_min) ;
    cudaFree(attr_width) ;
    cudaFree(attr_grid_size) ;
    cudaFree(partition_is_selected) ;
    for(unsigned i = 0  ; i < 5 ; i ++)
        cudaFree(partition_infos[i]) ;
    for(unsigned i = 0 ; i < 3 ; i ++) 
        cudaFree(idx_mapper[i]) ;
    
}

void multi_attr_test_batch_process_large_dataset_extreme_compressed_graph_pnum36(unsigned L = 80) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    unsigned preset_pnum = 36 ;
    unsigned preset_num = 1e8 ;
    unsigned preset_avg_num = (preset_num + preset_pnum - 1) / preset_pnum ; 
    unsigned average_num = 6250000 ;
    array<unsigned*,3> graph_infos = get_graph_info_host("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/deep100M/36/deep100M" 
    , ".cagra16" , preset_pnum , 16, preset_num) ;
    // unsigned offset4 = 4 * average_num * 16 ;
    // for(unsigned i = 0 ; i < 100 ; i ++) {
    //     for(unsigned j = 0 ; j < 16 ; j ++) {
    //         cout << graph_infos[0][offset4 + i * 16 + j] << " ," ;
    //     }
    //     cout << endl;
    // }



    // 每个分区对应向量id起始位置
    vector<unsigned> seg_partition_point_start_index(preset_pnum) ;
    seg_partition_point_start_index[0] = 0 ;
    for(int i = 1; i < preset_pnum ; i ++)
        seg_partition_point_start_index[i] = seg_partition_point_start_index[i - 1] + graph_infos[1][i - 1] ;

    unsigned degree = 16 , global_graph_edge_per_p = 2 ; 
    array<unsigned*,3> idx_mapper = get_idx_mapper(preset_num , preset_pnum , graph_infos) ;
    // vector<vector<unsigned>> global_edges ;
    // load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges) ;
    // unsigned *global_graph ;
    // unsigned global_graph_degree , global_graph_num_in_graph ;
    // cudaMalloc((void**) &global_graph , 1000000 * 16 * 16 * sizeof(unsigned)) ;
    // load_result_pined("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/global_graph_T.graphT" ,
    //  global_graph , global_graph_degree , global_graph_num_in_graph) ;
    // cout << "global graph : degree-" << global_graph_degree << ", num-" << global_graph_num_in_graph << endl ;



    // 压缩全局图
    
    // for(unsigned i = 0 ; i < 1000000 ; i ++) {
    //     cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    // }
    using attrType = double ;
    attrType *attrs_dev , *attrs ;
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    constexpr unsigned attr_dim =  2 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<attrType> grid_min(attr_dim , 0.0) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<attrType> grid_max(attr_dim , 1e8) ;
    vector<unsigned> grid_size_host = {6 , 6} ;
    vector<attrType> attr_width_host(attr_dim) ;
    // cout << "grid size host size : " << grid_size_host.size() << endl ;
    for(unsigned i = 0 ; i < attr_dim ; i ++) {
        // cout << "grid_max " << grid_max[i] << ", grid_min " << grid_min[i] << ", grid_size_host : " << grid_size_host[i] << endl ;
        attr_width_host[i] = (grid_max[i] - grid_min[i]) / grid_size_host[i] ;
        cout << "attr_width_host[i] : " << attr_width_host[i] << "," ;
    }
    cout << "随机生成属性" << endl ;
    attrs = new attrType[attr_dim * 100000000] ;
    re_generate<attrType>(attrs , attr_dim , preset_num , grid_min.data() , grid_max.data() , grid_size_host.data() , graph_infos[1] , seg_partition_point_start_index.data() , preset_pnum) ;
    cout << "属性生成完成" << endl ;
    // save_attrs("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/attrs.bin" ,100000000 , attr_dim, attrs) ;
    // load_attrs<attrType>("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/attrs.bin", attrs) ;
    size_t attrs_dev_byteSize = static_cast<size_t>(attr_dim) * 100000000 * sizeof(attrType) ;
    cudaMalloc((void**) &attrs_dev , attrs_dev_byteSize) ;
    cudaMemcpy(attrs_dev , attrs , attrs_dev_byteSize , cudaMemcpyHostToDevice) ;

    cout << "sizeof(half) = " << sizeof(half) << endl ;

    // 先载入数据集和图索引
    unsigned num , dim  ;
    __half* data_   ;
    cout << "开始载入数据集" << endl ;
    data_ = load_dataset_return_pined_pointer("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs" ,
     num , dim , 10.0) ;

    __half* query_data_load ;
    unsigned query_num = 1 , query_dim = dim ;
    vector<unsigned> qids(query_num * 10) ;
    /**
        unordered_set<unsigned> qids_contains ;
        mt19937 rng ;
        uniform_int_distribution<> intd(0 , num - 1) ;
        unsigned qids_ofst = 0 ;
        while(qids_ofst < query_num * 10) {
            unsigned t = intd(rng) ;
            if(!qids_contains.count(t)) {
                qids_contains.insert(t) ;
                qids[qids_ofst ++] = t ;
            }
        }
        random_shuffle(qids.begin() , qids.end()) ;
    **/
    // save_query_ids("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/qids.bin" , query_num , qids) ;
    load_query_ids("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/qids.bin" , query_num , qids) ;
    query_num = 10 ; 
    // qids[0] = 0 ;

    query_data_load = new __half[query_num * query_dim] ;
    for(unsigned i = 0 ; i < query_num ; i ++) {
        // for(unsigned j = 0 ; j < query_dim ; j ++)
        //     query_data_load[i * query_dim + j] = data_[qids[i] * query_dim + j] ;
        size_t ofst = static_cast<size_t>(qids[i]) * static_cast<size_t>(query_dim) ;
        memcpy(query_data_load + i * query_dim , data_ + ofst , query_dim * sizeof(__half)) ;
    }
    // query_num = 1 ; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;

    half *query_data_half_dev ;
    
    cudaMalloc((void**) &query_data_half_dev , query_num * query_dim * sizeof(__half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(query_data_half_dev , query_data_load , query_num * query_dim * sizeof(__half) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num ; 
    
    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    vector<attrType> l_bound_host(query_num * attr_dim) , r_bound_host(query_num * attr_dim) ;
    if(selectivity == 0.0) {
        cout << "sel = " << selectivity << endl ;
        load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        query_range_x.reserve(query_num) ;
        for(unsigned i = 1 ; i < 10 ; i ++)
            query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        query_range_y = query_range_x ; 

        // query_range_y[0] = {245000 , 255000} , query_range_x[0] = {245000 , 255000} ;
        // for(unsigned i = 1 ; i < query_num ; i ++)
        //     query_range_y[i] = query_range_y[0] , query_range_x[i] = query_range_x[0] ;
        query_range_x[0] = {0 , 16666666} ;
        query_range_y[0] = {0 , 16666666 * 5} ;
        for(unsigned i = 1; i < query_num ;i  ++)
            query_range_x[i] = query_range_x[0] , query_range_y[i] = query_range_y[0] ;
        // for(unsigned i = 1 ; i < query_num ;i  ++) {
        //     query_range_x[i].first *= 100 , query_range_x[i].second *= 100 ; 
        //     query_range_y[i].first *= 100 , query_range_y[i].second *= 100 ;
        // }

            // query_range_x[i] = query_range_x[0] , query_range_y[i] = query_range_y[0] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ;j < attr_dim ; j ++)
            l_bound_host[i * attr_dim] = (attrType)query_range_x[i].first , r_bound_host[i * attr_dim] = (attrType)query_range_x[i].second ;
            l_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].first , r_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].second ;
        }

        cout << "range 0 (" << query_range_x[0].first << "," << query_range_x[0].second << ")" << endl ;
        cout << "range 1 (" << query_range_x[1].first << "," << query_range_x[1].second << ")" << endl ;
        cout << "range 2 (" << query_range_x[2].first << "," << query_range_x[2].second << ")" << endl ;
        cout << "range 5 (" << query_range_x[5].first << "," << query_range_x[5].second << ")" << endl ;
        // query_range_y[0] = query_range_x[0] = {490000 , 510000} ;
    
        // query_num = 1; 
        // query_range_x[0].first = 250000 , query_range_x[0].second = 499999 ;
        // query_range_y[0].first = 250000 , query_range_y[0].second = 499999 ;
        // for(unsigned i = 0 ; i < query_num ;i ++) {
        //     query_range_x[i] = query_range_x.front() ;
        //     query_range_y[i] = query_range_y.front() ;
        // }
    } else {
        cout << "selectivity : " << selectivity << endl ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_x , query_num) ;
        // generate_range_by_selectivity(selectivity , 0.0 , 1000000.0 , query_range_y , query_num) ;
        generate_range_multi_attr_with_selectivity<attrType>(l_bound_host.data() , r_bound_host.data() , attr_dim , grid_min.data() , grid_max.data() , query_num , selectivity) ;
        cout << "随机生成属性, 完成" << endl ;
    }


    attrType* l_bound , * r_bound , *l_bound_dev , *r_bound_dev , *attr_min , *attr_width ;
    unsigned *attr_grid_size ;
    // l_bound = new float[query_num * attr_dim] , r_bound = new float[query_num * attr_dim] ;
    l_bound = l_bound_host.data() , r_bound = r_bound_host.data() ;

    cudaMalloc((void**) &l_bound_dev , attr_dim * query_num * sizeof(attrType)) ;
    cudaMalloc((void**) &r_bound_dev , attr_dim * query_num * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_min , attr_dim * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_width , attr_dim * sizeof(attrType)) ;
    cudaMalloc((void**) &attr_grid_size , attr_dim * sizeof(unsigned)) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    cudaMemcpy(l_bound_dev , l_bound , query_num * attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(r_bound_dev , r_bound , query_num * attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_min , attr_min + attr_dim , 0.0) ;
    cudaMemcpy(attr_min , grid_min.data() , attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_width , attr_width + attr_dim , 250000.0) ;
    cudaMemcpy(attr_width , attr_width_host.data() , attr_dim * sizeof(attrType) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_grid_size , attr_grid_size + attr_dim , 4) ;
    cudaMemcpy(attr_grid_size , grid_size_host.data() , attr_dim * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    unsigned qnum = query_num ;
    unsigned pnum = thrust::reduce(
        thrust::device , 
        attr_grid_size , attr_grid_size + attr_dim ,
        1 , thrust::multiplies<unsigned>() 
    ) ;
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    bool *partition_is_selected ; 
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg_batch_process<attr_dim , attrType>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , 0.05 , partition_is_selected) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    // ************************************* 以下声明一些要用到的数据结构, 以及调用函数 -----------------------------------
    // vector<vector<unsigned>> batch_info(4 , vector<unsigned>(4)) ;
    // for(int i =0 ; i < 16 ; i ++)
    //     batch_info[i / 4][i % 4] = i ; 
    // vector<vector<unsigned>> batch_info = {
        // {0 , 1 , 2} ,
        // {3 , 4 , 5} 
        // {6 , 7 , 8} ,
        // {9 , 10 , 11} ,
        // {12,  13 , 14} ,
        // {15} 
    // } ;

    size_t free_bytes, total_bytes;

    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    double free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    double total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "准备工作完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;


    // vector<vector<unsigned>> batch_info = {
    //     {14, 10, 15},
    //     {7, 11, 13},
    //     {9, 5, 6},
    //     {4, 0, 1},
    //     {2, 12, 8},
    //     {3}
    // } ;

    // vector<vector<unsigned>> batch_info = {
    //     {14, 10, 15 , 7},
    //     {11, 13, 9 , 5},
    //     {6 , 4 , 0 , 1},
    //     {2, 12, 8 , 3}
    // } ;
    // vector<vector<unsigned>> batch_info(6 , vector<unsigned>(6)) ;
    // for(unsigned i = 0 ; i < preset_pnum ; i ++)
    //     batch_info[i / 6][i % 6] = i ; 
    vector<vector<unsigned>> batch_info = {
        {0 , 1 , 2 , 3 , 4} 
        // {5 , 6 , 7 , 8 , 9} ,
        // {10 , 11 , 12 , 13 , 14} ,
        // {15 , 16 , 17, 18 , 19} ,
        // {20 , 21 , 22 , 23 , 24} ,
        // {25 , 26 , 27 , 28 , 29} ,
        // {30 , 31 , 32 , 33 , 34} ,
        // {35}
    } ;
  

    
    // unsigned* spokesman_gpu ;
    // cudaMalloc((void**) & spokesman_gpu , num * sizeof(unsigned)) ;
    // cudaMemcpy(spokesman_gpu , spokesman.data() , num * sizeof(unsigned) , cudaMemcpyHostToDevice) ;

    // 压缩全局图
    // array<unsigned*,5> compressed_global_graph = compress_global_graph_extreme(graph_infos, degree , global_graph , 16 , num , global_graph_edge_per_p , seg_partition_point_start_index.data()) ;
    array<unsigned*,5> compressed_global_graph ;
    load_global_graph_extreme("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep100M/global_graph_T.extremeT" , compressed_global_graph) ;
    // cudaFreeHost(global_graph) ;
    unsigned total_spokesman_num = accumulate(compressed_global_graph[1] , compressed_global_graph[1] + 16 , 0) ;
    cout << "全局图压缩完成--------" << endl ; 

    // cout << "global graph view : [" ;
    // for(unsigned i = 0 ; i < 400 ; i ++) {
    //     cout << compressed_global_graph[0][i] ;
    //     if(i < 399)
    //         cout << " ," ;
    // }
    // cout << "]" << endl ;


    // vector<vector<unsigned>> batch_info(16 , vector<unsigned>(1)) ;
    // for(unsigned i =0 ; i < 16 ; i ++)
    //     batch_info[i][0] = i ; 
    unsigned batch_size = 0 ;
    for(unsigned i = 0 ; i < batch_info.size() ;i  ++)
        batch_size = max((unsigned) batch_info[i].size() , batch_size) ;
    cout << "batch size : " << batch_size << endl ;
    // random_shuffle(batch_info.begin() , batch_info.end()) ;

    unsigned reserve_points_num = *max_element(graph_infos[1] , graph_infos[1] + pnum) ;
    __half *data_half_buffer1 , *data_half_buffer2 ;
    size_t data_half_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(reserve_points_num) * static_cast<size_t>(dim) * sizeof(__half) ;
    cudaMalloc((void**) &data_half_buffer1 , data_half_buffer_size) ;
    cudaMalloc((void**) &data_half_buffer2 , data_half_buffer_size) ;
    array<__half*,2> vec_buffer = {data_half_buffer1 , data_half_buffer2} ;
    unsigned* graph_buffer0 , *graph_buffer1 ; 
    size_t graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(degree) * static_cast<size_t>(reserve_points_num) * sizeof(unsigned) ;
    cudaMalloc((void**) &graph_buffer0 , graph_buffer_size) ;
    cudaMalloc((void**) &graph_buffer1 , graph_buffer_size) ;
    array<unsigned*,2> graph_buffer = {graph_buffer0 , graph_buffer1} ;
    unsigned* global_graph_buffer0 , *global_graph_buffer1 ;
    // 此处超出内存限制
    // size_t global_graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(global_graph_degree) * static_cast<size_t>(average_num) * sizeof(unsigned) ;
    // size_t global_graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<const size_t>(batch_size) * static_cast<size_t>(global_graph_edge_per_p) * static_cast<size_t>(average_num) * sizeof(unsigned) ;
    unsigned max_spokesman_num = *max_element(compressed_global_graph[1] , compressed_global_graph[1] + pnum) ;
    size_t global_graph_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(global_graph_edge_per_p) * static_cast<size_t>(max_spokesman_num) * static_cast<size_t>(batch_size) * sizeof(unsigned) ;
    cudaMalloc((void**) &global_graph_buffer0 , global_graph_buffer_size) ;
    cudaMalloc((void**) &global_graph_buffer1 , global_graph_buffer_size) ;
    array<unsigned*,2> global_graph_buffer = {global_graph_buffer0 , global_graph_buffer1} ;
    unsigned* generic_buffer0 , *generic_buffer1 ; 
    cudaMalloc((void**) &generic_buffer0 , 10 * 1024 * 1024 * sizeof(unsigned)) ;
    cudaMalloc((void**) &generic_buffer1 , 10 * 1024 * 1024 * sizeof(unsigned)) ;
    array<unsigned*,2> generic_buffer = {generic_buffer0 , generic_buffer1} ;
    unsigned* stored_pos_global_idx ;
    cudaMalloc((void**) &stored_pos_global_idx , num * sizeof(unsigned)) ;
    array<unsigned*,4> new_idx_mapper = {idx_mapper[0] , idx_mapper[1] , idx_mapper[2] , stored_pos_global_idx} ;


    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cudaError_t err2 = cudaMemGetInfo(&free_bytes, &total_bytes) ;
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "缓冲区申请完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;
    cout << "准备工作完成, 开始查找 :" << endl ;
    s = chrono::high_resolution_clock::now() ;
    // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
    //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
    //     global_graph , 16 , pnum , L) ;
    

    vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter_batch_process_extreme_compressed_graph<attrType>(batch_info.size() , batch_info , degree , data_ , vec_buffer, graph_buffer , global_graph_buffer ,
        generic_buffer , query_data_half_dev , dim , qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        attr_dim  , graph_infos , new_idx_mapper , seg_partition_point_start_index  , partition_infos , L , 10 , 
        compressed_global_graph[0] , global_graph_edge_per_p , pnum , partition_is_selected , 10 , num , compressed_global_graph ,total_spokesman_num , 16) ;
    e = chrono::high_resolution_clock::now() ;
    cost3 = chrono::duration<double>(e - s).count() ;
    cout << "finish search" << endl ;

    cudaFree(data_half_buffer1) ;
    cudaFree(data_half_buffer2) ;
    cudaFree(graph_buffer0) ;
    cudaFree(graph_buffer1) ;
    cudaFree(global_graph_buffer0) ;
    cudaFree(global_graph_buffer1) ;
    cudaFree(generic_buffer0) ;
    cudaFree(generic_buffer1) ;
    cudaFree(stored_pos_global_idx) ;
    cudaFreeHost(compressed_global_graph[0]) ;
    cudaFreeHost(compressed_global_graph[1]) ;
    cudaFreeHost(compressed_global_graph[2]) ;
    cudaFree(compressed_global_graph[3]) ;
    cudaFree(compressed_global_graph[4]) ;
    // cudaFree(spokesman_gpu) ;

    cout << "gpu 暴力:" << endl ;
    auto gpu_bf_s = chrono::high_resolution_clock::now() ;
    vector<vector<unsigned>> gpu_bf_result ;
    vector<vector<float>> gpu_bf_result_dis ; 
    gpu_bf_result = prefiltering_and_bruteforce<attrType>(query_data_half_dev , data_ , qnum ,  10 , dim , num , attrs_dev,
       l_bound_dev , r_bound_dev , attr_dim , gpu_bf_result_dis) ;
    auto gpu_bf_e = chrono::high_resolution_clock::now() ;
    float gpu_bf_cost = chrono::duration<double>(gpu_bf_e - gpu_bf_s).count() ;
    cout << "gpu 暴力 结束: " << gpu_bf_cost << endl ;

    // cout << "ground truth dis [" ;
    // for(unsigned i = 0 ; i < gpu_bf_result_dis[0].size() ; i ++) {
    //     cout << gpu_bf_result_dis[0][i] ;
    //     if(i < gpu_bf_result_dis[0].size() - 1)
    //         cout << " ," ;
    // }
    // cout << "]" << endl ;

    // save_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt", query_num , 10, gpu_bf_result) ;
    // load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt" , gpu_bf_result) ;
    // load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt" , gpu_bf_result) ;
    cudaFreeHost(data_) ;
    // cudaFreeHost(global_graph) ;
    cudaFreeHost(graph_infos[0]) ;
    cudaFreeHost(graph_infos[1]) ;
    cudaFreeHost(graph_infos[2]) ;
    /**
        float* data_float , * query_data_load_float ;
        cout << "开始暴力搜索, 载入数据--------------------" << endl ;
        load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs" , data_float , num , dim) ;
        cout << "num : " << num << ", dim : " << dim << endl ;
        query_data_load_float = new float[query_num * query_dim] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ; j < query_dim ; j ++)
            // query_data_load_float[i * query_dim + j] = data_float[qids[i] * query_dim + j] ;
            size_t ofst = static_cast<size_t>(qids[i]) * static_cast<size_t>(query_dim) ;
            memcpy(query_data_load_float + i * query_dim , data_float + ofst , query_dim * sizeof(float)) ;
        }
        // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
        vector<vector<unsigned>> ground_truth(qnum) ;
        vector<vector<float>> gt_dis(qnum) ;
        auto cpu_bf_s = chrono::high_resolution_clock::now() ;
        #pragma omp parallel for num_threads(10)
        for(unsigned i = 0 ; i < qnum ;i  ++) {
            vector<float> gt_d ; 
            ground_truth[i] = naive_brute_force_with_filter_store_dis<attrType>(query_data_load_float + i * dim , data_float , dim , num , 10 ,
                l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim , gt_d) ;
            gt_dis[i] = gt_d ; 
        }
        auto cpu_bf_e = chrono::high_resolution_clock::now() ;
        float cpu_bf_cost = chrono::duration<double>(cpu_bf_e - cpu_bf_s).count() ;
        reverse(ground_truth[0].begin() , ground_truth[0].end()) ;
        reverse(gt_dis[0].begin() , gt_dis[0].end()) ;
        cout << "gt id :[" ;
        for(unsigned i = 0 ; i < ground_truth[0].size() ;i  ++) {
            cout << ground_truth[0][i] ;
            if(i < ground_truth[0].size() - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;
        cout << "gt dis :[" ;
        for(unsigned i = 0 ; i < gt_dis[0].size() ;i  ++) {
            cout << gt_dis[0][i] ;
            if(i < gt_dis[0].size() - 1)
                cout << " ," ;
        }
        cout << "]" << endl ;
        // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
        float total_recall = cal_recall_show_partition_multi_attrs<attrType>(result , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;

        
        cout << "recall : " << total_recall << endl  ;
    **/
    cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
    cout << "获取分区列表用时:" << cost1 << endl ;
    cout << "对分区列表排序用时:" << cost2 << endl ;
    cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
    cout << "核函数 + 数据运输、准备工作:" << other_kernal_cost << endl ;
    cout << "数据传输时间: " << mem_trans_cost << endl ;
    cout << "搜索用时:" << cost3 << endl ;
    cout << "batch cost : [" ;
    for(unsigned i = 0 ; i < batch_process_cost.size() ; i ++) {
        cout << batch_process_cost[i] ;
        if(i < batch_process_cost.size() - 1)
            cout << " ," ;
    }
    cout << "]" << endl ; 
    cout << "batch mem load cost : [" ;
    for(unsigned i = 0 ; i < batch_data_trans_cost.size() ; i ++) {
        cout << batch_data_trans_cost[i] ;
        if(i < batch_data_trans_cost.size() - 1)
            cout << " ," ;
    }
    cout << "]" << endl ;
    cout << "gpu brute force 时间: " << gpu_bf_cost << endl ;
    gpu_bf_result.resize(qnum) ;
    // cout << "cpu brute force 时间: " << cpu_bf_cost << endl ;
    float total_recall = cal_recall_show_partition_multi_attrs<attrType>(result , gpu_bf_result , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;
    cout << " recall : " << total_recall << endl ;


    rc_recall1 = total_recall ;
    rc_cost1 = select_path_kernal_cost ;

    // cudaFree(global_graph) ;
    cudaFree(attrs_dev) ;
    cudaFree(query_data_half_dev) ;
    cudaFree(l_bound_dev) ;
    cudaFree(r_bound_dev) ;
    cudaFree(attr_min) ;
    cudaFree(attr_width) ;
    cudaFree(attr_grid_size) ;
    cudaFree(partition_is_selected) ;
    for(unsigned i = 0  ; i < 5 ; i ++)
        cudaFree(partition_infos[i]) ;
    for(unsigned i = 0 ; i < 3 ; i ++) 
        cudaFree(idx_mapper[i]) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    
}


void test_single_graph_large_datset(unsigned L = 80) {
 // 先测试一下select_path_V3 是否靠谱
    
    // unsigned* graph ; 
    // float* data_ ;
    unsigned sub_par_id = 0 ;
    unsigned num , dim  ;
    __half* data_   ;
    data_ = load_dataset_return_pined_pointer("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs" , num , dim , 10.0f) ;
    // array<unsigned*,3> graph_infos = get_graph_info_host_avg_size("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/deep100M/deep100M"  , ".cagra32" , 16 , 6250000 , 32) ;
    array<unsigned*,3> graph_infos = get_graph_info_host("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/deep100M/36/deep100M"  , ".cagra16" , 36 , 100000000 , 16) ;
    // unsigned average_num = num / 16 , degree = 32 ; 
    unsigned pnum = 36 , degree = 16 ;
    unsigned average_num = graph_infos[1][0] ;
    cout << "graph_size[" ;
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cout << graph_infos[1][i] ;
        if(i < pnum - 1)
            cout << " ," ;
    }
    cout << "]" << endl;
    cout << "graph_start_index[" ;
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cout << graph_infos[2][i] ;
        if(i < pnum - 1)
            cout << " ," ;
    }
    cout << "]" << endl ;

    __half* data_load_dev ; 
    size_t data_load_byteSize = static_cast<size_t>(average_num) * static_cast<size_t>(dim) * sizeof(__half) ;
    cudaMalloc((void**) &data_load_dev , data_load_byteSize) ;
    size_t data_load_offset = static_cast<size_t>(average_num) * static_cast<size_t>(dim) * static_cast<size_t>(sub_par_id) ;
    cout << "data load offset : " << data_load_offset << endl ;
    cudaMemcpy(data_load_dev , data_ + data_load_offset , data_load_byteSize , cudaMemcpyHostToDevice) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;

    unsigned* graph_dev ;
    cudaMalloc((void**) &graph_dev , average_num * degree * sizeof(unsigned)) ;
    cout << "graph start index : " << graph_infos[2][sub_par_id] << endl ;
    cudaMemcpy(graph_dev , graph_infos[0] + graph_infos[2][sub_par_id], average_num * degree * sizeof(unsigned) , cudaMemcpyHostToDevice) ;


    __half* query_data_load ;
    unsigned query_num = 1000 , query_dim = dim ;
    cudaMalloc((void**) &query_data_load , query_num * query_dim * sizeof(__half)) ;
    // size_t data_load_num = static_cast<size_t>(average_num) * static_cast<size_t>(dim) ;
    cudaMemcpy(query_data_load , data_ , query_num * dim * sizeof(__half) , cudaMemcpyHostToDevice) ;

    unsigned TOPM = L , DEGREE = degree , DIM = dim ;
    const unsigned RETURN_SIZE = TOPM ;
    unsigned* result_gpu ;
    float* result_dis_gpu ;
    cudaMalloc((void**) &result_gpu , RETURN_SIZE * average_num * sizeof(unsigned)) ;
    cudaMalloc((void**) &result_dis_gpu , RETURN_SIZE * average_num * sizeof(float)) ;
    unsigned K = degree ; 
    mt19937 rng ;
    unordered_set<unsigned> us ;
    vector<unsigned> ent_pts(K) ;
    unsigned* ent_pts_dev ;
    cudaMalloc((void**) &ent_pts_dev , K * sizeof(unsigned)) ;
    uniform_int_distribution<> dist(0 , average_num - 1) ;
    int ent_cnt = 0 ;
    while(ent_cnt < K) {
        unsigned ep = dist(rng) ;
        if(!us.count(ep)) {
            us.insert(ep) ;
            ent_pts[ent_cnt ++] = ep ;
        }
    }
    cudaMemcpy(ent_pts_dev , ent_pts.data() , K * sizeof(unsigned) , cudaMemcpyHostToDevice) ;

    unsigned qnum = query_num ;
    cout << "准备工作完成, 开始搜索" << endl ;
    dim3 grid_s(qnum , 1 , 1) , block_s(32 , 6 , 1) ;
    
    unsigned byteSize = 
     (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float)  + (DIM/4) * sizeof(half4) + alignof(half4) ;
    cout << "byteSize : " << (byteSize * 1.0 / 1024) << "KB" << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    select_path_v3<<<grid_s , block_s , byteSize>>>(graph_dev , data_load_dev, query_data_load, result_gpu , result_dis_gpu,  ent_pts_dev ,  degree ,TOPM , dim) ;
    // select_path_v4<<<grid_s , block_s>>>(graph_dev, data_load_dev, query_data_load, result_gpu, result_dis_gpu, ent_pts_dev , TOPM , degree , DIM ) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    auto e = chrono::high_resolution_clock::now() ;
    float cost = chrono::duration<double>(e - s).count() ;
    cout << "用时:" << cost << endl ;

    vector<unsigned> result(RETURN_SIZE * qnum) ;
    cudaMemcpy(result.data() , result_gpu , RETURN_SIZE * qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // cout << "{" << endl ;
    // for(unsigned i = 0 ; i < qnum ; i ++) {
    //     cout << "[" ;
    //     for(unsigned j = 0 ; j < RETURN_SIZE ; j ++) {
    //         cout << result[i * RETURN_SIZE + j] ;
    //         if(j < RETURN_SIZE - 1)
    //             cout << " ," ;
    //     }
    //     cout << "]" << endl ;
    // }
    // cout << "}" << endl ; 
    vector<unsigned> result2(10 * qnum) ;
    for(unsigned i = 0 ; i < query_num ; i ++) {
        for(unsigned j = 0  ;j  < 10 ; j  ++)
            result2[i * 10 + j] = result[i * RETURN_SIZE + j] ;
    }
    result = result2 ; 

    cout << "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/deep100M/100K10_36_GT"+ to_string(sub_par_id) + ".gt" << endl ;
    unsigned* gt ;
    unsigned gt_num , gt_k ; 
    load_result("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/deep100M/100K10_36_GT"+ to_string(sub_par_id) + ".gt" , gt , gt_k , gt_num) ;
    gt_num = query_num ;
    cout << "gt_num : " << gt_num << ", gt_k : " << gt_k  << endl ;



    vector<float> result_dis_cpu(query_num * L) ;
    cudaMemcpy(result_dis_cpu.data() , result_dis_gpu , query_num * RETURN_SIZE *  sizeof(float) , cudaMemcpyDeviceToHost) ;
    for(unsigned i = 0 ; i < query_num / 10 ;i  ++) {
        cout << "q" << i << "[" ;
        for(unsigned j = 0 ; j < 10 ; j ++)
            cout << result[i * 10 + j] << "-" << result_dis_cpu[i * RETURN_SIZE + j] << " ," ;
        cout << "]" << endl ;
    }
    vector<vector<unsigned>> gt_vec(query_num , vector<unsigned>(gt_k)) ;
    for(unsigned i = 0 ; i < query_num ;i  ++) {
        for(unsigned j = 0 ; j < gt_k ; j ++) {
            gt_vec[i][j] = gt[i * gt_k + j] ;
        }
    }

    vector<vector<unsigned>> res_vec(query_num , vector<unsigned>(gt_k)) ;
    for(unsigned i = 0 ; i < query_num ;i  ++) {
        for(unsigned j = 0 ; j < gt_k ; j ++) {
            res_vec[i][j] = result[i * gt_k + j] ;
        }
    }
    float rc = cal_recall(res_vec , gt_vec, false) ;
    cout << "rc recall : " << rc << endl ;
    // for(unsigned i = 0 ; i < query_num ;i  ++) {
    //     cout << "q" << i << "[" ;
    //     for(unsigned j = 0 ; j < 10 ; j ++)
    //         cout << result[i * RETURN_SIZE + j] << "-" << result_dis_cpu[i * RETURN_SIZE + j] << " ," ;
    //     cout << "]" << endl ;
    // }
} 


void test_deep1M(unsigned L = 80) {
    // 先测试一下select_path_V3 是否靠谱
    
    // unsigned* graph ; 
    // float* data_ ;
    unsigned num , dim  ;
    float* data_host  ;
    load_data("/home/yhr/workspace/cuda_clustering/data/sift_base.fvecs" , data_host , num , dim) ;
    cout << "num : " << num << ", dim : " << dim << endl ;
    float* data_dev ; 
    cudaMalloc((void**) &data_dev , num * dim * sizeof(float)) ;
    cudaMemcpy(data_dev , data_host , num * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    __half* data_load_dev ; 
    // size_t data_load_byteSize = static_cast<size_t>(average_num) * static_cast<size_t>(dim) * sizeof(__half) ;
    size_t data_load_byteSize = num * dim * sizeof(__half) ;
    cudaMalloc((void**) &data_load_dev , data_load_byteSize) ;
    // cudaMemcpy(data_load_dev , data_ , data_load_byteSize , cudaMemcpyHostToDevice) ;
    f2h<<<num , 256>>>(data_dev, data_load_dev, num ,dim , 0.01f);
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cout << "数据集转换完成" << endl ;


    unsigned* graph , degree , num_in_graph ;
    load_result("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/sift1M/sift1M.cagra32" , graph , degree , num_in_graph) ;
    cout << "degree : " << degree << "num_in_graph" << num_in_graph << endl ;
    unsigned* graph_dev ;
    cudaMalloc((void**) &graph_dev , num_in_graph * degree * sizeof(unsigned)) ;
    cudaMemcpy(graph_dev , graph ,  num_in_graph * degree * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    cout << "图载入完成" << endl ;

    __half* query_data_load ;
    unsigned query_num = 1000 , query_dim = dim ;
    cudaMalloc((void**) &query_data_load , query_num * query_dim * sizeof(__half)) ;
    // size_t data_load_num = static_cast<size_t>(average_num) * static_cast<size_t>(dim) ;
    cudaMemcpy(query_data_load , data_load_dev , query_num * dim * sizeof(__half) , cudaMemcpyDeviceToDevice) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cout << "查询信息生成完成" << endl ;
    unsigned TOPM = L , DEGREE = degree , DIM = dim ;
    const unsigned RETURN_SIZE = TOPM ;
    unsigned* result_gpu ;
    float* result_dis_gpu ;
    cudaMalloc((void**) &result_gpu , RETURN_SIZE * query_num * sizeof(unsigned)) ;
    cudaMalloc((void**) &result_dis_gpu , RETURN_SIZE * query_num * sizeof(float)) ;
    unsigned K = degree ; 
    mt19937 rng ;
    unordered_set<unsigned> us ;
    vector<unsigned> ent_pts(K) ;
    unsigned* ent_pts_dev ;
    cudaMalloc((void**) &ent_pts_dev , K * sizeof(unsigned)) ;
    uniform_int_distribution<> dist(0 , num - 1) ;
    int ent_cnt = 0 ;
    while(ent_cnt < K) {
        unsigned ep = dist(rng) ;
        if(!us.count(ep)) {
            us.insert(ep) ;
            ent_pts[ent_cnt ++] = ep ;
        }
    }
    for(unsigned i = 0 ; i < 32 ; i ++)
        ent_pts[i] = i ; 
    cudaMemcpy(ent_pts_dev , ent_pts.data() , K * sizeof(unsigned) , cudaMemcpyHostToDevice) ;

    unsigned qnum = query_num ;
    cout << "准备工作完成, 开始搜索" << endl ;
    dim3 grid_s(qnum , 1 , 1) , block_s(32 , 6 , 1) ;
    
    unsigned byteSize = 
    (TOPM + DEGREE * 2) * sizeof(unsigned) + (TOPM + DEGREE * 2) * sizeof(float)  + (DIM/4) * sizeof(half4) + alignof(half4) ;
    cout << "byteSize : " << (byteSize * 1.0 / 1024) << "KB" << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    select_path_v3<<<grid_s , block_s , byteSize>>>(graph_dev , data_load_dev, query_data_load, result_gpu , result_dis_gpu,  ent_pts_dev ,  degree ,TOPM , dim) ;
    // select_path_v4<<<grid_s , block_s>>>(graph_dev, data_load_dev, query_data_load, result_gpu, result_dis_gpu, ent_pts_dev , TOPM , degree , DIM ) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    auto e = chrono::high_resolution_clock::now() ;
    float cost = chrono::duration<double>(e - s).count() ;
    cout << "用时:" << cost << endl ;

    vector<unsigned> result(RETURN_SIZE * qnum) ;
    cudaMemcpy(result.data() , result_gpu , RETURN_SIZE * qnum * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // cout << "{" << endl ;
    // for(unsigned i = 0 ; i < qnum ; i ++) {
    //     cout << "[" ;
    //     for(unsigned j = 0 ; j < RETURN_SIZE ; j ++) {
    //         cout << result[i * RETURN_SIZE + j] ;
    //         if(j < RETURN_SIZE - 1)
    //             cout << " ," ;
    //     }
    //     cout << "]" << endl ;
    // }
    // cout << "}" << endl ; 
    vector<unsigned> result2(10 * qnum) ;
    for(unsigned i = 0 ; i < query_num ; i ++) {
        for(unsigned j = 0  ;j  < 10 ; j  ++)
            result2[i * 10 + j] = result[i * RETURN_SIZE + j] ;
    }
    result = result2 ; 


    unsigned* gt ;
    unsigned gt_num , gt_k ; 
    load_result("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/sift1M/100K10_GT.gt" , gt , gt_k , gt_num) ;
    gt_num = query_num ;
    cout << "gt_num : " << gt_num << ", gt_k : " << gt_k  << endl ;
    int  cnt = 0 ; 
    // for(unsigned i = 0 ; i < gt_num * gt_k ; i ++) {
    //     cnt += (result[i] == gt[i]) ;
    //     // if(i < 20) {
    //     //     cout << "result : " << result[i] << ", gt : " << gt[i] << endl ;
    //     // }
    // }
    float recall = 0.0 ;
    for(int i = 0 ; i < gt_num ;i  ++) {
        int tt = 0 ; 
        for(int j = 0 ; j < gt_k ; j ++) {
            tt += (result[i * gt_k + j] == gt[i * gt_k + j]) ;
        }
        recall += (tt * 1.0f / gt_k) ;
    }
    recall /= gt_num ;
    // cout << "cnt : " << cnt << endl ;
    // cout << (cnt * 1.0f / qnum) ;
    cout << "recall : " << recall << endl ;


    vector<float> result_dis_cpu(query_num * L) ;
    cudaMemcpy(result_dis_cpu.data() , result_dis_gpu , query_num * RETURN_SIZE *  sizeof(float) , cudaMemcpyDeviceToHost) ;
    // for(unsigned i = query_num *L - 1000 ;i  < query_num * L ;i  ++) {
    //     cout << result[i] << "-" << result_dis_cpu[i] << " ," ;
    // }
    for(unsigned i = 0 ; i < query_num / 10 ;i  ++) {
        cout << "q" << i << "[" ;
        for(unsigned j = 0 ; j < 10 ; j ++)
            cout << result[i * 10 + j] << "-" << result_dis_cpu[i * RETURN_SIZE + j] << " ," ;
        cout << "]" << endl ;
    }

    vector<vector<unsigned>> ground_truth(query_num) ;
    // vector<vector<float>> gt_dis(qnum) ;
    #pragma omp parallel for num_threads(10)
    for(unsigned i = 0 ; i < query_num ;i  ++) {
        // vector<float> gt_d ; 
        ground_truth[i] = naive_brute_force(data_host + i * dim, data_host , dim , num , 10) ;
        // gt_dis[i] = gt_d ; 
    }
    vector<vector<unsigned>> res_vec(query_num , vector<unsigned>(10)) ;
    vector<vector<unsigned>> gt_vec(query_num , vector<unsigned>(10)) ;
    for(unsigned i = 0 ; i < query_num ;i  ++) {
        for(unsigned j = 0 ; j < 10 ; j ++) {
            res_vec[i][j] = result[i * 10 + j] ;
        }
    }
    float rc = cal_recall(res_vec , ground_truth) ;
    cout << rc << endl ;

    for(unsigned i = 0 ; i < query_num ;i  ++) {
        for(unsigned j = 0 ; j < 10 ; j ++) {
            gt_vec[i][j] = gt[i * 10 + j] ;
        }
    }
    rc = cal_recall(res_vec , gt_vec) ;
    cout << rc << endl ;


    // for(unsigned i = 0 ;i  < 10  ; i++)
        // cout << data_host[dim + i] << " ;" ;
} 


void check_bitonic_sort() {
    sort_test<<<1 , 32 * 6>>> () ;
    cudaDeviceSynchronize() ;
}

void check_test_prefilter_bf() {
    // 先载入数据集和图索引
    unsigned num , dim  ;
    float* data_  ;
    load_data("/home/yhr/workspace/cuda_clustering/data/sift_base.fvecs" , data_ , num , dim) ;
    cout << "num : " << num << ", dim : " << dim << endl ;
    float* query_data_load ;
    unsigned query_num , query_dim ;
    load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_range_query.fvecs" , query_data_load , query_num , query_dim) ;
    // query_num = 500; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;
    // query_num *= 10 ;
    float* tmp_query_data_load = new float[10 *query_num * query_dim] ;
    for(int i = 0 ; i < 10 ; i ++)
        memcpy(tmp_query_data_load + i * query_num * query_dim  , query_data_load , query_num * query_dim * sizeof(float)) ;
    query_num *= 10 ;
    delete [] query_data_load ;
    query_data_load = tmp_query_data_load ;
    // for(unsigned i = 0 ; i )
    // TOPM = 20 , DIM = dim ; 
    // ************** 若要修改查询数量, 请修改如下变量 **********************
    // query_num = 1 ;
    // for(int i = 0 ; i < dim ; i ++)
    //     query_data_load[i] = query_data_load[613 * dim + i] ;

    float* data_dev ;
    half *data_half_dev ;
    float* query_data_dev ;
    half *query_data_half_dev ;
    cudaMalloc((void**)&data_dev , num * dim * sizeof(float)) ;
    cudaMalloc((void**) &data_half_dev , num * dim * sizeof(half)) ;
    cudaMalloc((void**) &query_data_dev , query_num * query_dim   * sizeof(float)) ;
    unsigned query_num_with_pad = (query_num + 15) / 16 * 16 ;
    cout << "query_num_with_pad : " << query_num_with_pad << endl ;
    cudaMalloc((void**) &query_data_half_dev , query_num_with_pad * query_dim * sizeof(half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(data_dev , data_ , num * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(query_data_dev , query_data_load , query_num * query_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num ; 
    // 调用实例
    f2h<<<points_num, 256>>>(data_dev, data_half_dev, points_num); // 先将数据转换成fp16
    // cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    f2h<<<query_num, 256>>>(query_data_dev, query_data_half_dev, query_num); // 将查询转换成fp16
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    if(query_num != query_num_with_pad)
        thrust::fill(thrust::device , query_data_half_dev + query_num * query_dim , query_data_half_dev + query_num_with_pad * query_dim , __float2half(0.0f)) ;
    
    unsigned* ids ;
    float* ret_dis ; 
    cudaMalloc((void**) &ids , query_num * 6250 * 2 * sizeof(unsigned)) ;
    cudaMalloc((void**) &ret_dis , query_num * 6250 * 2 * sizeof(unsigned)) ;
    thrust::sequence(thrust::device , ids , ids + query_num * 6250 * 2) ;
    thrust::transform(
        thrust::device , 
        ids , ids + query_num * 6250 * 2 ,
        ids , 
        [] __host__ __device__ (unsigned x) {
            return x % 1000000 ;
        }
    ) ;

    auto s = chrono::high_resolution_clock::now() ;
    dis_cal_test_prefilter<<<query_num , dim3(32 , 6 , 1)>>>(data_half_dev , query_data_half_dev , 6250 * 2 , dim , ids , ret_dis) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    auto e = chrono::high_resolution_clock::now() ;

    float cost = chrono::duration<double>(e - s).count() ;
    cout << "prefilter bruteforce cost : " << cost << endl ;
}


int main1(int argc , char** argv){ 

    showMemInfo("进入主函数") ;
    cout << "输入selectivity : " ;
    // cin >> selectivity ;
    selectivity = stod(argv[1]) ;
    // selectivity = 0.0 ;
    cout << "输入L1 : " ;
    unsigned L1  ;
    // cin >> L1 ; 
    L1 = atoi(argv[2]) ;
    // cout << "输入L2 : " ;
    // cin >> L2 ;

   
    // check_all_process_version0(L1) ;
    // check_all_process(L1) ;

    // multi_attr_test_batch_process(L1) ; 
    // multi_attr_test_batch_process_large_dataset(L1) ;
    // multi_attr_test_batch_process_large_dataset_compressed_graph_T(L1) ;
    // multi_attr_test_batch_process_large_dataset_compressed_graph_T36(L1) ;
    
    // multi_attr_test_batch_process_large_dataset_extreme_compressed_graph(L1) ;
    // multi_attr_test_batch_process_large_dataset_extreme_compressed_graph_pnum36(L1) ;
    // multi_attr_test_batch_process_large_dataset_compressed_graph_T36_dataset_resident(L1) ;
    // test_single_graph_large_datset(L1) ;
    // test_deep1M(L1) ;
    multi_attr_test(L1) ; 
    cout << "with prefilter : " << endl ;
    cout << "recall : " << rc_recall1 << endl ;
    cout << "cost : " << rc_cost1 << endl ;
    // cout << "without prefilter : " << endl ;
    // cout << "recall : " << rc_recall2 << endl ;
    // cout << "cost : " << rc_cost2 << endl ;
    // check_bitonic_sort() ;
    // check_test_prefilter_bf() ;
    return 0 ;
}