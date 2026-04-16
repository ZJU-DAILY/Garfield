#include "../big_dataset_adapterV3.cuh"
#include "../check.cu"

using namespace std ;

template<unsigned attr_dim = 1>
void multi_attr_test_batch_process_large_dataset_compressed_graph_T36_dataset_resident_new(string dataset_filepath , string dataset_filepath_q , string query_data_path ,  string cagra_prefix , string cagra_suffix , string global_graph_path , unsigned degree , unsigned global_graph_edge_per_p ,  vector<unsigned> ef_list ,string attrFile , string rangeFile , string gtFile ,  string resultFile , double prefilter_threshold ,vector<unsigned> grid_shape , 
    float f2h_factor) {
    
    // 先载入数据集和图索引
    unsigned num , dim  ;
    float* data_ ;
    int8_t* data_int8_dev ;
    float* scales ; 
    int* zero_points ;
    cout << "开始载入数据集" << endl ;

    smart::production_adapter::load_dataset_return_int8_arr_infile(dataset_filepath , dataset_filepath_q ,data_ , num , dim , f2h_factor, data_int8_dev , scales , zero_points) ;
    cout << "数据集载入完成: num:" << num << ", dim :" << dim << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    smart::production_adapter::show_dataset_int8_lines(data_ , data_int8_dev , num , dim , scales , zero_points , 1) ;
    
    showMemInfo("算法开始") ;

    unsigned pnum = std::accumulate(grid_shape.begin(), grid_shape.end(), 1, std::multiplies<int>());

    array<unsigned*,3> graph_infos = get_graph_info_host(cagra_prefix , cagra_suffix , pnum , num , degree) ;
    showMemInfo("图索引载入完成") ;




    unsigned preset_num = num , preset_pnum = pnum ;

    // unsigned degree = 16 , global_graph_edge_per_p = 2 ; 
    array<unsigned*,3> idx_mapper = get_idx_mapper(num , pnum , graph_infos) ;
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
    load_result_pined(global_graph_path ,
     global_graph , global_graph_degree , global_graph_num_in_graph) ;
    cout << "global graph : degree-" << global_graph_degree << ", num-" << global_graph_num_in_graph << endl ;
    showMemInfo("全局图载入完成") ;
    // for(unsigned i = 0 ; i < 1000000 ; i ++) {
    //     cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    // }
    using attrType = unsigned ;
    attrType *attrs_dev , *attrs ;
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    // constexpr unsigned attr_dim =  1 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<attrType> grid_min(attr_dim , 0.0) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<attrType> grid_max(attr_dim , 1e8) ;
    vector<unsigned> grid_size_host = grid_shape ;
    vector<attrType> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++) {
        attr_width_host[i] = (grid_max[i] - grid_min[i] + grid_size_host[i] - 1) / grid_size_host[i] ;
        cout << "attr_width_host[i] : " << attr_width_host[i] << "," ;
    }
    // attrs = new attrType[attr_dim * 100000000] ;
    // re_generate<attrType>(attrs , attr_dim , 100000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    // re_generate<attrType>(attrs , attr_dim , preset_num , grid_min.data() , grid_max.data() , grid_size_host.data() , graph_infos[1] , seg_partition_point_start_index.data() , preset_pnum) ;
    // save_attrs("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/attrs.bin" ,100000000 , attr_dim, attrs) ;
    // load_attrs<attrType>("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/big_dataset/deep100M/attributes/attr100M1D.attrs", attrs) ;
    load_attrs<attrType>(attrFile , attrs) ;
    size_t attrs_dev_byteSize = static_cast<size_t>(attr_dim) * num * sizeof(attrType) ;
    cudaMalloc((void**) &attrs_dev , attrs_dev_byteSize) ;
    cudaMemcpy(attrs_dev , attrs , attrs_dev_byteSize , cudaMemcpyHostToDevice) ;
    delete [] attrs ;
    cout << "sizeof(half) = " << sizeof(half) << endl ;

    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    double free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    double total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "属性分配完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;


    


    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "数据转换完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;

    float* query_data_load , * query_data_load_dev ;
    unsigned query_num = 1 , query_dim = dim ;
    // string query_data_path = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep1m_query.fvecs" ;
    load_data(query_data_path.c_str() , query_data_load , query_num , query_dim) ;
    cout << "查询读取成功: qnum = " << query_num  <<  ", qdim = " << query_dim << endl ;  

    half *query_data_half_dev ;
    
    cudaMalloc((void**) &query_data_half_dev , query_num * query_dim * sizeof(__half)) ;
    cudaMalloc((void**) &query_data_load_dev , query_num * query_dim * sizeof(float)) ;
    // 将数据复制到设备内存
    cudaMemcpy(query_data_load_dev , query_data_load , query_num * query_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    f2h<<<query_num, 256>>>(query_data_load_dev , query_data_half_dev , query_num , dim , 10.0f);
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    unsigned points_num = num ; 
    
    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    vector<attrType> l_bound_host(query_num * attr_dim) , r_bound_host(query_num * attr_dim) ;
    if(true) {
        cout << "sel = " << selectivity << endl ;
        // load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift1M/sift_query_range.bin", query_range_x, 1000) ;
        // load_range<attrType>("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/big_dataset/deep100M/attributes/attr100M1D_Q10K.range" , l_bound_host.data() ,r_bound_host.data() , attr_dim , query_num) ;
        load_range<attrType>(rangeFile , l_bound_host.data() ,r_bound_host.data() , attr_dim , query_num) ;
        /**
        query_range_x.reserve(query_num) ;
        for(unsigned i = 1 ; i < 10 ; i ++)
            query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        query_range_y = query_range_x ; 
        **/
        // query_range_x[0] = {0 ,16666600 } , query_range_y[0] = {0 , 16666600 * 4} ;
        // for(unsigned i = 1 ; i < query_num ; i ++)
        //     query_range_y[i] = query_range_y[0] , query_range_x[i] = query_range_x[0] ;
        // query_range_x[0] = {0 , 24999999} ;
        // query_range_y[0] = {0 , 24999999} ;
        /**
        for(unsigned i = 0 ; i < query_num ;i  ++) {
            query_range_x[i].first *= 100 , query_range_x[i].second *= 100 ; 
            query_range_y[i].first *= 100 , query_range_y[i].second *= 100 ;
            // query_range_x[i] = query_range_x[0] , query_range_y[i] = query_range_y[0] ;
        }

            // query_range_x[i] = query_range_x[0] , query_range_y[i] = query_range_y[0] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ;j < attr_dim ; j ++)
            l_bound_host[i * attr_dim] = (attrType)query_range_x[i].first , r_bound_host[i * attr_dim] = (attrType)query_range_x[i].second ;
            l_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].first , r_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].second ;
        }
        **/
        // cout << "range 0 (" << query_range_x[0].first << "," << query_range_x[0].second << ")" << endl ;
        // cout << "range 1 (" << query_range_x[1].first << "," << query_range_x[1].second << ")" << endl ;
        // cout << "range 2 (" << query_range_x[2].first << "," << query_range_x[2].second << ")" << endl ;
        // cout << "range 5 (" << query_range_x[5].first << "," << query_range_x[5].second << ")" << endl ;
        for(int i = 0 ;i < 5 ; i ++) 
            cout <<"range " << i << " (" << l_bound_host[i] << "," << r_bound_host[i] << ")" << endl ;
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
        // generate_range_multi_attr_with_selectivity<attrType>(l_bound_host.data() , r_bound_host.data() , attr_dim , grid_min.data() , grid_max.data() , query_num , selectivity) ;
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

    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    // size_t free_bytes, total_bytes;

    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "属性, 查询范围分配完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;

    bool *partition_is_selected ; 
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg_batch_process<attr_dim , attrType>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , prefilter_threshold , partition_is_selected) ;
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
    vector<unsigned> batch_pid_list(36) ;
    for(unsigned i = 0 ; i < 36 ; i ++)
        batch_pid_list[i] = i ;
    random_shuffle(batch_pid_list.begin() , batch_pid_list.end()) ;
    vector<vector<unsigned>> batch_info(9 , vector<unsigned>(4)) ;
    for(unsigned i = 0 ; i < pnum ; i ++)
        batch_info[i / 4][i % 4] = batch_pid_list[i] ; 

    // vector<vector<unsigned>> batch_info = {
    //     {0 , 1 , 2 , 3}
    // } ;

    // vector<vector<unsigned>> batch_info = {
    //     {20, 21, 19, 18},
    //     {12, 13, 14, 15},
    //     {16, 17, 23, 22},
    //     {28, 29, 26, 27}, 
    //     {8, 9, 10, 11}, 
    //     {24, 25, 30, 31}, 
    //     {1, 0, 6, 7}, 
    //     {32, 33, 34, 35}, 
    //     {2, 4, 5, 3}
    // } ;

    // vector<vector<unsigned>> batch_info(16 , vector<unsigned>(1)) ;
    // for(unsigned i =0 ; i < 16 ; i ++)
    //     batch_info[i][0] = i ; 
    unsigned batch_size = 0 ;
    for(unsigned i = 0 ; i < batch_info.size() ; i ++)
        batch_size = max(batch_size , (unsigned) batch_info[i].size()) ;
    unsigned reserve_points_num = *max_element(graph_infos[1] , graph_infos[1] + pnum) ;
    // random_shuffle(batch_info.begin() , batch_info.end()) ;
    /**
    __half *data_half_buffer1 , *data_half_buffer2 ;
    // size_t data_half_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(reserve_points_num) * static_cast<size_t>(dim) * sizeof(__half) ;
    size_t data_half_buffer_size = static_cast<const size_t>(num) * static_cast<const size_t>(dim) * sizeof(__half) ;
    cudaMalloc((void**) &data_half_buffer1 , data_half_buffer_size) ;
    cudaMemcpy(data_half_buffer1 , data_ , data_half_buffer_size , cudaMemcpyHostToDevice) ;
    // cudaMalloc((void**) &data_half_buffer2 , data_half_buffer_size) ;
    array<__half*,2> vec_buffer = {data_half_buffer1 , data_half_buffer1} ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    **/
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
    vector<vector<unsigned>> gpu_bf_result ;
    // load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/big_dataset/deep100M/ground_truth/1D.gt" , gpu_bf_result) ;
    load_ground_truth(gtFile , gpu_bf_result) ;
    // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
    //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
    //     global_graph , 16 , pnum , L) ;
    ofstream resultFileOs(resultFile , ios::out) ;
    resultFileOs << "ef,recall,qps,total_cost,pre_cost,search_cost" << endl ;
    for(int efid = 0 ; efid < ef_list.size() ; efid ++) {

        unsigned ef = ef_list[efid] ;
        vector<vector<unsigned>> result = smart::big_dataset_adapter::batch_graph_search_gpu_with_prefilter_batch_process_compressed_graph_T_dataset_resident<attrType>(batch_info.size() , batch_info , degree , data_ , data_int8_dev , scales , zero_points, graph_buffer , global_graph_buffer ,
        generic_buffer , query_data_load ,  query_data_half_dev , dim , qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        attr_dim  , graph_infos , new_idx_mapper , seg_partition_point_start_index  , partition_infos , ef , 10 , 
        global_graph , global_graph_edge_per_p , pnum , partition_is_selected , 200 , num , 16) ;

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
        // cout << "gpu brute force 时间: " << gpu_bf_cost << endl ;
        gpu_bf_result.resize(qnum) ;
        // cout << "cpu brute force 时间: " << cpu_bf_cost << endl ;
        float total_recall = cal_recall_show_partition_multi_attrs<attrType>(result , gpu_bf_result , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , false) ;
        cout << " recall : " << total_recall << endl ;
        
        batch_process_cost.clear() ;
        batch_data_trans_cost.clear() ;
        float pre_cost = COST::cost1[0] ;
        float total_search_cost = pre_cost + other_kernal_cost ;
        float qps = qnum * 1.0 / total_search_cost ; 
        resultFileOs << ef << "," << total_recall << "," << qps  << "," << total_search_cost << "," << pre_cost << "," << other_kernal_cost << endl ;
        cout << "ef : " << ef << ", recall : " << total_recall << ", qps : " << qps  << ", latency : " << total_search_cost << ", precost : " << pre_cost << ", other_kernal_cost : " << other_kernal_cost << endl ;
        
    }    

    

    resultFileOs.close() ;
    e = chrono::high_resolution_clock::now() ;
    cost3 = chrono::duration<double>(e - s).count() ;
    cout << "finish search" << endl ;

    // cudaFree(data_half_buffer1) ;
    // (data_half_buffer2) ;
    cudaFree(graph_buffer0) ;
    cudaFree(graph_buffer1) ;
    cudaFree(global_graph_buffer0) ;
    cudaFree(global_graph_buffer1) ;
    cudaFree(generic_buffer0) ;
    cudaFree(generic_buffer1) ;
    cudaFree(stored_pos_global_idx) ;



    // for(unsigned i = 0 ; i < 10  ;i ++)
    //     cout << gpu_bf_result_dis[0][i] << " ," ;

    // save_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/100Kquery.gt", query_num , 10, gpu_bf_result) ;
    
    // load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt" , gpu_bf_result) ;
    delete [] data_ ;
    cudaFreeHost(global_graph) ;
    cudaFreeHost(graph_infos[0]) ;
    cudaFreeHost(graph_infos[1]) ;
    cudaFreeHost(graph_infos[2]) ;



// 
    // rc_recall1 = total_recall ;
    // rc_cost1 = select_path_kernal_cost ;

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



template<unsigned attr_dim = 1>
void multi_attr_test_batch_process_large_dataset_compressed_graph_T36_dataset_resident_u8(vector<unsigned> ef_list ,string attrFile , string rangeFile , string gtFile ,  string resultFile , double prefilter_threshold ,vector<unsigned> grid_shape) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    // print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    showMemInfo("算法开始") ;
    unsigned average_num = 6250000 ;
    array<unsigned*,3> graph_infos = get_graph_info_host("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/sift100M/sift100M" 
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
    load_result_pined("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/sift100M/global_graph_T36.graphT" ,
     global_graph , global_graph_degree , global_graph_num_in_graph) ;
    cout << "global graph : degree-" << global_graph_degree << ", num-" << global_graph_num_in_graph << endl ;
    showMemInfo("全局图载入完成") ;
    // for(unsigned i = 0 ; i < 1000000 ; i ++) {
    //     cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    // }
    using attrType = unsigned ;
    attrType *attrs_dev , *attrs ;
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    // constexpr unsigned attr_dim =  1 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<attrType> grid_min(attr_dim , 0.0) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<attrType> grid_max(attr_dim , 1e8) ;
    vector<unsigned> grid_size_host = grid_shape ;
    vector<attrType> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++) {
        attr_width_host[i] = (grid_max[i] - grid_min[i] + grid_size_host[i] - 1) / grid_size_host[i] ;
        cout << "attr_width_host[i] : " << attr_width_host[i] << "," ;
    }
    // attrs = new attrType[attr_dim * 100000000] ;
    // re_generate<attrType>(attrs , attr_dim , 100000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    // re_generate<attrType>(attrs , attr_dim , preset_num , grid_min.data() , grid_max.data() , grid_size_host.data() , graph_infos[1] , seg_partition_point_start_index.data() , preset_pnum) ;
    // save_attrs("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/attrs.bin" ,100000000 , attr_dim, attrs) ;
    // load_attrs<attrType>("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/big_dataset/deep100M/attributes/attr100M1D.attrs", attrs) ;
    load_attrs<attrType>(attrFile , attrs) ;
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
    // float* data_ ;
    uint8_t* data_ ;
    uint8_t* data_u8_dev ;
    float* data_ori ;
    // float* scales ; 
    // int* zero_points ;
    cout << "开始载入数据集" << endl ;
    // data_ = load_dataset_return_pined_pointer("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs" ,
    //  num , dim , 10.0) ;
    string dataset_filepath = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift100M/dataset/learn.100M.u8bin" ;
    load_data_u8(dataset_filepath, data_, num, dim) ;

    cout << "数据集载入完成 : num - " << num  << ", dim - " << dim << endl ; 
    size_t total_data_size = static_cast<size_t>(num) * static_cast<size_t>(dim) * sizeof(uint8_t) ;
    cudaMalloc((void**) &data_u8_dev , total_data_size) ;
    cudaMemcpy(data_u8_dev , data_ , total_data_size , cudaMemcpyHostToDevice) ;
    delete [] data_ ;
    load_data_u8(dataset_filepath , data_ori , num , dim) ;
    cout << "float32 版本数据集加载完毕: num - " << num << ", dim - " << dim << endl ;


    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    free_gb = free_bytes / 1024.0 / 1024.0 / 1024.0 ;
    total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0 ;
    cout << "数据转换完成, 现有剩余显存 : " << free_gb << "/" << total_gb << endl ;

    float* query_ori ;
    uint8_t* query_data_load , * query_data_load_dev ;
    unsigned query_num = 1 , query_dim = dim ;
    string query_data_path = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift100M/dataset/sift100M_u8_q10k.u8bin" ;
    load_data_u8(query_data_path , query_data_load , query_num , query_dim) ;
    cout << "查询读取成功: qnum = " << query_num  <<  ", qdim = " << query_dim << endl ;  

    delete [] query_data_load ; 
    load_data_u8(query_data_path , query_ori , query_num , query_dim) ;
    cout << "float32 版本的查询载入完毕: query_num = " << query_num << ", query_dim = " << query_dim << endl ;
    // half *query_data_half_dev ;
    
    cudaMalloc((void**) &query_data_load_dev , query_num * query_dim * sizeof(uint8_t)) ;
    // cudaMalloc((void**) &query_data_load_dev , query_num * query_dim * sizeof(float)) ;
    // 将数据复制到设备内存
    cudaMemcpy(query_data_load_dev , query_data_load , query_num * query_dim * sizeof(uint8_t) , cudaMemcpyHostToDevice) ;

    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    unsigned points_num = num ; 
    
    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    vector<attrType> l_bound_host(query_num * attr_dim) , r_bound_host(query_num * attr_dim) ;
    if(true) {
        cout << "sel = " << selectivity << endl ;
        // load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift1M/sift_query_range.bin", query_range_x, 1000) ;
        // load_range<attrType>("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/big_dataset/deep100M/attributes/attr100M1D_Q10K.range" , l_bound_host.data() ,r_bound_host.data() , attr_dim , query_num) ;
        load_range<attrType>(rangeFile , l_bound_host.data() ,r_bound_host.data() , attr_dim , query_num) ;
        /**
        query_range_x.reserve(query_num) ;
        for(unsigned i = 1 ; i < 10 ; i ++)
            query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        query_range_y = query_range_x ; 
        **/
        // query_range_x[0] = {0 ,16666600 } , query_range_y[0] = {0 , 16666600 * 4} ;
        // for(unsigned i = 1 ; i < query_num ; i ++)
        //     query_range_y[i] = query_range_y[0] , query_range_x[i] = query_range_x[0] ;
        // query_range_x[0] = {0 , 24999999} ;
        // query_range_y[0] = {0 , 24999999} ;
        /**
        for(unsigned i = 0 ; i < query_num ;i  ++) {
            query_range_x[i].first *= 100 , query_range_x[i].second *= 100 ; 
            query_range_y[i].first *= 100 , query_range_y[i].second *= 100 ;
            // query_range_x[i] = query_range_x[0] , query_range_y[i] = query_range_y[0] ;
        }

            // query_range_x[i] = query_range_x[0] , query_range_y[i] = query_range_y[0] ;
        for(unsigned i = 0 ; i < query_num ; i ++) {
            // for(unsigned j = 0 ;j < attr_dim ; j ++)
            l_bound_host[i * attr_dim] = (attrType)query_range_x[i].first , r_bound_host[i * attr_dim] = (attrType)query_range_x[i].second ;
            l_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].first , r_bound_host[i * attr_dim + 1] = (attrType) query_range_y[i].second ;
        }
        **/
        // cout << "range 0 (" << query_range_x[0].first << "," << query_range_x[0].second << ")" << endl ;
        // cout << "range 1 (" << query_range_x[1].first << "," << query_range_x[1].second << ")" << endl ;
        // cout << "range 2 (" << query_range_x[2].first << "," << query_range_x[2].second << ")" << endl ;
        // cout << "range 5 (" << query_range_x[5].first << "," << query_range_x[5].second << ")" << endl ;
        for(int i = 0 ;i < 5 ; i ++) 
            cout <<"range " << i << " (" << l_bound_host[i] << "," << r_bound_host[i] << ")" << endl ;
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
        // generate_range_multi_attr_with_selectivity<attrType>(l_bound_host.data() , r_bound_host.data() , attr_dim , grid_min.data() , grid_max.data() , query_num , selectivity) ;
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
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg_batch_process<attr_dim , attrType>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , prefilter_threshold , partition_is_selected) ;
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
    // vector<unsigned> batch_pid_list(36) ;
    // for(unsigned i = 0 ; i < 36 ; i ++)
    //     batch_pid_list[i] = i ;
    // random_shuffle(batch_pid_list.begin() , batch_pid_list.end()) ;
    // vector<vector<unsigned>> batch_info(9 , vector<unsigned>(4)) ;
    // for(unsigned i = 0 ; i < pnum ; i ++)
    //     batch_info[i / 4][i % 4] = i ; 
    // batch_info.resize(1) ;
    // vector<vector<unsigned>> batch_info = {
    //     {0 , 1 , 2 , 3}
    // } ;

    vector<vector<unsigned>> batch_info = {
        {20, 21, 19, 18},
        {12, 13, 14, 15},
        {16, 17, 23, 22},
        {28, 29, 26, 27}, 
        {8, 9, 10, 11}, 
        {24, 25, 30, 31}, 
        {1, 0, 6, 7}, 
        {32, 33, 34, 35}, 
        {2, 4, 5, 3}
    } ;

    // vector<vector<unsigned>> batch_info(16 , vector<unsigned>(1)) ;
    // for(unsigned i =0 ; i < 16 ; i ++)
    //     batch_info[i][0] = i ; 
    unsigned batch_size = 0 ;
    for(unsigned i = 0 ; i < batch_info.size() ; i ++)
        batch_size = max(batch_size , (unsigned) batch_info[i].size()) ;
    unsigned reserve_points_num = *max_element(graph_infos[1] , graph_infos[1] + pnum) ;
    // random_shuffle(batch_info.begin() , batch_info.end()) ;
    /**
    __half *data_half_buffer1 , *data_half_buffer2 ;
    // size_t data_half_buffer_size = static_cast<const size_t>(batch_size) * static_cast<size_t>(reserve_points_num) * static_cast<size_t>(dim) * sizeof(__half) ;
    size_t data_half_buffer_size = static_cast<const size_t>(num) * static_cast<const size_t>(dim) * sizeof(__half) ;
    cudaMalloc((void**) &data_half_buffer1 , data_half_buffer_size) ;
    cudaMemcpy(data_half_buffer1 , data_ , data_half_buffer_size , cudaMemcpyHostToDevice) ;
    // cudaMalloc((void**) &data_half_buffer2 , data_half_buffer_size) ;
    array<__half*,2> vec_buffer = {data_half_buffer1 , data_half_buffer1} ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    **/
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
    vector<vector<unsigned>> gpu_bf_result ;
    // load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/big_dataset/deep100M/ground_truth/1D.gt" , gpu_bf_result) ;
    load_ground_truth(gtFile , gpu_bf_result) ;
    // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
    //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
    //     global_graph , 16 , pnum , L) ;
    ofstream resultFileOs(resultFile , ios::out) ;
    resultFileOs << "ef,recall,qps,total_cost,pre_cost,search_cost" << endl ;
    for(int efid = 0 ; efid < ef_list.size() ; efid ++) {

        unsigned ef = ef_list[efid] ;
        vector<vector<unsigned>> result = smart::big_dataset_adapter::batch_graph_search_gpu_with_prefilter_batch_process_compressed_graph_T_dataset_resident_u8<attrType>(batch_info.size() , batch_info , degree , data_ori , data_u8_dev , graph_buffer , global_graph_buffer ,
        generic_buffer , query_ori , query_data_load_dev , dim , qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        attr_dim  , graph_infos , new_idx_mapper , seg_partition_point_start_index  , partition_infos , ef , 10 , 
        global_graph , global_graph_edge_per_p , pnum , partition_is_selected , 200 , num , 16) ;

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
        // cout << "gpu brute force 时间: " << gpu_bf_cost << endl ;
        gpu_bf_result.resize(qnum) ;
        // cout << "cpu brute force 时间: " << cpu_bf_cost << endl ;
        float total_recall = cal_recall_show_partition_multi_attrs<attrType>(result , gpu_bf_result , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , false) ;
        cout << " recall : " << total_recall << endl ;
        
        batch_process_cost.clear() ;
        batch_data_trans_cost.clear() ;
        float pre_cost = COST::cost1[0] ;
        float total_search_cost = pre_cost + other_kernal_cost ;
        float qps = qnum * 1.0 / total_search_cost ; 
        resultFileOs << ef << "," << total_recall << "," << qps  << "," << total_search_cost << "," << pre_cost << "," << other_kernal_cost << endl ;
        cout << "ef : " << ef << ", recall : " << total_recall << ", qps : " << qps  << ", latency : " << total_search_cost << ", precost : " << pre_cost << ", other_kernal_cost : " << other_kernal_cost << endl ;
        
    }    

    

    resultFileOs.close() ;
    e = chrono::high_resolution_clock::now() ;
    cost3 = chrono::duration<double>(e - s).count() ;
    cout << "finish search" << endl ;

    // cudaFree(data_half_buffer1) ;
    // (data_half_buffer2) ;
    cudaFree(graph_buffer0) ;
    cudaFree(graph_buffer1) ;
    cudaFree(global_graph_buffer0) ;
    cudaFree(global_graph_buffer1) ;
    cudaFree(generic_buffer0) ;
    cudaFree(generic_buffer1) ;
    cudaFree(stored_pos_global_idx) ;
    cudaFree(query_data_load_dev) ;
    cudaFree(data_u8_dev) ;



    // for(unsigned i = 0 ; i < 10  ;i ++)
    //     cout << gpu_bf_result_dis[0][i] << " ," ;

    // save_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/100Kquery.gt", query_num , 10, gpu_bf_result) ;
    
    // load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/deep1M/10kquery.gt" , gpu_bf_result) ;
    delete [] data_ ;
    cudaFreeHost(global_graph) ;
    cudaFreeHost(graph_infos[0]) ;
    cudaFreeHost(graph_infos[1]) ;
    cudaFreeHost(graph_infos[2]) ;



// 
    // rc_recall1 = total_recall ;
    // rc_cost1 = select_path_kernal_cost ;

    cudaFree(global_graph) ;
    cudaFree(attrs_dev) ;
    // cudaFree(query_data_half_dev) ;
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



int main(int argc , char** argv) {
    // unsigned L = stoi(argv[1]) ;
    constexpr unsigned attr_dim = 1 ; 
    vector<unsigned> shape = {36} ;

    string result_file_name ;
    string attrFile ;
    string rangeFile ;
    string gtFile ;

    string dataset_filepath , dataset_filepath_q , query_data_path , cagra_prefix , cagra_suffix , global_graph_path ;
    unsigned degree , global_graph_edge_per_p ;
    float f2h_factor ;
    float threshold ;

    

    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
    
        if (arg == "--result_file")
            result_file_name = std::string(argv[i + 1]);
    
        if (arg == "--attr_file")
            attrFile = std::string(argv[i + 1]);
    
        if (arg == "--range_file")
            rangeFile = std::string(argv[i + 1]);
    
        if (arg == "--gt_file")
            gtFile = std::string(argv[i + 1]);
    
        if (arg == "--dataset")
            dataset_filepath = std::string(argv[i + 1]);
    
        if (arg == "--dataset_q")
            dataset_filepath_q = std::string(argv[i + 1]);
    
        if (arg == "--query")
            query_data_path = std::string(argv[i + 1]);
    
        if (arg == "--cagra_prefix")
            cagra_prefix = std::string(argv[i + 1]);
    
        if (arg == "--cagra_suffix")
            cagra_suffix = std::string(argv[i + 1]);
    
        if (arg == "--global_graph")
            global_graph_path = std::string(argv[i + 1]);
    
        if (arg == "--degree")
            degree = std::stoi(argv[i + 1]);
    
        if (arg == "--global_edge")
            global_graph_edge_per_p = std::stoi(argv[i + 1]);
    
        if (arg == "--f2h_factor")
            f2h_factor = std::stof(argv[i + 1]);
    
        if (arg == "--threshold")
            threshold = std::stof(argv[i + 1]);
    }
    cout << "threshold : " << threshold << endl ;

    vector<unsigned> ef_list = {1500, 1400, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10 ,10} ;
    // vector<unsigned> ef_list = {64} ;
    reverse(ef_list.begin() , ef_list.end()) ;
    string D = to_string(attr_dim) ;


    // multi_attr_test_batch_process_large_dataset_compressed_graph_T36_dataset_resident_new<attr_dim>(ef_list ,attrFile ,rangeFile ,gtFile ,result_file_name ,threshold ,shape) ;
    multi_attr_test_batch_process_large_dataset_compressed_graph_T36_dataset_resident_new<attr_dim>(dataset_filepath ,dataset_filepath_q , query_data_path , cagra_prefix , cagra_suffix , global_graph_path , degree , global_graph_edge_per_p ,ef_list , attrFile , rangeFile , gtFile , result_file_name , ,threshold ,shape , f2h_factor) ;

}