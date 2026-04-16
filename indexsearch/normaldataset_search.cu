#include "check.cu"
#include "full_graph_search.cuh"
#include "real_dataset_adapter.cuh"
#include <numeric>
using namespace std ;

unsigned test_r_bound ;
void sift_experiment_run(vector<unsigned>& ef_list) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    array<unsigned*,3> graph_infos = get_graph_info("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/sift" , ".cagra" , 16 , 16, 1000000) ;
    array<unsigned*,3> idx_mapper = get_idx_mapper(1000000 , 16 , 62500) ;
    vector<vector<unsigned>> global_edges ;
    load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges);
    unsigned *global_graph ;
    cudaMalloc((void**) &global_graph , 1000000 * 16 * 16 * sizeof(unsigned)) ;

    for(unsigned i = 0 ; i < 1000000 ; i ++) {
        cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    }
    float *attrs_dev , *attrs ;
    int * attrs_int ;  
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    // using attrType = int ;
    constexpr unsigned attr_dim =  1 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<float> grid_min(attr_dim , 0.0f) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<float> grid_max(attr_dim , 1000000.0f) ;
    vector<unsigned> grid_size_host = {16} ;
    vector<float> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++)
        attr_width_host[i] = (grid_max[i] - grid_min[i]) / grid_size_host[i] ;
    attrs = new float[attr_dim * 1000000] ;
    attrs_int = new int[attr_dim * 1000000] ;
    unsigned attr_num , attr_dim_tmp ;
    load_attrs<int>("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M1D.attrs", attrs_int , attr_dim_tmp, attr_num) ;
    // re_generate(attrs , attr_dim , 1000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    for(unsigned i = 0 ; i < attr_num * attr_dim ; i ++)
        attrs[i] = (float) attrs_int[i] ;
    delete [] attrs_int ; 

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
    load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_range_query.fvecs" , query_data_load , query_num , query_dim) ;
    // query_num = 500; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;
    // query_num *= 10 ;
    /**
    float* tmp_query_data_load = new float[10 *query_num * query_dim] ;
    for(int i = 0 ; i < 10 ; i ++)
        memcpy(tmp_query_data_load + i * query_num * query_dim  , query_data_load , query_num * query_dim * sizeof(float)) ;
    query_num *= 10 ;
    delete [] query_data_load ;
    query_data_load = tmp_query_data_load ;
    **/
    // for(unsigned i = 0 ; i )
    // TOPM = 20 , DIM = dim ; 
    // ************** 若要修改查询数量, 请修改如下变量 **********************
    // query_num = 3 ;
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
    // unsigned test_r_bound = 124999 ;
    if(true) {
        // load_range("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M1D.range", query_range_x, 1000) ;
        load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        // query_range_x.reserve(query_num) ;
        // for(unsigned i = 1 ; i < 10 ; i ++)
        //     query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        // query_range_y = query_range_x ;
        // query_range_x[0] = {0 , 249999} ;
        // query_range_y[0] = {0 , 249999} ;
        // query_range_x[1] = {0 , 499999} , query_range_y[1] = {0 , 249999} ;
        // query_range_x[2] = {0 , 749999} , query_range_y[2] = {0 , 249999} ;
        // query_range_x[3] = {0 , 499999} , query_range_y[3] = {0 , 499999} ;
        // for(unsigned i = 4 ; i < query_num ; i ++)
        //     query_range_x[i] = query_range_x[i % 4] , query_range_y[i] = query_range_y[i % 4] ;
        // query_range_y = query_range_x ; 
        for(unsigned i = 0 ; i < query_num ; i ++)
            query_range_x[i] = {0 , test_r_bound} ;

        for(unsigned i = 0 ; i < query_num ; i ++) {
            for(unsigned j = 0 ;j < attr_dim ; j ++)
                l_bound_host[i * attr_dim] = (float)query_range_x[i].first , r_bound_host[i * attr_dim] = (float)query_range_x[i].second ;
                // l_bound_host[i * attr_dim + 1] = query_range_y[i].first , r_bound_host[i * attr_dim + 1] = query_range_y[i].second ;
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

    // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
    vector<vector<unsigned>> ground_truth(qnum) ;
        
    #pragma omp parallel for num_threads(10)
    for(unsigned i = 0 ; i < qnum ;i  ++)
        ground_truth[i] = naive_brute_force_with_filter(query_data_load + i * dim , data_ , dim , num , 10 ,
            l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim) ;

    cout << "开始批量查找" << endl ;        
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;

    //  以下载入聚类信息
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
    auto inner_product_s = chrono::high_resolution_clock::now() ;
    // find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    find_norm<<<(cnum_with_pad + 3) / 4 , dim3(32 , 4 , 1)>>>(cluster_centers_half_dev , c_norm , dim , cnum_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__ ) ;
    auto inner_product_e = chrono::high_resolution_clock::now() ;

    float inner_product_cost = chrono::duration<float>(inner_product_e - inner_product_s).count() ;

    unsigned num_c_p_offset = 0 ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cudaMemcpy(cluster_p_info + i * cnum , num_c_p[i].data() , cnum * sizeof(unsigned) , cudaMemcpyHostToDevice ) ;
    }

    // 找到存在交集的分区, 并按照聚类将其排序
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg<attr_dim>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , 0.0) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    

    cout << partition_infos[0] << "," <<  partition_infos[1] << "," <<  partition_infos[2] << endl ;
    // cnum = 16 , qnum = 16 ;
    s = chrono::high_resolution_clock::now() ;
    find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    get_partitions_access_order_V2(partition_infos , pnum , cnum , qnum , cluster_p_info , 4 , cluster_centers_half_dev , query_data_half_dev , dim , query_data_load , cluster_centers , q_norm , c_norm , num)  ;
    e = chrono::high_resolution_clock::now() ;
    cost2 = chrono::duration<double>(e - s).count() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;

    ofstream result_file("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/results/smart/dual_result_" + to_string(test_r_bound) + ".csv" , ios::out) ;
    result_file << "ef,recall,qps,total_cost,cost1,cost2,cost3,kernal_cost" << endl ;

    
    // 载入 ground truth 
    vector<vector<unsigned>> gt ;
    load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/ground_truth/sift1m_q1k_k10_unify_dataset.gt" , gt) ;


    // 由于改变的只有L的大小, 故只需将cost3的部分重复执行
    for(unsigned efid = 0 ; efid < ef_list.size() ; efid ++) {

        // check_process2(attr_dim , attr_min ,attr_width ,attr_grid_size ,l_bound ,r_bound ,partition_infos , query_num , query_data_load , dim , data_) ;
        unsigned L = ef_list[efid] ;
        cout << "L : " << L << endl ; 
        s = chrono::high_resolution_clock::now() ;
        vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
            attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
            global_graph , 16 , pnum , 10) ;
        e = chrono::high_resolution_clock::now() ;
        cost3 = chrono::duration<double>(e - s).count() ;
        cout << "finish search" << endl ;

        
        // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
        float total_recall = cal_recall_show_partition_multi_attrs(result , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , false) ;
        cout << "recall : " << total_recall << endl  ;

        // float total_cost = cost1 + cost2 + cost3 ; 
        cost1 = COST::cost1[efid] , cost2 = COST::cost2[efid] , cost3 = COST::cost3[efid] ;
        float total_cost = cost1 + cost2 + cost3 ; 
        float qps = qnum * 1.0 / total_cost ; 
        result_file << L << "," << total_recall << "," << qps << "," << total_cost << "," << cost1 << "," << cost2 << "," << cost3 << "," << select_path_kernal_cost << endl ;

        cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
        cout << "获取分区列表用时:" << cost1 << endl ;
        cout << "对分区列表排序用时:" << cost2 << endl ;
        cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
        cout << "搜索用时:" << cost3 << endl ;

        // 计算 ground truth recall
        // float gt_recall = cal_recall_show_partition_multi_attrs(gt , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;
        // cout << "ground truth recall : " << gt_recall << endl ;
    }
    // rc_recall1 = total_recall ;
    // rc_cost1 = select_path_kernal_cost ;
    cout << "1000点的内积计算时间: " << inner_product_cost << endl ;
    result_file.close() ;
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

void sift_postfiltering_test(vector<unsigned>& ef_list) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    array<unsigned*,3> graph_infos = get_graph_info("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/sift" , ".cagra" , 16 , 16, 1000000) ;
    array<unsigned*,3> idx_mapper = get_idx_mapper(1000000 , 16 , 62500) ;
    vector<vector<unsigned>> global_edges ;
    load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges);
    unsigned *global_graph ;
    cudaMalloc((void**) &global_graph , 1000000 * 16 * 16 * sizeof(unsigned)) ;

    for(unsigned i = 0 ; i < 1000000 ; i ++) {
        cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    }
    float *attrs_dev , *attrs ;
    int * attrs_int ;  
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    // using attrType = int ;
    constexpr unsigned attr_dim =  1 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<float> grid_min(attr_dim , 0.0f) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<float> grid_max(attr_dim , 1000000.0f) ;
    vector<unsigned> grid_size_host = {16} ;
    vector<float> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++)
        attr_width_host[i] = (grid_max[i] - grid_min[i]) / grid_size_host[i] ;
    attrs = new float[attr_dim * 1000000] ;
    attrs_int = new int[attr_dim * 1000000] ;
    unsigned attr_num , attr_dim_tmp ;
    load_attrs<int>("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M1D.attrs", attrs_int , attr_dim_tmp, attr_num) ;
    // re_generate(attrs , attr_dim , 1000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    for(unsigned i = 0 ; i < attr_num * attr_dim ; i ++)
        attrs[i] = (float) attrs_int[i] ;
    delete [] attrs_int ; 

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
    load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_range_query.fvecs" , query_data_load , query_num , query_dim) ;
    // query_num = 500; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;
    // query_num *= 10 ;
    /**
    float* tmp_query_data_load = new float[10 *query_num * query_dim] ;
    for(int i = 0 ; i < 10 ; i ++)
        memcpy(tmp_query_data_load + i * query_num * query_dim  , query_data_load , query_num * query_dim * sizeof(float)) ;
    query_num *= 10 ;
    delete [] query_data_load ;
    query_data_load = tmp_query_data_load ;
    **/
    // for(unsigned i = 0 ; i )
    // TOPM = 20 , DIM = dim ; 
    // ************** 若要修改查询数量, 请修改如下变量 **********************
    // query_num = 500 ;
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
    // unsigned test_r_bound = 124999 ;
    if(true) {
        load_range("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M1D.range", query_range_x, 1000) ;
        // load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        // query_range_x.reserve(query_num) ;
        // for(unsigned i = 1 ; i < 10 ; i ++)
        //     query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        // query_range_y = query_range_x ;
        // query_range_x[0] = {0 , 249999} ;
        // query_range_y[0] = {0 , 249999} ;
        // query_range_x[1] = {0 , 499999} , query_range_y[1] = {0 , 249999} ;
        // query_range_x[2] = {0 , 749999} , query_range_y[2] = {0 , 249999} ;
        // query_range_x[3] = {0 , 499999} , query_range_y[3] = {0 , 499999} ;
        // for(unsigned i = 4 ; i < query_num ; i ++)
        //     query_range_x[i] = query_range_x[i % 4] , query_range_y[i] = query_range_y[i % 4] ;
        // query_range_y = query_range_x ; 
        // for(unsigned i = 0 ; i < query_num ; i ++)
        //     query_range_x[i] = {0 , test_r_bound} ;

        for(unsigned i = 0 ; i < query_num ; i ++) {
            for(unsigned j = 0 ;j < attr_dim ; j ++)
                l_bound_host[i * attr_dim] = (float)query_range_x[i].first , r_bound_host[i * attr_dim] = (float)query_range_x[i].second ;
                // l_bound_host[i * attr_dim + 1] = query_range_y[i].first , r_bound_host[i * attr_dim + 1] = query_range_y[i].second ;
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

    // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
    vector<vector<unsigned>> ground_truth(qnum) ;
        
    #pragma omp parallel for num_threads(10)
    for(unsigned i = 0 ; i < qnum ;i  ++)
        ground_truth[i] = naive_brute_force_with_filter(query_data_load + i * dim , data_ , dim , num , 10 ,
            l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim) ;

    cout << "开始批量查找" << endl ;        
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;

    //  以下载入聚类信息
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
    auto inner_product_s = chrono::high_resolution_clock::now() ;
    // find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    find_norm<<<(cnum_with_pad + 3) / 4 , dim3(32 , 4 , 1)>>>(cluster_centers_half_dev , c_norm , dim , cnum_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__ ) ;
    auto inner_product_e = chrono::high_resolution_clock::now() ;

    float inner_product_cost = chrono::duration<float>(inner_product_e - inner_product_s).count() ;

    unsigned num_c_p_offset = 0 ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cudaMemcpy(cluster_p_info + i * cnum , num_c_p[i].data() , cnum * sizeof(unsigned) , cudaMemcpyHostToDevice ) ;
    }

    // 找到存在交集的分区, 并按照聚类将其排序
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg<attr_dim>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , 0.0) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    

    cout << partition_infos[0] << "," <<  partition_infos[1] << "," <<  partition_infos[2] << endl ;
    // cnum = 16 , qnum = 16 ;
    s = chrono::high_resolution_clock::now() ;
    find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    get_partitions_access_order_V2(partition_infos , pnum , cnum , qnum , cluster_p_info , 4 , cluster_centers_half_dev , query_data_half_dev , dim , query_data_load , cluster_centers , q_norm , c_norm , num)  ;
    e = chrono::high_resolution_clock::now() ;
    cost2 = chrono::duration<double>(e - s).count() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;

    ofstream result_file("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/results/smart/result_with_postfiler.csv" , ios::out) ;
    result_file << "ef,recall,qps,total_cost,cost1,cost2,cost3,kernal_cost" << endl ;

    
    // 载入 ground truth 
    vector<vector<unsigned>> gt ;
    load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/ground_truth/sift1m_q1k_k10.gt" , gt) ;
    gt.resize(qnum) ;
    // 每个点的分区编号
    unsigned* pid_list ;
    cudaMalloc((void**) &pid_list , num * sizeof(unsigned)) ;
    // thrust::sequence(pid_list , pid_list + num) ;
    auto cit = thrust::make_counting_iterator<unsigned>(0) ;
    thrust::transform(
        thrust::device ,
        cit , cit + num ,
        pid_list ,
        [] __host__ __device__ (unsigned i) {
            return i / 62500 ;    
        }
    ) ;
    unsigned* bigGraph ;
    unsigned bigDegree = 32 ;
    unsigned degree = 16 ;
    cudaMalloc((void**) &bigGraph , num * bigDegree * sizeof(unsigned)) ;
    smart::full_graph_search::assemble_edges<<<num , bigDegree>>>(graph_infos[0], degree , graph_infos[2] , graph_infos[1] ,
        global_graph , 16, 2 , pnum , idx_mapper[0] ,
        idx_mapper[1] , idx_mapper[2] , pid_list , bigGraph) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cout << "全局图组装完成 : bigDegree -  " << bigDegree << endl ;

    // 由于改变的只有L的大小, 故只需将cost3的部分重复执行
    for(unsigned efid = 0 ; efid < ef_list.size() ; efid ++) {

        // check_process2(attr_dim , attr_min ,attr_width ,attr_grid_size ,l_bound ,r_bound ,partition_infos , query_num , query_data_load , dim , data_) ;
        unsigned L = ef_list[efid] ;
        cout << "L : " << L << endl ; 
        s = chrono::high_resolution_clock::now() ;
        // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
        //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
        //     global_graph , 16 , pnum , 10) ;
        // vector<vector<unsigned>> result = smart::full_graph_search::smart_postfiltering_V2(graph_infos , 16 , data_half_dev , query_data_half_dev ,dim ,qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        //     attr_dim  , idx_mapper , partition_infos , pid_list , L , 10 , 
        //     global_graph , 16 , pnum) ;
        unsigned postfilter_threshold = 10 ; 
        vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter_and_postfilter(graph_infos , 16 ,data_half_dev , query_data_half_dev ,dim ,qnum ,l_bound_dev ,r_bound_dev , attrs_dev , 
            attr_dim  , idx_mapper ,partition_infos , L , 10 , 
            global_graph , 16 , pnum , 10 , postfilter_threshold , bigGraph , bigDegree) ;        
        e = chrono::high_resolution_clock::now() ;
        cost3 = chrono::duration<double>(e - s).count() ;
        cout << "finish search" << endl ;

        
        // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
        float total_recall = cal_recall_show_partition_multi_attrs(result , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , false) ;
        cout << "recall : " << total_recall << endl  ;

        cost1 = COST::cost1[0] , cost2 = COST::cost2[0] , cost3 = COST::cost3[efid] ;
        float total_cost = cost1 + cost2 + cost3 ; 
        float qps = qnum * 1.0 / total_cost ; 
        result_file << L << "," << total_recall << "," << qps << "," << total_cost << "," << cost1 << "," << cost2 << "," << cost3 << "," << select_path_kernal_cost << endl ;

        cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
        cout << "获取分区列表用时:" << cost1 << endl ;
        cout << "对分区列表排序用时:" << cost2 << endl ;
        cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
        cout << "搜索用时:" << cost3 << endl ;

        // 计算 ground truth recall
        // float gt_recall = cal_recall_show_partition_multi_attrs(gt , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;
        // cout << "ground truth recall : " << gt_recall << endl ;
    }
    // rc_recall1 = total_recall ;
    // rc_cost1 = select_path_kernal_cost ;
    cout << "1000点的内积计算时间: " << inner_product_cost << endl ;
    result_file.close() ;
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
    cudaFree(pid_list) ;
    cudaFree(bigGraph) ;
}


void sift_test_2D(vector<unsigned>& ef_list) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    array<unsigned*,3> graph_infos = get_graph_info("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/sift" , ".cagra" , 16 , 16, 1000000) ;
    array<unsigned*,3> idx_mapper = get_idx_mapper(1000000 , 16 , 62500) ;
    vector<vector<unsigned>> global_edges ;
    load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges);
    unsigned *global_graph ;
    cudaMalloc((void**) &global_graph , 1000000 * 16 * 16 * sizeof(unsigned)) ;

    for(unsigned i = 0 ; i < 1000000 ; i ++) {
        cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    }
    float *attrs_dev , *attrs ;
    int * attrs_int ;  
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    // using attrType = int ;
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
    attrs_int = new int[attr_dim * 1000000] ;
    unsigned attr_num , attr_dim_tmp ;
    load_attrs<int>("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M2D_unif.attrs", attrs_int , attr_dim_tmp, attr_num) ;
    // re_generate(attrs , attr_dim , 1000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    for(unsigned i = 0 ; i < attr_num * attr_dim ; i ++)
        attrs[i] = (float) attrs_int[i] ;
    delete [] attrs_int ; 

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
    load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_range_query.fvecs" , query_data_load , query_num , query_dim) ;
    // query_num = 500; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;
    // query_num *= 10 ;
    /**
    float* tmp_query_data_load = new float[10 *query_num * query_dim] ;
    for(int i = 0 ; i < 10 ; i ++)
        memcpy(tmp_query_data_load + i * query_num * query_dim  , query_data_load , query_num * query_dim * sizeof(float)) ;
    query_num *= 10 ;
    delete [] query_data_load ;
    query_data_load = tmp_query_data_load ;
    **/
    // for(unsigned i = 0 ; i )
    // TOPM = 20 , DIM = dim ; 
    // ************** 若要修改查询数量, 请修改如下变量 **********************
    // query_num = 500 ;
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
    // unsigned test_r_bound = 124999 ;
    if(true) {
        // load_range("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M1D.range", query_range_x, 1000) ;
        vector<int> l_bound_int(query_num * attr_dim) , r_bound_int(query_num * attr_dim) ;
        load_range<int>("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M2D.range" , l_bound_int.data() ,r_bound_int.data() , attr_dim , query_num) ;
        // load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        // query_range_x.reserve(query_num) ;
        // for(unsigned i = 1 ; i < 10 ; i ++)
        //     query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        // query_range_y = query_range_x ;
        // query_range_x[0] = {0 , 249999} ;
        // query_range_y[0] = {0 , 249999} ;
        // query_range_x[1] = {0 , 499999} , query_range_y[1] = {0 , 249999} ;
        // query_range_x[2] = {0 , 749999} , query_range_y[2] = {0 , 249999} ;
        // query_range_x[3] = {0 , 499999} , query_range_y[3] = {0 , 499999} ;
        // for(unsigned i = 4 ; i < query_num ; i ++)
        //     query_range_x[i] = query_range_x[i % 4] , query_range_y[i] = query_range_y[i % 4] ;
        // query_range_y = query_range_x ; 
        // for(unsigned i = 0 ; i < query_num ; i ++)
        //     query_range_x[i] = {0 , test_r_bound} ;

        for(unsigned i = 0 ; i < query_num ; i ++) {
            for(unsigned j = 0 ;j < attr_dim ; j ++)
                l_bound_host[i * attr_dim + j] = (float)l_bound_int[i * attr_dim + j] , r_bound_host[i * attr_dim + j] = (float)r_bound_int[i * attr_dim + j] ;
                // l_bound_host[i * attr_dim + 1] = query_range_y[i].first , r_bound_host[i * attr_dim + 1] = query_range_y[i].second ;
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

    // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
    vector<vector<unsigned>> ground_truth(qnum) ;
        
    // #pragma omp parallel for num_threads(10)
    // for(unsigned i = 0 ; i < qnum ;i  ++)
    //     ground_truth[i] = naive_brute_force_with_filter(query_data_load + i * dim , data_ , dim , num , 10 ,
    //         l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim) ;

    cout << "开始批量查找" << endl ;        
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;

    //  以下载入聚类信息
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
    auto inner_product_s = chrono::high_resolution_clock::now() ;
    // find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    find_norm<<<(cnum_with_pad + 3) / 4 , dim3(32 , 4 , 1)>>>(cluster_centers_half_dev , c_norm , dim , cnum_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__ ) ;
    auto inner_product_e = chrono::high_resolution_clock::now() ;

    float inner_product_cost = chrono::duration<float>(inner_product_e - inner_product_s).count() ;

    unsigned num_c_p_offset = 0 ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cudaMemcpy(cluster_p_info + i * cnum , num_c_p[i].data() , cnum * sizeof(unsigned) , cudaMemcpyHostToDevice ) ;
    }

    // 找到存在交集的分区, 并按照聚类将其排序
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg<attr_dim>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , 0.0) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    

    cout << partition_infos[0] << "," <<  partition_infos[1] << "," <<  partition_infos[2] << endl ;
    // cnum = 16 , qnum = 16 ;
    s = chrono::high_resolution_clock::now() ;
    find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    get_partitions_access_order_V2(partition_infos , pnum , cnum , qnum , cluster_p_info , 4 , cluster_centers_half_dev , query_data_half_dev , dim , query_data_load , cluster_centers , q_norm , c_norm , num)  ;
    e = chrono::high_resolution_clock::now() ;
    cost2 = chrono::duration<double>(e - s).count() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;

    ofstream result_file("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/results/smart/2D_unif.csv" , ios::out) ;
    result_file << "ef,recall,qps,total_cost,cost1,cost2,cost3,kernal_cost" << endl ;

    
    // 载入 ground truth 
    vector<vector<unsigned>> gt ;
    load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/ground_truth/sift1m_q1k_k10_2D_unif.gt" , gt) ;
    gt.resize(qnum) ;
    // 每个点的分区编号
    unsigned* pid_list ;
    cudaMalloc((void**) &pid_list , num * sizeof(unsigned)) ;
    // thrust::sequence(pid_list , pid_list + num) ;
    auto cit = thrust::make_counting_iterator<unsigned>(0) ;
    thrust::transform(
        thrust::device ,
        cit , cit + num ,
        pid_list ,
        [] __host__ __device__ (unsigned i) {
            return i / 62500 ;    
        }
    ) ;
    unsigned* bigGraph ;
    unsigned bigDegree = 32 ;
    unsigned degree = 16 ;
    cudaMalloc((void**) &bigGraph , num * bigDegree * sizeof(unsigned)) ;
    smart::full_graph_search::assemble_edges<<<num , bigDegree>>>(graph_infos[0], degree , graph_infos[2] , graph_infos[1] ,
        global_graph , 16, 2 , pnum , idx_mapper[0] ,
        idx_mapper[1] , idx_mapper[2] , pid_list , bigGraph) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cout << "全局图组装完成 : bigDegree -  " << bigDegree << endl ;

    // 由于改变的只有L的大小, 故只需将cost3的部分重复执行
    for(unsigned efid = 0 ; efid < ef_list.size() ; efid ++) {

        // check_process2(attr_dim , attr_min ,attr_width ,attr_grid_size ,l_bound ,r_bound ,partition_infos , query_num , query_data_load , dim , data_) ;
        unsigned L = ef_list[efid] ;
        cout << "L : " << L << endl ; 
        s = chrono::high_resolution_clock::now() ;
        // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
        //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
        //     global_graph , 16 , pnum , 10) ;
        // vector<vector<unsigned>> result = smart::full_graph_search::smart_postfiltering_V2(graph_infos , 16 , data_half_dev , query_data_half_dev ,dim ,qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        //     attr_dim  , idx_mapper , partition_infos , pid_list , L , 10 , 
        //     global_graph , 16 , pnum) ;
        unsigned postfilter_threshold = 10 ; 
        vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter_and_postfilter(graph_infos , 16 ,data_half_dev , query_data_half_dev ,dim ,qnum ,l_bound_dev ,r_bound_dev , attrs_dev , 
            attr_dim  , idx_mapper ,partition_infos , L , 10 , 
            global_graph , 16 , pnum , 10 , postfilter_threshold , bigGraph , bigDegree) ;        
        e = chrono::high_resolution_clock::now() ;
        cost3 = chrono::duration<double>(e - s).count() ;
        cout << "finish search" << endl ;

        
        // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
        float total_recall = cal_recall_show_partition_multi_attrs(result , gt , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , false) ;
        cout << "recall : " << total_recall << endl  ;

        cost1 = COST::cost1[0] , cost2 = COST::cost2[0] , cost3 = COST::cost3[efid] ;
        float total_cost = cost1 + cost2 + cost3 ; 
        float qps = qnum * 1.0 / total_cost ; 
        result_file << L << "," << total_recall << "," << qps << "," << total_cost << "," << cost1 << "," << cost2 << "," << cost3 << "," << select_path_kernal_cost << endl ;

        cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
        cout << "获取分区列表用时:" << cost1 << endl ;
        cout << "对分区列表排序用时:" << cost2 << endl ;
        cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
        cout << "搜索用时:" << cost3 << endl ;

        // 计算 ground truth recall
        // float gt_recall = cal_recall_show_partition_multi_attrs(gt , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;
        // cout << "ground truth recall : " << gt_recall << endl ;
    }
    // rc_recall1 = total_recall ;
    // rc_cost1 = select_path_kernal_cost ;
    cout << "1000点的内积计算时间: " << inner_product_cost << endl ;
    result_file.close() ;
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
    cudaFree(pid_list) ;
    cudaFree(bigGraph) ;
}


void sift_test_multiD(vector<unsigned>& ef_list) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    array<unsigned*,3> graph_infos = get_graph_info("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/sift" , ".cagra" , 16 , 16, 1000000) ;
    array<unsigned*,3> idx_mapper = get_idx_mapper(1000000 , 16 , 62500) ;
    vector<vector<unsigned>> global_edges ;
    load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges);
    unsigned *global_graph ;
    cudaMalloc((void**) &global_graph , 1000000 * 16 * 16 * sizeof(unsigned)) ;

    for(unsigned i = 0 ; i < 1000000 ; i ++) {
        cudaMemcpy(global_graph + i * 16 * 16 , global_edges[i].data() , 16 * 16 * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    }
    float *attrs_dev , *attrs ;
    int * attrs_int ;  
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    // using attrType = int ;
    constexpr unsigned attr_dim =  4 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<float> grid_min(attr_dim , 0.0f) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<float> grid_max(attr_dim , 1000000.0f) ;
    vector<unsigned> grid_size_host = {2 , 2, 2 , 2} ;
    vector<float> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++)
        attr_width_host[i] = (grid_max[i] - grid_min[i]) / grid_size_host[i] ;
    attrs = new float[attr_dim * 1000000] ;
    attrs_int = new int[attr_dim * 1000000] ;
    unsigned attr_num , attr_dim_tmp ;
    load_attrs<int>("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M4D.attrs", attrs_int , attr_dim_tmp, attr_num) ;
    // re_generate(attrs , attr_dim , 1000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    for(unsigned i = 0 ; i < attr_num * attr_dim ; i ++)
        attrs[i] = (float) attrs_int[i] ;
    delete [] attrs_int ; 

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
    load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_range_query.fvecs" , query_data_load , query_num , query_dim) ;
    // query_num = 500; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;
    // query_num *= 10 ;
    /**
    float* tmp_query_data_load = new float[10 *query_num * query_dim] ;
    for(int i = 0 ; i < 10 ; i ++)
        memcpy(tmp_query_data_load + i * query_num * query_dim  , query_data_load , query_num * query_dim * sizeof(float)) ;
    query_num *= 10 ;
    delete [] query_data_load ;
    query_data_load = tmp_query_data_load ;
    **/
    // for(unsigned i = 0 ; i )
    // TOPM = 20 , DIM = dim ; 
    // ************** 若要修改查询数量, 请修改如下变量 **********************
    // query_num = 500 ;
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
    // unsigned test_r_bound = 124999 ;
    if(true) {
        // load_range("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M1D.range", query_range_x, 1000) ;
        vector<int> l_bound_int(query_num * attr_dim) , r_bound_int(query_num * attr_dim) ;
        load_range<int>("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M4D.range" , l_bound_int.data() ,r_bound_int.data() , attr_dim , query_num) ;
        // load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        // query_range_x.reserve(query_num) ;
        // for(unsigned i = 1 ; i < 10 ; i ++)
        //     query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        // query_range_y = query_range_x ;
        // query_range_x[0] = {0 , 249999} ;
        // query_range_y[0] = {0 , 249999} ;
        // query_range_x[1] = {0 , 499999} , query_range_y[1] = {0 , 249999} ;
        // query_range_x[2] = {0 , 749999} , query_range_y[2] = {0 , 249999} ;
        // query_range_x[3] = {0 , 499999} , query_range_y[3] = {0 , 499999} ;
        // for(unsigned i = 4 ; i < query_num ; i ++)
        //     query_range_x[i] = query_range_x[i % 4] , query_range_y[i] = query_range_y[i % 4] ;
        // query_range_y = query_range_x ; 
        // for(unsigned i = 0 ; i < query_num ; i ++)
        //     query_range_x[i] = {0 , test_r_bound} ;

        for(unsigned i = 0 ; i < query_num ; i ++) {
            for(unsigned j = 0 ;j < attr_dim ; j ++)
                l_bound_host[i * attr_dim + j] = (float)l_bound_int[i * attr_dim + j] , r_bound_host[i * attr_dim + j] = (float)r_bound_int[i * attr_dim + j] ;
                // l_bound_host[i * attr_dim + 1] = query_range_y[i].first , r_bound_host[i * attr_dim + 1] = query_range_y[i].second ;
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

    // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
    vector<vector<unsigned>> ground_truth(qnum) ;
        
    // #pragma omp parallel for num_threads(10)
    // for(unsigned i = 0 ; i < qnum ;i  ++)
    //     ground_truth[i] = naive_brute_force_with_filter(query_data_load + i * dim , data_ , dim , num , 10 ,
    //         l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim) ;

    cout << "开始批量查找" << endl ;        
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;

    //  以下载入聚类信息
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
    auto inner_product_s = chrono::high_resolution_clock::now() ;
    // find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    find_norm<<<(cnum_with_pad + 3) / 4 , dim3(32 , 4 , 1)>>>(cluster_centers_half_dev , c_norm , dim , cnum_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__ ) ;
    auto inner_product_e = chrono::high_resolution_clock::now() ;

    float inner_product_cost = chrono::duration<float>(inner_product_e - inner_product_s).count() ;

    unsigned num_c_p_offset = 0 ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cudaMemcpy(cluster_p_info + i * cnum , num_c_p[i].data() , cnum * sizeof(unsigned) , cudaMemcpyHostToDevice ) ;
    }

    // 找到存在交集的分区, 并按照聚类将其排序
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg<attr_dim>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , 0.35) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    

    cout << partition_infos[0] << "," <<  partition_infos[1] << "," <<  partition_infos[2] << endl ;
    // cnum = 16 , qnum = 16 ;
    s = chrono::high_resolution_clock::now() ;
    find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    get_partitions_access_order_V2(partition_infos , pnum , cnum , qnum , cluster_p_info , 4 , cluster_centers_half_dev , query_data_half_dev , dim , query_data_load , cluster_centers , q_norm , c_norm , num)  ;
    e = chrono::high_resolution_clock::now() ;
    cost2 = chrono::duration<double>(e - s).count() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;

    ofstream result_file("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/results/smart/4D_035.csv" , ios::out) ;
    result_file << "ef,recall,qps,total_cost,cost1,cost2,cost3,kernal_cost" << endl ;

    
    // 载入 ground truth 
    vector<vector<unsigned>> gt ;
    load_ground_truth("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/ground_truth/sift1m_q1k_k10_4D.gt" , gt) ;
    gt.resize(qnum) ;
    // 每个点的分区编号
    unsigned* pid_list ;
    cudaMalloc((void**) &pid_list , num * sizeof(unsigned)) ;
    // thrust::sequence(pid_list , pid_list + num) ;
    auto cit = thrust::make_counting_iterator<unsigned>(0) ;
    thrust::transform(
        thrust::device ,
        cit , cit + num ,
        pid_list ,
        [] __host__ __device__ (unsigned i) {
            return i / 62500 ;    
        }
    ) ;
    unsigned* bigGraph ;
    unsigned bigDegree = 32 ;
    unsigned degree = 16 ;
    cudaMalloc((void**) &bigGraph , num * bigDegree * sizeof(unsigned)) ;
    smart::full_graph_search::assemble_edges<<<num , bigDegree>>>(graph_infos[0], degree , graph_infos[2] , graph_infos[1] ,
        global_graph , 16, 2 , pnum , idx_mapper[0] ,
        idx_mapper[1] , idx_mapper[2] , pid_list , bigGraph) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cout << "全局图组装完成 : bigDegree -  " << bigDegree << endl ;

    // 由于改变的只有L的大小, 故只需将cost3的部分重复执行
    for(unsigned efid = 0 ; efid < ef_list.size() ; efid ++) {

        // check_process2(attr_dim , attr_min ,attr_width ,attr_grid_size ,l_bound ,r_bound ,partition_infos , query_num , query_data_load , dim , data_) ;
        unsigned L = ef_list[efid] ;
        cout << "L : " << L << endl ; 
        s = chrono::high_resolution_clock::now() ;
        // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
        //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
        //     global_graph , 16 , pnum , 10) ;
        // vector<vector<unsigned>> result = smart::full_graph_search::smart_postfiltering_V2(graph_infos , 16 , data_half_dev , query_data_half_dev ,dim ,qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        //     attr_dim  , idx_mapper , partition_infos , pid_list , L , 10 , 
        //     global_graph , 16 , pnum) ;
        unsigned postfilter_threshold = 10 ; 
        vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter_and_postfilter(graph_infos , 16 ,data_half_dev , query_data_half_dev ,dim ,qnum ,l_bound_dev ,r_bound_dev , attrs_dev , 
            attr_dim  , idx_mapper ,partition_infos , L , 10 , 
            global_graph , 16 , pnum , 10 , postfilter_threshold , bigGraph , bigDegree) ;        
        e = chrono::high_resolution_clock::now() ;
        cost3 = chrono::duration<double>(e - s).count() ;
        cout << "finish search" << endl ;

        
        // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
        float total_recall = cal_recall_show_partition_multi_attrs(result , gt , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , false) ;
        cout << "recall : " << total_recall << endl  ;

        cost1 = COST::cost1[0] , cost2 = COST::cost2[0] , cost3 = COST::cost3[efid] ;
        float total_cost = cost1 + cost2 + cost3 ; 
        float qps = qnum * 1.0 / total_cost ; 
        result_file << L << "," << total_recall << "," << qps << "," << total_cost << "," << cost1 << "," << cost2 << "," << cost3 << "," << select_path_kernal_cost << endl ;

        cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
        cout << "获取分区列表用时:" << cost1 << endl ;
        cout << "对分区列表排序用时:" << cost2 << endl ;
        cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
        cout << "搜索用时:" << cost3 << endl ;

        // 计算 ground truth recall
        // float gt_recall = cal_recall_show_partition_multi_attrs(gt , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;
        // cout << "ground truth recall : " << gt_recall << endl ;
    }
    // rc_recall1 = total_recall ;
    // rc_cost1 = select_path_kernal_cost ;
    cout << "1000点的内积计算时间: " << inner_product_cost << endl ;
    result_file.close() ;
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
    cudaFree(pid_list) ;
    cudaFree(bigGraph) ;
}


template<unsigned attr_dim = 1>
void dataset_test_multiD(string cagra_graph_savepath_prefix , string cagra_graph_savepath_suffix , string global_graph_savepath , 
    string attrs_savepath , string range_savepath , string dataset_path , string query_path , string cluster_savepath  , string ground_truth_savepath , string result_savepath
    ,vector<unsigned>& ef_list , vector<unsigned> grid_size_outer , float prefilter_threshold , unsigned postfilter_threshold, bool testGT = false) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    array<unsigned*,3> graph_infos = get_graph_info(cagra_graph_savepath_prefix , cagra_graph_savepath_suffix , 16 , 16, 1000000) ;
    array<unsigned*,3> idx_mapper = get_idx_mapper(1000000 , 16 , 62500) ;
    vector<vector<unsigned>> global_edges ;
    // load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges);
    unsigned *global_graph , *global_graph_cpu ;
    unsigned global_graph_degree , global_graph_point_num ;
    load_result(global_graph_savepath, global_graph_cpu , global_graph_degree , global_graph_point_num) ;
    cout <<  "global graph , degree : " << global_graph_degree <<  ", num : " << global_graph_point_num << endl ;
 
    cudaMalloc((void**) &global_graph , global_graph_degree * global_graph_point_num * sizeof(unsigned)) ;
    cudaMemcpy(global_graph , global_graph_cpu , global_graph_degree * global_graph_point_num * sizeof(unsigned) , cudaMemcpyHostToDevice) ;

    // for(unsigned i = 0 ; i < 10 ; i ++) {
    //     for(unsigned j = 0 ; j < 32 ; j ++)
    //         cout << global_graph_cpu[i * 32 + j] << ", " ;
    //     cout << endl ;
    // }
    

    float *attrs_dev , *attrs ;
    int * attrs_int ;  
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    // using attrType = int ;
    // constexpr unsigned attr_dim =  4 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<float> grid_min(attr_dim , 0.0f) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<float> grid_max(attr_dim , 1000000.0f) ;
    // vector<unsigned> grid_size_host = {2 , 2, 2 , 2} ;
    vector<unsigned> grid_size_host = grid_size_outer ;
    vector<float> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++)
        attr_width_host[i] = (grid_max[i] - grid_min[i]) / grid_size_host[i] ;
    attrs = new float[attr_dim * 1000000] ;
    // attrs_int = new int[attr_dim * 1000000] ;
    unsigned attr_num , attr_dim_tmp ;
    // load_attrs<int>("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M4D.attrs", attrs_int , attr_dim_tmp, attr_num) ;
    load_attrs<int>(attrs_savepath, attrs_int , attr_dim_tmp, attr_num) ;
    // re_generate(attrs , attr_dim , 1000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    for(unsigned i = 0 ; i < attr_num * attr_dim ; i ++)
        attrs[i] = (float) attrs_int[i] ;
    delete [] attrs_int ; 

    cudaMalloc((void**) &attrs_dev , attr_dim * 1000000 * sizeof(float)) ;
    cudaMemcpy(attrs_dev , attrs , attr_dim * 1000000 * sizeof(float) , cudaMemcpyHostToDevice) ;

    cout << "sizeof(half) = " << sizeof(half) << endl ;

    // 先载入数据集和图索引
    unsigned num , dim  ;
    float* data_  ;
    // load_data("/home/yhr/workspace/cuda_clustering/data/sift_base.fvecs" , data_ , num , dim) ;
    load_data(dataset_path , data_ , num , dim) ;
    cout << "num : " << num << ", dim : " << dim << endl ;
    float* query_data_load ;
    unsigned query_num , query_dim ;
    // load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_range_query.fvecs" , query_data_load , query_num , query_dim) ;
    load_data(query_path , query_data_load , query_num , query_dim) ;
    // query_num = 500; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;
    // ************** 若要修改查询数量, 请修改如下变量 **********************
    // query_num = 1 ;

    float* data_dev ;
    half *data_half_dev ;
    float* query_data_dev ;
    half *query_data_half_dev ;
    cudaMalloc((void**)&data_dev , num * dim * sizeof(float)) ;
    cudaMalloc((void**) &data_half_dev , num * dim * sizeof(__half)) ;
    cudaMalloc((void**) &query_data_dev , query_num * query_dim   * sizeof(float)) ;
    unsigned query_num_with_pad = (query_num + 15) / 16 * 16 ;
    cout << "query_num_with_pad : " << query_num_with_pad << endl ;
    cudaMalloc((void**) &query_data_half_dev , query_num_with_pad * query_dim * sizeof(__half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(data_dev , data_ , num * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(query_data_dev , query_data_load , query_num * query_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num ; 
    // 调用实例
    f2h<<<points_num, 256>>>(data_dev, data_half_dev, points_num , dim ,  0.1f); // 先将数据转换成fp16
    // cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    f2h<<<query_num, 256>>>(query_data_dev, query_data_half_dev, query_num , dim , 0.1f); // 将查询转换成fp16
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    if(query_num != query_num_with_pad)
        thrust::fill(thrust::device , query_data_half_dev + query_num * query_dim , query_data_half_dev + query_num_with_pad * query_dim , __float2half(0.0f)) ;

    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    vector<float> l_bound_host(query_num * attr_dim) , r_bound_host(query_num * attr_dim) ;
    // unsigned test_r_bound = 124999 ;
    if(true) {
        // load_range("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M1D.range", query_range_x, 1000) ;
        vector<int> l_bound_int(query_num * attr_dim) , r_bound_int(query_num * attr_dim) ;
        load_range<int>(range_savepath , l_bound_int.data() ,r_bound_int.data() , attr_dim , query_num) ;
        // load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        // query_range_x.reserve(query_num) ;
        // for(unsigned i = 1 ; i < 10 ; i ++)
        //     query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        // query_range_y = query_range_x ;
        // query_range_x[0] = {0 , 249999} ;
        // query_range_y[0] = {0 , 249999} ;
        // query_range_x[1] = {0 , 499999} , query_range_y[1] = {0 , 249999} ;
        // query_range_x[2] = {0 , 749999} , query_range_y[2] = {0 , 249999} ;
        // query_range_x[3] = {0 , 499999} , query_range_y[3] = {0 , 499999} ;
        // for(unsigned i = 4 ; i < query_num ; i ++)
        //     query_range_x[i] = query_range_x[i % 4] , query_range_y[i] = query_range_y[i % 4] ;
        // query_range_y = query_range_x ; 
        // for(unsigned i = 0 ; i < query_num ; i ++)
        //     query_range_x[i] = {0 , test_r_bound} ;

        for(unsigned i = 0 ; i < query_num ; i ++) {
            for(unsigned j = 0 ;j < attr_dim ; j ++)
                l_bound_host[i * attr_dim + j] = (float)l_bound_int[i * attr_dim + j] , r_bound_host[i * attr_dim + j] = (float)r_bound_int[i * attr_dim + j] ;
                // l_bound_host[i * attr_dim + 1] = query_range_y[i].first , r_bound_host[i * attr_dim + 1] = query_range_y[i].second ;
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

    // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
    vector<vector<unsigned>> ground_truth(qnum) ;
    vector<vector<unsigned>> gt ;
    load_ground_truth(ground_truth_savepath , gt) ;
    if(testGT) {
        #pragma omp parallel for num_threads(10)
        for(unsigned i = 0 ; i < qnum ;i  ++)
            ground_truth[i] = naive_brute_force_with_filter(query_data_load + i * dim , data_ , dim , num , 10 ,
                l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim) ;
    } else {
        // 载入 ground truth 
        // vector<vector<unsigned>> gt ;
        
        swap(ground_truth , gt) ;
        // 每个点的分区编号
    }
    cout << "开始批量查找" << endl ;        
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;

    //  以下载入聚类信息
    vector<unsigned> cluster_id ;
    float* cluster_centers;
    vector<vector<unsigned>> num_c_p ;
    load_clustering(cluster_id , cluster_savepath , cluster_centers , dim , num_c_p , data_) ;
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
    f2h<<<cnum, 256>>>(cluster_centers_dev, cluster_centers_half_dev, cnum, dim , 0.1f);
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    if(cnum != cnum_with_pad)
        thrust::fill(thrust::device , cluster_centers_half_dev + cnum * dim , cluster_centers_half_dev + cnum_with_pad * dim , __float2half(0.0f)) ;
    // 在此处计算得到q_norm 和 c_norm 
    // ***************************************************
    float *q_norm , *c_norm ;
    cudaMalloc((void**) &q_norm , query_num_with_pad * sizeof(float)) ;
    cudaMalloc((void**) &c_norm , cnum_with_pad * sizeof(float)) ;
    auto inner_product_s = chrono::high_resolution_clock::now() ;
    // find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    find_norm<<<(cnum_with_pad + 3) / 4 , dim3(32 , 4 , 1)>>>(cluster_centers_half_dev , c_norm , dim , cnum_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__ ) ;
    auto inner_product_e = chrono::high_resolution_clock::now() ;

    float inner_product_cost = chrono::duration<float>(inner_product_e - inner_product_s).count() ;

    unsigned num_c_p_offset = 0 ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cudaMemcpy(cluster_p_info + i * cnum , num_c_p[i].data() , cnum * sizeof(unsigned) , cudaMemcpyHostToDevice ) ;
    }

    // 找到存在交集的分区, 并按照聚类将其排序
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg<attr_dim>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , prefilter_threshold) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    

    cout << partition_infos[0] << "," <<  partition_infos[1] << "," <<  partition_infos[2] << endl ;
    // cnum = 16 , qnum = 16 ;
    s = chrono::high_resolution_clock::now() ;
    find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    get_partitions_access_order_V2(partition_infos , pnum , cnum , qnum , cluster_p_info , 4 , cluster_centers_half_dev , query_data_half_dev , dim , query_data_load , cluster_centers , q_norm , c_norm , num)  ;
    e = chrono::high_resolution_clock::now() ;
    cost2 = chrono::duration<double>(e - s).count() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;

    ofstream result_file(result_savepath , ios::out) ;
    result_file << "ef,recall,qps,total_cost,cost1,cost2,cost3,kernal_cost" << endl ;

    
    
    unsigned* pid_list ;
    cudaMalloc((void**) &pid_list , num * sizeof(unsigned)) ;
    // thrust::sequence(pid_list , pid_list + num) ;
    auto cit = thrust::make_counting_iterator<unsigned>(0) ;
    thrust::transform(
        thrust::device ,
        cit , cit + num ,
        pid_list ,
        [] __host__ __device__ (unsigned i) {
            return i / 62500 ;    
        }
    ) ;
    unsigned* bigGraph ;
    unsigned bigDegree = 32 ;
    unsigned degree = 16 ;
    cudaMalloc((void**) &bigGraph , num * bigDegree * sizeof(unsigned)) ;
    auto assemble_edges_s = chrono::high_resolution_clock::now() ;
    smart::full_graph_search::assemble_edges<<<num , bigDegree>>>(graph_infos[0], degree , graph_infos[2] , graph_infos[1] ,
        global_graph , 2, 2 , pnum , idx_mapper[0] ,
        idx_mapper[1] , idx_mapper[2] , pid_list , bigGraph) ;
    cudaDeviceSynchronize() ;
    auto assemble_edges_e = chrono::high_resolution_clock::now() ;
    float assemble_edges_cost = chrono::duration<float>(assemble_edges_e - assemble_edges_s).count() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cout << "全局图组装完成 : bigDegree -  " << bigDegree << endl ;

    // 由于改变的只有L的大小, 故只需将cost3的部分重复执行
    for(unsigned efid = 0 ; efid < ef_list.size() ; efid ++) {

        // check_process2(attr_dim , attr_min ,attr_width ,attr_grid_size ,l_bound ,r_bound ,partition_infos , query_num , query_data_load , dim , data_) ;
        unsigned L = ef_list[efid] ;
        cout << "L : " << L << endl ; 
        s = chrono::high_resolution_clock::now() ;
        // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
        //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
        //     global_graph , 16 , pnum , 10) ;
        // vector<vector<unsigned>> result = smart::full_graph_search::smart_postfiltering_V2(graph_infos , 16 , data_half_dev , query_data_half_dev ,dim ,qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        //     attr_dim  , idx_mapper , partition_infos , pid_list , L , 10 , 
        //     global_graph , 16 , pnum) ;
        // unsigned postfilter_threshold = 16 ; 
        vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter_and_postfilter(graph_infos , degree ,data_half_dev , query_data_half_dev ,dim ,qnum ,l_bound_dev ,r_bound_dev , attrs_dev , 
            attr_dim  , idx_mapper ,partition_infos , L , 10 , 
            global_graph , 2 , pnum , 10 , postfilter_threshold , bigGraph , bigDegree) ;        
        e = chrono::high_resolution_clock::now() ;
        cost3 = chrono::duration<double>(e - s).count() ;
        cout << "finish search" << endl ;

        
        // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
        float total_recall = cal_recall_show_partition_multi_attrs(result , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , false) ;
        cout << "recall : " << total_recall << endl  ;

        cost1 = COST::cost1[0] , cost2 = COST::cost2[0] , cost3 = COST::cost3[efid] ;
        float total_cost = cost1 + cost2 + cost3 ; 
        float qps = qnum * 1.0 / total_cost ; 
        result_file << L << "," << total_recall << "," << qps << "," << total_cost << "," << cost1 << "," << cost2 << "," << cost3 << "," << select_path_kernal_cost << endl ;

        cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
        cout << "获取分区列表用时:" << cost1 << endl ;
        cout << "对分区列表排序用时:" << cost2 << endl ;
        cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
        cout << "搜索用时:" << cost3 << endl ;

        // 计算 ground truth recall
        if(testGT) {
            float gt_recall = cal_recall_show_partition_multi_attrs(gt , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;
            cout << "ground truth recall : " << gt_recall << endl ;
        }
    }
    // rc_recall1 = total_recall ;
    // rc_cost1 = select_path_kernal_cost ;
    cout << "1000点的内积计算时间: " << inner_product_cost << endl ;
    cout << "组装全局边时间: " << assemble_edges_cost << endl ;
    result_file.close() ;
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
    cudaFree(pid_list) ;
    cudaFree(bigGraph) ;
}

template<unsigned attr_dim = 1>
void dataset_test_multiD(string cagra_graph_savepath_prefix , string cagra_graph_savepath_suffix , string global_graph_savepath , 
    string attrs_savepath , string range_savepath , string dataset_path , string query_path , string cluster_savepath  , string ground_truth_savepath , string result_savepath
    ,vector<unsigned>& ef_list , vector<unsigned> grid_size_outer , float f2h_factor , float prefilter_threshold , unsigned postfilter_threshold, bool testGT = false) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    array<unsigned*,3> graph_infos = get_graph_info(cagra_graph_savepath_prefix , cagra_graph_savepath_suffix , 16 , 16, 1000000) ;
    array<unsigned*,3> idx_mapper = get_idx_mapper(1000000 , 16 , 62500) ;
    vector<vector<unsigned>> global_edges ;
    // load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges);
    unsigned *global_graph , *global_graph_cpu ;
    unsigned global_graph_degree , global_graph_point_num ;
    load_result(global_graph_savepath, global_graph_cpu , global_graph_degree , global_graph_point_num) ;
    cout <<  "global graph , degree : " << global_graph_degree <<  ", num : " << global_graph_point_num << endl ;
 
    cudaMalloc((void**) &global_graph , global_graph_degree * global_graph_point_num * sizeof(unsigned)) ;
    cudaMemcpy(global_graph , global_graph_cpu , global_graph_degree * global_graph_point_num * sizeof(unsigned) , cudaMemcpyHostToDevice) ;

    // for(unsigned i = 0 ; i < 10 ; i ++) {
    //     for(unsigned j = 0 ; j < 32 ; j ++)
    //         cout << global_graph_cpu[i * 32 + j] << ", " ;
    //     cout << endl ;
    // }
    

    float *attrs_dev , *attrs ;
    int * attrs_int ;  
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    // using attrType = int ;
    // constexpr unsigned attr_dim =  4 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<float> grid_min(attr_dim , 0.0f) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<float> grid_max(attr_dim , 1000000.0f) ;
    // vector<unsigned> grid_size_host = {2 , 2, 2 , 2} ;
    vector<unsigned> grid_size_host = grid_size_outer ;
    vector<float> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++)
        attr_width_host[i] = (grid_max[i] - grid_min[i]) / grid_size_host[i] ;
    attrs = new float[attr_dim * 1000000] ;
    // attrs_int = new int[attr_dim * 1000000] ;
    unsigned attr_num , attr_dim_tmp ;
    // load_attrs<int>("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M4D.attrs", attrs_int , attr_dim_tmp, attr_num) ;
    load_attrs<int>(attrs_savepath, attrs_int , attr_dim_tmp, attr_num) ;
    // re_generate(attrs , attr_dim , 1000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    for(unsigned i = 0 ; i < attr_num * attr_dim ; i ++)
        attrs[i] = (float) attrs_int[i] ;
    delete [] attrs_int ; 

    cudaMalloc((void**) &attrs_dev , attr_dim * 1000000 * sizeof(float)) ;
    cudaMemcpy(attrs_dev , attrs , attr_dim * 1000000 * sizeof(float) , cudaMemcpyHostToDevice) ;

    cout << "sizeof(half) = " << sizeof(half) << endl ;

    // 先载入数据集和图索引
    unsigned num , dim  ;
    float* data_  ;
    // load_data("/home/yhr/workspace/cuda_clustering/data/sift_base.fvecs" , data_ , num , dim) ;
    load_data(dataset_path , data_ , num , dim) ;
    cout << "num : " << num << ", dim : " << dim << endl ;
    float* query_data_load ;
    unsigned query_num , query_dim ;
    // load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_range_query.fvecs" , query_data_load , query_num , query_dim) ;
    load_data(query_path , query_data_load , query_num , query_dim) ;
    // query_num = 500; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;
    // ************** 若要修改查询数量, 请修改如下变量 **********************
    // query_num = 1 ;

    float* data_dev ;
    half *data_half_dev ;
    float* query_data_dev ;
    half *query_data_half_dev ;
    cudaMalloc((void**)&data_dev , num * dim * sizeof(float)) ;
    cudaMalloc((void**) &data_half_dev , num * dim * sizeof(__half)) ;
    cudaMalloc((void**) &query_data_dev , query_num * query_dim   * sizeof(float)) ;
    unsigned query_num_with_pad = (query_num + 15) / 16 * 16 ;
    cout << "query_num_with_pad : " << query_num_with_pad << endl ;
    cudaMalloc((void**) &query_data_half_dev , query_num_with_pad * query_dim * sizeof(__half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(data_dev , data_ , num * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(query_data_dev , query_data_load , query_num * query_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num ; 
    // 调用实例
    f2h<<<points_num, 256>>>(data_dev, data_half_dev, points_num , dim , f2h_factor); // 先将数据转换成fp16
    // cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    f2h<<<query_num, 256>>>(query_data_dev, query_data_half_dev, query_num , dim , f2h_factor); // 将查询转换成fp16
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    if(query_num != query_num_with_pad)
        thrust::fill(thrust::device , query_data_half_dev + query_num * query_dim , query_data_half_dev + query_num_with_pad * query_dim , __float2half(0.0f)) ;

    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    vector<float> l_bound_host(query_num * attr_dim) , r_bound_host(query_num * attr_dim) ;
    // unsigned test_r_bound = 124999 ;
    if(true) {
        // load_range("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M1D.range", query_range_x, 1000) ;
        vector<int> l_bound_int(query_num * attr_dim) , r_bound_int(query_num * attr_dim) ;
        load_range<int>(range_savepath , l_bound_int.data() ,r_bound_int.data() , attr_dim , query_num) ;
        // load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        // query_range_x.reserve(query_num) ;
        // for(unsigned i = 1 ; i < 10 ; i ++)
        //     query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        // query_range_y = query_range_x ;
        // query_range_x[0] = {0 , 249999} ;
        // query_range_y[0] = {0 , 249999} ;
        // query_range_x[1] = {0 , 499999} , query_range_y[1] = {0 , 249999} ;
        // query_range_x[2] = {0 , 749999} , query_range_y[2] = {0 , 249999} ;
        // query_range_x[3] = {0 , 499999} , query_range_y[3] = {0 , 499999} ;
        // for(unsigned i = 4 ; i < query_num ; i ++)
        //     query_range_x[i] = query_range_x[i % 4] , query_range_y[i] = query_range_y[i % 4] ;
        // query_range_y = query_range_x ; 
        // for(unsigned i = 0 ; i < query_num ; i ++)
        //     query_range_x[i] = {0 , test_r_bound} ;

        for(unsigned i = 0 ; i < query_num ; i ++) {
            for(unsigned j = 0 ;j < attr_dim ; j ++)
                l_bound_host[i * attr_dim + j] = (float)l_bound_int[i * attr_dim + j] , r_bound_host[i * attr_dim + j] = (float)r_bound_int[i * attr_dim + j] ;
                // l_bound_host[i * attr_dim + 1] = query_range_y[i].first , r_bound_host[i * attr_dim + 1] = query_range_y[i].second ;
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

    // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
    vector<vector<unsigned>> ground_truth(qnum) ;
    vector<vector<unsigned>> gt ;
    load_ground_truth(ground_truth_savepath , gt) ;
    if(testGT) {
        #pragma omp parallel for num_threads(10)
        for(unsigned i = 0 ; i < qnum ;i  ++)
            ground_truth[i] = naive_brute_force_with_filter(query_data_load + i * dim , data_ , dim , num , 10 ,
                l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim) ;
    } else {
        // 载入 ground truth 
        // vector<vector<unsigned>> gt ;
        
        swap(ground_truth , gt) ;
        // 每个点的分区编号
    }
    cout << "开始批量查找" << endl ;        
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;

    //  以下载入聚类信息
    vector<unsigned> cluster_id ;
    float* cluster_centers;
    vector<vector<unsigned>> num_c_p ;
    load_clustering(cluster_id , cluster_savepath , cluster_centers , dim , num_c_p , data_) ;
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
    f2h<<<cnum, 256>>>(cluster_centers_dev, cluster_centers_half_dev, cnum, dim , f2h_factor);
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    if(cnum != cnum_with_pad)
        thrust::fill(thrust::device , cluster_centers_half_dev + cnum * dim , cluster_centers_half_dev + cnum_with_pad * dim , __float2half(0.0f)) ;
    // 在此处计算得到q_norm 和 c_norm 
    // ***************************************************
    float *q_norm , *c_norm ;
    cudaMalloc((void**) &q_norm , query_num_with_pad * sizeof(float)) ;
    cudaMalloc((void**) &c_norm , cnum_with_pad * sizeof(float)) ;
    auto inner_product_s = chrono::high_resolution_clock::now() ;
    // find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    find_norm<<<(cnum_with_pad + 3) / 4 , dim3(32 , 4 , 1)>>>(cluster_centers_half_dev , c_norm , dim , cnum_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__ ) ;
    auto inner_product_e = chrono::high_resolution_clock::now() ;

    float inner_product_cost = chrono::duration<float>(inner_product_e - inner_product_s).count() ;

    unsigned num_c_p_offset = 0 ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cudaMemcpy(cluster_p_info + i * cnum , num_c_p[i].data() , cnum * sizeof(unsigned) , cudaMemcpyHostToDevice ) ;
    }

    // 找到存在交集的分区, 并按照聚类将其排序
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg<attr_dim>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , prefilter_threshold) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    

    cout << partition_infos[0] << "," <<  partition_infos[1] << "," <<  partition_infos[2] << endl ;
    // cnum = 16 , qnum = 16 ;
    s = chrono::high_resolution_clock::now() ;
    find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    get_partitions_access_order_V2(partition_infos , pnum , cnum , qnum , cluster_p_info , 4 , cluster_centers_half_dev , query_data_half_dev , dim , query_data_load , cluster_centers , q_norm , c_norm , num)  ;
    e = chrono::high_resolution_clock::now() ;
    cost2 = chrono::duration<double>(e - s).count() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;

    ofstream result_file(result_savepath , ios::out) ;
    result_file << "ef,recall,qps,total_cost,cost1,cost2,cost3,kernal_cost" << endl ;

    
    
    unsigned* pid_list ;
    cudaMalloc((void**) &pid_list , num * sizeof(unsigned)) ;
    // thrust::sequence(pid_list , pid_list + num) ;
    auto cit = thrust::make_counting_iterator<unsigned>(0) ;
    thrust::transform(
        thrust::device ,
        cit , cit + num ,
        pid_list ,
        [] __host__ __device__ (unsigned i) {
            return i / 62500 ;    
        }
    ) ;
    unsigned* bigGraph ;
    unsigned bigDegree = 32 ;
    unsigned degree = 16 ;
    cudaMalloc((void**) &bigGraph , num * bigDegree * sizeof(unsigned)) ;
    auto assemble_edges_s = chrono::high_resolution_clock::now() ;
    smart::full_graph_search::assemble_edges<<<num , bigDegree>>>(graph_infos[0], degree , graph_infos[2] , graph_infos[1] ,
        global_graph , 2, 2 , pnum , idx_mapper[0] ,
        idx_mapper[1] , idx_mapper[2] , pid_list , bigGraph) ;
    cudaDeviceSynchronize() ;
    auto assemble_edges_e = chrono::high_resolution_clock::now() ;
    float assemble_edges_cost = chrono::duration<float>(assemble_edges_e - assemble_edges_s).count() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cout << "全局图组装完成 : bigDegree -  " << bigDegree << endl ;

    // 由于改变的只有L的大小, 故只需将cost3的部分重复执行
    for(unsigned efid = 0 ; efid < ef_list.size() ; efid ++) {

        // check_process2(attr_dim , attr_min ,attr_width ,attr_grid_size ,l_bound ,r_bound ,partition_infos , query_num , query_data_load , dim , data_) ;
        unsigned L = ef_list[efid] ;
        cout << "L : " << L << endl ; 
        s = chrono::high_resolution_clock::now() ;
        // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
        //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
        //     global_graph , 16 , pnum , 10) ;
        // vector<vector<unsigned>> result = smart::full_graph_search::smart_postfiltering_V2(graph_infos , 16 , data_half_dev , query_data_half_dev ,dim ,qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        //     attr_dim  , idx_mapper , partition_infos , pid_list , L , 10 , 
        //     global_graph , 16 , pnum) ;
        // unsigned postfilter_threshold = 16 ; 
        vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter_and_postfilter(graph_infos , degree ,data_half_dev , query_data_half_dev ,dim ,qnum ,l_bound_dev ,r_bound_dev , attrs_dev , 
            attr_dim  , idx_mapper ,partition_infos , L , 10 , 
            global_graph , 2 , pnum , 10 , postfilter_threshold , bigGraph , bigDegree) ;        
        e = chrono::high_resolution_clock::now() ;
        cost3 = chrono::duration<double>(e - s).count() ;
        cout << "finish search" << endl ;

        
        // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
        float total_recall = cal_recall_show_partition_multi_attrs(result , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , false) ;
        cout << "recall : " << total_recall << endl  ;

        cost1 = COST::cost1[0] , cost2 = COST::cost2[0] , cost3 = COST::cost3[efid] ;
        float total_cost = cost1 + cost2 + cost3 ; 
        float qps = qnum * 1.0 / total_cost ; 
        result_file << L << "," << total_recall << "," << qps << "," << total_cost << "," << cost1 << "," << cost2 << "," << cost3 << "," << select_path_kernal_cost << endl ;

        cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
        cout << "获取分区列表用时:" << cost1 << endl ;
        cout << "对分区列表排序用时:" << cost2 << endl ;
        cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
        cout << "搜索用时:" << cost3 << endl ;

        // 计算 ground truth recall
        if(testGT) {
            float gt_recall = cal_recall_show_partition_multi_attrs(gt , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;
            cout << "ground truth recall : " << gt_recall << endl ;
        }
    }
    // rc_recall1 = total_recall ;
    // rc_cost1 = select_path_kernal_cost ;
    cout << "1000点的内积计算时间: " << inner_product_cost << endl ;
    cout << "组装全局边时间: " << assemble_edges_cost << endl ;
    result_file.close() ;
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
    cudaFree(pid_list) ;
    cudaFree(bigGraph) ;
}



template<unsigned attr_dim = 1 , typename T = float>
void real_dataset_test_multiD(string cagra_graph_savepath_prefix , string cagra_graph_savepath_suffix , string global_graph_savepath , string grid_info_savepath ,
    string attrs_savepath , string range_savepath , string dataset_path , string query_path , string cluster_savepath  , string ground_truth_savepath , string result_savepath
    ,vector<unsigned>& ef_list , vector<unsigned> grid_size_outer ,unsigned degree, float f2h_factor ,  T prefilter_threshold , unsigned postfilter_threshold, bool testGT = false) {
    // unsigned L = 80; 
    cout << "请输入L :" ;
    // cin >> L ; 
    print_sym() ;
    // sort_test<<<1 , 32 * 6>>> () ;
    // cudaDeviceSynchronize() ;
    array<unsigned*,3> graph_infos = get_graph_info(cagra_graph_savepath_prefix , cagra_graph_savepath_suffix , 16 , degree, 1000000) ;
    
    // 先载入 grid info
    real_dataset_adapter::GridInfo g_info ;
    real_dataset_adapter::load_grid_infos(grid_info_savepath , g_info) ;    
    // array<unsigned*,3> idx_mapper = get_idx_mapper(1000000 , 16 , 62500) ;
    vector<unsigned> graph_size_host(16) ;
    cudaMemcpy(graph_size_host.data() , graph_infos[1] , 16 * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    cout << "graph size : [" ;
    for(int i = 0 ; i < 16 ; i ++) {
        cout << graph_size_host[i] ;
        if(i < 15)
            cout << "," ;
    }
    cout << "]" << endl ;

    real_dataset_adapter::GridInfoGpu g_info_gpu ;
    real_dataset_adapter::load2GPU(g_info , g_info_gpu) ;

    vector<unsigned> check_pids = {96874 ,135128 ,141421 ,147441 ,150750 ,222072 ,242008 ,305488 ,435467 ,477754} ;
    cout << "check & g_info  : ------" <<endl ; 
    for(int i = 0 ; i < check_pids.size()  ; i ++)
        cout << g_info.global_pid[check_pids[i]] << "," ;
    cout << endl ;
    g_info.showInfo() ;


    array<unsigned*,3> idx_mapper = get_idx_mapper(g_info.num , 16 , graph_size_host.data() , (unsigned*) g_info.global_pid.data()) ;
    vector<vector<unsigned>> global_edges ;
    // load_result("/home/yhr/workspace/2D_range_filter/from_lzg/nsg-master/sift_nn.result", global_edges);
    unsigned *global_graph , *global_graph_cpu ;
    unsigned global_graph_degree , global_graph_point_num ;
    load_result(global_graph_savepath, global_graph_cpu , global_graph_degree , global_graph_point_num) ;
    cout <<  "global graph , degree : " << global_graph_degree <<  ", num : " << global_graph_point_num << endl ;
 
    cudaMalloc((void**) &global_graph , global_graph_degree * global_graph_point_num * sizeof(unsigned)) ;
    cudaMemcpy(global_graph , global_graph_cpu , global_graph_degree * global_graph_point_num * sizeof(unsigned) , cudaMemcpyHostToDevice) ;

    // for(unsigned i = 0 ; i < 10 ; i ++) {
    //     for(unsigned j = 0 ; j < 32 ; j ++)
    //         cout << global_graph_cpu[i * 32 + j] << ", " ;
    //     cout << endl ;
    // }
    

    T *attrs_dev , *attrs ;
    int * attrs_int ;  
    // attrs = generate_attrs() ;
    cout << "开始生成属性----------" << endl ;
    // using attrType = int ;
    // constexpr unsigned attr_dim =  4 ;
    // vector<float> grid_min = {0.0f , 0.0f , 0.0f , 0.0f} ;
    vector<T> grid_min(attr_dim , 0.0f) ;
    // vector<float> grid_max = {1000000.0f , 1000000.0f ,1000000.0f , 1000000.0f } ;
    vector<T> grid_max(attr_dim , 1000000.0f) ;
    // vector<unsigned> grid_size_host = {2 , 2, 2 , 2} ;
    vector<unsigned> grid_size_host = grid_size_outer ;
    vector<T> attr_width_host(attr_dim) ;
    for(unsigned i = 0 ; i < attr_dim ; i ++)
        attr_width_host[i] = (grid_max[i] - grid_min[i]) / grid_size_host[i] ;
    attrs = new T[attr_dim * 1000000] ;
    // attrs_int = new int[attr_dim * 1000000] ;
    unsigned attr_num , attr_dim_tmp ;
    // load_attrs<int>("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M4D.attrs", attrs_int , attr_dim_tmp, attr_num) ;
    load_attrs<int>(attrs_savepath, attrs_int , attr_dim_tmp, attr_num) ;
    cout << "attr_dim : " << attr_dim_tmp << ", attr_num : " << attr_num << endl ;
    // re_generate(attrs , attr_dim , 1000000 , grid_min.data() , grid_max.data() , grid_size_host.data()) ;
    for(unsigned i = 0 ; i < attr_num * attr_dim ; i ++)
        attrs[i] = (T) attrs_int[i] ;
    delete [] attrs_int ; 

    cudaMalloc((void**) &attrs_dev , attr_dim * 1000000 * sizeof(T)) ;
    cudaMemcpy(attrs_dev , attrs , attr_dim * 1000000 * sizeof(T) , cudaMemcpyHostToDevice) ;

    cout << "sizeof(half) = " << sizeof(half) << endl ;

    // 先载入数据集和图索引
    unsigned num , dim  ;
    float* data_  ;
    // load_data("/home/yhr/workspace/cuda_clustering/data/sift_base.fvecs" , data_ , num , dim) ;
    load_data(dataset_path , data_ , num , dim) ;
    // float* data_tmp = new float[1000000] ;
    // memcpy(data_tmp , data_  + num - 1000000 , 1000000 * )
    // data_ = data_ + num * dim - 1000000 * dim ;
    // num = 1000000 ;
    cout << "num : " << num << ", dim : " << dim << endl ;
    float* query_data_load ;
    unsigned query_num , query_dim ;
    // load_data("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_range_query.fvecs" , query_data_load , query_num , query_dim) ;
    load_data(query_path , query_data_load , query_num , query_dim) ;
    // query_num = 500; 
    cout << "qnum : " << query_num << ", qdim : " << query_dim << endl ;
    // ************** 若要修改查询数量, 请修改如下变量 **********************
    // query_num = 993 ;

    float* data_dev ;
    half *data_half_dev ;
    float* query_data_dev ;
    half *query_data_half_dev ;
    cudaMalloc((void**)&data_dev , num * dim * sizeof(float)) ;
    cudaMalloc((void**) &data_half_dev , num * dim * sizeof(__half)) ;
    cudaMalloc((void**) &query_data_dev , query_num * query_dim   * sizeof(float)) ;
    unsigned query_num_with_pad = (query_num + 15) / 16 * 16 ;
    cout << "query_num_with_pad : " << query_num_with_pad << endl ;
    cudaMalloc((void**) &query_data_half_dev , query_num_with_pad * query_dim * sizeof(__half)) ;
    // 将数据复制到设备内存
    cudaMemcpy(data_dev , data_ , num * dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(query_data_dev , query_data_load , query_num * query_dim * sizeof(float) , cudaMemcpyHostToDevice) ;
    unsigned points_num = num ; 
    // 调用实例
    f2h<<<points_num, 256>>>(data_dev, data_half_dev, points_num , dim ,  f2h_factor); // 先将数据转换成fp16
    // cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    f2h<<<query_num, 256>>>(query_data_dev, query_data_half_dev, query_num , dim , f2h_factor); // 将查询转换成fp16
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__) ;
    if(query_num != query_num_with_pad)
        thrust::fill(thrust::device , query_data_half_dev + query_num * query_dim , query_data_half_dev + query_num_with_pad * query_dim , __float2half(0.0f)) ;

    //载入查询范围
    vector<pair<unsigned,unsigned>> query_range_x ;
    vector<pair<unsigned , unsigned>> query_range_y ; 
    vector<T> l_bound_host(query_num * attr_dim) , r_bound_host(query_num * attr_dim) ;
    // unsigned test_r_bound = 124999 ;
    if(true) {
        // load_range("/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M1D.range", query_range_x, 1000) ;
        vector<int> l_bound_int(query_num * attr_dim) , r_bound_int(query_num * attr_dim) ;
        load_range<int>(range_savepath , l_bound_int.data() ,r_bound_int.data() , attr_dim , query_num) ;
        // l_bound_int[0] = 1937 , r_bound_int[0] = 2002 ;
        // load_range("/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_query_range.bin", query_range_x, 1000) ;
        // query_range_x.reserve(query_num) ;
        // for(unsigned i = 1 ; i < 10 ; i ++)
        //     query_range_x.insert(query_range_x.end() , query_range_x.begin() , query_range_x.begin() + 1000) ;
        // query_range_y = query_range_x ;
        // query_range_x[0] = {0 , 249999} ;
        // query_range_y[0] = {0 , 249999} ;
        // query_range_x[1] = {0 , 499999} , query_range_y[1] = {0 , 249999} ;
        // query_range_x[2] = {0 , 749999} , query_range_y[2] = {0 , 249999} ;
        // query_range_x[3] = {0 , 499999} , query_range_y[3] = {0 , 499999} ;
        // for(unsigned i = 4 ; i < query_num ; i ++)
        //     query_range_x[i] = query_range_x[i % 4] , query_range_y[i] = query_range_y[i % 4] ;
        // query_range_y = query_range_x ; 
        // for(unsigned i = 0 ; i < query_num ; i ++)
        //     query_range_x[i] = {0 , test_r_bound} ;

        for(unsigned i = 0 ; i < query_num ; i ++) {
            for(unsigned j = 0 ;j < attr_dim ; j ++)
                l_bound_host[i * attr_dim + j] = (T)l_bound_int[i * attr_dim + j] , r_bound_host[i * attr_dim + j] = (T)r_bound_int[i * attr_dim + j] ;
                // l_bound_host[i * attr_dim + 1] = query_range_y[i].first , r_bound_host[i * attr_dim + 1] = query_range_y[i].second ;
        }
        
        for(unsigned i = 992 ; i < query_num ; i ++) {
            for(unsigned j = 0; j < attr_dim ; j ++)
                // cout << "bound : 992(" << l_bound_host[992 * attr_dim + j] << "," << r_bound_host[992 * attr_dim + j] << ")" << endl ;
                l_bound_host[i * attr_dim + j] = l_bound_host[(i - 10) * attr_dim + j] ,
                r_bound_host[i * attr_dim + j] = r_bound_host[(i - 10) * attr_dim + j] ;
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

    T* l_bound , * r_bound , *l_bound_dev , *r_bound_dev , *attr_min , *attr_width ;
    unsigned *attr_grid_size ;
    // l_bound = new float[query_num * attr_dim] , r_bound = new float[query_num * attr_dim] ;
    l_bound = l_bound_host.data() , r_bound = r_bound_host.data() ;
    // for(unsigned i = 0 ; i < query_num ; i ++) {
    //     // for(unsigned j = 0 ;j  < attr_dim ; j ++)
    //         l_bound[i * attr_dim] = (float) query_range_x[i].first , r_bound[i * attr_dim] = (float) query_range_x[i].second ; 
    //         l_bound[i * attr_dim + 1] = (float) query_range_y[i].first , r_bound[i * attr_dim + 1] = (float) query_range_y[i].second ;
    // }
    cudaMalloc((void**) &l_bound_dev , attr_dim * query_num * sizeof(T)) ;
    cudaMalloc((void**) &r_bound_dev , attr_dim * query_num * sizeof(T)) ;
    cudaMalloc((void**) &attr_min , attr_dim * sizeof(T)) ;
    cudaMalloc((void**) &attr_width , attr_dim * sizeof(T)) ;
    cudaMalloc((void**) &attr_grid_size , attr_dim * sizeof(unsigned)) ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;


    cudaMemcpy(l_bound_dev , l_bound , query_num * attr_dim * sizeof(T) , cudaMemcpyHostToDevice) ;
    cudaMemcpy(r_bound_dev , r_bound , query_num * attr_dim * sizeof(T) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_min , attr_min + attr_dim , 0.0) ;
    cudaMemcpy(attr_min , grid_min.data() , attr_dim * sizeof(T) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_width , attr_width + attr_dim , 250000.0) ;
    cudaMemcpy(attr_width , attr_width_host.data() , attr_dim * sizeof(T) , cudaMemcpyHostToDevice) ;
    // thrust::fill(thrust::device , attr_grid_size , attr_grid_size + attr_dim , 4) ;
    cudaMemcpy(attr_grid_size , grid_size_host.data() , attr_dim * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    unsigned qnum = query_num ;
    unsigned pnum = thrust::reduce(
        thrust::device , 
        attr_grid_size , attr_grid_size + attr_dim ,
        1 , thrust::multiplies<unsigned>() 
    ) ;

    // 需要完成: 补全global_edges , 计算cpu版本 ground truth, 计算recall
    vector<vector<unsigned>> ground_truth(qnum) ;
    vector<vector<unsigned>> gt ;
    load_ground_truth(ground_truth_savepath , gt) ;
    gt.resize(qnum) ;
    if(testGT) {
        #pragma omp parallel for num_threads(10)
        for(unsigned i = 0 ; i < qnum ;i  ++)
            ground_truth[i] = naive_brute_force_with_filter(query_data_load + i * dim , data_ , dim , num , 10 ,
                l_bound + attr_dim * i , r_bound + attr_dim * i , attrs , attr_dim) ;
    } else {
        // 载入 ground truth 
        // vector<vector<unsigned>> gt ;
        
        swap(ground_truth , gt) ;
        // 每个点的分区编号
    }
    cout << "开始批量查找" << endl ;        
    float cost1 = 0.0 , cost2 = 0.0 , cost3 = 0.0 ;

    //  以下载入聚类信息
    vector<unsigned> cluster_id ;
    float* cluster_centers;
    vector<vector<unsigned>> num_c_p ;
    load_clustering(cluster_id , cluster_savepath , cluster_centers , dim , num_c_p , data_) ;
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
    f2h<<<cnum, 256>>>(cluster_centers_dev, cluster_centers_half_dev, cnum, dim , f2h_factor);
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    if(cnum != cnum_with_pad)
        thrust::fill(thrust::device , cluster_centers_half_dev + cnum * dim , cluster_centers_half_dev + cnum_with_pad * dim , __float2half(0.0f)) ;
    // 在此处计算得到q_norm 和 c_norm 
    // ***************************************************
    float *q_norm , *c_norm ;
    cudaMalloc((void**) &q_norm , query_num_with_pad * sizeof(float)) ;
    cudaMalloc((void**) &c_norm , cnum_with_pad * sizeof(float)) ;
    auto inner_product_s = chrono::high_resolution_clock::now() ;
    // find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    find_norm<<<(cnum_with_pad + 3) / 4 , dim3(32 , 4 , 1)>>>(cluster_centers_half_dev , c_norm , dim , cnum_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__ ) ;
    auto inner_product_e = chrono::high_resolution_clock::now() ;

    float inner_product_cost = chrono::duration<float>(inner_product_e - inner_product_s).count() ;

    unsigned num_c_p_offset = 0 ; 
    for(unsigned i = 0 ; i < pnum ; i ++) {
        cudaMemcpy(cluster_p_info + i * cnum , num_c_p[i].data() , cnum * sizeof(unsigned) , cudaMemcpyHostToDevice ) ;
    }

    // 找到存在交集的分区, 并按照聚类将其排序
    cout << "intersection_area- qnum : " << qnum << ", pnum : " << pnum << endl ;
    auto s = chrono::high_resolution_clock::now() ;
    // array<unsigned* , 5> partition_infos = intersection_area_mark_little_seg<attr_dim>(l_bound_dev , r_bound_dev , attr_min , attr_width , attr_grid_size , attr_dim , qnum , pnum , prefilter_threshold) ;
    // array<unsigned* , 5> partition_infos = real_dataset_adapter::intersection_area_mark_little_seg<attr_dim>(l_bound_dev , r_bound_dev , g_info_gpu.l_quantiles , g_info_gpu.r_quantiles , g_info_gpu.dim_offset , g_info_gpu.grid_size , attr_dim , qnum , pnum , prefilter_threshold) ;
    array<unsigned* , 5> partition_infos = real_dataset_adapter::intersection_area_mark_little_seg_V2<attr_dim , T>(l_bound_dev ,r_bound_dev , g_info_gpu.l_quantiles , g_info_gpu.r_quantiles , g_info_gpu.dim_offset , g_info_gpu.grid_size , attr_dim , qnum , pnum , num , attrs_dev , prefilter_threshold) ;
    auto e = chrono::high_resolution_clock::now() ;
    cost1 = chrono::duration<double>(e - s).count() ;
    cout << "intersection_area execution finished !" << endl ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    // int t ;
    // cin >> t ; 

    
    // 查看排序后的分区顺序
    // vector<unsigned> psize_host(10) ;
    // cudaMemcpy(psize_host.data() , partition_infos[1] , 10 * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // // unsigned psize_host_acc = std::accmulate(psize_host.begin() , psize_host.end() , 0) ;
    // unsigned psize_host_acc = 0 ;
    // for(int i = 0 ; i < 10 ; i ++)
    //     psize_host_acc += psize_host[i] ;
    // cout << "psize_host_acc :" << psize_host_acc << endl ;
    // vector<unsigned> partition_infos1_host(100) ;
    // cudaMemcpy(partition_infos1_host.data() , partition_infos[0] , 100 * sizeof(unsigned) , cudaMemcpyDeviceToHost) ;
    // int ppp_offset = 0 ;
    // for(int i = 0 ; i < 10 ; i ++) {
    //     cout << "[" ;
    //     for(int j = 0 ; j < psize_host[i] ; j ++) {
    //         cout << partition_infos[ppp_offset ++] ;
    //         if(j < psize_host[i] - 1)
    //             cout << "," ;
    //     }
    //     cout << "]" << endl ;
    // }
    // int t ; 
    // cin >> t ;
    /////////////////////////////////////////////






    cout << partition_infos[0] << "," <<  partition_infos[1] << "," <<  partition_infos[2] << endl ;
    // cnum = 16 , qnum = 16 ;
    s = chrono::high_resolution_clock::now() ;
    find_norm<<<(query_num_with_pad + 5) / 6 , dim3(32 , 6 , 1)>>>(query_data_half_dev , q_norm , dim , query_num_with_pad) ;
    cudaDeviceSynchronize() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    get_partitions_access_order_V2(partition_infos , pnum , cnum , qnum , cluster_p_info , 4 , cluster_centers_half_dev , query_data_half_dev , dim , query_data_load , cluster_centers , q_norm , c_norm , num)  ;
    e = chrono::high_resolution_clock::now() ;
    cost2 = chrono::duration<double>(e - s).count() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;

    ofstream result_file(result_savepath , ios::out) ;
    result_file << "ef,recall,qps,total_cost,cost1,cost2,cost3,kernal_cost" << endl ;

    
    
    unsigned* pid_list = g_info_gpu.global_pid ; 
    // cudaMalloc((void**) &pid_list , num * sizeof(unsigned)) ;
    // // thrust::sequence(pid_list , pid_list + num) ;
    // auto cit = thrust::make_counting_iterator<unsigned>(0) ;
    // thrust::transform(
    //     thrust::device ,
    //     cit , cit + num ,
    //     pid_list ,
    //     [] __host__ __device__ (unsigned i) {
    //         return i / 62500 ;    
    //     }
    // ) ;
    unsigned* bigGraph ;
    // unsigned bigDegree = 32 ;
    unsigned bigDegree = 32 ;
    // unsigned degree = 32 ;

    // 从文件中写入bigDegree
    // unsigned *bigGraph_cpu ;
    // unsigned bigGraph_degree_cpu , bigGraph_num_cpu ;
    // load_result("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/year/dblp.bigGraph16", bigGraph_cpu , bigGraph_degree_cpu , bigGraph_num_cpu) ;
    // cout <<  "global graph , degree : " << bigGraph_degree_cpu <<  ", num : " << bigGraph_num_cpu << endl ;

    //***********************************
    cudaMalloc((void**) &bigGraph , num * bigDegree * sizeof(unsigned)) ;
    // cudaMemcpy(bigGraph , bigGraph_cpu , num * bigDegree * sizeof(unsigned) , cudaMemcpyHostToDevice) ;
    auto assemble_edges_s = chrono::high_resolution_clock::now() ;
    smart::full_graph_search::assemble_edges<<<num , bigDegree>>>(graph_infos[0], degree , graph_infos[2] , graph_infos[1] ,
        global_graph , 2, 2 , pnum , idx_mapper[0] ,
        idx_mapper[1] , idx_mapper[2] , pid_list , bigGraph) ;
    cudaDeviceSynchronize() ;
    auto assemble_edges_e = chrono::high_resolution_clock::now() ;
    float assemble_edges_cost = chrono::duration<float>(assemble_edges_e - assemble_edges_s).count() ;
    CHECK(cudaGetLastError() , __LINE__ , __FILE__) ;
    cout << "全局图组装完成 : bigDegree -  " << bigDegree << endl ;

    // 由于改变的只有L的大小, 故只需将cost3的部分重复执行
    for(unsigned efid = 0 ; efid < ef_list.size() ; efid ++) {

        // check_process2(attr_dim , attr_min ,attr_width ,attr_grid_size ,l_bound ,r_bound ,partition_infos , query_num , query_data_load , dim , data_) ;
        unsigned L = ef_list[efid] ;
        cout << "L : " << L << endl ; 
        s = chrono::high_resolution_clock::now() ;
        // vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter(graph_infos , 16 ,  data_half_dev , query_data_half_dev ,dim , qnum , l_bound_dev , r_bound_dev , 
        //     attrs_dev , attr_dim , idx_mapper , partition_infos ,  L , 10 , 
        //     global_graph , 16 , pnum , 10) ;
        // vector<vector<unsigned>> result = smart::full_graph_search::smart_postfiltering_V2(graph_infos , 16 , data_half_dev , query_data_half_dev ,dim ,qnum , l_bound_dev , r_bound_dev , attrs_dev , 
        //     attr_dim  , idx_mapper , partition_infos , pid_list , L , 10 , 
        //     global_graph , 16 , pnum) ;
        // unsigned postfilter_threshold = 16 ; 
        vector<vector<unsigned>> result = batch_graph_search_gpu_with_prefilter_and_postfilter(graph_infos , degree ,data_half_dev , query_data_half_dev ,dim ,qnum ,l_bound_dev ,r_bound_dev , attrs_dev , 
            attr_dim  , idx_mapper ,partition_infos , L , 10 , 
            global_graph , 2 , pnum , 10 , postfilter_threshold , bigGraph , bigDegree) ;        
        e = chrono::high_resolution_clock::now() ;
        cost3 = chrono::duration<double>(e - s).count() ;
        cout << "finish search" << endl ;

        
        // float total_recall = cal_recall_show_partition(result , ground_truth ,query_range_x , query_range_y , 250000.0 , 250000.0, true) ;
        float total_recall = cal_recall_show_partition_multi_attrs_show_attr(result , ground_truth , l_bound , r_bound , attrs , attr_dim , attr_width_host.data() , grid_min.data() , false) ;
        cout << "recall : " << total_recall << endl  ;

        cost1 = COST::cost1[0] , cost2 = COST::cost2[0] , cost3 = COST::cost3[efid] ;
        float total_cost = cost1 + cost2 + cost3 ; 
        float qps = qnum * 1.0 / total_cost ; 
        result_file << L << "," << total_recall << "," << qps << "," << total_cost << "," << cost1 << "," << cost2 << "," << cost3 << "," << select_path_kernal_cost << endl ;

        cout << "总用时:" << (cost1 + cost2 + cost3) << endl ;
        cout << "获取分区列表用时:" << cost1 << endl ;
        cout << "对分区列表排序用时:" << cost2 << endl ;
        cout << "select path 核函数执行时间:" << select_path_kernal_cost << endl ;
        cout << "搜索用时:" << cost3 << endl ;

        // 计算 ground truth recall
        if(testGT) {
            float gt_recall = cal_recall_show_partition_multi_attrs(gt , ground_truth , l_bound , r_bound , attr_dim , attr_width_host.data() , grid_min.data() , true) ;
            cout << "ground truth recall : " << gt_recall << endl ;
        }
    }
    // rc_recall1 = total_recall ;
    // rc_cost1 = select_path_kernal_cost ;
    cout << "1000点的内积计算时间: " << inner_product_cost << endl ;
    cout << "组装全局边时间: " << assemble_edges_cost << endl ;
    result_file.close() ;
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
    cudaFree(pid_list) ;
    cudaFree(bigGraph) ;
}

int main(int argc , char** argv) {

    constexpr unsigned attr_dim = 1 ; 
    vector<unsigned> shape = {16} ;
    string attrD = to_string(attr_dim) ;
    string datasetName = "dblp" ;
    string cagra_graph_savepath_prefix = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/2D/dblp_2D" ;
    string cagra_graph_savepath_suffix = ".cagra16" ;
    string global_graph_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/2D/dblp_2D.graph32" ;
    // string global_graph_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/year/dblp.bigGraph16" ;
    // string grid_info_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/grid/dblp_2D.grid" ;
    string attrs_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/attributes/attr1M2D.attrs" ;
    string range_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/attributes/2D.range_mapped" ;
    string dataset_path = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/DBLP/embeddings_output/dblp1M.fvecs" ;
    string query_path = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/DBLP/embeddings_output/dblp_query.fvecs" ;
    string cluster_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/clustering/dblp_clustering1000.bin" ;
    string ground_truth_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/ground_truth/2D.gt" ;
    string result_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/results/smart/2D/" + attrD + "D.csv" ;
    float prefilter_threshold ;
    unsigned postfilter_threshold ;
    float f2h_factor ;
    // unsigned degree ;
    
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            // paths["data_vector"] = argv[i + 1];
            dataset_path = string(argv[i + 1]) ;
        if (arg == "--query_path")
            // paths["query_vector"] = argv[i + 1];
            query_path = string(argv[i + 1]) ;
        if (arg == "--range")
            // paths["range_prefix"] = argv[i + 1];
            range_savepath = string(argv[i + 1]) ;
        if (arg == "--gt")
            // paths["groundtruth_prefix"] = argv[i + 1];
            ground_truth_savepath = string(argv[i + 1]) ;
        if (arg == "--cagra_prefix")
            // paths["index"] = argv[i + 1];
            cagra_graph_savepath_prefix = string(argv[i + 1]) ;
        if (arg == "--cagra_suffix")
            cagra_graph_savepath_suffix = string(argv[i + 1]) ;
        if (arg == "--global_graph")
            // paths["result_saveprefix"] = argv[i + 1];
            global_graph_savepath = string(argv[i + 1]) ;
        // if (arg == "--grid")
        //     grid_info_savepath = string(argv[i + 1]) ;
        if (arg == "--attrs")
            attrs_savepath = string(argv[i + 1]) ;
        if (arg == "--cluster")
            cluster_savepath = string(argv[i + 1]);
        if (arg == "--result")
            result_savepath = string(argv[i + 1]) ;
        if (arg == "--prefilter")
            prefilter_threshold = stof(argv[i + 1]) ;
        if (arg == "--postfilter")
            postfilter_threshold = stoi(argv[i + 1]) ;
        if (arg == "--f2h_factor")
            f2h_factor = stof(argv[i + 1] ) ;
        // if (arg == "--degree")
        //     degree = stoi(argv[i + 1]) ;
    }


    // ef , qps , recall 
    // unsigned L = 20 ;
    // vector<unsigned> ef_list = {1400, 700, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10} ;
    vector<unsigned> ef_list = {1700, 1400, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10 ,10} ;
    // unsigned ef = stoi(argv[1]) ;
    // vector<unsigned> ef_list = {10} ;
    reverse(ef_list.begin() , ef_list.end()) ;
    // ef_list.resize(8) ;
    // cin >> test_r_bound ;
    // test_r_bound = 999999 ;
    // test_r_bound = stoi(argv[1]) ;
    // sift_experiment_run(ef_list) ;
    // sift_postfiltering_test(ef_list) ;
    // sift_test_multiD(ef_list) ;

    /**
    string cagra_graph_savepath_prefix = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/2D/dblp_2D" ;
    string cagra_graph_savepath_suffix = ".cagra16" ;
    string global_graph_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/2D/dblp_2D.graph32" ;
    // string global_graph_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/year/dblp.bigGraph16" ;
    string grid_info_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/grid/dblp_2D.grid" ;
    string attrs_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/attributes/attr1M2D.attrs" ;
    string range_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/attributes/2D.range_mapped" ;
    string dataset_path = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/DBLP/embeddings_output/dblp1M.fvecs" ;
    string query_path = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/DBLP/embeddings_output/dblp_query.fvecs" ;
    string cluster_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/clustering/dblp_clustering1000.bin" ;
    string ground_truth_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/ground_truth/2D.gt" ;
    string result_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/results/smart/2D/" + attrD + "D.csv" ;
    **/
    // real_dataset_test_multiD<attr_dim>(cagra_graph_savepath_prefix , cagra_graph_savepath_suffix ,global_graph_savepath , grid_info_savepath ,
    //    attrs_savepath ,range_savepath ,dataset_path , query_path ,cluster_savepath  , ground_truth_savepath ,  result_savepath ,
    //    ef_list , shape ,degree , f2h_factor,  prefilter_threshold , postfilter_threshold , false) ;
    dataset_test_multiD<attr_dim>(cagra_graph_savepath_prefix , cagra_graph_savepath_suffix ,global_graph_savepath , 
        attrs_savepath ,range_savepath ,dataset_path , query_path ,cluster_savepath  , ground_truth_savepath ,  result_savepath ,
        ef_list , shape , f2h_factor , prefilter_threshold , postfilter_threshold , false) ;
    return 0 ;
}


// 注: 替换数据集后, f2h的系数要变
int old_main(int argc , char** argv) {
    // ef , qps , recall 
    // unsigned L = 20 ;
    // vector<unsigned> ef_list = {1400, 700, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10} ;
    vector<unsigned> ef_list = {1700, 1400, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10 ,10} ;
    // unsigned ef = stoi(argv[1]) ;
    // vector<unsigned> ef_list = {128} ;
    reverse(ef_list.begin() , ef_list.end()) ;
    // ef_list.resize(8) ;
    // cin >> test_r_bound ;
    // test_r_bound = 999999 ;
    // test_r_bound = stoi(argv[1]) ;
    // sift_experiment_run(ef_list) ;
    // sift_postfiltering_test(ef_list) ;
    // sift_test_multiD(ef_list) ;
    constexpr unsigned attr_dim = 1 ; 
    vector<unsigned> shape = {16} ;
    string attrD = to_string(attr_dim) ;
    string datasetName = "sift1m" ;
    string cagra_graph_savepath_prefix = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/sift" ;
    string cagra_graph_savepath_suffix = ".cagra" ;
    string global_graph_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/cagra_indexes/sift1M/sift1m_global.graph32" ;
    string attrs_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/sift1m/attributes/attr1M" + attrD + "D.attrs" ;
    string range_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/selectivity/attributes/attr1M" + attrD + "D25S@.range" ;
    string dataset_path = "/home/yhr/workspace/cuda_clustering/data/sift_base.fvecs" ;
    string query_path = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/sift_range_query.fvecs" ;
    string cluster_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/clustering1000.bin" ;
    string ground_truth_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/selectivity/sift1m/ground_truth/sift1m_q1k_k10_" + attrD + "D_25S@.gt" ;
    string result_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/selectivity/sift1m/results/smart/" + attrD + "D_25S@_0-0@.csv" ;
    dataset_test_multiD<attr_dim>(cagra_graph_savepath_prefix , cagra_graph_savepath_suffix ,global_graph_savepath , 
       attrs_savepath ,range_savepath ,dataset_path , query_path ,cluster_savepath  , ground_truth_savepath ,  result_savepath ,
       ef_list , shape ,  -0.1 , 0 , false) ;
    return 0 ;
}


int __________main(int argc , char** argv) {

    constexpr unsigned attr_dim = 1 ; 
    vector<unsigned> shape = {16} ;
    string attrD = to_string(attr_dim) ;
    string datasetName = "dblp" ;
    string cagra_graph_savepath_prefix = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/2D/dblp_2D" ;
    string cagra_graph_savepath_suffix = ".cagra16" ;
    string global_graph_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/2D/dblp_2D.graph32" ;
    // string global_graph_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/year/dblp.bigGraph16" ;
    string grid_info_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/grid/dblp_2D.grid" ;
    string attrs_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/attributes/attr1M2D.attrs" ;
    string range_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/attributes/2D.range_mapped" ;
    string dataset_path = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/DBLP/embeddings_output/dblp1M.fvecs" ;
    string query_path = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/DBLP/embeddings_output/dblp_query.fvecs" ;
    string cluster_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/clustering/dblp_clustering1000.bin" ;
    string ground_truth_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/ground_truth/2D.gt" ;
    string result_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/results/smart/2D/" + attrD + "D.csv" ;
    float prefilter_threshold ;
    unsigned postfilter_threshold ;
    float f2h_factor ;
    unsigned degree ;
    
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            // paths["data_vector"] = argv[i + 1];
            dataset_path = string(argv[i + 1]) ;
        if (arg == "--query_path")
            // paths["query_vector"] = argv[i + 1];
            query_path = string(argv[i + 1]) ;
        if (arg == "--range")
            // paths["range_prefix"] = argv[i + 1];
            range_savepath = string(argv[i + 1]) ;
        if (arg == "--gt")
            // paths["groundtruth_prefix"] = argv[i + 1];
            ground_truth_savepath = string(argv[i + 1]) ;
        if (arg == "--cagra_prefix")
            // paths["index"] = argv[i + 1];
            cagra_graph_savepath_prefix = string(argv[i + 1]) ;
        if (arg == "--cagra_suffix")
            cagra_graph_savepath_suffix = string(argv[i + 1]) ;
        if (arg == "--global_graph")
            // paths["result_saveprefix"] = argv[i + 1];
            global_graph_savepath = string(argv[i + 1]) ;
        if (arg == "--grid")
            grid_info_savepath = string(argv[i + 1]) ;
        if (arg == "--attrs")
            attrs_savepath = string(argv[i + 1]) ;
        if (arg == "--cluster")
            cluster_savepath = string(argv[i + 1]);
        if (arg == "--result")
            result_savepath = string(argv[i + 1]) ;
        if (arg == "--prefilter")
            prefilter_threshold = stof(argv[i + 1]) ;
        if (arg == "--postfilter")
            postfilter_threshold = stoi(argv[i + 1]) ;
        if (arg == "--f2h_factor")
            f2h_factor = stof(argv[i + 1] ) ;
        if (arg == "--degree")
            degree = stoi(argv[i + 1]) ;
    }


    // ef , qps , recall 
    // unsigned L = 20 ;
    // vector<unsigned> ef_list = {1400, 700, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10} ;
    vector<unsigned> ef_list = {1700, 1400, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10 ,10} ;
    // unsigned ef = stoi(argv[1]) ;
    // vector<unsigned> ef_list = {10} ;
    reverse(ef_list.begin() , ef_list.end()) ;
    // ef_list.resize(8) ;
    // cin >> test_r_bound ;
    // test_r_bound = 999999 ;
    // test_r_bound = stoi(argv[1]) ;
    // sift_experiment_run(ef_list) ;
    // sift_postfiltering_test(ef_list) ;
    // sift_test_multiD(ef_list) ;

    /**
    string cagra_graph_savepath_prefix = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/2D/dblp_2D" ;
    string cagra_graph_savepath_suffix = ".cagra16" ;
    string global_graph_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/2D/dblp_2D.graph32" ;
    // string global_graph_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/year/dblp.bigGraph16" ;
    string grid_info_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/data/grid/dblp_2D.grid" ;
    string attrs_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/attributes/attr1M2D.attrs" ;
    string range_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/attributes/2D.range_mapped" ;
    string dataset_path = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/DBLP/embeddings_output/dblp1M.fvecs" ;
    string query_path = "/home/yhr/workspace/2D_range_filter/from_lzg/dataset/DBLP/embeddings_output/dblp_query.fvecs" ;
    string cluster_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/clustering/dblp_clustering1000.bin" ;
    string ground_truth_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/ground_truth/2D.gt" ;
    string result_savepath = "/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/results/smart/2D/" + attrD + "D.csv" ;
    **/
    real_dataset_test_multiD<attr_dim>(cagra_graph_savepath_prefix , cagra_graph_savepath_suffix ,global_graph_savepath , grid_info_savepath ,
       attrs_savepath ,range_savepath ,dataset_path , query_path ,cluster_savepath  , ground_truth_savepath ,  result_savepath ,
       ef_list , shape ,degree , f2h_factor,  prefilter_threshold , postfilter_threshold , false) ;
    return 0 ;
}

