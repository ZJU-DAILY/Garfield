import numpy as np
import cupy as cp
from cuvs.neighbors import cagra
from cuvs.neighbors import brute_force
import time
import argparse

def read_fvecs(file_path , d = 128):
    with open(file_path, 'rb') as f:
        # 读取文件的前两个整数：维度和数据点数
        dtype = np.dtype([
            ('dim', np.int32),
            ('features', np.float32, d)  # 固定形状
        ])
        
        data = np.fromfile(f, dtype=dtype)
        dimensions = data['dim']  # 形状：(num,)
        features = data['features']  # 形状：(num, 128)

        print(f"读取了 {len(data)} 行数据")
        print(f"特征向量形状: {features.shape}")
        print(f"第一行的维数: {dimensions[0]}")
        print(f"第一行的前5个特征: {features[0, :5]}")
        # 将第一列去掉
        
        
    return dimensions[0] , len(data) , features 

def read_binary_file(file_path):
    with open(file_path, 'rb') as f:
        # 读取前8个字节，解析为两个unsigned int (num, dim)
        header = np.fromfile(f, dtype=np.uint32, count=2)
        num, dim = header[0], header[1]
        
        # 计算后续字节的大小：num * dim 个 uint8 数据
        data = np.fromfile(f, dtype=np.uint8, count=np.int64(num) * np.int64(dim))
        
        # 将字节数据重塑为 (num, dim) 的形状
        data = data.reshape((num, dim))
        data = data.astype(np.float32)
    return dim , num , data

# 示例：读取 fvecs 文件
# file_path = '/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs'
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description="Index parameters")
    parser.add_argument("--pnum" , required=True , type=int , help='Partition num for Grid')
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the fvecs data file"
    )
    parser.add_argument(
        "--dimension" , type = int , required=True , help="dimension for dataset"
    )
    parser.add_argument(
        "--graph_degree", type=int, required=True, help="Graph degree for each partition"
    )
    parser.add_argument(
        "--cagra_index_save_prefix", type=str, required=True, help="equal to the path used in cagra_build.py"
    )
    parser.add_argument(
        "--edges_num", type=int, required=True, help="global edges num"
    )
    
    parser.add_argument(
        "--savepath", type=int, required=True, help="global graph save path"
    )
    parser.add_argument(
        "--searchEf", type=int, default=64, required=False, help="global graph save path"
    )
    args = parser.parse_args()
    (
        pnum , 
        file_path , 
        preset_d , 
        degree ,
        cagra_index_save_prefix , 
        edges_num ,
        savepath ,
        searchEf 
    ) = (
        args.pnum , 
        args.dataset_path , 
        args.dimension , 
        args.graph_degree , 
        args.cagra_index_save_prefix ,
        args.edges_num ,
        args.savepath , 
        args.searchEf
    )
    # pnum = 16
    # D = 4
    saveFlag = True
    savePerformance = False
    # datasetName = 'dblp'
    # file_path = f'/home/yhr/workspace/2D_range_filter/from_lzg/dataset/DBLP/embeddings_output/dblp1M.fvecs'
    # performance_path = f'/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/results/smart-new/{D}D/buildcost.txt'
    # performance_path = f'cuvs-indexes/youtube/1D/test/buildcost.txt'
    # savepath = f'/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/youtube/{D}D/dblp_global.graph32'
    dim , num , xb = read_fvecs(file_path , preset_d)
    # num = 1000000
    # xb = xb[-num : , :]
    print(xb.shape)



    k = edges_num
    # result = np.zeros((pnum , num * 2) , dtype=np.int32)
    result = np.zeros([num , pnum * k] , dtype = np.int32)
    print(result.shape)
    # average_num = int((num + pnum - 1) / pnum)
    # dataset = cp.asarray(xb[0 : average_num , :])

    s = time.perf_counter() 
    index_load_cost = 0.0
    pure_search_cost = 0.0
    for i in range(pnum) :
        index_load_s = time.perf_counter() 
        index = cagra.load(f'{cagra_index_save_prefix}{i}.cagra{degree}')
        index_load_e = time.perf_counter() 
        index_load_cost += (index_load_e - index_load_s)
        
        batch_size = 62500
        batch_num = int((num + batch_size - 1)/ batch_size)
        print(f"epoch {i} : batch_size : {batch_size} , batch_num : {batch_num}")
        search_params = cagra.SearchParams(
            max_queries=batch_size,
            itopk_size=searchEf, 
            # search_width = 1 
        )
        locals = time.perf_counter()
        for j in range(batch_num) :
            start = j * batch_size 
            end = (j + 1) * batch_size if j < batch_num - 1 else num
            
            queries = cp.asarray(xb[start : end , :])
            if j == 0 : 
                print(f'queries.shape : {queries.shape}')
            pure_search_s = time.perf_counter() 
            distances, neighbors = cagra.search(search_params, index, queries,k)
            pure_search_e = time.perf_counter()
            pure_search_cost += (pure_search_e - pure_search_s)
            
            neighbors_cpu = cp.asnumpy(neighbors)
            distances_cpu = cp.asnumpy(distances)
            del queries
            del neighbors
            del distances
            
            # neighbors.flatten(order='C')
            flat = neighbors_cpu.ravel() 
            # result[i , j * batch_size * 2 : (j + 1) * batch_size * 2] = flat 
            result[start : end , i * k : (i + 1) * k] = neighbors_cpu 
            
            
        locale = time.perf_counter()
        del index 
        print(f"index {i} finished , 用时: {locale - locals}s")
    e = time.perf_counter() 
    print(f"构建完成 , 用时: {e - s}s")
    print(f"载入索引用时: {index_load_cost}")
    print(f"纯搜索用时: {pure_search_cost}")

    # 释放掉不需要的内存
    del xb 




    # if savePerformance == True :
    #     with open(performance_path , 'a', encoding = 'utf-8') as file :
    #         file.write(f"python : full graph build cost : {pure_search_cost}\n")

    if saveFlag == True :
        n_point = np.array([result.shape[0]] , np.int32)
        g_degree = np.array([result.shape[1]] , np.int32)
        bolb = n_point.tobytes() + g_degree.tobytes() + result.tobytes() 
        print(f'result : {result}')
        with open(savepath , 'wb') as f :
            f.write(bolb)
        print('global graph 文件保存完成')