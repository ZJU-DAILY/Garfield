import numpy as np
import cupy as cp
from cuvs.neighbors import cagra
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
        "--index_save_prefix", type=str, required=True, help="Index Save Prefix"
    )
    parser.add_argument("--cagra_index_save_prefix", type=str, required=True, help="cagra Index Save Prefix (for global graph build)")
    args = parser.parse_args()
    
    (
        pnum , 
        file_path , 
        preset_d , 
        degree ,
        index_save_prefix , 
        cagra_index_save_prefix
    ) = (
        args.pnum , 
        args.dataset_path , 
        args.dimension , 
        args.graph_degree , 
        args.index_save_prefix , 
        args.cagra_index_save_prefix
    )
    
    # pnum = 36
    # 示例：读取 fvecs 文件
    # file_path = '/home/yhr/workspace/cuda_clustering/data/sift_base.fvecs'
    saveFlag = True
    cagraSave = True
    graphSave = True
    performanceSave = False
    # datasetName = 'deep100M'
    # file_path = f'/home/yhr/workspace/2D_range_filter/from_lzg/dataset/deep100M/deep100M_base.fvecs'
    # performance_savepath = f'/home/yhr/workspace/2D_range_filter/from_lzg/experiments/ablation/{datasetName}/results/buildcost.txt'
    performance_savepath = ''
    # dim , num , xb = read_binary_file(file_path)
    dim , num , xb = read_fvecs(file_path , preset_d)
    print(xb.shape)
    # dataset = cp.asarray(xb)

    k = 10
    # dataset = cp.random.random_sample((n_samples, n_features),dtype=cp.float32)
    build_params = cagra.IndexParams(metric="sqeuclidean" , graph_degree=degree , build_algo = 'nn_descent')
    # index = cagra.build(build_params, dataset)
    # pnum = 36

    average_num = int((num + pnum - 1) / pnum)

    print(f"每个分区内点的数量: {average_num}")
    pure_build_cost = 0.0 
    totals = time.perf_counter()  
    for i in range(0 , pnum) :
        real_load = average_num if i < pnum - 1 else (num - i * average_num)
        dataseti = cp.asarray(xb[i * average_num : i * average_num + real_load , :])
        print(f"{i} : {dataseti.shape}")
        s = time.perf_counter() 
        index = cagra.build(build_params , dataseti)
        e = time.perf_counter() 
        pure_build_cost += (e - s)
        print(f"finish {i} , 构建用时: {e - s}s")
        # 保存完整索引
        if saveFlag :
            full_index_savepath = f'{cagra_index_save_prefix}{i}.cagra{degree}' 
            # 写入格式, 每行前4个字节存k, 其余k * 4 个字节存号
            if cagraSave == True :
                cagra.save(full_index_savepath , index) 
        

            if graphSave == True : 
                savepath = f'{index_save_prefix}{i}.cagra{degree}'
                # 保存自定义格式索引
                n_point = np.array([real_load] , dtype = np.int32)
                g_degree = np.array([index.graph_degree] , dtype = np.int32)
                blob = n_point.tobytes() + g_degree.tobytes() + cp.asarray(index.graph).tobytes() 
                with open(savepath , "wb") as file :
                    file.write(blob)
                print(f"{n_point} , {g_degree}")
        del dataseti 
        del index
        # del blob
    totale = time.perf_counter() 
    print(f'总用时 :  {totale - totals}s')
    print(f"纯构建用时 : {pure_build_cost}")

    if performanceSave == True :
        with open(performance_savepath , 'a' , encoding = 'utf-8') as file :
            file.write(f"cagra M : {16} , pnum : {pnum} , N : {num} , dim : {dim}\nbuildcost : {pure_build_cost}\n")
        