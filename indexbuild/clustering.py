import cupy as cp
import numpy as np
from cuvs.cluster.kmeans import fit, KMeansParams, predict
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


if __name__ == '__main__' :
    
    parser = argparse.ArgumentParser(description="Index parameters")
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the fvecs data file"
    )
    parser.add_argument(
        "--dimension" , type = int , required=True , help="dimension for dataset"
    )
    parser.add_argument(
        "--cluster_num" , type = int , required=True , help="cluster number"
    )
    parser.add_argument(
        "--cluseter_save_path", type=str, required=True, help="Path for cluster saving"
    )
    args = parser.parse_args()
    (
        file_path , 
        preset_d , 
        cnum , 
        savepath
    ) = (
        args.dataset_path , 
        args.dimension , 
        args.cluster_num , 
        args.cluseter_save_path 
    )
    
    datasetName = 'youtube'
    # 示例：读取 fvecs 文件
    # file_path = f'/home/yhr/workspace/2D_range_filter/from_lzg/dataset/YouTube/youtube1m/rgb.fvecs'
    # result_save_path = f'/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/{datasetName}/results/smart/buildcost.txt'
    saveFlag =  True
    dim , num , xb = read_fvecs(file_path, preset_d)
    print(xb.shape)
    X = cp.asarray(xb)

    s = time.perf_counter()
    params = KMeansParams(n_clusters=cnum)
    centroids, inertia, n_iter = fit(params, X)
    labels, inertia = predict(params, X, centroids)
    e = time.perf_counter() 

    labels = cp.asnumpy(labels)
    centroids = cp.asnumpy(centroids)

    print(f"inertia : {inertia}")
    print(f"n_iter : {n_iter}")
    print(f"centroids.shape : {centroids.shape}")
    print(f"centroids : {centroids}")

    print(f"聚类时间: {e - s}")
    print(labels)

    if saveFlag == True :
        n = np.array([labels.size] , dtype=np.int32)
        blob = n.tobytes() + labels.tobytes() 

        with open(savepath , "wb") as f :
            f.write(blob)
        print(f'centroids.dtype : {centroids.dtype}')
        
        
        # byte_list = []
        # for row in centroids :
        #     nd = np.array([dim] , dtype = np.int32)
        #     byte_list.append(nd.tobytes() + row.tobytes())
        # centroids_bytes = b''.join(byte_list)
        # with open("/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/clustering/youtube_clustering1000.fvecs" , 'wb') as f :
        #     f.write(centroids_bytes)
        # print("中心点, 文件保存完成")
            
    # with open(result_save_path , 'a' , encoding = 'utf-8') as file : 
    #     file.write(f"聚类参数, n_clusters = 1000, 聚类时间: {e - s}\n")
        
    print("全部文件保存完成")