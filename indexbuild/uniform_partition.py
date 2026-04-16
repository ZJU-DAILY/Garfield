from collections import defaultdict
from bisect import bisect_right
import math
import random
import numpy as np
import os

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

def read_attrs_safe(path):
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        num, dim = header

        expected_size = 8 + num * dim * 4
        actual_size = os.path.getsize(path)

        if actual_size != expected_size:
            raise ValueError("文件大小不匹配")

        data = np.fromfile(f, dtype=np.int32)

    return data.reshape(num, dim)


def read_grid_info(filepath):
    """
    读取指定格式的二进制文件。

    文件格式：
    1. 前两个 int32:
        - N
        - dim
    2. 接着 dim 个 int32:
        - dSize[0:dim]
    3. 接着 dim 个维度块，每个维度块包含：
        - l_quantiles: dSize[i] 个 int32
        - r_quantiles: dSize[i] 个 int32
        - pid_list:    N 个 int32
    4. 最后：
        - global_pid: N 个 int32

    返回：
        {
            "N": int,
            "dim": int,
            "dSize": np.ndarray,                 # shape (dim,)
            "dims": [
                {
                    "l_quantiles": np.ndarray,   # shape (dSize[i],)
                    "r_quantiles": np.ndarray,   # shape (dSize[i],)
                    "pid_list": np.ndarray       # shape (N,)
                },
                ...
            ],
            "global_pid": np.ndarray             # shape (N,)
        }
    """
    with open(filepath, "rb") as f:
        # 1. 读取 N 和 dim
        head = np.fromfile(f, dtype=np.int32, count=2)
        if head.size < 2:
            raise ValueError("文件头不足，无法读取 N 和 dim。")

        N, dim = map(int, head)

        # 2. 读取 dSize
        dSize = np.fromfile(f, dtype=np.int32, count=dim)
        if dSize.size < dim:
            raise ValueError("文件内容不足，无法读取完整 dSize。")

        dims_data = []

        # 3. 逐维读取
        for i in range(dim):
            size_i = int(dSize[i])

            l_quantiles = np.fromfile(f, dtype=np.int32, count=size_i)
            if l_quantiles.size < size_i:
                raise ValueError(f"第 {i} 维: l_quantiles 数据不足。")

            r_quantiles = np.fromfile(f, dtype=np.int32, count=size_i)
            if r_quantiles.size < size_i:
                raise ValueError(f"第 {i} 维: r_quantiles 数据不足。")

            pid_list = np.fromfile(f, dtype=np.int32, count=N)
            if pid_list.size < N:
                raise ValueError(f"第 {i} 维: pid_list 数据不足。")

            dims_data.append({
                "l_quantiles": l_quantiles,
                "r_quantiles": r_quantiles,
                "pid_list": pid_list,
            })

        # 4. 读取 global_pid
        global_pid = np.fromfile(f, dtype=np.int32, count=N)
        if global_pid.size < N:
            raise ValueError("global_pid 数据不足。")

        # 可选：检查文件是否正好读完
        remain = np.fromfile(f, dtype=np.int32)
        if remain.size > 0:
            print(f"警告：文件末尾还有 {remain.size} 个 int32 未读取。")

    return {
        "N": N,
        "dim": dim,
        "dSize": dSize,
        "dims": dims_data,
        "global_pid": global_pid,
    }

def partition_ordered_contiguous(
    objects,
    num_partitions=16,
    value_key="value",
    id_key="id",
    allow_extreme_split=True,
    extreme_split_threshold=None,
):
    """
    将对象按数值属性划分为有序连续分区（0..num_partitions-1）。

    修正版策略：
    - 默认不拆同值；
    - 若切点落在同值段中间，且该段大小 >= extreme_split_threshold，
      则优先允许在段内切分（前提 allow_extreme_split=True）；
    - 这样可主动拆分超大同值组（例如某值出现 200000 > target_size）。

    返回:
        assignment: dict[obj_id] = partition_id
        report: dict
    """
    n = len(objects)
    if n == 0:
        return {}, {
            "num_objects": 0,
            "num_partitions": num_partitions,
            "target_size": 0,
            "partition_sizes": [0] * num_partitions,
            "max_size": 0,
            "min_size": 0,
            "size_range": 0,
            "std_dev": 0.0,
            "split_values_count": 0,
            "split_values": {},
            "partition_value_ranges": [],
            "cut_positions": [],
        }

    sorted_objs = sorted(objects, key=lambda x: x[value_key])

    target_size = n / num_partitions
    if extreme_split_threshold is None:
        extreme_split_threshold = 2*math.ceil(target_size)

    # 构建同值 run: (value, start, end, count)
    runs = []
    i = 0
    while i < n:
        v = sorted_objs[i][value_key]
        j = i + 1
        while j < n and sorted_objs[j][value_key] == v:
            j += 1
        runs.append((v, i, j, j - i))
        i = j

    run_starts = [r[1] for r in runs]

    cut_positions = []
    prev_cut = 0

    for p in range(1, num_partitions):
        ideal = round(p * n / num_partitions)
        min_remaining = num_partitions - p

        # 保证切点合法
        ideal = max(prev_cut + 1, min(ideal, n - min_remaining))

        # 找到 ideal 所在 run
        idx = bisect_right(run_starts, ideal) - 1
        idx = max(0, min(idx, len(runs) - 1))
        v, s, e, c = runs[idx]

        if s < ideal < e:
            # 落在同值段内部
            left_cut, right_cut = s, e

            left_ok = (left_cut > prev_cut) and (n - left_cut >= min_remaining)
            right_ok = (right_cut > prev_cut) and (n - right_cut >= min_remaining)

            # 关键改动：超大组优先允许内部切分
            if allow_extreme_split and c >= extreme_split_threshold:
                cut = ideal
            else:
                # 不拆优先：选最近合法边界
                candidates = []
                if left_ok:
                    candidates.append((abs(left_cut - ideal), left_cut))
                if right_ok:
                    candidates.append((abs(right_cut - ideal), right_cut))

                if candidates:
                    candidates.sort(key=lambda x: (x[0], x[1]))
                    cut = candidates[0][1]
                else:
                    if not allow_extreme_split:
                        raise ValueError(
                            f"分区 {p} 无法不拆同值并保持切点合法。"
                        )
                    cut = ideal
        else:
            cut = ideal

        # 再次防御式约束
        if cut <= prev_cut:
            cut = prev_cut + 1
        if n - cut < min_remaining:
            cut = n - min_remaining

        cut_positions.append(cut)
        prev_cut = cut

    boundaries = [0] + cut_positions + [n]

    assignment = {}
    partition_sizes = [0] * num_partitions
    partition_ranges = []

    for pid in range(num_partitions):
        l, r = boundaries[pid], boundaries[pid + 1]
        partition_sizes[pid] = r - l

        if l < r:
            min_v = sorted_objs[l][value_key]
            max_v = sorted_objs[r - 1][value_key]
        else:
            min_v = max_v = None
        partition_ranges.append((min_v, max_v))

        for k in range(l, r):
            assignment[sorted_objs[k][id_key]] = pid

    # 统计同值被分到多个分区
    value_partitions = defaultdict(set)
    for obj in sorted_objs:
        value_partitions[obj[value_key]].add(assignment[obj[id_key]])

    split_values_exact = {
        v: sorted(list(pids))
        for v, pids in value_partitions.items()
        if len(pids) > 1
    }

    mean_size = n / num_partitions
    var = sum((x - mean_size) ** 2 for x in partition_sizes) / num_partitions
    std_dev = var ** 0.5

    report = {
        "num_objects": n,
        "num_partitions": num_partitions,
        "target_size": target_size,
        "partition_sizes": partition_sizes,
        "max_size": max(partition_sizes),
        "min_size": min(partition_sizes),
        "size_range": max(partition_sizes) - min(partition_sizes),
        "std_dev": std_dev,
        "cut_positions": cut_positions,
        "split_values_count": len(split_values_exact),
        "split_values": split_values_exact,      # value -> [partition_ids]
        "partition_value_ranges": partition_ranges
    }

    return assignment, report

def grid_save(savepath , p_infos , global_pid , grid) :
    # 先写grid
    bolb = np.array([len(global_pid)] , np.int32).tobytes()
    bolb += np.array([len(grid)] , np.int32).tobytes() 
    bolb += grid.tobytes() 
    # 循环写入p_infos
    for l_quantiles , r_quantiles , pid_list in p_infos :
        bolb += l_quantiles.tobytes() 
        bolb += r_quantiles.tobytes() 
        bolb += pid_list.tobytes() 
    bolb += global_pid.tobytes() 
    
    with open(savepath , 'wb') as f :
        f.write(bolb)
    print(f'文件写入完成: {savepath}')
    
def grid_gen(attrs , d , dSize) :
    # data = []
    p_infos = []
    
    for i in range(d) :
        values , counts = np.unique(attrs[: , i] , return_counts=True)
        oid = 0
        data = []
        for v, cnt in zip(values.tolist() , counts.tolist()) :
            for _ in range(cnt):
                data.append({"id": oid, "value": v})
                oid += 1

        random.shuffle(data)

        assignment, report = partition_ordered_contiguous(
            data,
            num_partitions=dSize[i].item(),
            value_key="value",
            id_key="id",
            allow_extreme_split=True,
            extreme_split_threshold=None,  # 默认 target_size
        )
        
        
        print("target_size:", report["target_size"])
        print("partition_sizes:", report["partition_sizes"])
        print("split_values_count:", report["split_values_count"])
        print("value=10 partitions:", report["split_values"].get(10, []))

        # if i == 0 : 
        #     report["partition_value_ranges"][4] = (2010 , 2010)

        # 新增：输出每个分区的属性区间
        print("\npartition value ranges:")
        for pid, (vmin, vmax) in enumerate(report["partition_value_ranges"]):
            size = report["partition_sizes"][pid]
            print(f"P{pid:02d}: size={size}, value_range=[{vmin}, {vmax}]")


    
        print("\npartition index boundaries:")
        boundaries = [0] + report["cut_positions"] + [report["num_objects"]]
        
        # if i == 0 :
            
        #         #    if i == 0 :
        #     boundaries[5] = 345001
        #     boundaries[4] = 287369
        
        for pid in range(report["num_partitions"]):
            l, r = boundaries[pid], boundaries[pid + 1]
            print(f"P{pid:02d}: index_range=[{l}, {r})")
        
 
        
        # global_pid = np.zeros([1000000] , np.int32) 
        # attrs = read_attrs_safe('/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/attributes/year.attrs')
        # print(f'attrs : {attrs}')
        local_pid = np.zeros([1000000] , np.int32)
        indices = np.argsort(attrs[: , i] , kind='stable')
        print(f"indices : {indices}")
        for pid in range(report["num_partitions"]):
            l, r = boundaries[pid], boundaries[pid + 1]
            # print(f"P{pid:02d}: index_range=[{l}, {r})")
            local_pid[indices[l : r]] = pid 
        print(local_pid)
        unique_vals, counts = np.unique(local_pid, return_counts=True)
        print(f'd : {i} , local pid show : ')
        print(list(zip(unique_vals.tolist() , counts.tolist())))
        
        print(report["partition_value_ranges"])
        # 保存边界, 映射关系
        # grid = np.array([16] , np.int32)
        # p_info = []
        l_b = np.zeros([dSize[i].item()] , np.int32)
        r_b = np.zeros([dSize[i].item()] , np.int32)
        for pid, (vmin, vmax) in enumerate(report["partition_value_ranges"]):
            # size = report["partition_sizes"][pid]
            # print(f"P{pid:02d}: size={size}, value_range=[{vmin}, {vmax}]")
            l_b[pid] = vmin 
            r_b[pid] = vmax
        p_infos.append((l_b , r_b , local_pid))
    # globla_pid = np.zeros([1000000] , np.int32)
    factors = np.cumprod(dSize[::-1])
    factors = np.concatenate((factors[::-1][1:] , [1]))
    print(f'factors : {factors}')
    def find_pid(idx) :
        pid = 0
        # factor = 1
        for i in range(d) :
            pid += factors[i] * p_infos[i][2][idx]
        return pid 

    global_pid = list(map(find_pid , range(len(attrs))))
    global_pid = np.array(global_pid , np.int32) 

    print('show globla pid info : ')
    values , counts = np.unique(global_pid , return_counts=True)
    print(list(zip(values.tolist() , counts.tolist())))
    
    return p_infos , global_pid 

from tools import grid_scatter_V2

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Index parameters")
    parser.add_argument("--attrs_filename" , required=True , type=str , help='file stored attributes')
    parser.add_argument("--grid_savepath" , required=True , type=str , help='file stored Grid info')
    parser.add_argument("--grid_shape" , required=True , type=int , nargs="+" ,help='Grid shape for index')
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
    parser.add_argument(
        "--cagra_index_save_prefix", type=str, required=True, help="cagra Index Save Prefix (for global graph build)"
    )
    

    args = parser.parse_args()
    (   
        attrs_filename , 
        grid_savepath , 
        grid_shape , 
        file_path , 
        preset_d , 
        degree ,
        index_save_prefix ,
        cagra_index_save_prefix 
    ) = (
        args.attrs_filename , 
        args.grid_savepath , 
        args.grid_shape , 
        args.dataset_path , 
        args.dimension , 
        args.graph_degree ,
        args.index_save_prefix ,
        args.cagra_index_save_prefix 
    )
    
    attrs = read_attrs_safe(attrs_filename)
    print(f'attrs.shape : {attrs.shape}')
    print(f'len(attrs) : {len(attrs)}')
    D = attrs.shape[1]
    grid = np.array(grid_shape , np.int32)
    # p_info , global_pid = grid_gen(attrs , D , grid)

    
    p_info , global_pid = grid_scatter_V2(attrs , grid)



    # grid_info = read_grid_info(f"/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/dblp/degree32/{D}DV2/{D}D.grid")

    # print("N =", grid_info["N"])
    # print("dim =", grid_info["dim"])
    # print("dSize =", grid_info["dSize"])
    # print("global_pid.shape =", grid_info["global_pid"].shape)

    # for i, item in enumerate(grid_info["dims"]):
    #     print(f"--- 第 {i} 维 ---")
    #     print("l_quantiles:", item["l_quantiles"].shape)
    #     print("r_quantiles:", item["r_quantiles"].shape)
    #     print("pid_list:", item["pid_list"].shape)
        
    # global_pid = grid_info['global_pid']
    # p_info = grid_info['dims']
    # grid = grid_info['dSize']
    g_vals , g_cnts = np.unique(global_pid , return_counts=True)
    print(list(zip(g_vals.tolist() , g_cnts.tolist())))
    pnum = np.prod(grid)
    # input() 
    grid_save(grid_savepath , p_info , global_pid , grid)

    
    # 根据global id 的映射建图
    savePureIndex = False
    saveCagraIndex = False
    savePerformance = True
    # datasetName = 'dblp'
    # file_path = f'/home/yhr/workspace/2D_range_filter/from_lzg/dataset/DBLP/embeddings_output/dblp1M.fvecs'
    # performance_savepath = f'/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/results/smart-new/{D}D/buildcost.txt'
    dim , num , xb = read_fvecs(file_path , preset_d)
    # num = 1000000
    # xb = xb[-num: , :]
    print(xb.shape)
    # dataset = cp.asarray(xb)

    k = 10
    # dataset = cp.random.random_sample((n_samples, n_features),dtype=cp.float32)
    build_params = cagra.IndexParams(metric="sqeuclidean" , graph_degree=degree , build_algo = 'nn_descent' , intermediate_graph_degree=96, nn_descent_niter=50)
    # index = cagra.build(build_params, dataset)
    # pnum = 36
    # pnum = 16
    # average_num = int((num + pnum - 1) / pnum)

    # print(f"每个分区内点的数量: {average_num}")
    pure_build_cost = 0.0 
    totals = time.perf_counter()  
    saveFlag = True
    # i = 0 
    for i in range(pnum) :
    # for i in range(1) :
        # real_load = average_num if i < pnum - 1 else (num - i * average_num)
        maski = (global_pid == i)
        dataseti = cp.asarray(xb[maski])
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
            if saveCagraIndex == True :
                cagra.save(full_index_savepath , index) 
            savepath = f'{index_save_prefix}{i}.cagra{degree}'
            # 保存自定义格式索引
            if savePureIndex == True :
                n_point = np.array([dataseti.shape[0]] , dtype = np.int32)
                g_degree = np.array([index.graph_degree] , dtype = np.int32)
                blob = n_point.tobytes() + g_degree.tobytes() + cp.asarray(index.graph).tobytes() 
                with open(savepath , "wb") as file :
                    file.write(blob)
                print(f"{n_point} , {g_degree}")
        del dataseti 
        del index
        # i += 1
        # del blob
    totale = time.perf_counter() 
    print(f'总用时 :  {totale - totals}s')
    print(f"纯构建用时 : {pure_build_cost}")

    # if savePerformance == True :
    #     with open(performance_savepath , 'a' , encoding = 'utf-8') as file :
    #         file.write(f"cagra M : {32} , pnum : {pnum} , N : {num} , dim : {dim}\nbuildcost : {pure_build_cost}\n")