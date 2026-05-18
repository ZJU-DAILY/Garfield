import numpy as np
import os
import cupy as cp
from cuvs.neighbors import cagra
import time 

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


# 左界和右界都保存, 按闭区间存储
def grid_scatter(attrs , grid) :
    num , dim = attrs.shape
    attrDim = len(grid)
    # quantiles = np.zeros([grid[0] - 1] , dtype = np.int32)
    #对每一维度单独处理
    # for i in range(dim) :
    #     grid_i = grid[i] - 1
    #     quantiles = np.percentile(attrs[:,i] , np.linspace(0 , 100 , grid_i))
    #     print(quantiles)
    # 排序, 直接在分位点的位置取元素, 区间范围为左闭右闭
    attr1 = attrs[:, 0]
    sorted_attr1 = np.sort(attr1)
    indices = np.argsort(attr1 , kind = 'stable')
    print(f'indices : {indices}')
    print(f'indices.shape : {indices.shape}')
    avg_num = int((num + grid[0].item() - 1) / grid[0].item())
    quantiles = []
    quantiles_left = []
    for i in range(1 , grid[0].item() + 1) :
        quantiles.append(int(avg_num * i if avg_num * i < num else num))
        quantiles_left.append(int(avg_num * (i - 1)))
    
    for i in range(len(quantiles)) :
        quantiles[i] -= 1 
    print(quantiles_left)
    print(quantiles)
    # quantiles -= 1
    pid_list = np.zeros([num] , np.int32)
    for i in range(16) :
        start = i * avg_num 
        end = (i + 1) * avg_num if i < grid[0].item() - 1 else num
        print(f'start : {start} , end : {end}')
        print(f'indices[start:end] : {indices[start:end]}')
        pid_list[indices[start:end]] = i  

    print(pid_list)
    return sorted_attr1[np.array(quantiles_left , dtype = np.int32)] , sorted_attr1[np.array(quantiles , dtype = np.int32)] , pid_list  

# file_path = f'/home/yhr/workspace/2D_range_filter/from_lzg/dataset/DBLP/embeddings_output/paper_title_abstract.fvecs'
# dim , num , xb = read_fvecs(file_path , 128)
# print(xb.shape)
# d = 1

def grid_scatter_V2(attrs , grid) :
    num = attrs.shape[0]
    dim = len(grid)
    
    # 返回元组列表, 列表中的每个元组对应当前维度的划分方案, (下界, 上界, 当前维度的id)
    # (A , B) , A为上述元组列表, B为全局分区号
    p_infos = []
    # 处理每一个分区
    for i in range(dim) :
        attrs_i = attrs[: , i]
        sorted_attr1 = np.sort(attrs_i)
        indices = np.argsort(attrs_i , kind = 'stable')
        grid_size = grid[i].item() 
        avg_num = int((num + grid_size - 1) / grid_size)
        quantiles_left = [j * avg_num for j in range(grid_size)]
        quantiles_right = [((j + 1) * avg_num - 1) if (j < grid_size - 1) else (num - 1) for j in range(grid_size)]
        idx_list = np.zeros([num] , np.int32)
        for j in range(grid_size) :
            start = avg_num * j 
            end = avg_num * (j + 1) if j < grid_size - 1 else num 
            idx_list[indices[start:end]] = j 
        p_infos.append((sorted_attr1[quantiles_left] , sorted_attr1[quantiles_right] , idx_list))
        
    # global_pid = np.zeros([num] , np.int32) 
    factors = np.cumprod(grid[::-1])
    factors = np.concatenate((factors[::-1][1:] , [1]))
    print(f'factors : {factors}')
    def find_pid(idx) :
        pid = 0
        # factor = 1
        for i in range(dim) :
            pid += factors[i] * p_infos[i][2][idx]
        return pid 

    global_pid = list(map(find_pid , range(num)))
    global_pid = np.array(global_pid , np.int32) 
    
    return p_infos , global_pid

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


# import numpy as np

def split_sorted_array_dp(arr, S):
    arr = np.asarray(arr)
    N = len(arr)
    if N == 0 or S <= 0:
        return []

    unique_vals, counts = np.unique(arr, return_counts=True)
    
    print(list(zip(unique_vals.tolist(),  counts.tolist())))
    M = len(counts)

    if S >= M:
        intervals = []
        start = 0
        for c in counts:
            intervals.append((start, start + c - 1))
            start += c
        return intervals

    target = N / S
    prefix = np.zeros(M + 1, dtype=np.int64)
    prefix[1:] = np.cumsum(counts)

    def seg_cost(j, i):
        # 块 j 到 i-1
        seg_sum = prefix[i] - prefix[j]
        return (seg_sum - target) ** 2

    dp = np.full((M + 1, S + 1), np.inf)
    prev = np.full((M + 1, S + 1), -1, dtype=int)
    dp[0][0] = 0

    for i in range(1, M + 1):
        for k in range(1, min(i, S) + 1):
            for j in range(k - 1, i):
                val = dp[j][k - 1] + seg_cost(j, i)
                if val < dp[i][k]:
                    dp[i][k] = val
                    prev[i][k] = j

    # 回溯块区间
    cuts = []
    i, k = M, S
    while k > 0:
        j = prev[i][k]
        cuts.append((j, i))
        i, k = j, k - 1
    cuts.reverse()

    # 转成元素下标区间
    intervals = []
    for bstart, bend in cuts:
        l = prefix[bstart]
        r = prefix[bend] - 1
        intervals.append((int(l), int(r)))
    
    print(intervals)
    intervals_len = np.zeros([S] , np.int32)
    for i in range(S) :
        intervals_len[i] = intervals[i][1] - intervals[i][0] + 1
    print(f"intervals len : {intervals_len}")

    return intervals

# 有bug, 作废
def split_sorted_array_into_s_intervals(arr, S):
    """
    将有序数组 arr 分成 S 个连续区间：
    - 每个区间元素数尽量接近
    - 相同元素必须在同一区间内

    返回：
        intervals: [(l1, r1), (l2, r2), ...]   下标闭区间
    """
    arr = np.asarray(arr)
    N = len(arr)
    if N == 0 or S <= 0:
        return []

    # 最多只能按“不同值块数”来分
    unique_vals, counts = np.unique(arr, return_counts=True)
    
    for val , cnt in zip(unique_vals , counts) :
        print(f'{val} : {cnt}')
    M = len(counts)

    if S >= M:
        # 每个值块单独成段即可，最多只能有 M 段
        intervals = []
        start = 0
        for c in counts:
            intervals.append((start, start + c - 1))
            start += c
        return intervals

    target = N / S

    intervals = []
    block_start = 0       # 当前段从第几个块开始
    elem_start = 0        # 当前段起始元素下标
    curr_sum = 0
    used_blocks_prefix = np.cumsum(counts)

    i = 0
    remaining_segments = S

    while i < M and remaining_segments > 1:
        remaining_blocks = M - i
        if remaining_blocks == remaining_segments:
            # 后面每个块必须单独成段
            break

        curr_sum += counts[i]

        # 看看当前是否适合切
        if curr_sum >= target:
            prev_sum = curr_sum - counts[i]

            # 比较：在 i 前切，还是在 i 处切
            if abs(prev_sum - target) <= abs(curr_sum - target) and i > block_start:
                # 在 i-1 结束当前段
                end_block = i - 1
                seg_size = prev_sum
                elem_end = elem_start + seg_size - 1
                intervals.append((elem_start, elem_end))

                elem_start = elem_end + 1
                block_start = i
                curr_sum = counts[i]
            else:
                # 在 i 结束当前段
                end_block = i
                seg_size = curr_sum
                elem_end = elem_start + seg_size - 1
                intervals.append((elem_start, elem_end))

                elem_start = elem_end + 1
                block_start = i + 1
                curr_sum = 0
                i += 1

            remaining_segments -= 1

            # 重新计算后续目标长度
            remaining_N = N - elem_start
            if remaining_segments > 0:
                target = remaining_N / remaining_segments
            continue

        i += 1

    # 剩余部分作为最后若干段
    remaining_counts = counts[block_start:]
    remaining_blocks = len(remaining_counts)

    if remaining_segments == 1:
        intervals.append((elem_start, N - 1))
    else:
        # 后面每块单独一段
        pos = elem_start
        for c in remaining_counts:
            intervals.append((pos, pos + c - 1))
            pos += c

    print(f"intervals : {intervals}")
    intervals_len = np.zeros([S] , np.int32)
    for i in range(S) :
        intervals_len[i] = intervals[i][1] - intervals[i][0] + 1
    print(f"intervals len : {intervals_len}")
    return intervals


def run() :
    # 载入几个bin文件
    attrs = read_attrs_safe('/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/attributes/year.attrs')
    gridShape = np.array([16], dtype=np.int32)
    p_infos , global_pid = grid_scatter_V2(attrs , gridShape)
    l_quantiles ,r_quantiles, pid_list = p_infos[0]
    print(l_quantiles)
    print(r_quantiles)
    print(pid_list)

    cnt = {}
    for i in range(16) :
        cnt[i] = 0

    for i in pid_list :
        cnt[i] += 1 
    print(cnt)
    
    print(global_pid)
    print(global_pid == pid_list)
    saveFlag = True
    
    
    if saveFlag == True :
        grid_save('../data/grid/dblp1D_n_citation.grid', p_infos , global_pid , gridShape)
    
    # input()
    
    datasetName = 'dblp'
    file_path = f'/home/yhr/workspace/2D_range_filter/from_lzg/dataset/DBLP/embeddings_output/paper_title_abstract.fvecs'
    performance_savepath = f'/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/{datasetName}/results/smart/buildcost.txt'
    dim , num , xb = read_fvecs(file_path , 768)
    num = 1000000
    xb = xb[-num: , :]
    print(xb.shape)
    # dataset = cp.asarray(xb)

    k = 10
    # dataset = cp.random.random_sample((n_samples, n_features),dtype=cp.float32)
    build_params = cagra.IndexParams(metric="sqeuclidean" , graph_degree=16 , build_algo = 'nn_descent')
    # index = cagra.build(build_params, dataset)
    # pnum = 36
    # pnum = 16
    # average_num = int((num + pnum - 1) / pnum)
    pnum = np.prod(gridShape)
    # print(f"每个分区内点的数量: {average_num}")
    pure_build_cost = 0.0 
    totals = time.perf_counter()  
    # i = 0 
    for i in range(pnum) :
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
            full_index_savepath = f'/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/cuvs-indexes/dblp_n_citation{i}.cagra16' 
            # 写入格式, 每行前4个字节存k, 其余k * 4 个字节存号
            cagra.save(full_index_savepath , index) 
        
        
            savepath = f'/home/yhr/workspace/2D_range_filter/from_lzg/gpu_index/py_tools/indexes/dblp_n_citation{i}.cagra16'
            # 保存自定义格式索引
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

    with open(performance_savepath , 'a' , encoding = 'utf-8') as file :
        file.write(f"cagra M : {16} , pnum : {pnum} , N : {num} , dim : {dim}\nbuildcost : {pure_build_cost}\n")
# cnt = {}
# for i in range(len(quantiles) + 1) :
#     cnt[i] = 0 
# cnt[len(quantiles) + 1] = 0 
# for i in attrs :
#     index = np.searchsorted(quantiles , i , side = 'right')[0]
#     # print(index)
#     index = index.item()
#     cnt[index] += 1
#     if i[0].item() > 2018 : 
#         print(i) 
    
# print(cnt)

if __name__ == '__main__' :
    attrs = read_attrs_safe('/home/yhr/workspace/2D_range_filter/from_lzg/experiments/datasets/dblp/attributes/year.attrs')
    
    # split_sorted_array_into_s_intervals(np.sort(attrs , kind='stable'), 16)
    # 把索引映射回属性
    sorted_list = np.sort(attrs , kind='stable')
    indices = np.argsort(attrs , kind='stable')
    intervals = split_sorted_array_dp(np.sort(attrs , kind='stable'), 16)
    N = len(attrs) 
    print(f'N : {N}')
    global_idx = np.zeros([N] , np.int32)
    
    # for i in range(16) :
    #     global_idx[indices[]]