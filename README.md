# <img src="https://github.com/ZhonggenLi/Garfield/blob/main/Garfield.jpg" width = 10%>Garfield: A GPU-Accelerated Framework for Multi-Attribute Range Filtered Approximate Nearest Neighbor Search 

## Introduction

Garfield is a GPU-accelerated framework for multi-attribute range filtered ANNS that overcomes these bottlenecks by co-designing a lightweight index structure and hardwareaware indexing process. Garfield introduces GMG index, which partitions data into cells and builds a graph index per cell. By adding only a constant number of cross-cell edges, GMG guarantees a predictable, linear storage and indexing overhead. During query processing, Garfield employs a cluster-guided cell-ordering strategy that reorders query-relevant cells, enabling a highly efficient cell-by-cell traversal on the GPU that aggressively reuses current candidates as entry points across cells. For large-scale datasets exceeding GPU memory, Garfield features a cell-oriented out-of-core pipeline. It dynamically schedules cells to minimize the number of active queries per batch and overlaps GPU computation with CPU-to-GPU index streaming.

## Development

We implement our search pipeline using C++ and CUDA. We use CMake to compile and build the project. We compile the project directly using NVIDIA's `nvcc` compiler, without relying on any build system.
For index construction, clustering, and global graph construction, we rely on the Python cuVS library, specifically leveraging `cuvs.neighbors.cagra` and `cuvs.cluster.kmeans` for implementation.
The code for this project is located in the "Garfield" directory. The files in the "Garfeild" directory are organized as follows.

 - indexsearch: It includes `.cu` files for index searching.
 - indexbuild: It includes `.py` files for index building, partitioning and clustering.
 
The steps for building the project are as follows:

```bash
nvcc  -O3  -o  your_executable_file  your_cuda_main_program.cu  -arch=sm_86  -Xcompiler  -fopenmp  --extended-lambda  -w  -Xcompiler  -mcmodel=medium  -Xcompiler  \"-Wl,--no-relax\"  -lcublas
```
The CUDA source file `your_cuda_main_program.cu` is compiled, and the resulting executable is placed in the specified `your_executable_file` directory.
## Examples

All your files and folders are presented as a tree in the file explorer. You can switch from one to another by clicking a file in the tree.

### Intra-cell Index Build

 1. For simulated dataset.
 ```bash
python3 indexbuild/cagra_build.py \
  --pnum 16 \
  --dataset_path ./data/sift.fvecs \
  --dimension 128 \
  --graph_degree 16 \
  --index_save_prefix ./indexes/sift \
  --cagra_index_save_prefix ./indexes/cagra/sift
 ```
 The meanings of the parameters are as follows.
 - pnum: partition number.
 - dataset_path: the file contains all the vector data, stored in `.fvecs`.
 - dimension: dimensions of dataset.
 - graph_degree: degree of graph to build.
 - index_save_prefix: prefix of the Garfield index file name.
 - cagra_index_save_prefix: prefix of the Cagra index file name.

2. For real dataset.
```bash
python3 uniform_partition.py \
  --attrs_filename ./data/attrs.bin \
  --grid_savepath ./output/grid_info.grid \
  --grid_shape 4 4 \
  --dataset_path ./data/sift.fvecs \
  --dimension 128 \
  --graph_degree 32 \
  --index_save_prefix ./indexes/sift \
  --cagra_index_save_prefix ./indexes/cagra/sift
```
 The meanings of the parameters are as follows.
 - attrs_filename: the file contains all the attributes of each points, the first 8 bytes of the file store two `unsigned` variables (`num` for the number of points and `dim` for the attribute dimensionality), followed by `num` consecutive rows, each containing `dim` attributes for a point, stored as `unsigned` integers.
 - grid_savepath: grid infomation for partitioning scheme.
 - grid_shape: a list representing the number of partitions in each attributes dimension.
 - dataset_path: the file contains all the vector data, stored in `.fvecs`.
 - dimension: dimensions of dataset.
 - graph_degree: degree of graph to build.
 - index_save_prefix: prefix of the Garfield index file name.
 - cagra_index_save_prefix: prefix of the Cagra index file name.

 ### Inter-cell Index Build
 ```bash
python3 global_graph_build.py \  
--pnum  16 \  
--dataset_path ./data/sift.fvecs \  
--dimension  128 \  
--graph_degree  16 \  
--cagra_index_save_prefix ./indexes/cagra/sift \  
--edges_num  2 \  
--savepath ./indexes/global_graph.graph32 \  
--searchEf  64
```
The meanings of the parameters are as follows.
-   `pnum`: partition number.
-   `dataset_path`: the file contains all the vector data, stored in `.fvecs`.
-   `dimension`: dimensions of dataset.
-   `graph_degree`: graph degree for each partition.
-   `cagra_index_save_prefix`: prefix of the CAGRA index file name (must be consistent with the one used in `cagra_build.py`).
-   `edges_num`: number of edges for the global graph.
-   `savepath`: path to save the global graph.
-   `searchEf`: search parameter for global graph construction (default: 64).
 
### Clustering
```bash
python3 clustering.py \
  --dataset_path ./data/sift.fvecs \
  --dimension 128 \
  --cluster_num 1000 \
  --cluseter_save_path ./clustering/clustering1000.bin
```
 The meanings of the parameters are as follows.
 - dataset_path: the file contains all the vector data, stored in `.fvecs`.
 - dimension: dimensions of dataset.
 - cluster_num: number of cluster centers.
 - cluseter_save_path: a `.bin` file used to store cluster centers.

### Approximate Nearest Neighbor Search
Index construction and clustering (not required for large-scale datasets) must be completed prior to running the search.
1. For simulated dataset.
```bash
./normaldataset_search \  
--data_path ./data/dataset.fvecs \  
--query_path ./data/query.fvecs \  
--range ./output/range \  
--gt ./output/groundtruth \  
--cagra_prefix ./index/cagra \  
--cagra_suffix .cagra16 \  
--global_graph ./index/global_graph.bin \  
--attrs ./data/attrs.bin \  
--cluster ./index/cluster.bin \  
--result ./output/result \  
--prefilter  0.0 \  
--postfilter  17 \  
--f2h_factor  1.0 \  
--degree  32
```
 The meanings of the parameters are as follows.
-   `data_path`: the file contains all base vectors, stored in `.fvecs`.
-   `query_path`: the file contains all query vectors, stored in `.fvecs`.
-   `range`: the file contains all query ranges, each record includes the lower and upper bounds of all attributes.
-   `gt`: the ground-truth file, stored in `.bin`.
-   `cagra_prefix`: prefix of the CAGRA index file name.
-   `cagra_suffix`: suffix of the CAGRA index file name.
-   `global_graph`: path to the global graph file.
-   `attrs`: the file containing all point attributes.
-   `cluster`: path to the clustering result file.
-   `result`: path to the search result file.
-   `prefilter`: threshold used for pre-filtering.
-   `postfilter`: threshold used for post-filtering.
-   `f2h_factor`: used as a scaling factor for converting `float` values to `__half`.
-   `degree`: graph degree, consistent with the value specified during construction.
2. For real dataset.
```bash
./realdataset_search \
  --data_path ./data/dataset.fvecs \
  --query_path ./data/query.fvecs \
  --range ./output/range \
  --gt ./output/groundtruth \
  --cagra_prefix ./index/cagra \
  --cagra_suffix .cagra32 \
  --global_graph ./index/global_graph.bin \
  --grid ./index/grid_info.bin \
  --attrs ./data/attrs.bin \
  --cluster ./index/cluster.bin \
  --result ./output/result \
  --prefilter 0.5 \
  --postfilter 20 \
  --f2h_factor 1.0 \
  --degree 32
```
 The meanings of the parameters are as follows.
-   `data_path`: the file contains all base vectors, stored in `.fvecs`.
-   `query_path`: the file contains all query vectors, stored in `.fvecs`.
-   `range`: the file contains all query ranges, each record includes the lower and upper bounds of all attributes.
-   `gt`: the ground-truth file, stored in `.bin`.
-   `cagra_prefix`: prefix of the CAGRA index file name.
-   `cagra_suffix`: suffix of the CAGRA index file name.
-   `global_graph`: path to the global graph file.
- `grid`: path to grid information file.
-   `attrs`: the file containing all point attributes.
-   `cluster`: path to the clustering result file.
-   `result`: path to the search result file.
-   `prefilter`: threshold used for pre-filtering.
-   `postfilter`: threshold used for post-filtering.
-   `f2h_factor`: used as a scaling factor for converting `float` values to `__half`.
-   `degree`: graph degree, consistent with the value specified during construction.

## Datasets
Each dataset can be obtained from the following links.
| Dataset | Cardinality | Dimensionality |Attributes| Link                                          |
| ------- | ----------- | -------------- | --------------|------------------------------- |
| Deep1M   | 1,000,000  | 96 |Uniform random |https://ieeexplore.ieee.org/document/7780595                       |
| SIFT1M   | 1,000,000  | 128  |Uniform random |http://corpus-texmex.irisa.fr/ |
| DBLP  | 1,000,000     | 768  |Year, #authors,#references, #citations |https://open.aminer.cn/open/article?id=655db2202ab17a072284bc0c    |
| YouTube     | 1,000,000   | 1024|Year, date, #views, #likes |https://research.google.com/youtube8m/download.html           |
| Deep100M   | 100,000,000   | 96 | Uniform random|https://ieeexplore.ieee.org/document/7780595                     |
| SIFT100M   | 100,000,000   | 128|Uniform random |https://big-ann-benchmarks.com/neurips21.html                |


