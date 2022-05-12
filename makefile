build_rapids_env:
	scripts/install_conda.sh

build_torch_env:
	scripts/torch_conda_env.sh

build_python:
	python3 setup.py install --user

build_cython: build_python
	python3 setup_cython.py install #https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/

build_cuda_code: build_rapids_env build_python build_cython
	/usr/local/cuda-11.5/bin/nvcc --shared -o libgpu_dim_reduction.so \
		cython/cuda_wrappers/gpu_dim_reduction.cpp \
		cython/cuda_kernels/gpu_kernels.cu \
		cython/utils/gpu_utils.cu \
		-Xcompiler -fPIC
	/usr/local/cuda-11.5/bin/nvcc --shared -o libgpu_graph_weights.so \
	   	cython/cuda_wrappers/gpu_graph_weights.cpp \
		cython/cuda_kernels/gpu_graph_cuda.cu \
		cython/utils/gpu_utils.cu \
		cython/utils/util.cpp \
		cython/utils/mem_util.cpp \
		-Xcompiler -fPIC
	python3 setup_cython_gpu.py install

# Create test across datasets
# Create test across parameters
# Create make targets that will produce all plots

run_cpu_test: build_python build_cython
	# FIXME -- make into a unit test
	# Basic test to make sure that every algorithm can be run on CPU
	### GIDR_DUN
	# Basic gidr_dun implementation
	python3 dim_reduce_dataset.py --num-points 6000
	# Gidr_dun for obtaining TSNE outputs
	python3 dim_reduce_dataset.py --optimize-method gidr_dun --num-points 6000 --normalized --momentum
	# Gidr_dun running on numba -- this is faster on large distributed systems than cython
	python3 dim_reduce_dataset.py --num-points 6000 --numba
	### UMAP
	# Run the original UMAP algorithm that gets installed with `pip install umap-learn`
	python3 dim_reduce_dataset.py --num-points 6000 --dr-algorithm umap
	# Run OUR implementation of UMAP in cython
	# 	- `dr-algorithm` means that we run the GIDR_DUN implementations
	# 	- `optimize-method` means that we run the UMAP optimization protocol
	python3 dim_reduce_dataset.py --dr-algorithm gidr_dun --optimize-method umap --num-points 6000
	# Run OUR implementation of UMAP in numba
	python3 dim_reduce_dataset.py --num-points 6000 --numba --umap
	### TSNE
	# Run the original TSNE algorithm that gets installed with `pip install scikit-learn`
	python3 dim_reduce_dataset.py --num-points 6000 --dr-algorithm tsne
	# Run OUR implementation of TSNE in cython
	# 	- `dr-algorithm` means that we run the GIDR_DUN implementations
	# 	- `optimize-method` means that we run the UMAP optimization protocol
	python3 dim_reduce_dataset.py --dr-algorithm gidr_dun --optimize-method tsne --num-points 6000

run_gpu_test: build_cuda_code
	# FIXME -- make into a unit test
	# Basic test to make sure that every algorithm can be run on GPU
	### GIDR_DUN
	python3 dim_reduce_dataset.py --gpu --num-points 60000
	### RAPIDS UMAP
	python3 dim_reduce_dataset.py --dr-algorithm rapids_umap --num-points 60000
	### RAPIDS TSNE
	python3 dim_reduce_dataset.py --dr-algorithm rapids_tsne --num-points 60000

clean:
	rm *.so *.o *.egg-info
