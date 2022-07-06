create_rapids_env:
	scripts/rapids_conda.sh

create_torch_env:
	scripts/torch_conda.sh

create_python_env:
	scripts/basic_conda.sh

install_python_env:
	scripts/check_conda_env.sh GiDR_DUN
	python3 setup.py install --user

install_cython_env: install_python_env
	# https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/ 
	scripts/check_conda_env.sh GiDR_DUN
	python3 setup_cython.py install --user

install_cuda_code: install_cython_env
	scripts/check_conda_env.sh GiDR_DUN_rapids
	# FIXME -- this should run on user's preferred cuda
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

run_cython_test: install_cython_env
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
	python3 dim_reduce_dataset.py --num-points 6000 --dr-algorithm original_umap
	# Run OUR implementation of UMAP in cython
	# 	- `dr-algorithm` means that we run the GIDR_DUN implementations
	# 	- `optimize-method` means that we run the UMAP optimization protocol
	python3 dim_reduce_dataset.py --dr-algorithm gidr_dun --optimize-method umap --num-points 6000
	# Run OUR implementation of UMAP in numba
	python3 dim_reduce_dataset.py --num-points 6000 --numba --optimize-method umap
	### TSNE
	# Run the original TSNE algorithm that gets installed with `pip install scikit-learn`
	python3 dim_reduce_dataset.py --num-points 6000 --dr-algorithm original_tsne
	# Run OUR implementation of TSNE in cython
	# 	- `dr-algorithm` means that we run the GIDR_DUN implementations
	# 	- `optimize-method` means that we run the UMAP optimization protocol
	python3 dim_reduce_dataset.py --dr-algorithm gidr_dun --optimize-method tsne --num-points 6000

run_gpu_test: install_cuda_code
	# FIXME -- make into a unit test
	# Basic test to make sure that every algorithm can be run on GPU
	### GIDR_DUN
	python3 dim_reduce_dataset.py --gpu --num-points 60000
	### RAPIDS UMAP
	python3 dim_reduce_dataset.py --dr-algorithm rapids_umap --num-points 60000
	### RAPIDS TSNE
	python3 dim_reduce_dataset.py --dr-algorithm rapids_tsne --num-points 60000

run_analysis: install_cython_env
	python3 run_analysis.py
	python3 experiment_utils/read_metrics.py

run_gpu_analysis: install_cuda_code
	python3 run_gpu_analysis.py --analysis-type runtimes
	python3 run_gpu_analysis.py --analysis-type data_size_sweep
	python3 run_gpu_analysis.py --analysis-type dim_size_sweep
	python3 experiment_utils/read_gpu_metrics.py

clean:
	rm *.so *.o *.egg-info
