create_gpu_env:
	GDR/scripts/gpu_conda.sh

create_torch_env:
	GDR/scripts/torch_conda.sh

create_python_env:
	GDR/scripts/basic_conda.sh

install_python_env:
	GDR/scripts/check_conda_env.sh GDR
	python3 setup.py install --user

install_cython_env: install_python_env
	# https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/ 
	GDR/scripts/check_conda_env.sh GDR
	python3 setup_cython.py install --user

install_cuda_code: install_python_env
	GDR/scripts/check_conda_env.sh GDR_gpu
	# FIXME -- this should run on user's preferred cuda
	/usr/local/cuda-11.5/bin/nvcc --shared -o libgpu_dim_reduction.so \
		GDR/cython/cuda_wrappers/gpu_dim_reduction.cpp \
		GDR/cython/cuda_kernels/gpu_kernels.cu \
		GDR/cython/utils/gpu_utils.cu \
		-Xcompiler -fPIC
	/usr/local/cuda-11.5/bin/nvcc --shared -o libgpu_graph_weights.so \
	   	GDR/cython/cuda_wrappers/gpu_graph_weights.cpp \
		GDR/cython/cuda_kernels/gpu_graph_cuda.cu \
		GDR/cython/utils/gpu_utils.cu \
		GDR/cython/utils/util.cpp \
		GDR/cython/utils/mem_util.cpp \
		-Xcompiler -fPIC
	python3 setup_cython_gpu.py build_ext --inplace

run_numba_test: install_python_env
	# FIXME -- make into a unit test
	python -m GDR.dim_reduce_dataset --num-points 5000 --numba
	python -m GDR.dim_reduce_dataset --num-points 5000 --numba --optimize-method umap

run_cython_test: install_cython_env
	# FIXME -- make into a unit test
	# Basic test to make sure that every algorithm can be run with cython
	### GIDR_DUN
	# Basic gidr_dun implementation
	python -m GDR.dim_reduce_dataset --num-threads 1 --num-points 5000 --optimize-method gidr_dun
	python -m GDR.dim_reduce_dataset --num-threads 1 --num-points 5000 --optimize-method umap
	python -m GDR.dim_reduce_dataset --num-threads 1 --num-points 5000 --optimize-method tsne

run_gpu_test: install_cuda_code
	# FIXME -- make into a unit test
	# Basic test to make sure that every algorithm can be run on GPU
	### GIDR_DUN
	python -m GDR.dim_reduce_dataset --gpu --num-points 60000
	### RAPIDS UMAP
	python -m GDR.dim_reduce_dataset --dr-algorithm rapids_umap --num-points 60000
	### RAPIDS TSNE
	python -m GDR.dim_reduce_dataset --dr-algorithm rapids_tsne --num-points 60000

clean:
	rm *.so *.o *.egg-info
