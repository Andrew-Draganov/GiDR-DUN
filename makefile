build_rapids_env:
	utils/install_conda.sh

build_torch_env:
	utils/torch_conda_env.sh

build_python:
	python3 setup.py install --user
	python3 nndescent/setup.py install --user

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

build_torch_gpu_code: build_torch_env build_python
	echo "Done."

run_gidr_dun_gpu_example:
	python3 dim_reduce_dataset.py --gpu --num-points 60000

run_gidr_dun_cython_example: build_cython
	python3 dim_reduce_dataset.py --num-points 60000

run_gidr_dun_numba_example: build_python
	python3 dim_reduce_dataset.py --num-points 60000 --numba

run_gidr_dun_tsne_example: build_cython
	python3 dim_reduce_dataset.py --optimize-method uniform_umap --num-points 60000 --normalized --momentum

#clean:
#	rm *.so *.o test.c
