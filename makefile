run_gpu_example:
	python3 dim_reduce_dataset.py --gpu --num-points 60000

run_example:
	python3 dim_reduce_dataset.py --num-points 60000

build_python:
	python3 setup.py install --user
	python3 nndescent/setup.py install --user

build_cython: build_python
	sudo apt-get install cython
	python3 setup_cython.py install #https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/

build_gpu_code: build_python build_cython
	/usr/local/cuda-11/bin/nvcc --shared -o libgpu_dim_reduction.so \
		cython/cuda_wrappers/gpu_dim_reduction.cpp \
		cython/cuda_kernels/gpu_kernels.cu \
		cython/utils/gpu_utils.cu \
		-Xcompiler -fPIC
	/usr/local/cuda-11/bin/nvcc --shared -o libgpu_graph_weights.so \
	   	cython/cuda_wrappers/gpu_graph_weights.cpp \
		cython/cuda_kernels/gpu_graph_cuda.cu \
		cython/utils/gpu_utils.cu \
		cython/utils/util.cpp \
		cython/utils/mem_util.cpp \
		-Xcompiler -fPIC
	python3 setup_cython_gpu.py install

run_tsne:
	python3 dim_reduce_dataset.py --optimize-method uniform_umap --num-points 60000 --normalized --momentum

#clean:
#	rm *.so *.o test.c
