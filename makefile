run: #this is working :D
	nvcc --shared -o libgpu_dim_reduction.so cython/gpu_dim_reduction.cpp cython/gpu_kernels.cu cython/GPU_utils.cu -Xcompiler -fPIC

	nvcc --shared -o libgpu_graph_weights.so cython/gpu_graph_weights.cpp cython/gpu_graph_cuda.cu cython/GPU_utils.cu cython/util.cpp cython/mem_util.cpp -Xcompiler -fPIC
	#
	# Setting up...
	python setup_cython_gpu.py install #https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/
	python setup_cython.py install #https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/
	#
	# Running...
	python dim_reduce_dataset.py --gpu --num-points 60000

run_tsne:
	python dim_reduce_dataset.py --optimize-method uniform_umap --num-points 60000 --normalized --momentum

#clean:
#	rm *.so *.o test.c

profile:
	nvcc -o cython/profile cython/profiling_test.cpp cython/gpu_kernels.cu cython/GPU_utils.cu
	nvprof ./cython/profile

