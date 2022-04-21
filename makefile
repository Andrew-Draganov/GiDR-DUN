run_gpu: #this is working :D
	nvcc --shared -o libgpu_dim_reduction.so cython/gpu_dim_reduction.cpp cython/gpu_kernels.cu cython/GPU_utils.cu -Xcompiler -fPIC

	#
	# Setting up...
	python setup_cython_gpu.py install #https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/
	#
	# Running...
	python run_analysis.py

run_tsne:
	python dim_reduce_dataset.py --optimize-method uniform_umap --num-points 60000 --normalized --momentum

#clean:
#	rm *.so *.o test.c

profile:
	nvcc -o cython/profile cython/profiling_test.cpp cython/gpu_kernels.cu cython/GPU_utils.cu
	nvprof ./cython/profile
