run: #this is working :D
	nvcc --shared -o libgpu_dim_reduction.so cython/gpu_dim_reduction.cpp cython/gpu_kernels.cu -Xcompiler -fPIC

	#
	# Setting up...
	python setup_cython.py install #https://stavshamir.github.io/python/making-your-c-library-callable-from-python-by-wrapping-it-with-cython/
	#
	# Running...
	python umap/dim_reduce_dataset.py --optimize-method cy_umap_uniform --num-points 60000

#clean:
#	rm *.so *.o test.c