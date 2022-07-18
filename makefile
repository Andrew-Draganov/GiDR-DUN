install_python_env:
	pip install -e .
	python3 setup.py build_ext --inplace

install_cython_code: install_python_env
	python3 setup_cython.py build_ext --inplace

install_cuda_code: install_python_env
	# Compile gradient optimizer cuda code
	nvcc --shared -o libgpu_dim_reduction.so \
		GDR/cython/cuda_wrappers/gpu_dim_reduction.cpp \
		GDR/cython/cuda_kernels/gpu_kernels.cu \
		GDR/cython/utils/gpu_utils.cu \
		-Xcompiler -fPIC
	#
	# Compile graph similarity cuda code
	nvcc --shared -o libgpu_graph_weights.so \
	   	GDR/cython/cuda_wrappers/gpu_graph_weights.cpp \
		GDR/cython/cuda_kernels/gpu_graph_cuda.cu \
		GDR/cython/utils/gpu_utils.cu \
		GDR/cython/utils/util.cpp \
		GDR/cython/utils/mem_util.cpp \
		-Xcompiler -fPIC
	#
	# Link cuda code to python through cython
	python3 setup_cython_gpu.py build_ext --inplace

run_test: install_python_env
	python -m GDR.dim_reduce_dataset --num-points 5000
	python -m GDR.dim_reduce_dataset --num-points 5000 --optimize-method umap
	echo 'Numba code runs successfully'

run_cython_test: install_cython_code
	python -m GDR.dim_reduce_dataset --cython --num-points 5000
	python -m GDR.dim_reduce_dataset --cython --num-points 5000 --optimize-method umap
	python -m GDR.dim_reduce_dataset --cython --num-points 5000 --optimize-method tsne
	echo 'Cython code runs successfully'

run_gpu_test: install_cuda_code
	python -m GDR.dim_reduce_dataset --gpu --num-points 60000
	echo 'GPU code runs successfully'

clean:
	rm -r *.so *.egg-info build/
