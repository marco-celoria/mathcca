export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/math_libs/12.6/lib64/

/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvcc -std=c++20 main_breadth.cu -I./../../include -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/math_libs/12.6/lib64/ -lcublas -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/math_libs/12.6/include -O3 -D_USE_DOUBLE_PRECISION -D_PINNED  -Xcompiler -fopenmp   -lgomp  -o main_breadth.x

/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvcc -std=c++20 main_depth.cu -I./../../include -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/math_libs/12.6/lib64/ -lcublas -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/math_libs/12.6/include -O3 -D_USE_DOUBLE_PRECISION -D_PINNED  -Xcompiler -fopenmp   -lgomp  -o main_depth.x

(-D_USE_DOUBLE_PRECISION)
(-D_PINNED)
(-D_HOST_CHECK)
(-D_THRUST)

/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/compute-sanitizer --leak-check full ./main.x
/usr/local/cuda-12.2/compute-sanitizer/compute-sanitizer  --leak-check full ./main.x

/usr/local/cuda-12.2/bin/nvcc -std=c++20 main_breadth.cu -I./../../include -L/usr/local/cuda-12.2/lib -lcublas -I/usr/local/cuda-12.2/lib/include -O3 -D_USE_DOUBLE_PRECISION -D_PINNED -o main_breadth.x

/usr/local/cuda-12.2/bin/nvcc -std=c++20 main_depth.cu -I./../../include -L/usr/local/cuda-12.2/lib -lcublas -I/usr/local/cuda-12.2/lib/include -O3 -D_USE_DOUBLE_PRECISION -D_PINNED -o main_depth.x

