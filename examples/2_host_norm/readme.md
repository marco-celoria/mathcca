/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main.cpp -I./../../include -o main.x   


[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main.cpp -I./../../include -O3 -o main.x          
[mceloria@dgx003 2_host_norm]$ ./main.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
DEBUG TRANSFORM_REDUCE_SUM STDPAR
5477.23 5477.23
dtor
[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main.cpp -I./../../include -o main.x          
[mceloria@dgx003 2_host_norm]$ ./main.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
DEBUG TRANSFORM_REDUCE_SUM STDPAR
4096 5477.23
dtor



export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib

g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -O3 main.cpp -o main.x
