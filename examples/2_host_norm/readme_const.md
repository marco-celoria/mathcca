# Intel

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib

[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include main_const.cpp -o main_const.x
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.254258
OMP frobenius norm Mkl
11585.2 27477.2
tm == 0.0529983
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR
11585.2 8192
ts == 2.70803
dtor

[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -O3 -Wall -Wextra -mtune=native -march=native -fstrict-aliasing -mprefer-vector-width=512 -ftree-vectorize -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm  main_const.cpp -o main_const.x
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.107497
OMP frobenius norm Mkl
11585.2 27477.2
tm == 0.0530146
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR
11585.2 8192
ts == 0.260435
dtor

[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -D_USE_DOUBLE_PRECISION main_const.cpp -o main_const.x
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.259355
OMP frobenius norm Mkl
27477.2 27477.2
tm == 0.117581
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR
27477.2 27477.2
ts == 2.7021
dtor

[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -O3 -Wall -Wextra -mtune=native -march=native -fstrict-aliasing -mprefer-vector-width=512 -ftree-vectorize -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm -D_USE_DOUBLE_PRECISION main_const.cpp -o main_const.x
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.125063
OMP frobenius norm Mkl
27477.2 27477.2
tm == 0.112561
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR
27477.2 27477.2
ts == 0.401972
dtor

#Nvidia


[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main_const.cpp -I./../../include -o main_const.x          
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.850825
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR
4096 27477.2
ts == 0.21801
dtor

[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main_const.cpp -I./../../include -O3 -o main_const.x          
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.447767
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR
16384 27477.2
ts == 0.207578
dtor

[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main_const.cpp -I./../../include -D_USE_DOUBLE_PRECISION -o main_const.x          
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 1.2035
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR
27477.2 27477.2
ts == 0.402959
dtor

[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main_const.cpp -I./../../include -D_USE_DOUBLE_PRECISION -O3 -o main_const.x         
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.896593
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR
27477.2 27477.2
ts == 0.408626
dtor


 
