# Intel

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib

[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include main_rand.cpp -o main_rand.x
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
OMP frobenius norm Mkl
5138.63 5163.755163.98
DEBUG TRANSFORM_REDUCE_SUM STDPAR
DEBUG REDUCE_SUM STDPAR
5138.63 5204.69 5163.98
1.67772e+07 4e+07
dtor

[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -O3 main_rand.cpp -o main_rand.x
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
OMP frobenius norm Mkl
5138.6 5163.795163.98
DEBUG TRANSFORM_REDUCE_SUM STDPAR
DEBUG REDUCE_SUM STDPAR
5138.6 5204.69 5163.98
3.99988e+07 4e+07
dtor

[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -D_USE_DOUBLE_PRECISION main_rand.cpp -o main_rand.x
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
OMP frobenius norm Mkl
5164.05 5164.055163.98
DEBUG TRANSFORM_REDUCE_SUM STDPAR
DEBUG REDUCE_SUM STDPAR
5164.05 5164.05 5163.98
4.00005e+07 4e+07
dtor

[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -O3 -D_USE_DOUBLE_PRECISION main_rand.cpp -o main_rand.x
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
OMP frobenius norm Mkl
5164.13 5164.135163.98
DEBUG TRANSFORM_REDUCE_SUM STDPAR
DEBUG REDUCE_SUM STDPAR
5164.13 5164.13 5163.98
4.00022e+07 4e+07
dtor

#Nvidia


[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main_rand.cpp -I./../../include -o main_rand.x          
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
DEBUG TRANSFORM_REDUCE_SUM STDPAR
DEBUG REDUCE_SUM STDPAR
4096 5163.64 5163.98
3.99972e+07 4e+07
dtor

[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main_rand.cpp -I./../../include -O3 -o main_rand.x          
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
DEBUG TRANSFORM_REDUCE_SUM STDPAR
DEBUG REDUCE_SUM STDPAR
5154.5 5163.87 5163.98
3.99983e+07 4e+07
dtor

[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main_rand.cpp -I./../../include -D_USE_DOUBLE_PRECISION -o main_rand.x          
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
DEBUG TRANSFORM_REDUCE_SUM STDPAR
DEBUG REDUCE_SUM STDPAR
5163.54 5163.54 5163.98
3.99947e+07 4e+07
dtor

[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main_rand.cpp -I./../../include -D_USE_DOUBLE_PRECISION -O3 -o main_rand.x         
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
DEBUG TRANSFORM_REDUCE_SUM STDPAR
DEBUG REDUCE_SUM STDPAR
5163.56 5163.56 5163.98
3.99951e+07 4e+07
dtor


 
