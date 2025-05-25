# Intel

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib

[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include main_rand.cpp -o main_rand.x
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.253785
OMP frobenius norm Mkl
11585.2 1586415864
tm == 0.053092
DEBUG TRANSFORM_REDUCE_SUM STDPAR
ts == 2.64896
DEBUG REDUCE_SUM STDPAR
11585.2 8192 15864
1.67772e+07 3.77498e+08
dtor

[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -O3 -Wall -Wextra -mtune=native -march=native -fstrict-aliasing -mprefer-vector-width=512 -ftree-vectorize -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm  main_rand.cpp -o main_rand.x
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.107424
OMP frobenius norm Mkl
11585.2 15863.615864
tm == 0.0530267
DEBUG TRANSFORM_REDUCE_SUM STDPAR
ts == 0.260574
DEBUG REDUCE_SUM STDPAR
11585.2 8192 15864
2.68435e+08 3.77498e+08
dtor

[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -D_USE_DOUBLE_PRECISION main_rand.cpp -o main_rand.x
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.258159
OMP frobenius norm Mkl
15864.1 15864.115864
tm == 0.108435
DEBUG TRANSFORM_REDUCE_SUM STDPAR
ts == 2.18522
DEBUG REDUCE_SUM STDPAR
15864.1 15864.1 15864
3.775e+08 3.77498e+08
dtor

[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -O3 -Wall -Wextra -mtune=native -march=native -fstrict-aliasing -mprefer-vector-width=512 -ftree-vectorize -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm -D_USE_DOUBLE_PRECISION main_rand.cpp -o main_rand.x
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.125009
OMP frobenius norm Mkl
15864.4 15864.415864
tm == 0.108283
DEBUG TRANSFORM_REDUCE_SUM STDPAR
ts == 0.403035
DEBUG REDUCE_SUM STDPAR
15864.4 15864.4 15864
3.77513e+08 3.77498e+08
dtor


#Nvidia


[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main_rand.cpp -I./../../include -o main_rand.x          
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.53173
DEBUG TRANSFORM_REDUCE_SUM STDPAR
ts == 0.223661
DEBUG REDUCE_SUM STDPAR
4096 15863.7 15864
3.77487e+08 3.77498e+08
dtor


[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main_rand.cpp -I./../../include -O3 -o main_rand.x          
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.18622
DEBUG TRANSFORM_REDUCE_SUM STDPAR
ts == 0.214295
DEBUG REDUCE_SUM STDPAR
15286 15863.8 15864
3.77498e+08 3.77498e+08
dtor

[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main_rand.cpp -I./../../include -D_USE_DOUBLE_PRECISION -o main_rand.x          
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.591859
DEBUG TRANSFORM_REDUCE_SUM STDPAR
ts == 0.414997
DEBUG REDUCE_SUM STDPAR
15864 15864 15864
3.77498e+08 3.77498e+08
dtor

[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar -D_STDPAR main_rand.cpp -I./../../include -D_USE_DOUBLE_PRECISION -O3 -o main_rand.x         
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
ustom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP
tb == 0.367737
DEBUG TRANSFORM_REDUCE_SUM STDPAR
ts == 0.423088
DEBUG REDUCE_SUM STDPAR
15863.4 15863.4 15864
3.77479e+08 3.77498e+08
dtor


 
