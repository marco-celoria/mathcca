<!--
SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Intel

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib

[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include main_rand.cpp -o main_rand.x
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
ustom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 11585.2 -:- 15864 	Error: 4278.71
Base  time: 0.253817
OMP frobenius norm Mkl

Does Mkl  result agree with real result? 15863.8 -:- 15864 	Error: 0.18457
Does Base result agree with Mkl  result? 11585.2 -:- 15863.8 	Error: 4278.53
Mkl  time: 0.0530508
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 8192 -:- 15864 	Error: 7671.95
Does Base   result agree with StdPar result? 11585.2 -:- 8192 	Error: 3393.24
StdPar  time: 3.35987
dtor



[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -O3 -Wall -Wextra -mtune=native -march=native -fstrict-aliasing -mprefer-vector-width=512 -ftree-vectorize -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm  main_rand.cpp -o main_rand.x
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 11585.2 -:- 15864 	Error: 4278.71
Base  time: 0.107452
OMP frobenius norm Mkl

Does Mkl  result agree with real result? 15864 -:- 15864 	Error: 0.0683594
Does Base result agree with Mkl  result? 11585.2 -:- 15864 	Error: 4278.78
Mkl  time: 0.0530195
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 8192 -:- 15864 	Error: 7671.95
Does Base   result agree with StdPar result? 11585.2 -:- 8192 	Error: 3393.24
StdPar  time: 0.261011
dtor



[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -D_USE_DOUBLE_PRECISION main_rand.cpp -o main_rand.x
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 15864.2 -:- 15864 	Error: 0.288634
Base  time: 0.258408
OMP frobenius norm Mkl

Does Mkl  result agree with real result? 15864.2 -:- 15864 	Error: 0.288634
Does Base result agree with Mkl  result? 15864.2 -:- 15864.2 	Error: 1.39698e-09
Mkl  time: 0.108619
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 15864.2 -:- 15864 	Error: 0.288634
Does Base   result agree with StdPar result? 15864.2 -:- 15864.2 	Error: 2.51021e-10
StdPar  time: 1.99449
dtor


[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -O3 -Wall -Wextra -mtune=native -march=native -fstrict-aliasing -mprefer-vector-width=512 -ftree-vectorize -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm -D_USE_DOUBLE_PRECISION main_rand.cpp -o main_rand.x
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 15864.6 -:- 15864 	Error: 0.677843
Base  time: 0.125476
OMP frobenius norm Mkl

Does Mkl  result agree with real result? 15864.6 -:- 15864 	Error: 0.677843
Does Base result agree with Mkl  result? 15864.6 -:- 15864.6 	Error: 4.91127e-11
Mkl  time: 0.10839
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 15864.6 -:- 15864 	Error: 0.677843
Does Base   result agree with StdPar result? 15864.6 -:- 15864.6 	Error: 3.23053e-09
StdPar  time: 0.405048
dtor


#Nvidia


[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar=gpu -D_STDPAR main_rand.cpp -I./../../include -o main_rand.x          
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 4096 -:- 15864 	Error: 11768
Base  time: 0.531576
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 15863.7 -:- 15864 	Error: 0.25
Does Base   result agree with StdPar result? 4096 -:- 15863.7 	Error: 11767.7
StdPar  time: 0.220084
dtor


[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar=gpu -D_STDPAR main_rand.cpp -I./../../include -O3 -o main_rand.x          
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 15287 -:- 15864 	Error: 576.963
Base  time: 0.184444
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 15864.1 -:- 15864 	Error: 0.194336
Does Base   result agree with StdPar result? 15287 -:- 15864.1 	Error: 577.157
StdPar  time: 0.213304
dtor


[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar=gpu -D_STDPAR main_rand.cpp -I./../../include -D_USE_DOUBLE_PRECISION -o main_rand.x          
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 15863.8 -:- 15864 	Error: 0.147412
Base  time: 0.592366
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 15863.8 -:- 15864 	Error: 0.147412
Does Base   result agree with StdPar result? 15863.8 -:- 15863.8 	Error: 1.23691e-08
StdPar  time: 0.409823
dtor


[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar=gpu -D_STDPAR main_rand.cpp -I./../../include -D_USE_DOUBLE_PRECISION -O3 -o main_rand.x         
[mceloria@dgx003 2_host_norm]$ ./main_rand.x 
custom ctor
DEBUG FILL_RAND OMP
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 15864.4 -:- 15864 	Error: 0.446532
Base  time: 0.369305
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 15864.4 -:- 15864 	Error: 0.446532
Does Base   result agree with StdPar result? 15864.4 -:- 15864.4 	Error: 6.38465e-10
StdPar  time: 0.418011
dtor


 
