<!--
SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Intel

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib

[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include main_const.cpp -o main_const.x
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 1448.15 -:- 2747.72 	Error: 1299.56
Base  time: 0.254437
OMP frobenius norm Mkl

Does Mkl  result agree with real result? 2747.72 -:- 2747.72 	Error: 0
Does Base result agree with Mkl  result? 1448.15 -:- 2747.72 	Error: 1299.56
Mkl  time: 0.0530801
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 1024 -:- 2747.72 	Error: 1723.72
Does Base   result agree with StdPar result? 1448.15 -:- 1024 	Error: 424.155
StdPar time: 2.1592
dtor




[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -O3 -Wall -Wextra -mtune=native -march=native -fstrict-aliasing -mprefer-vector-width=512 -ftree-vectorize -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm  main_const.cpp -o main_const.x
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 1448.15 -:- 2747.72 	Error: 1299.56
Base  time: 0.107633
OMP frobenius norm Mkl

Does Mkl  result agree with real result? 2747.72 -:- 2747.72 	Error: 0
Does Base result agree with Mkl  result? 1448.15 -:- 2747.72 	Error: 1299.56
Mkl  time: 0.0529764
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 1024 -:- 2747.72 	Error: 1723.72
Does Base   result agree with StdPar result? 1448.15 -:- 1024 	Error: 424.155
StdPar time: 0.261273
dtor



[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -D_USE_DOUBLE_PRECISION main_const.cpp -o main_const.x
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 2747.72 -:- 2747.72 	Error: 1.05818e-06
Base  time: 0.259593
OMP frobenius norm Mkl

Does Mkl  result agree with real result? 2747.72 -:- 2747.72 	Error: 2.62844e-10
Does Base result agree with Mkl  result? 2747.72 -:- 2747.72 	Error: 1.05791e-06
Mkl  time: 0.108415
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 2747.72 -:- 2747.72 	Error: 1.16884e-06
Does Base   result agree with StdPar result? 2747.72 -:- 2747.72 	Error: 1.10665e-07
StdPar time: 2.75618
dtor



[mceloria@dgx003 2_host_norm]$ g++ -std=c++20 -fopenmp -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -O3 -Wall -Wextra -mtune=native -march=native -fstrict-aliasing -mprefer-vector-width=512 -ftree-vectorize -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm -D_USE_DOUBLE_PRECISION main_const.cpp -o main_const.x
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 2747.72 -:- 2747.72 	Error: 1.05818e-06
Base  time: 0.125374
OMP frobenius norm Mkl

Does Mkl  result agree with real result? 2747.72 -:- 2747.72 	Error: 2.62844e-10
Does Base result agree with Mkl  result? 2747.72 -:- 2747.72 	Error: 1.05791e-06
Mkl  time: 0.108477
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 2747.72 -:- 2747.72 	Error: 1.16884e-06
Does Base   result agree with StdPar result? 2747.72 -:- 2747.72 	Error: 1.10665e-07
StdPar time: 0.403056
dtor



#Nvidia


[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar=gpu -D_STDPAR main_const.cpp -I./../../include -o main_const.x          
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 512 -:- 2747.72 	Error: 2235.72
Base  time: 0.851165
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 2747.74 -:- 2747.72 	Error: 0.0200195
Does Base   result agree with StdPar result? 512 -:- 2747.74 	Error: 2235.74
StdPar time: 0.218218
dtor



[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar=gpu -D_STDPAR main_const.cpp -I./../../include -O3 -o main_const.x          
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 2048 -:- 2747.72 	Error: 699.717
Base  time: 0.443141
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 2747.73 -:- 2747.72 	Error: 0.00830078
Does Base   result agree with StdPar result? 2048 -:- 2747.73 	Error: 699.726
StdPar time: 0.209671
dtor



[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar=gpu -D_STDPAR main_const.cpp -I./../../include -D_USE_DOUBLE_PRECISION -o main_const.x          
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 2747.72 -:- 2747.72 	Error: 2.18506e-05
Base  time: 1.20366
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 2747.72 -:- 2747.72 	Error: 1.86446e-11
Does Base   result agree with StdPar result? 2747.72 -:- 2747.72 	Error: 2.18506e-05
StdPar time: 0.406675
dtor



[mceloria@dgx003 2_host_norm]$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++ -std=c++20  -stdpar=gpu -D_STDPAR main_const.cpp -I./../../include -D_USE_DOUBLE_PRECISION -O3 -o main_const.x         
[mceloria@dgx003 2_host_norm]$ ./main_const.x 
custom ctor
DEBUG FILL_CONST STDPAR
OMP frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM OMP

Does Base result agree with real result? 2747.72 -:- 2747.72 	Error: 1.05818e-06
Base  time: 0.887049
STDPAR frobenius norm Base
DEBUG TRANSFORM_REDUCE_SUM STDPAR

Does StdPar result agree with real   result? 2747.72 -:- 2747.72 	Error: 2.27374e-11
Does Base   result agree with StdPar result? 2747.72 -:- 2747.72 	Error: 1.0582e-06
StdPar time: 0.408492
dtor


 
