<!--
SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>

SPDX-License-Identifier: AGPL-3.0-or-later
-->

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib

g++ -std=c++20 -fopenmp -D_USE_DOUBLE_PRECISION -I./include -I/u/dssc/mceloria/intel/oneapi/mkl/2025.1/include -L/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -D_MKL -L/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib -ltbb -D_STDPAR -I./../../include -Wall -Wextra -O3 -mtune=native -march=native -fstrict-aliasing -mprefer-vector-width=512 -ftree-vectorize -std=c++20 -fopenmp -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm -D_MKL -DUSE_DOUBLE_PRECISION main.cpp -o main.x

