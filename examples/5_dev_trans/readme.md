<!--
SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>

SPDX-License-Identifier: AGPL-3.0-or-later
-->

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/math_libs/12.6/lib64/

/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvcc -std=c++20 main.cu -I./../../include -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/math_libs/12.6/lib64/ -lcublas -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/math_libs/12.6/include -D_CUBLAS -O3  -o main.x

(-D_PINNED)

/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/compute-sanitizer --leak-check full ./main.x


