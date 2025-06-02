# SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message("Setting CMAKE_BUILD_TYPE to ${CMAKE_BUILD_TYPE}")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
  message("Setting CMAKE_CUDA_ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -G")
set(CMAKE_CXX_FLAGS_RELEASE "-O3" )

option(DEFINE_CUBLAS "Enable Cublas" ON)
option(DEFINE_THRUST "Enable Thrust" ON)
option(DEFINE_PINNED "Enable Pinned Host Memory" ON)

set(
        device_tests
        dev_copy_ctor_sp
	dev_copy_ctor_dp
	dev_copy_sp
	dev_copy_dp
	dev_fill_sp
	dev_fill_dp
	dev_matarith_sp
	dev_matarith_dp
	dev_matmul_sp
	dev_matmul_dp
	dev_move_ctor_sp
	dev_move_ctor_dp
	dev_norm_sp
	dev_norm_dp
	dev_reduce_sp
	dev_reduce_dp
	dev_trans_sp
	dev_trans_dp
	dev_trans_reduce_sp
	dev_trans_reduce_dp
)

# Import the module
include (CheckLanguage)
check_language(CUDA)
    
find_package(CUDAToolkit)
find_package(OpenMP)

if(CMAKE_CUDA_COMPILER)
     
  enable_language(CUDA)
    
  message("CUDA compiler is found")
     
  if( OpenMP_FOUND )
    message("OpenMP is found")	  
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp ${CUDA_NVCC_ARCH_FLAGS}")
  else()
    message("OpenMP is not found") 	    
  endif()
     
  foreach (name IN LISTS device_tests)
          
    add_executable("${name}" "${name}.cu")
           
    target_compile_features("${name}" PRIVATE cxx_std_20)
          
    target_include_directories("${name}" PRIVATE ${PROJECT_SOURCE_DIR}/include)
         
    if( CUDAToolkit_FOUND )
         
      message("CUDAToolkit_FOUND - ${CUDAToolkit_INCLUDE_DIRS}")
          
      target_link_libraries("${name}" PRIVATE CUDA::curand)
         
      if(DEFINE_CUBLAS)
        target_compile_definitions("${name}" PRIVATE _CUBLAS)
        target_link_libraries("${name}" PRIVATE CUDA::cublas)
      endif(DEFINE_CUBLAS)
          
      if(DEFINE_THRUST)
        target_compile_definitions("${name}" PRIVATE _THRUST)
      endif(DEFINE_THRUST)
        
    else()
           
      message("CUDAToolkit is not found.")
           
    endif()
         
    if(DEFINE_PINNED)
      target_compile_definitions("${name}" PRIVATE _PINNED)
    endif(DEFINE_PINNED)
          
    if( OpenMP_FOUND )
      target_link_libraries("${name}" PRIVATE "-lgomp")
    endif()
         
    target_link_libraries("${name}" PRIVATE GTest::gtest GTest::gtest_main)
     
    add_test(NAME "${name}_Test" COMMAND "${name}")
  
  endforeach()

  message("CMAKE_CXX_FLAGS= ${CMAKE_CXX_FLAGS}")
  message("CMAKE_CUDA_FLAGS= ${CMAKE_CUDA_FLAGS}")

else()

  message("CUDA compiler is not found.")

endif()

enable_testing()


