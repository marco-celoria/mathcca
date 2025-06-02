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
option(DEFINE_USE_DOUBLE_PRECISION "Enable Double Precision" ON)


# Import the module
include (CheckLanguage)
check_language(CUDA)

find_package(CUDAToolkit)
find_package(OpenMP)

if(CMAKE_CUDA_COMPILER)

  enable_language(CUDA)

  message("CUDA compiler is found")

  add_executable("dev_steams_depth"   "main_depth.cu")
  add_executable("dev_steams_breadth" "main_breadth.cu")
  
  target_compile_features("dev_steams_depth"   PRIVATE cxx_std_20)
  target_compile_features("dev_steams_breadth" PRIVATE cxx_std_20)
  
  target_include_directories("dev_steams_depth"   PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_include_directories("dev_steams_breadth" PRIVATE ${PROJECT_SOURCE_DIR}/include)
  
  if( OpenMP_FOUND )
    message("OpenMP is found")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp ${CUDA_NVCC_ARCH_FLAGS}")
    target_link_libraries("dev_steams_depth"   PRIVATE "-lgomp")
    target_link_libraries("dev_steams_breadth" PRIVATE "-lgomp")
  else()
    message("OpenMP is not found")
  endif()
    
  if(DEFINE_PINNED)
    target_compile_definitions("dev_steams_depth"   PRIVATE _PINNED)
    target_compile_definitions("dev_steams_breadth" PRIVATE _PINNED)
  endif(DEFINE_PINNED)
    
  if(DEFINE_USE_DOUBLE_PRECISION)
    target_compile_definitions("dev_steams_depth"   PRIVATE _USE_DOUBLE_PRECISION)
    target_compile_definitions("dev_steams_breadth" PRIVATE _USE_DOUBLE_PRECISION)
  endif(DEFINE_USE_DOUBLE_PRECISION)

  if( CUDAToolkit_FOUND )
      
    message("CUDAToolkit_FOUND - ${CUDAToolkit_INCLUDE_DIRS}")
         
    target_link_libraries("dev_steams_depth"   PRIVATE CUDA::curand)
    target_link_libraries("dev_steams_breadth" PRIVATE CUDA::curand)
      
    if(DEFINE_CUBLAS)
      target_compile_definitions("dev_steams_depth"   PRIVATE _CUBLAS)
      target_compile_definitions("dev_steams_breadth" PRIVATE _CUBLAS)
      target_link_libraries("dev_steams_depth"   PRIVATE CUDA::cublas)
      target_link_libraries("dev_steams_breadth" PRIVATE CUDA::cublas)
    endif(DEFINE_CUBLAS)
      
    if(DEFINE_THRUST)
      target_compile_definitions("dev_steams_depth"   PRIVATE _THRUST)
      target_compile_definitions("dev_steams_breadth" PRIVATE _THRUST)
    endif(DEFINE_THRUST)
     
  else()
    
    message("CUDAToolkit is not found.")
  
  endif()
    
  set_target_properties("dev_steams_depth"   PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties("dev_steams_breadth" PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
   
  message("CMAKE_CXX_FLAGS= ${CMAKE_CXX_FLAGS}")
    
  message("CMAKE_CUDA_FLAGS= ${CMAKE_CUDA_FLAGS}")
    
else()
   
  message("CUDA compiler is not found.")
   
endif()



