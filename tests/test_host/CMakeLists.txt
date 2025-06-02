# SPDX-FileCopyrightText: 2025 Marco Celoria <celoria.marco@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message("Setting CMAKE_BUILD_TYPE to ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")

set(CMAKE_CXX_FLAGS_RELEASE "-O3" )
      
option(DEFINE_MKL "Enable Cublas" ON)

option(DEFINE_STDPAR "Enable StdPar" ON)

set(
        host_tests
        host_copy_ctor_sp
	host_copy_ctor_dp
	host_copy_sp
	host_copy_dp
	host_fill_sp
	host_fill_dp
	host_matarith_sp
	host_matarith_dp
	host_matmul_sp
	host_matmul_dp
	host_move_ctor_sp
	host_move_ctor_dp
	host_norm_sp
	host_norm_dp
	host_reduce_sp
	host_reduce_dp
	host_trans_sp
	host_trans_dp
	host_trans_reduce_sp
	host_trans_reduce_dp
)

find_package(OpenMP)
find_package(MKL)
find_package(TBB)
find_package(CUDAToolkit)
find_package(OpenMP)

if( CUDAToolkit_FOUND )
	
  message("CUDAToolkit is found: CUDAToolkit_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}")

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
    message("Setting CMAKE_CUDA_ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES}")
  endif() 
  
  if(DEFINE_STDPAR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdpar=gpu")    
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdpar=multicore")
  endif(DEFINE_STDPAR) 
  
  if( OpenMP_FOUND )
    message("OpenMP is found.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mp=multicore")
  else()
    message("OpenMP is not found.")
  endif()
      
  foreach (name IN LISTS host_tests)
     
    add_executable("${name}" "${name}.cpp")
      
    target_compile_features("${name}" PRIVATE cxx_std_20)
     
    target_include_directories("${name}" PRIVATE ${PROJECT_SOURCE_DIR}/include)
      
    if(DEFINE_STDPAR)
      target_compile_definitions("${name}" PRIVATE _STDPAR)
    endif(DEFINE_STDPAR)
       
    if( OpenMP_FOUND )
      target_link_libraries("${name}" PRIVATE "-lgomp")
    endif()
     
    target_link_libraries("${name}" PRIVATE GTest::gtest GTest::gtest_main)
    
    add_test(NAME "${name}_Test" COMMAND "${name}")
     
  endforeach()
    
  message("CMAKE_CXX_FLAGS= ${CMAKE_CXX_FLAGS}") 
  message("CMAKE_CUDA_FLAGS= ${CMAKE_CUDA_FLAGS}")      

else()
    
  message("CUDAToolkit is found")
  
  set(CMAKE_CXX_FLAGS "-Wall -Wextra") 
  
  foreach (name IN LISTS host_tests)
    
    add_executable("${name}" "${name}.cpp")
     
    target_compile_features("${name}" PRIVATE cxx_std_20)
    
    target_include_directories("${name}" PRIVATE ${PROJECT_SOURCE_DIR}/include)

    if( MKL_FOUND )
      message("MKL is found.")
      target_link_libraries("${name}" PRIVATE MKL::MKL)
      if(DEFINE_MKL)
        target_compile_definitions("${name}" PRIVATE _MKL)
      endif(DEFINE_MKL)
    else()
      message("MKL is not found.")
    endif()
     
    if( TBB_FOUND )
      message("TBB is found.")
      target_link_libraries("${name}" PRIVATE TBB::tbb)
      target_link_libraries("${name}" PRIVATE -ltbb -lpthread )
      if(DEFINE_STDPAR)
        target_compile_definitions("${name}" PRIVATE _STDPAR)
      endif(DEFINE_STDPAR)
    endif()
      
    if( OpenMP_FOUND )
      target_link_libraries("${name}" PRIVATE OpenMP::OpenMP_CXX)
    else()
      message("OpenMP is not found.")
    endif()
     
    target_link_libraries("${name}" PRIVATE GTest::gtest GTest::gtest_main)
     
    add_test(NAME "${name}_Test" COMMAND "${name}")
      
  endforeach()
      
  message("CMAKE_CXX_FLAGS= ${CMAKE_CXX_FLAGS}")
       
endif()

enable_testing()


