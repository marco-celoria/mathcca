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

option(DEFINE_USE_DOUBLE_PRECISION "Enable Double Precision" ON)


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
     
  add_executable("host_norm_const" "main_const.cpp")
  add_executable("host_norm_rand"  "main_rand.cpp")
   
  target_compile_features("host_norm_const" PRIVATE cxx_std_20)
  target_compile_features("host_norm_rand"  PRIVATE cxx_std_20)
   
  target_include_directories("host_norm_const" PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_include_directories("host_norm_rand"  PRIVATE ${PROJECT_SOURCE_DIR}/include)
     
  if(DEFINE_USE_DOUBLE_PRECISION)
    target_compile_definitions("host_norm_const" PRIVATE _USE_DOUBLE_PRECISION)
    target_compile_definitions("host_norm_rand"  PRIVATE _USE_DOUBLE_PRECISION)
  endif(DEFINE_USE_DOUBLE_PRECISION)
     
  if(DEFINE_STDPAR)
    target_compile_definitions("host_norm_const" PRIVATE _STDPAR)
    target_compile_definitions("host_norm_rand"  PRIVATE _STDPAR)
  endif(DEFINE_STDPAR)
    
  if( OpenMP_FOUND )
    target_link_libraries("host_norm_const" PRIVATE "-lgomp")
    target_link_libraries("host_norm_rand"  PRIVATE "-lgomp")
  endif()
    
  set_target_properties("host_norm_const" PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties("host_norm_rand"  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    
  message("CMAKE_CXX_FLAGS= ${CMAKE_CXX_FLAGS}")
   
else()
     
  message("CUDAToolkit is not found")
    
  set(CMAKE_CXX_FLAGS "-Wall -Wextra")
   
  add_executable("host_norm_const" "main_const.cpp")
  add_executable("host_norm_rand"  "main_rand.cpp")
   
  target_compile_features("host_norm_const" PRIVATE cxx_std_20)
  target_compile_features("host_norm_rand"  PRIVATE cxx_std_20)
   
  target_include_directories("host_norm_const" PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_include_directories("host_norm_rand"  PRIVATE ${PROJECT_SOURCE_DIR}/include)
    
  if(DEFINE_USE_DOUBLE_PRECISION)
    target_compile_definitions("host_norm_const" PRIVATE _USE_DOUBLE_PRECISION)
    target_compile_definitions("host_norm_rand"  PRIVATE _USE_DOUBLE_PRECISION)
  endif(DEFINE_USE_DOUBLE_PRECISION)
     
  if(MKL_FOUND )
    message("MKL is found.")
    target_link_libraries("host_norm_const" PRIVATE MKL::MKL)
    target_link_libraries("host_norm_rand"  PRIVATE MKL::MKL)
    if(DEFINE_MKL)
      target_compile_definitions("host_norm_const" PRIVATE _MKL)	    
      target_compile_definitions("host_norm_rand" PRIVATE _MKL)	    
    endif(DEFINE_MKL)
  else()
    message("MKL is not found.")
  endif()
     
  if(TBB_FOUND )
    message("TBB is found.")
    target_link_libraries("host_norm_const" PRIVATE TBB::tbb)
    target_link_libraries("host_norm_rand"  PRIVATE TBB::tbb)
    target_link_libraries("host_norm_const" PRIVATE -ltbb -lpthread )
    target_link_libraries("host_norm_rand"  PRIVATE -ltbb -lpthread )
    if(DEFINE_STDPAR)
      target_compile_definitions("host_norm_const" PRIVATE _STDPAR)
      target_compile_definitions("host_norm_rand"  PRIVATE _STDPAR)
    endif(DEFINE_STDPAR)
  endif()

  if( OpenMP_FOUND )
    target_link_libraries("host_norm_const" PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries("host_norm_rand"  PRIVATE OpenMP::OpenMP_CXX)
  else()
    message("OpenMP is not found.")
  endif()
     
  set_target_properties("host_norm_const" PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties("host_norm_rand"  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    
  message("CMAKE_CXX_FLAGS= ${CMAKE_CXX_FLAGS}")

endif()


