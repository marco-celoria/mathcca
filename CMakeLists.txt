# Build
cmake_minimum_required(VERSION 3.12)

project(hello VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

add_library(${PROJECT_NAME} INTERFACE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

message(PROJECT_NAME="${PROJECT_NAME}")

#target_include_directories(
#    ${PROJECT_NAME}
#    INTERFACE
#    $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}>
#)

# Test
set(GTest_ROOT /u/dssc/mceloria/googletest)
add_subdirectory(${GTest_ROOT} "${CMAKE_CURRENT_BINARY_DIR}/googletest" EXCLUDE_FROM_ALL)
message(CMAKE_CURRENT_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}")

find_package(GTest REQUIRED)
add_executable(hello_test test/test.cpp)
target_link_libraries(hello_test PRIVATE GTest::gtest GTest::gtest_main)
target_include_directories(hello_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(AllAddTests hello_test)
enable_testing()


# Import the module
include (CheckLanguage)
# Check whether CUDA is supported
# export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/cuda/12.6/bin:$PATH
#set(CUDAToolkit_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/cuda/12.6/)
#export PATH=/u/dssc/mceloria/intel/oneapi/mkl/2025.1/

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/mkl/2025.1/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/dssc/mceloria/intel/oneapi/tbb/2022.1/lib
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # CUDA language is supported. It is safe to enable it.
  enable_language(CUDA)
  add_executable(main main.cu)
  target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR}/include)
  find_package(CUDAToolkit)
  IF( CUDAToolkit_FOUND )
    message("CUDAToolkit_FOUND - ${CUDAToolkit_INCLUDE_DIRS}")
    OPTION(_CUBLAS "Option description" ON)
    OPTION(_PARALG "Option description" ON)
    target_link_libraries(main PRIVATE CUDA::curand)
    target_link_libraries(main PRIVATE CUDA::cublas)
  else()
    message("CUDAToolkit is not found.")
  endif()
else()
  add_executable(main main.cpp)
  target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR}/include)
  find_package(MKL)
  IF(MKL_FOUND )
	  message("MKL is found.")
    target_link_libraries(main PUBLIC MKL::MKL)
    OPTION(DEFINE_MKL "Option description" ON)
    IF(DEFINE_MKL)
	    add_compile_definitions(-D_MKL)
    ENDIF(DEFINE_MKL)
  else()
	    message("MKL is not found.")
  endif()
  find_package(TBB)
  IF(TBB_FOUND )
    message("TBB is found.")
    target_link_libraries(main PUBLIC TBB::tbb)
    OPTION(DEFINE_PARALG "Option description" ON)
    IF(DEFINE_PARALG)
	    add_compile_definitions(-D_PARALG)
    ENDIF(DEFINE_PARALG)
  else()
    message("TBB is not found.")
  endif()	  
endif()	


