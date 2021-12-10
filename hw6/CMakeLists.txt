cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0104 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

set(CMAKE_C_COMPILER nvcc)
set(CMAKE_CXX_COMPILER nvcc)

project(hw5 LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

find_package(OpenCV REQUIRED)
list(TRANSFORM OpenCV_LIBS PREPEND -l)
string(REPLACE ";" "," OpenCV_LIBS_NVCC "${OpenCV_LIBS}")

include(CheckLanguage)
check_language(CUDA)

#set(CMAKE_VERBOSE_MAKEFILE TRUE)

list(APPEND SRCS
  src/main.cpp
  src/gold_standard.cpp
  src/util.cpp
  src/superpixel_gslic.cu
  )

add_executable(homework6 ${SRCS})

set_target_properties(homework6 PROPERTIES
  CUDA_ARCHITECTURES "52;60;61;70;75;80;86"
  )

target_include_directories(homework6 PRIVATE
  /usr/local/cuda/samples/common/inc
  ${OpenCV_INCLUDE_DIRS}
  src
  )

target_compile_features(homework6 PRIVATE cxx_std_14)

target_compile_options(homework6 PRIVATE
  --threads 0
  --compiler-options -Wall,-Werror
  $<$<CONFIG:Debug>:-g;-G>
  )

target_compile_definitions(homework6 PRIVATE
  )

target_link_libraries(homework6 PUBLIC
  --linker-options ${OpenCV_LIBS_NVCC}
  )
