cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0104 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

#set(CMAKE_VERBOSE_MAKEFILE TRUE)

set(CMAKE_C_COMPILER nvcc)
set(CMAKE_CXX_COMPILER nvcc)

################################################################################
## PROJECT hw6
################################################################################

project(hw6 LANGUAGES CXX CUDA)

# Dependencies

include(CheckLanguage)
check_language(CUDA)

find_package(CUDAToolkit)

find_package(OpenCV REQUIRED)
list(TRANSFORM OpenCV_LIBS PREPEND -l)
string(REPLACE ";" "," OpenCV_LIBS_NVCC "${OpenCV_LIBS}")

################################################################################
## TARGET homework6
################################################################################

add_executable(homework6
  src/main.cu
  src/util.cpp
  src/gold_standard.cpp
  src/superpixel_gslic.cu
  )

set_target_properties(homework6 PROPERTIES
  CUDA_ARCHITECTURES "52;60;61;70;75;80;86"
  CUDA_SEPARABLE_COMPILATION ON
  )

target_compile_features(homework6 PRIVATE cxx_std_14)

target_compile_options(homework6 PRIVATE
  --threads 0
  $<$<CONFIG:Debug>:-G>
  --relocatable-device-code=true
  --compiler-options -Wall,-Werror
  )

target_include_directories(homework6 PRIVATE
  /usr/local/cuda/samples/common/inc
  ${OpenCV_INCLUDE_DIRS}
  src
  )

target_link_libraries(homework6 PRIVATE
  -lcudadevrt
  --linker-options ${OpenCV_LIBS_NVCC}
  )

add_custom_command(TARGET homework6
  POST_BUILD
  COMMAND ln -fs homework6 homework6-no-gold
  VERBATIM
  )
