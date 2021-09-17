cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
# Force generation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

project(hw2)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# set(CMAKE_VERBOSE_MAKEFILE TRUE)

list(APPEND SRCS
  src/main.cpp
  src/superpixel_slic.cpp
  src/util.cpp
  )

add_executable(homework2 ${SRCS})
target_include_directories(homework2 PRIVATE src)
target_link_libraries(homework2 ${OpenCV_LIBS})
target_compile_options(homework2 PRIVATE -pedantic -Wall -Werror -march=native -g)
