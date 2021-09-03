cmake_minimum_required(VERSION 3.10)

project(hw0)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# set(CMAKE_VERBOSE_MAKEFILE TRUE)

list(APPEND SRCS
  src/main.cpp
  )

add_executable(homework0 ${SRCS})
target_link_libraries(homework0 ${OpenCV_LIBS})
target_compile_options(homework0 PRIVATE -pedantic -Wall -Werror)
