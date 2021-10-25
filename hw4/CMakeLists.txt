cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
# Force generation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

project(hw4)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# set(CMAKE_VERBOSE_MAKEFILE TRUE)

list(APPEND SRCS
  src/main.cpp
  src/superpixel_slic.cpp
  src/util.cpp
  )

add_executable(homework4 ${SRCS})
target_include_directories(homework4 PRIVATE
  src
  ${OpenCV_INCLUDE_DIRS}
  )
target_link_libraries(homework4 PUBLIC
  Threads::Threads
  ${OpenCV_LIBS}
  )
target_compile_options(homework4 PRIVATE -pedantic -Wall -Werror -march=native -g)
