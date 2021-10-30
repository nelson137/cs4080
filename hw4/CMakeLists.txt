cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
# Force generation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE CACHE INTERNAL "")

# Use MPI wrapper compilers
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpic++)

project(hw4)

find_package(OpenCV REQUIRED)

find_package(MPI REQUIRED)
# Get include dirs used by mpic++ so they show up in compile_commands.json to
# make Intellisense happy
execute_process(
  COMMAND mpic++ --showme:incdirs
  COMMAND tr " " ";"
  OUTPUT_VARIABLE MY_MPI_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

set(CMAKE_CXX_STANDARD 14)
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
  ${MY_MPI_INCLUDE_DIRS}
  )
target_link_libraries(homework4 PUBLIC
  "$<$<CONFIG:DEBUG>:asan>"
  ${OpenCV_LIBS}
  )
# CMake automatically adds -g and -O3 for DEBUG and RELEASE, respecitively
target_compile_options(homework4 PRIVATE
  -pedantic -Wall -Werror
  "$<$<CONFIG:DEBUG>:-fsanitize=address>"
  "$<$<CONFIG:RELEASE>:-march=native>"
  )
# CMake already adds -DNDEBUG for RELEASE but I don't like the double negative
target_compile_definitions(homework4 PRIVATE
  "$<$<CONFIG:DEBUG>:DEBUG>"
  )
