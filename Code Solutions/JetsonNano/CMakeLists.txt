cmake_minimum_required(VERSION 3.0)
project(TFG_SCANNER)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

add_executable(main src/main.cc)

# RealSense Library
find_package(realsense2 REQUIRED)
include_directories(include ${realsense_INCLUDE_DIR})
target_include_directories(main PRIVATE ${realsense_INCLUDE_DIR})
target_link_libraries(main ${realsense2_LIBRARY})

# Point Cloud Library
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(main ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FILTERS_LIBRARIES})

# JsonCpp Library
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})
target_link_libraries(main ${JSONCPP_LIBRARIES})
configure_file(parameters.json parameters.json COPYONLY)

# CUDA Library
find_package(CUDA 10.2 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(main ${CUDA_LIBRARIES})

# CUDA ICP Local Library
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(libcudaicp SHARED IMPORTED)
set_target_properties(libcudaicp PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/libcudaicp.so)
target_link_libraries(main libcudaicp)
