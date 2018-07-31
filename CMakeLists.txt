cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(MBS)

set(CMAKE_CXX_FLAGS "-std=c++11")


find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/opt/intel/ipp/include")
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

link_directories("/opt/intel/ipp/lib/intel64/")
set(IPP_LIBS -lippi -lipps -lippcore -lippcv)
message(${IPP_LIBS})

add_executable (MBS src/main.cpp src/MBS.cpp
        src/getParam.cpp include/MBS.hpp)
target_link_libraries (MBS ${PCL_LIBRARIES} ${freenect2_LIBRARIES} ${IPP_LIBS} ${OpenCV_LIBS} ${BOOST_LIBS})
