cmake_minimum_required(VERSION 2.8.12)
SET(VERSION_REQ_CUDA "8.0")

project(OpenPoseTest)

#for openpose begin:
add_definitions(-std=c++11)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

find_package(CUDA ${VERSION_REQ_CUDA} REQUIRED)
find_package(GFlags)
find_package(Glog)
#find_package(OpenCV REQUIRED)
find_package(OpenPose REQUIRED)
find_package(ZED 2 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenPose_INCLUDE_DIRS} ${GFLAGS_INCLUDE_DIR} ${GLOG_INCLUDE_DIR})# ${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/opencv3.4.6/include)
include_directories(${ZED_INCLUDE_DIRS})
#openpose end

#for opencv libs begin:
link_directories(${ZED_LIBRARY_DIR})
link_directories(/usr/local/opencv3.4.6/lib/)
link_directories(${CUDA_LIBRARY_DIRS})
#opencv end

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

#Create target
add_executable(${PROJECT_NAME} "op_zed.cpp")

target_link_libraries(${PROJECT_NAME} ${CUDA_nppi_LIBRARY} ${CUDA_LIBRARIES} ${OpenPose_LIBS} ${GFLAGS_LIBRARY} ${GLOG_LIBRARY} ${ZED_LIBRARIES})

#for openpose:
#target_link_libraries(${PROJECT_NAME} ${OpenPose_LIBS} ${GFLAGS_LIBRARY} ${GLOG_LIBRARY})# ${OpenCV_LIBS})
