cmake_minimum_required(VERSION 3.13)
project(dysparsemap LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR} CACHE STRING "" FORCE)
if(NOT EIGEN3_INCLUDE_DIR)
    set(EIGEN3_INCLUDE_DIR /home/vlad/src/eigen)
endif()

find_package(Eigen3 REQUIRED)
find_package(Dynet REQUIRED)

add_subdirectory(ad3qp)
add_subdirectory(dysparsemap)

file(GLOB TESTS test/*.cpp)
foreach(testin ${TESTS})
    get_filename_component(testout ${testin} NAME)
    string(REPLACE ".cpp" "" testout ${testout})
    add_executable(${testout} ${testin})
    target_link_libraries(${testout} PUBLIC ad3qp dysparsemap)
endforeach(testin ${TESTS})

export(TARGETS ad3qp dysparsemap
    NAMESPACE DySparseMAP::
    FILE DySparseMAPConfig.cmake)
export(PACKAGE DySparseMAP)
