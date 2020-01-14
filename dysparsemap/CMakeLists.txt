add_library(dysparsemap src/sparsemap.cpp)
target_compile_options(dysparsemap PUBLIC -Ofast)
target_compile_options(dysparsemap PRIVATE -Werror -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(dysparsemap
    ad3qp
    Eigen3::Eigen
    Dynet::dynet
)

target_include_directories(dysparsemap
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

    PRIVATE
    src
)
