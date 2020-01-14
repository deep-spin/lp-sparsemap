add_library(ad3qp

    ad3/Factor.cpp
    ad3/FactorGraph.cpp
    ad3/GenericFactor.cpp
    ad3/Utils.cpp
)

target_link_libraries(ad3qp Eigen3::Eigen)
target_compile_options(ad3qp PUBLIC -Ofast -funroll-loops)
target_compile_options(ad3qp PRIVATE
    -Werror
    -Wall
    -Wextra
    -Wno-unused-parameter)
target_include_directories(ad3qp
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
