cmake_minimum_required(VERSION 3.16)
project(QuasiGraph VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard to C++20 for modern features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler optimizations for performance
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /DDEBUG")
endif()

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/Graph.cpp
    src/QuasiPolynomial.cpp
    src/IndependentSet.cpp
    src/StructuralDecomposition.cpp
    src/SocialNetworkAnalysis.cpp
)

# Header files
set(HEADERS
    include/QuasiGraph/Graph.h
    include/QuasiGraph/QuasiPolynomial.h
    include/QuasiGraph/IndependentSet.h
    include/QuasiGraph/StructuralDecomposition.h
    include/QuasiGraph/SocialNetworkAnalysis.h
    include/QuasiGraph/QuasiGraph.h
)

# Create library
add_library(QuasiGraph STATIC ${SOURCES} ${HEADERS})

# Create executable for examples
add_executable(quasigraph_examples examples/main.cpp)
target_link_libraries(quasigraph_examples QuasiGraph)

# Create executable for benchmarks
add_executable(quasigraph_bench benchmarks/benchmark.cpp)
target_link_libraries(quasigraph_bench QuasiGraph)

# Installation
install(TARGETS QuasiGraph
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES ${HEADERS} DESTINATION include/QuasiGraph)

# Testing
enable_testing()
add_subdirectory(tests)

# Documentation
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in 
                   ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()
