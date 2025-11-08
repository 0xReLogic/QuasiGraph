# Test configuration for QuasiGraph

# Simple test runner without GTest
add_executable(basic_tests basic_tests.cpp)
target_link_libraries(basic_tests QuasiGraph)

# Add as CTest
add_test(NAME BasicTests COMMAND basic_tests)
