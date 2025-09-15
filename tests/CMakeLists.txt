enable_testing()

# Analysis tests
add_executable(analyze-tests AnalyzeTests.test.cpp)
target_link_libraries(analyze-tests gtest_main keyrita-core)
add_test(analyze-gtests analyze-tests analyze-gtests)

# State tests
add_executable(state-tests StateTests.test.cpp)
target_link_libraries(state-tests gtest_main keyrita-core)
add_test(state-gtests state-tests state-gtests)
target_include_directories(state-tests PUBLIC api PRIVATE ../private_include)
