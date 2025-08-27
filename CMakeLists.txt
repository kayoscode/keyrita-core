cmake_minimum_required(VERSION 3.20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_testing()
set(CMAKE_COLOR_DIAGNOSTICS ON)

set(EXE_NAME keyrita-core)
project(${EXE_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SRC
        src/main.cpp
)

set(HEADERS
        # Public api
        api/keyrita_core/State/MatrixAlloc.hpp
        api/keyrita_core/State/MatrixQuery.hpp
        api/keyrita_core/State/MatrixUtils.hpp
        api/keyrita_core/State/MatrixState.hpp
        api/keyrita_core/State/ScalarState.hpp
        api/keyrita_core/State/State.hpp
        api/keyrita_core/State/StateBase.hpp

        # Private utils
        private_include/DebugBreak.hpp
        private_include/Timer.hpp
)

# External libs
include(FetchDependencies.cmake)

add_executable(${EXE_NAME}_exec ${SRC} ${HEADERS})
add_library(${EXE_NAME} ${SRC} ${HEADERS})

target_include_directories(${EXE_NAME}_exec PUBLIC api PRIVATE private_include)
target_include_directories(${EXE_NAME} PUBLIC api PRIVATE private_include)

target_link_libraries(${EXE_NAME}_exec cpp-events)
target_link_libraries(${EXE_NAME} cpp-events)

# Add tests
add_subdirectory(tests)