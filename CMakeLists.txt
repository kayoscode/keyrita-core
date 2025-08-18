cmake_minimum_required(VERSION 3.20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_testing()
set(CMAKE_COLOR_DIAGNOSTICS ON)

set(EXE_NAME keyrita-core)
project(${EXE_NAME} LANGUAGES CXX)

set(SRC
        src/main.cpp
)

set(HEADERS
)

# External libs
include(FetchDependencies.cmake)

add_executable(${EXE_NAME}_exec ${SRC} ${HEADERS})
add_library(${EXE_NAME} ${SRC} ${HEADERS})

target_include_directories(${EXE_NAME} PUBLIC api PRIVATE_HEADER private_include)
# target_link_libraries(${EXE_NAME}
# )

# Add tests
add_subdirectory(tests)