cmake_minimum_required(VERSION 3.14)

project(CDL
    VERSION 0.1.0
    DESCRIPTION "Celeres Data Loader — O(1)-memory shuffling for deep learning"
    LANGUAGES CXX
)

# ============================================================================
# Build Options
# ============================================================================
option(CDL_BUILD_TESTS "Build C++ unit tests" OFF)
option(CDL_BUILD_PYTHON "Build Python bindings (requires pybind11)" OFF)

# ============================================================================
# C++ Standard and Compiler Flags
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position-independent code (required for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler-specific flags
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# ============================================================================
# CDL Header-Only Library
# ============================================================================
add_library(cdl INTERFACE)
target_include_directories(cdl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cdl/ext/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cdl INTERFACE cxx_std_17)

# ============================================================================
# Python Bindings (pybind11)
# ============================================================================
if(CDL_BUILD_PYTHON)
    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND)
        # Fall back to pip-installed pybind11
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(PYBIND11_CMAKE_DIR)
            list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
            find_package(pybind11 CONFIG REQUIRED)
        else()
            # Last resort: use pip to find pybind11
            execute_process(
                COMMAND pip show pybind11 --files
                OUTPUT_VARIABLE PYBIND11_INFO
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
                OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
            find_package(pybind11 CONFIG REQUIRED)
        endif()
    endif()

    pybind11_add_module(_cdl_native
        src/cdl/ext/bindings.cpp
    )
    target_link_libraries(_cdl_native PRIVATE cdl)
    target_compile_definitions(_cdl_native PRIVATE CDL_VERSION_STRING="${PROJECT_VERSION}")

    # Install the Python extension into the built wheel
    install(TARGETS _cdl_native LIBRARY DESTINATION .)
endif()

# ============================================================================
# C++ Tests
# ============================================================================
if(CDL_BUILD_TESTS)
    add_executable(cdl_test_bijection tests/cpp/test_bijection.cpp)
    target_link_libraries(cdl_test_bijection PRIVATE cdl)
    enable_testing()
    add_test(NAME bijection_test COMMAND cdl_test_bijection)
endif()

# ============================================================================
# Installation
# ============================================================================
install(DIRECTORY src/cdl/ext/include/cdl DESTINATION include)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "CDL Configuration Summary")
message(STATUS "=========================")
message(STATUS "Version:       ${PROJECT_VERSION}")
message(STATUS "Build Tests:   ${CDL_BUILD_TESTS}")
message(STATUS "Build Python:  ${CDL_BUILD_PYTHON}")
message(STATUS "")
