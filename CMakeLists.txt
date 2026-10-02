cmake_minimum_required(VERSION 3.18)
project(spicyr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Eigen (headers only): a system copy if there is one, else EIGEN_INCLUDE_DIR, else download 3.4.0.
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  if(DEFINED ENV{EIGEN_INCLUDE_DIR})
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES $ENV{EIGEN_INCLUDE_DIR})
  else()
    include(FetchContent)
    FetchContent_Declare(eigen
      URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
      SOURCE_SUBDIR do-not-configure)
    FetchContent_MakeAvailable(eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${eigen_SOURCE_DIR})
  endif()
endif()

find_package(Threads REQUIRED)
file(GLOB CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/core/src/*.cpp)
pybind11_add_module(_core src/bindings.cpp ${CORE_SOURCES})
target_include_directories(_core PRIVATE src/core/include)
target_link_libraries(_core PRIVATE Eigen3::Eigen Threads::Threads)
install(TARGETS _core DESTINATION spicyr)
