cmake_minimum_required(VERSION 2.6)
project(uncOptim)

IF(UNIX)
  IF(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  ENDIF()
ENDIF()

INCLUDE_DIRECTORIES(${uncOptim_SOURCE_DIR}/include)
add_library(uncoptim src/uncoptim.cpp)

SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
INSTALL_TARGETS(/lib uncoptim)
