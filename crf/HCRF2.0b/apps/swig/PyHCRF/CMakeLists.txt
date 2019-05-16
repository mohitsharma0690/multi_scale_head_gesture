cmake_minimum_required(VERSION 2.6)
project("PyHCRF")
#PUBLIC VERSION
ADD_DEFINITIONS (-D_PUBLIC)

#Where we want to install the files
SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../../..)
# This is for swig
FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

# And we also need to have a python library
FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
message(${PYTHON_INCLUDE_PATH})
SET(CMAKE_SWIG_FLAGS "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# This is for gnu c++ compiler, openmp support
if(OPENMP GREATER 0)
  message("Compiling with openmp support")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# On unix, we need fPIC to compile shared library
IF(UNIX)
  MESSAGE("fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  # We also have some C code in the code base
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
ENDIF()


# We need hCRF project
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/../../../libs/shared/hCRF hCRF)
INCLUDE_DIRECTORIES(${hCRF_SOURCE_DIR}/include/)

SET_SOURCE_FILES_PROPERTIES(PyHCRF.i PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(PyHCRF.i PROPERTIES SWIG_FLAGS "-I${hCRF_SOURCE_DIR}/include/")
SWIG_ADD_MODULE("PyHCRF" python PyHCRF.i)
SWIG_LINK_LIBRARIES("PyHCRF" ${PYTHON_LIBRARIES})


# All the library we need
SWIG_LINK_LIBRARIES("PyHCRF" hCRF)
SWIG_LINK_LIBRARIES("PyHCRF" cgDescent)
SWIG_LINK_LIBRARIES("PyHCRF" lbfgs)

SWIG_LINK_LIBRARIES("PyHCRF" uncoptim)


INSTALL(FILES ${PROJECT_BINARY_DIR}/PyHCRF.py
  ${PROJECT_BINARY_DIR}/_PyHCRF.so
   DESTINATION bin)

