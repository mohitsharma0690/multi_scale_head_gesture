cmake_minimum_required(VERSION 2.6)
project("matHCRF")
#PUBLIC VERSION
ADD_DEFINITIONS (-D_PUBLIC)

# We have a few setup for the mex file extension to do
if(${UNIX})
  if (${APPLE})
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".mexmaci")
    set(CMAKE_SHARED_MODULE_SUFFIX  ".mexmaci")

  else()
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
    set(CMAKE_SHARED_MODULE_PREFIX  "")
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
      set(CMAKE_SHARED_LIBRARY_SUFFIX ".mexa64")
      set(CMAKE_SHARED_MODULE_SUFFIX  ".mexa64")
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86" OR ${CMAKE_SYSTEM_PROCESSOR}
        STREQUAL "i686")
      set(CMAKE_SHARED_LIBRARY_SUFFIX ".mexglx")
      set(CMAKE_SHARED_MODULE_SUFFIX  ".mexglx")
    endif()
  endif()
elseif(${WIN£)
  if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".mexw64")
    set(CMAKE_SHARED_MODULE_SUFFIX  ".mexw64")
  elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86" OR ${CMAKE_SYSTEM_PROCESSOR}
      STREQUAL "i686")
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".mexw32")
    set(CMAKE_SHARED_MODULE_SUFFIX  ".mexw32")
  endif()
endif()

set(CMAKE_CXX_FLAGS "-Wall")

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
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/../../libs/shared/hCRF hCRF)
INCLUDE_DIRECTORIES(${hCRF_SOURCE_DIR}/include/)

ADD_LIBRARY("matHCRF" MODULE "src/matHCRF.cpp")
# All the library we need
TARGET_LINK_LIBRARIES("matHCRF" hCRF)
TARGET_LINK_LIBRARIES("matHCRF" cgDescent)
TARGET_LINK_LIBRARIES("matHCRF" lbfgs)
TARGET_LINK_LIBRARIES("matHCRF" uncoptim)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR} "../../libs/3rdParty/matlab/include/")

SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../..)
INSTALL(TARGETS matHCRF
  LIBRARY DESTINATION bin)

