cmake_minimum_required(VERSION 2.6)
project(hCRF)
#PUBLIC VERSION
ADD_DEFINITIONS(-DUSELBFGS -D_PUBLIC)

# We want -fPIC on linux, to be able to link to dynamic library
# IF(${UNIX})
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
# ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra ")

# For debugging. We also want the extra warning of Effective c++
IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
  ADD_DEFINITIONS(-D_DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++")
ENDIF()

# This is for gnu c++ compiler OPENMP support.
# We have to compile with vec_features (to avoid shared object)
if(OPENMP GREATER 0)
  message("Compiling with openmp support")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  ADD_DEFINITIONS(-D_VEC_FEATURES)
endif()

if (PROFILING GREATER 0)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# We need to add other subdirectory
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/../../3rdParty/cgDescent  cgdescent)
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/../../3rdParty/liblbfgs  lbfgs)
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/../../3rdParty/uncOptim  uncOptim)
# The file we need to include
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${ASA_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${cgDescent_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${liblbfgs_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${owlqn_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${uncOptim_SOURCE_DIR}/include)

# This is the list of source files
SET(LIB_SRC src/dataset.cpp src/Dummys.cpp src/EdgeFeatures.cpp
	src/evaluator.cpp src/evaluatorCRF.cpp src/evaluatorHCRF.cpp
	src/evaluatorLDCRF.cpp src/featuregenerator.cpp src/FeaturesOne.cpp src/gradient.cpp
	src/GradientCRF.cpp src/GradientFD.cpp src/GradientHCRF.cpp
	src/GradientLDCRF.cpp src/InferenceEngineBrute.cpp src/InferenceEngine.cpp
	src/InferenceEngineFB.cpp src/LabelEdgeFeatures.cpp src/model.cpp
	src/MyFeatures.cpp src/MyToolbox.cpp
	src/OptimizerCG.cpp src/optimizer.cpp src/OptimizerLBFGS.cpp
	src/OptimizerUncOptim.cpp src/RawFeatures.cpp
	src/RawFeaturesSquare.cpp src/toolbox.cpp
	src/ToolboxCRF.cpp src/ToolboxGHCRF.cpp src/ToolboxHCRF.cpp
	src/ToolboxLDCRF.cpp src/WindowRawFeatures.cpp src/matrix.cpp src/matrixSSE2.cpp)

add_library(hCRF ${LIB_SRC})

SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
INSTALL_TARGETS(/lib hCRF)
