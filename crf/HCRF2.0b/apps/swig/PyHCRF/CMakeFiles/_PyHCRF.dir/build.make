# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF

# Include any dependencies generated for this target.
include CMakeFiles/_PyHCRF.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/_PyHCRF.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/_PyHCRF.dir/flags.make

PyHCRFPYTHON_wrap.cxx: PyHCRF.i
	$(CMAKE_COMMAND) -E cmake_progress_report /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Swig source"
	/usr/bin/cmake -E make_directory /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF
	/usr/bin/swig2.0 -python -I/home/mohits1/Projects/HCRF2.0b/libs/shared/hCRF/include/ -outdir /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF -c++ -I/usr/include/python2.7 -I/home/mohits1/Projects/HCRF2.0b/libs/shared/hCRF/include -o /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF/PyHCRFPYTHON_wrap.cxx /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF/PyHCRF.i

PyHCRF.py: PyHCRFPYTHON_wrap.cxx

CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o: CMakeFiles/_PyHCRF.dir/flags.make
CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o: PyHCRFPYTHON_wrap.cxx
	$(CMAKE_COMMAND) -E cmake_progress_report /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o -c /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF/PyHCRFPYTHON_wrap.cxx

CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF/PyHCRFPYTHON_wrap.cxx > CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.i

CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF/PyHCRFPYTHON_wrap.cxx -o CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.s

CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o.requires:
.PHONY : CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o.requires

CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o.provides: CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o.requires
	$(MAKE) -f CMakeFiles/_PyHCRF.dir/build.make CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o.provides.build
.PHONY : CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o.provides

CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o.provides.build: CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o

# Object files for target _PyHCRF
_PyHCRF_OBJECTS = \
"CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o"

# External object files for target _PyHCRF
_PyHCRF_EXTERNAL_OBJECTS =

_PyHCRF.so: CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o
_PyHCRF.so: CMakeFiles/_PyHCRF.dir/build.make
_PyHCRF.so: /usr/lib/x86_64-linux-gnu/libpython2.7.so
_PyHCRF.so: hCRF/libhCRF.a
_PyHCRF.so: hCRF/cgdescent/libcgDescent.a
_PyHCRF.so: hCRF/lbfgs/liblbfgs.a
_PyHCRF.so: hCRF/uncOptim/libuncoptim.a
_PyHCRF.so: CMakeFiles/_PyHCRF.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared module _PyHCRF.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/_PyHCRF.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/_PyHCRF.dir/build: _PyHCRF.so
.PHONY : CMakeFiles/_PyHCRF.dir/build

CMakeFiles/_PyHCRF.dir/requires: CMakeFiles/_PyHCRF.dir/PyHCRFPYTHON_wrap.cxx.o.requires
.PHONY : CMakeFiles/_PyHCRF.dir/requires

CMakeFiles/_PyHCRF.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/_PyHCRF.dir/cmake_clean.cmake
.PHONY : CMakeFiles/_PyHCRF.dir/clean

CMakeFiles/_PyHCRF.dir/depend: PyHCRFPYTHON_wrap.cxx
CMakeFiles/_PyHCRF.dir/depend: PyHCRF.py
	cd /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF /home/mohits1/Projects/HCRF2.0b/apps/swig/PyHCRF/CMakeFiles/_PyHCRF.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/_PyHCRF.dir/depend
