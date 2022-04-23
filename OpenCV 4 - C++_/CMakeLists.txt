# Top-Level CmakeLists.txt to build AIA projects
project( aia )
cmake_minimum_required(VERSION 3.1)
cmake_policy(SET CMP0054 OLD) 
cmake_policy(SET CMP0045 OLD)

# hide unused predefined variables
set (CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE INTERNAL "")

# include cmake scripts
include(cmake/utils.cmake)

# require C++11
set (CMAKE_CXX_STANDARD 11)

# set up library and binary directories
if(NOT LIBRARY_OUTPUT_PATH)
  set (LIBRARY_OUTPUT_PATH ${aia_BINARY_DIR}/libs CACHE INTERNAL "Single output directory for building all libraries.")
endif(NOT LIBRARY_OUTPUT_PATH)
if(NOT EXECUTABLE_OUTPUT_PATH)
  set (EXECUTABLE_OUTPUT_PATH ${aia_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
endif(NOT EXECUTABLE_OUTPUT_PATH)

# set up OpenCV library
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV library not found or not properly installed")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

# set up "Release" and "Debug" as default build mode
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING 
        "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." 
        FORCE) # Default to Release
endif()

# disable annoying warnings on MSVC compilers
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
	warnings_disable(CMAKE_CXX_FLAGS /wd4290) # vs2012: can't properly handle exceptions
	warnings_disable(CMAKE_CXX_FLAGS /wd4996) # vs2012: complains about unsafe standard C++ functions
	warnings_disable(CMAKE_CXX_FLAGS /wd4530) # vs2012: C++ exception handler used, but unwind semantics are not enabled
	warnings_disable(CMAKE_CXX_FLAGS /wd4503) # vs2012: decorated name length exceeded
endif()

# disable annoying warning on Clang
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-varargs")
endif()


# define folder where example images are stored
add_definitions(-DEXAMPLE_IMAGES_PATH="${aia_SOURCE_DIR}/example-images")

# build modules
add_subdirectory( 3rdparty )
add_subdirectory( myproject )
add_subdirectory( gray_shades )
add_subdirectory( hello_world )
add_subdirectory( median_compare )
add_subdirectory( gamma_correction )
add_subdirectory( histogram_calculation )
add_subdirectory( image_normalization )
add_subdirectory( mammogram_log )
add_subdirectory( histogram_equalization_color )
add_subdirectory( histogram_equalization )
add_subdirectory( denoise_salt_and_pepper )
add_subdirectory( denoise_gaussian )
add_subdirectory( gradient_image )
add_subdirectory( binarization )
add_subdirectory( sharpening_laplacian )
add_subdirectory( sharpening_photoshop )
add_subdirectory( object_picking )
add_subdirectory( opening_closing )
add_subdirectory( junctions_matching )
add_subdirectory( contour_filling )
add_subdirectory( hough_line_detection )
add_subdirectory( long_characters_reconstruction )
add_subdirectory( tools_opening_by_reconstruction )
add_subdirectory( traffic_light_detection )
add_subdirectory( traffic_light_recognition )
add_subdirectory( alternating_sequential_filtering )
add_subdirectory( ilumination_correction )
add_subdirectory( grayscale_morphological_reconstruction )
add_subdirectory( morphological_gradient )
add_subdirectory( gradient_based_edge_detection )
add_subdirectory( LoG_based_edge_detection )