cmake_minimum_required(VERSION 3.10)

include(CheckSymbolExists)
  
project(simdhelpers VERSION 1.4)

enable_testing()


set(CMAKE_CXX_STANDARD_REQUIRED true)

set(src_dir src)
set(inc_dir include)
set(lib_dir lib)
set(simdhelpers_SRCs
  ${src_dir}/utils.cpp
  ${src_dir}/compress.cpp
  ${src_dir}/expand.cpp
  ${src_dir}/array_utils.cpp
  )

set(simdhelpers_dir "${CMAKE_CURRENT_SOURCE_DIR}/")

set(LIBRARY_OUTPUT_PATH ${lib_dir})


add_library(simdhelpers-obj OBJECT ${simdhelpers_SRCs})
set_target_properties(simdhelpers-obj PROPERTIES POSITION_INDEPENDENT_CODE ON) # set -fpic

add_library(simdhelpers-slib STATIC $<TARGET_OBJECTS:simdhelpers-obj>)
set_target_properties(simdhelpers-slib PROPERTIES OUTPUT_NAME simdhelpers POSITION_INDEPENDENT_CODE ON) # set -fpic

#list(APPEND lsl_none_target_list simdhelpers-obj simdhelpers-slib)
#list(APPEND lsl_target_list simdhelpers-obj simdhelpers-slib)
target_include_directories(simdhelpers-obj PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${inc_dir}/>)
target_include_directories(simdhelpers-slib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${inc_dir}/>)

export(TARGETS simdhelpers-slib NAMESPACE simdhelpers:: FILE "${lib_dir}/cmake/simdhelpers/${target-scalar-name}-config.cmake")

#add_library(simdhelpers ${simdhelpers_SRCs})
#add_library(simdhelpers::simdhelpers ALIAS simdhelpers)


if (SIMDHELPERS_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
