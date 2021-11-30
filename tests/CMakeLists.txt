find_package(Catch2 REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/tests
  )

file (GLOB all_SRCs
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/tests/*.cpp"
  )

add_executable(tests ${all_SRCs})
target_link_libraries(tests PRIVATE Catch2::Catch2)
