find_package(Catch2 REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/tests
  )

file (GLOB all_SRCs
  "${PROJECT_SOURCE_DIR}/tests/*.cpp"
  )

link_libraries(simdhelpers)

add_executable(simdhelpers_tests ${all_SRCs})
target_link_libraries(simdhelpers_tests PRIVATE Catch2::Catch2)

set_target_properties(simdhelpers_tests PROPERTIES OUTPUT_NAME tests)
