find_package(Catch2 REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/tests
  )

file (GLOB all_SRCs
  "${PROJECT_SOURCE_DIR}/tests/*.cpp"
)

include_directories("${simdhelpers_dir}/include")


add_executable(simdhelpers_tests ${simdhelpers_SRCs} ${tests_SRCs})
target_link_libraries(simdhelpers_tests PRIVATE simdhelpers-slib)
target_link_libraries(simdhelpers_tests PRIVATE Catch2::Catch2)

set_target_properties(simdhelpers_tests PROPERTIES OUTPUT_NAME tests)
