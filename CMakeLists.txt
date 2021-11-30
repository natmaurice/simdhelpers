cmake_minimum_required(VERSION 3.10)

include(CheckSymbolExists)
  
project(simdhelpers VERSION 1.4)

enable_testing()


# Export compilation commands.
# The purpose here is to make retrieving optimisation options easier.
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)


set(WARN_FLAGS "-Wall -Wextra -pedantic -Wno-unused-parameter -Wduplicated-branches -Wduplicated-cond -Wdangling-else")

set(ARCH_FLAGS "-march=native")

set(CMAKE_CXX_FLAGS "${WARN_FLAGS} ${ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "") # Clear default compiler flags

add_compile_options(
  -g
  "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb;-fsanitize=address;-fno-omit-frame-pointer;-fsanitize=undefined>"
#  "$<$<CONFIG:DEBUG>:${WARN_FLAGS}>"    
  )

add_compile_options(
  "$<$<CONFIG:RELEASE>:-O3;-march=native>"
  "$<$<CONFIG:RELEASE>:-mtune=native>"
  "$<$<CONFIG:RELEASE>:-march=native>"
  "$<$<CONFIG:RELEASE>:-fstrict-aliasing>"
  "$<$<CONFIG:RELEASE>:-DNDEBUG>"
#  "$<$<CONFIG:RELEASE>:${WARN_FLAGS}>"
  )

add_compile_options(
  "$<$<CONFIG:RelWithDebInfo>:-O3>"
  "$<$<CONFIG:RelWithDebInfo>:-march=native>"
#  "$<$<CONFIG:RELEASE>:${WARN_FLAGS}>"
  )


add_link_options(
  -g
  "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb;-fsanitize=address;-fno-omit-frame-pointer;-fsanitize=undefined>"    
  )

#     -fsanitize=address -fno-omit-frame-pointer -fsanitize=undefined

set(COMMON_LIBRARIES "m")

add_subdirectory(tests)
