# Driver that runs an IREE pass pipeline in process with the collector
# installed. It links the IREE compiler out of an existing IREE build tree.
#
#   cmake -G Ninja -S examples/iree -B build/iree-harness \
#         -DIREE_ROOT=/path/to/iree -DIREE_BUILD=/path/to/iree/build
#   cmake --build build/iree-harness
#
# Expect several minutes and a multi-GB binary: it statically links the whole
# compiler plus MLIR and LLVM, with debug info.
#
# The IREE build in question is not an archive build: its compiler objects stay
# as individual .o files, so the link line is assembled by globbing the build
# tree rather than by importing CMake targets. `-Wl,--start-group` is required
# because those objects reference each other in cycles.
cmake_minimum_required(VERSION 3.20)
project(mlir_compile_analysis_iree LANGUAGES CXX)

set(IREE_ROOT "" CACHE PATH "IREE source root")
if(NOT IREE_ROOT)
  message(FATAL_ERROR "pass -DIREE_ROOT=/path/to/iree")
endif()
set(IREE_BUILD "${IREE_ROOT}/build" CACHE PATH "IREE build root")
set(LLVM_SRC "${IREE_ROOT}/third_party/llvm-project")
set(LLVM_BUILD "${IREE_BUILD}/llvm-project")

if(NOT EXISTS "${IREE_BUILD}/compiler/src/iree/compiler/API")
  message(FATAL_ERROR
    "no IREE compiler objects under '${IREE_BUILD}/compiler/src'; "
    "pass -DIREE_BUILD=")
endif()

add_executable(iree-harness iree-harness.cpp)
target_compile_features(iree-harness PRIVATE cxx_std_17)
# Same footing as IREE's own compiler objects: no RTTI, no exceptions, and
# assertions live (IREE strips -DNDEBUG from its compiler flags in
# iree_fix_ndebug, so -UNDEBUG keeps this translation unit consistent).
target_compile_options(iree-harness PRIVATE -UNDEBUG -fno-rtti -fno-exceptions
                                            -w)
target_compile_definitions(iree-harness PRIVATE
  IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl
  IREE_HAVE_LIBBACKTRACE=1
  IREE_THREADING_ENABLE=1)
target_include_directories(iree-harness PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
# SYSTEM hides upstream warnings; they are not ours to fix.
target_include_directories(iree-harness SYSTEM PRIVATE
  "${IREE_ROOT}" "${IREE_BUILD}"
  "${IREE_ROOT}/compiler/src" "${IREE_BUILD}/compiler/src"
  "${IREE_ROOT}/runtime/src" "${IREE_BUILD}/runtime/src"
  "${IREE_BUILD}/runtime/src/iree/base/internal/flatcc"
  "${IREE_ROOT}/compiler/bindings/c" "${IREE_BUILD}/compiler/bindings/c"
  "${IREE_ROOT}/third_party/flatcc/include"
  "${IREE_ROOT}/third_party/llvm-project/llvm/include"
  "${LLVM_BUILD}/include"
  "${IREE_ROOT}/third_party/llvm-project/mlir/include"
  "${LLVM_BUILD}/tools/mlir/include"
  "${IREE_ROOT}/third_party/llvm-project/lld/include"
  "${LLVM_BUILD}/tools/lld/include")

# IREE ships its libraries as archives under <build>/lib, so the link line is
# assembled from archives rather than from the individual object files. Runtime
# pieces (the tflite shim among them), stablehlo and the vendored flatcc and
# libbacktrace archives complete the set. `-Wl,--start-group` is required
# because the compiler archives reference each other in cycles.
file(GLOB _iree_archives
  "${IREE_BUILD}/lib/*.a"
  "${IREE_BUILD}/build_tools/third_party/*/*.a"
  "${IREE_BUILD}/llvm-external-projects/*/*.a"
  "${IREE_BUILD}/llvm-external-projects/*/*/*.a")
# stablehlo is built as plain objects outside the <build>/lib archives.
file(GLOB_RECURSE _stablehlo_objects
  "${IREE_BUILD}/llvm-external-projects/stablehlo/*.o")
# tblgen tools and integration builders each carry their own main().
list(FILTER _stablehlo_objects EXCLUDE REGEX "_tblgen|/integrations/|/tools/")

file(GLOB_RECURSE _runtime_archives "${IREE_BUILD}/runtime/*.a")

file(GLOB _mlir_llvm_archives
  "${LLVM_BUILD}/lib/libMLIR*.a" "${LLVM_BUILD}/lib/libLLVM*.a"
  "${LLVM_BUILD}/lib/liblld*.a")

if(NOT _iree_archives)
  message(FATAL_ERROR "no IREE archives under '${IREE_BUILD}/lib'")
endif()

target_link_libraries(iree-harness PRIVATE
  -Wl,--start-group ${_iree_archives} ${_runtime_archives} ${_stablehlo_objects}
  ${_mlir_llvm_archives}
  -Wl,--end-group
  pthread dl m z rt lzma)
