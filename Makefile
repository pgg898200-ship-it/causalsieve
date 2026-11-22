UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(origin CXX),default)
ifeq ($(UNAME_S),Darwin)
CXX := clang++
else
CXX := g++
endif
endif

CXXFLAGS := -std=c++20 -Wall -Wextra -Werror -fno-math-errno -fno-trapping-math

ifeq ($(UNAME_S),Darwin)
SHLIB_EXT   := dylib
SHLIB_FLAGS  = -dynamiclib -install_name @rpath/$(notdir $@)
DL_LIB      :=
else
SHLIB_EXT   := so
SHLIB_FLAGS := -shared
DL_LIB      := -ldl
endif

IS_X86_64 := $(filter x86_64 amd64,$(UNAME_M))

ifneq ($(IS_X86_64),)
# Multi-ISA runtime CPUID dispatch on x86_64 (even when CAUSALSIEVE_PORTABLE=1):
# core + scalar TU compile portable baseline x86-64; AVX2 TU compiles with -mavx2 -mfma.
CORE_OPT_FLAGS   := -O3 -DCAUSALSIEVE_HAS_AVX2_TU=1
SCALAR_TU_FLAGS  := -O3
AVX2_TU_FLAGS    := -O3 -mavx2 -mfma -DCAUSALSIEVE_BUILD_AVX2=1
SCALAR_ONLY_FLAGS := -O3 -DCAUSALSIEVE_FORCE_SCALAR=1 -mno-avx2 -mno-fma
ASAN_CORE_FLAGS  := -O2 -g -DCAUSALSIEVE_HAS_AVX2_TU=1 -fsanitize=address,undefined -fno-omit-frame-pointer
ASAN_SCALAR_FLAGS := -O2 -g -fsanitize=address,undefined -fno-omit-frame-pointer
ASAN_AVX2_FLAGS  := -O2 -g -mavx2 -mfma -DCAUSALSIEVE_BUILD_AVX2=1 -fsanitize=address,undefined -fno-omit-frame-pointer
else
# ARM64 (NEON is baseline with -O3) and other architectures
CORE_OPT_FLAGS   := -O3
SCALAR_TU_FLAGS  := -O3
AVX2_TU_FLAGS    :=
SCALAR_ONLY_FLAGS := -O3 -DCAUSALSIEVE_FORCE_SCALAR=1
ASAN_CORE_FLAGS  := -O2 -g -fsanitize=address,undefined -fno-omit-frame-pointer
ASAN_SCALAR_FLAGS := -O2 -g -fsanitize=address,undefined -fno-omit-frame-pointer
ASAN_AVX2_FLAGS  :=
endif

ASAN_LINK_FLAGS := -fsanitize=address,undefined

BUILD_DIR := build
OBJ_DIR   := $(BUILD_DIR)/obj
SRC_DIR   := csrc

SHLIB        := $(BUILD_DIR)/libcausalsieve.$(SHLIB_EXT)
SHLIB_SCALAR := $(BUILD_DIR)/libcausalsieve_scalar.$(SHLIB_EXT)
HEADERS      := $(SRC_DIR)/causalsieve.h $(SRC_DIR)/causalsieve_core.hpp $(SRC_DIR)/causalsieve_kernels.hpp

PIC_OBJS := $(OBJ_DIR)/causalsieve.pic.o $(OBJ_DIR)/kernels_scalar.pic.o
OPT_OBJS := $(OBJ_DIR)/causalsieve.opt.o $(OBJ_DIR)/kernels_scalar.opt.o
ASAN_OBJS := $(OBJ_DIR)/causalsieve.asan.o $(OBJ_DIR)/kernels_scalar.asan.o

ifneq ($(IS_X86_64),)
PIC_OBJS  += $(OBJ_DIR)/kernels_avx2.pic.o
OPT_OBJS  += $(OBJ_DIR)/kernels_avx2.opt.o
ASAN_OBJS += $(OBJ_DIR)/kernels_avx2.asan.o
endif

SCALAR_PIC_OBJS := $(OBJ_DIR)/causalsieve.scalar.pic.o $(OBJ_DIR)/kernels_scalar.scalar.pic.o
SCALAR_OPT_OBJS := $(OBJ_DIR)/causalsieve.scalar.opt.o $(OBJ_DIR)/kernels_scalar.scalar.opt.o

.PHONY: all test test-asan bench clean

all: $(SHLIB) \
     $(SHLIB_SCALAR) \
     $(BUILD_DIR)/causalsieve-bin \
     $(BUILD_DIR)/test_core_cpp_asan \
     $(BUILD_DIR)/test_core_cpp_opt \
     $(BUILD_DIR)/test_core_cpp_scalar

$(BUILD_DIR) $(OBJ_DIR):
	mkdir -p $@

# --- PIC object files for multi-ISA shared library ---
$(OBJ_DIR)/causalsieve.pic.o: $(SRC_DIR)/causalsieve.cpp $(HEADERS) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CORE_OPT_FLAGS) -fPIC -I$(SRC_DIR) -c $(SRC_DIR)/causalsieve.cpp -o $@

$(OBJ_DIR)/kernels_scalar.pic.o: $(SRC_DIR)/causalsieve_kernels_scalar.cpp $(SRC_DIR)/causalsieve_kernels.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(SCALAR_TU_FLAGS) -fPIC -I$(SRC_DIR) -c $(SRC_DIR)/causalsieve_kernels_scalar.cpp -o $@

$(OBJ_DIR)/kernels_avx2.pic.o: $(SRC_DIR)/causalsieve_kernels_avx2.cpp $(SRC_DIR)/causalsieve_kernels.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(AVX2_TU_FLAGS) -fPIC -I$(SRC_DIR) -c $(SRC_DIR)/causalsieve_kernels_avx2.cpp -o $@

# --- Non-PIC object files for CLI & optimized C++ test runner ---
$(OBJ_DIR)/causalsieve.opt.o: $(SRC_DIR)/causalsieve.cpp $(HEADERS) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CORE_OPT_FLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/causalsieve.cpp -o $@

$(OBJ_DIR)/kernels_scalar.opt.o: $(SRC_DIR)/causalsieve_kernels_scalar.cpp $(SRC_DIR)/causalsieve_kernels.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(SCALAR_TU_FLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/causalsieve_kernels_scalar.cpp -o $@

$(OBJ_DIR)/kernels_avx2.opt.o: $(SRC_DIR)/causalsieve_kernels_avx2.cpp $(SRC_DIR)/causalsieve_kernels.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(AVX2_TU_FLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/causalsieve_kernels_avx2.cpp -o $@

# --- ASan + UBSan object files ---
$(OBJ_DIR)/causalsieve.asan.o: $(SRC_DIR)/causalsieve.cpp $(HEADERS) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(ASAN_CORE_FLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/causalsieve.cpp -o $@

$(OBJ_DIR)/kernels_scalar.asan.o: $(SRC_DIR)/causalsieve_kernels_scalar.cpp $(SRC_DIR)/causalsieve_kernels.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(ASAN_SCALAR_FLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/causalsieve_kernels_scalar.cpp -o $@

$(OBJ_DIR)/kernels_avx2.asan.o: $(SRC_DIR)/causalsieve_kernels_avx2.cpp $(SRC_DIR)/causalsieve_kernels.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(ASAN_AVX2_FLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/causalsieve_kernels_avx2.cpp -o $@

# --- Scalar-only reference object files (CAUSALSIEVE_FORCE_SCALAR=1) ---
$(OBJ_DIR)/causalsieve.scalar.pic.o: $(SRC_DIR)/causalsieve.cpp $(HEADERS) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(SCALAR_ONLY_FLAGS) -fPIC -I$(SRC_DIR) -c $(SRC_DIR)/causalsieve.cpp -o $@

$(OBJ_DIR)/kernels_scalar.scalar.pic.o: $(SRC_DIR)/causalsieve_kernels_scalar.cpp $(SRC_DIR)/causalsieve_kernels.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(SCALAR_ONLY_FLAGS) -fPIC -I$(SRC_DIR) -c $(SRC_DIR)/causalsieve_kernels_scalar.cpp -o $@

$(OBJ_DIR)/causalsieve.scalar.opt.o: $(SRC_DIR)/causalsieve.cpp $(HEADERS) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(SCALAR_ONLY_FLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/causalsieve.cpp -o $@

$(OBJ_DIR)/kernels_scalar.scalar.opt.o: $(SRC_DIR)/causalsieve_kernels_scalar.cpp $(SRC_DIR)/causalsieve_kernels.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(SCALAR_ONLY_FLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/causalsieve_kernels_scalar.cpp -o $@

# --- Linked binaries and shared libraries ---
$(SHLIB): $(PIC_OBJS) | $(BUILD_DIR)
	$(CXX) $(SHLIB_FLAGS) $(PIC_OBJS) -o $@ -lm

$(SHLIB_SCALAR): $(SCALAR_PIC_OBJS) | $(BUILD_DIR)
	$(CXX) $(SHLIB_FLAGS) $(SCALAR_PIC_OBJS) -o $@ -lm

$(BUILD_DIR)/causalsieve-bin: $(SRC_DIR)/cli_main.cpp $(OPT_OBJS) $(SRC_DIR)/causalsieve.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -O3 -I$(SRC_DIR) $(SRC_DIR)/cli_main.cpp $(OPT_OBJS) -o $@ -lm

$(BUILD_DIR)/test_core_cpp_asan: $(SRC_DIR)/test_core_cpp.cpp $(ASAN_OBJS) $(SRC_DIR)/causalsieve.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ASAN_SCALAR_FLAGS) -I$(SRC_DIR) $(SRC_DIR)/test_core_cpp.cpp $(ASAN_OBJS) -o $@ $(ASAN_LINK_FLAGS) $(DL_LIB) -lm

$(BUILD_DIR)/test_core_cpp_opt: $(SRC_DIR)/test_core_cpp.cpp $(OPT_OBJS) $(SRC_DIR)/causalsieve.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -O3 -I$(SRC_DIR) $(SRC_DIR)/test_core_cpp.cpp $(OPT_OBJS) -o $@ $(DL_LIB) -lm

$(BUILD_DIR)/test_core_cpp_scalar: $(SRC_DIR)/test_core_cpp.cpp $(SCALAR_OPT_OBJS) $(SRC_DIR)/causalsieve.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SCALAR_ONLY_FLAGS) -I$(SRC_DIR) $(SRC_DIR)/test_core_cpp.cpp $(SCALAR_OPT_OBJS) -o $@ $(DL_LIB) -lm

test-asan: $(BUILD_DIR)/test_core_cpp_asan
	./$(BUILD_DIR)/test_core_cpp_asan

test: all
	./$(BUILD_DIR)/test_core_cpp_asan
	./$(BUILD_DIR)/test_core_cpp_opt --parity-so ./$(SHLIB) ./$(SHLIB_SCALAR) --bench
	./$(BUILD_DIR)/test_core_cpp_scalar --bench

bench: $(BUILD_DIR)/test_core_cpp_opt $(BUILD_DIR)/test_core_cpp_scalar
	./$(BUILD_DIR)/test_core_cpp_opt --bench
	./$(BUILD_DIR)/test_core_cpp_scalar --bench

clean:
	rm -rf $(BUILD_DIR)
