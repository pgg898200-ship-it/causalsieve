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

ifeq ($(CAUSALSIEVE_PORTABLE),1)
SIMD_FLAGS      := -O3
ASAN_SIMD_FLAGS :=
ifneq ($(filter x86_64 amd64,$(UNAME_M)),)
SCALAR_FLAGS    := -O3 -DCAUSALSIEVE_FORCE_SCALAR -mno-avx2 -mno-fma
else
SCALAR_FLAGS    := -O3 -DCAUSALSIEVE_FORCE_SCALAR
endif
else ifneq ($(filter x86_64 amd64,$(UNAME_M)),)
SIMD_FLAGS      := -O3 -mavx2 -mfma
ASAN_SIMD_FLAGS := -mavx2 -mfma
SCALAR_FLAGS    := -O3 -DCAUSALSIEVE_FORCE_SCALAR -mno-avx2 -mno-fma
else ifneq ($(filter arm64 aarch64,$(UNAME_M)),)
SIMD_FLAGS      := -O3
ASAN_SIMD_FLAGS :=
SCALAR_FLAGS    := -O3 -DCAUSALSIEVE_FORCE_SCALAR
else
SIMD_FLAGS      := -O3
ASAN_SIMD_FLAGS :=
SCALAR_FLAGS    := -O3 -DCAUSALSIEVE_FORCE_SCALAR
endif

ASAN_FLAGS := $(strip -O2 -g $(ASAN_SIMD_FLAGS) -fsanitize=address,undefined -fno-omit-frame-pointer)

BUILD_DIR := build
SRC_DIR   := csrc

SHLIB        := $(BUILD_DIR)/libcausalsieve.$(SHLIB_EXT)
SHLIB_SCALAR := $(BUILD_DIR)/libcausalsieve_scalar.$(SHLIB_EXT)
HEADERS      := $(SRC_DIR)/causalsieve.h $(SRC_DIR)/causalsieve_core.hpp

.PHONY: all test test-asan bench clean

all: $(SHLIB) \
     $(SHLIB_SCALAR) \
     $(BUILD_DIR)/causalsieve-bin \
     $(BUILD_DIR)/test_core_cpp_asan \
     $(BUILD_DIR)/test_core_cpp_opt \
     $(BUILD_DIR)/test_core_cpp_scalar

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(SHLIB): $(SRC_DIR)/causalsieve.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SIMD_FLAGS) -fPIC $(SHLIB_FLAGS) -I$(SRC_DIR) $(SRC_DIR)/causalsieve.cpp -o $@ -lm

$(SHLIB_SCALAR): $(SRC_DIR)/causalsieve.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SCALAR_FLAGS) -fPIC $(SHLIB_FLAGS) -I$(SRC_DIR) $(SRC_DIR)/causalsieve.cpp -o $@ -lm

$(BUILD_DIR)/causalsieve-bin: $(SRC_DIR)/cli_main.cpp $(SRC_DIR)/causalsieve.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SIMD_FLAGS) -I$(SRC_DIR) $(SRC_DIR)/cli_main.cpp $(SRC_DIR)/causalsieve.cpp -o $@ -lm

$(BUILD_DIR)/test_core_cpp_asan: $(SRC_DIR)/test_core_cpp.cpp $(SRC_DIR)/causalsieve.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ASAN_FLAGS) -I$(SRC_DIR) $(SRC_DIR)/test_core_cpp.cpp $(SRC_DIR)/causalsieve.cpp -o $@ $(DL_LIB) -lm

$(BUILD_DIR)/test_core_cpp_opt: $(SRC_DIR)/test_core_cpp.cpp $(SRC_DIR)/causalsieve.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SIMD_FLAGS) -I$(SRC_DIR) $(SRC_DIR)/test_core_cpp.cpp $(SRC_DIR)/causalsieve.cpp -o $@ $(DL_LIB) -lm

$(BUILD_DIR)/test_core_cpp_scalar: $(SRC_DIR)/test_core_cpp.cpp $(SRC_DIR)/causalsieve.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SCALAR_FLAGS) -I$(SRC_DIR) $(SRC_DIR)/test_core_cpp.cpp $(SRC_DIR)/causalsieve.cpp -o $@ $(DL_LIB) -lm

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
