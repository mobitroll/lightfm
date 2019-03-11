CXX ?= g++

# path #
SRC_PATH = .
BUILD_PATH = build
BIN_PATH = bin
TEST_SRC_PATH = test
TEST_BUILD_PATH = $(BUILD_PATH)/test

# executable # 
BIN_NAME = fastlightfmpred.so
TEST_BIN_NAME = test

# extensions #
SRC_EXT = cpp

# code lists #
# Find all source files in the source directory, sorted by
# most recently modified
SOURCES = $(shell find $(SRC_PATH) -name '*.$(SRC_EXT)' -not -path './$(TEST_SRC_PATH)/*' | sort -k 1nr | cut -f2-)
TEST_SOURCES = $(shell find $(TEST_SRC_PATH) -name '*.$(SRC_EXT)')
# Set the object file names, with the source directory stripped
# from the path, and the build path prepended in its place
OBJECTS = $(SOURCES:$(SRC_PATH)/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)
TEST_OBJECTS = $(TEST_SOURCES:$(TEST_SRC_PATH)/%.$(SRC_EXT)=$(TEST_BUILD_PATH)/%.o)
# Set the dependency files that will be used to add header dependencies
DEPS = $(OBJECTS:.o=.d)

# flags #
COMPILE_FLAGS = -std=c++11 -Wall -Wextra -g -fPIC
INCLUDES = -I include/ -I /usr/local/include
# Space-separated pkg-config libraries used by this project
LIBS = -lz

.PHONY: default_target
default_target: release

.PHONY: release
release: export CXXFLAGS := $(CXXFLAGS) $(COMPILE_FLAGS)
release: dirs
	@$(MAKE) all

.PHONY: dirs
dirs:
	@echo "Creating directories"
	@mkdir -p $(dir $(OBJECTS))
	@mkdir -p $(BIN_PATH)
	@mkdir -p $(TEST_BUILD_PATH)

.PHONY: clean
clean:
	@echo "Deleting directories"
	@$(RM) -r $(BUILD_PATH)
	@$(RM) -r $(BIN_PATH)

.PHONY: test
test: dirs $(BIN_PATH)/test
	$(BIN_PATH)/test --success

# checks the executable and symlinks to the output
.PHONY: all
all: $(BIN_PATH)/$(BIN_NAME)

# Creation of the executable
$(BIN_PATH)/$(BIN_NAME): $(OBJECTS)
	@echo "Linking: $@"
	@echo "$<"
	$(CXX) $(OBJECTS) -o $@ -shared $(LIBS)

# Tests
$(BIN_PATH)/$(TEST_BIN_NAME): $(TEST_OBJECTS)
	@echo "Linking: $@"
	$(CXX) $(TEST_OBJECTS) -o $@

# Add dependency files, if they exist
-include $(DEPS)

# Source file rules
# After the first compilation they will be joined with the rules from the
# dependency files to provide header dependencies
$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_EXT)
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MP -MMD -c $< -o $@

# Tests
$(TEST_BUILD_PATH)/%.o: $(TEST_SRC_PATH)/%.$(SRC_EXT)
	@echo "Compiling: $< -> $@"
	$(CXX) -std=c++11 -I $(TEST_SRC_PATH) $(INCLUDES) -Wall -c $< -o $@