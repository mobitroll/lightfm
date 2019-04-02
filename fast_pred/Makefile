CXX ?= g++

# path #
SRC_PATH = src
BUILD_PATH = build
BIN_PATH = bin
TEST_SRC_PATH = test
TEST_BUILD_PATH = $(BUILD_PATH)/test

# executable # 
SO_NAME = libfastlightfmpred.so
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
COMPILE_FLAGS = -std=c++14 -Wall -Wextra -fPIC
INCLUDES = -I include/ -I /usr/local/include
# Space-separated pkg-config libraries used by this project
LIBS = -lzip

.PHONY: default_target
default_target: so

.PHONY: example
example: export CXXFLAGS := $(CXXFLAGS) $(COMPILE_FLAGS) -g -DDEBUG
example: dirs $(OBJECTS)
	@echo "Making example app: $@"
	$(CXX)  $(INCLUDES) -I $(SRC_PATH) example/main.cpp $(OBJECTS) -o $(BIN_PATH)/$@ $(LIBS) $(CXXFLAGS)

.PHONY: install
install: so
	cp $(BIN_PATH)/$(SO_NAME) /usr/local/lib/

.PHONY: release
release: export CXXFLAGS := $(CXXFLAGS) $(COMPILE_FLAGS) -O3
release: so

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
test: export CXXFLAGS := $(CXXFLAGS) $(COMPILE_FLAGS) -g -DDEBUG
test: dirs $(BIN_PATH)/$(TEST_BIN_NAME)
	$(BIN_PATH)/$(TEST_BIN_NAME)

# checks the executable and symlinks to the output

.PHONY: so
so: export CXXFLAGS := $(CXXFLAGS) $(COMPILE_FLAGS) -g -DDEBUG
so: dirs $(BIN_PATH)/$(SO_NAME)

# Creation of the executable
$(BIN_PATH)/$(SO_NAME): $(OBJECTS)
	@echo "Linking: $@"
	$(CXX) $(OBJECTS) -o $@ -shared -install_name,$(SO_NAME)  $(LIBS)

# Tests
$(BIN_PATH)/$(TEST_BIN_NAME): $(TEST_OBJECTS) $(OBJECTS)
	@echo "Linking: $@"
	$(CXX) $(TEST_OBJECTS) $(OBJECTS) -o $@ $(LIBS)

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
	$(CXX) $(CXXFLAGS) -I $(TEST_SRC_PATH) -I $(SRC_PATH) $(INCLUDES) -c $< -o $@
