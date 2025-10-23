CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -g -O3
LIBS = -lm

SRC_DIR = src
BUILD_DIR = build
BIN = $(BUILD_DIR)/main

SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRC_FILES))


all: $(BIN)
	@$(BIN)

$(BIN): $(OBJ_FILES)
	@$(CC) $(OBJ_FILES) -o $@ $(LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

clean:
	@rm -rf $(BUILD_DIR)
