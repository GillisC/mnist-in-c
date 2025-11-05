CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -g -O1
LIBS = -lm

SRC_DIR = src
BUILD_DIR = build
BIN = $(BUILD_DIR)/main

SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRC_FILES))

all: $(BIN)

run: $(BIN)
	$(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN): $(OBJ_FILES)
	$(CC) $(OBJ_FILES) -o $@ $(LIBS)

clean:
	rm -rf $(BUILD_DIR)
