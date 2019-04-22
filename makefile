NAME   = markerDetection

ARUCO_PATH = aruco_tx2

SRC_DIRS = src
INC_DIRS = include
OBJ_DIR  = obj

SRC_DIRS += $(ARUCO_PATH)/src
INC_DIRS += $(ARUCO_PATH)/include/

INC_DIRS += /usr/local/lib
INC_DIRS += /usr/bin/local/include

CC       = g++
SOURCES  = markerDetection.cpp
SOURCES += rsCam.cpp
SOURCES += detectPoseRealsense.cpp
SOURCES += kabsch.cpp

include $(ARUCO_PATH)/Makefile.inc


CFLAGS  += -Wall -g -std=gnu++11
CFLAGS  += $(addprefix -I,$(INC_DIRS))

LDLIBS   = -L/usr/local/lib/

OPENCV   = $(shell pkg-config opencv --cflags --libs)
LDLIBS  += -lrealsense2 $(OPENCV)
LDLIBS  += -lpthread

OBJ      = $(filter %.o,$(SOURCES:%.cpp=$(OBJ_DIR)/%.o))

vpath %.hpp   $(INC_DIRS)
vpath %.cpp $(SRC_DIRS)

.PHONY: all
all: $(NAME)

$(NAME): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(OBJ): $(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(NAME)
