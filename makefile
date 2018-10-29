NAME   = presentation

CC     = g++
SRC    = rsPres.cpp
OBJ    = $(subst .cpp,.o,$(SRC))

OPENCV   = $(shell pkg-config opencv --cflags --libs)
INC      = -I/usr/bin/local/include/
CPPFLAGS = -Wall -g -std=gnu++11
LDFLAGS  = -g
LDLIBS   = -L/usr/local/lib/ -lrealsense2 $(OPENCV)

all: $(NAME)

$(NAME): $(OBJ)
	$(CC) $(CPPFLAGS) $(LDFLAGS) -o $(NAME) $(OBJ) $(LDLIBS)

.PHONY: clean

clean:
	rm -f $(OBJ) $(NAME)
