CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

SRCS = FWH.cpp sound_direc.cpp sound_pressure.cpp calc_load.cpp cal_direction_vec.cpp
OBJS = $(SRCS:.cpp=.o)

TARGET = out

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)
