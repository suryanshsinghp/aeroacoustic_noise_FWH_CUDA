NVCC = nvcc
NVCCFLAGS = -std=c++17 -O3 -arch=sm_80 -rdc=true -use_fast_math -lineinfo #arch needed for atomics

SRCS = FWH.cu sound_direc.cu sound_pressure.cu calc_load.cu cal_direction_vec.cu
OBJS = $(SRCS:.cu=.o)
DLINK_OBJ = device_link.o
TARGET = out

all: $(TARGET)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(DLINK_OBJ): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -dlink $^ -o $@

$(TARGET): $(OBJS) $(DLINK_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

clean:
	rm -f $(OBJS) $(DLINK_OBJ) $(TARGET)

run: $(TARGET)
	./$(TARGET)
