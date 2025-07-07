#ifndef GLOBAL
#define GLOBAL

struct SimConsts
{
    double PI;
    double lscale;
    double cscale;
    int numElem;
    int numTstep;
    int bodyNum;
};

// extern __constant__ SimConsts d_SimConsts;
extern SimConsts h_SimConsts;

inline __host__ __device__ int idx(int row, int col, int width)
{
    return col + row * width;
}

#endif