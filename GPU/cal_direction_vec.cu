#include <math.h>
#include "cal_direction_vec.cuh"
#include "global.cuh"

__device__ void cal_direction_vec(double *rr,
                                  double x,
                                  double y,
                                  double z,
                                  double &rx,
                                  double &ry,
                                  double &rz,
                                  double &inv_r0)
{
    rx = rr[0] - x;
    ry = rr[1] - y;
    rz = rr[2] - z;

    inv_r0 = rsqrtf(rx * rx + ry * ry + rz * rz); // NOTE this is single precision
    rx *= inv_r0;
    ry *= inv_r0;
    rz *= inv_r0;
}