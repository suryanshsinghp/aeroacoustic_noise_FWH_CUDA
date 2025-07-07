#include <cmath>
#include "cal_direction_vec.hpp"
#include "global.hpp"

void cal_direction_vec(double (&rr)[3],
                       double x,
                       double y,
                       double z,
                       double &rx,
                       double &ry,
                       double &rz,
                       double &r0)
{
#ifdef TIME_DIREC
    Timer timer(__FUNCTION__, "ns");
#endif
    rx = rr[0] - x;
    ry = rr[1] - y;
    rz = rr[2] - z;

    r0 = std::sqrt(rx * rx + ry * ry + rz * rz);

    rx /= r0;
    ry /= r0;
    rz /= r0;
}