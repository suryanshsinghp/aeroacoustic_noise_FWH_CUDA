#ifndef CALDIREC
#define CALDIREC

__device__ void cal_direction_vec(double *rr,
                                  double x,
                                  double y,
                                  double z,
                                  double &rx,
                                  double &ry,
                                  double &rz,
                                  double &inv_r0);

#endif