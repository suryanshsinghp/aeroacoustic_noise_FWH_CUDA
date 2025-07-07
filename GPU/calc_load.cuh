#ifndef CALC_LOAD
#define CALC_LOAD

__device__ void cal_load(double time,
                         int j,
                         double *rr,
                         double &pp,
                         double *t,
                         double *xc,
                         double *yc,
                         double *zc,
                         double *uc,
                         double *vc,
                         double *wc,
                         double *p0c,
                         double *p1c,
                         double *NormalXc,
                         double *NormalYc,
                         double *NormalZc);

#endif