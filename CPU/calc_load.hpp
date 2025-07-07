#ifndef CALC_LOAD
#define CALC_LOAD

void cal_load(double time,
              int j,
              double (&rr)[3],
              double (&term)[3],
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