#ifndef SOUND_DIREC // or just use #pragma once
#define SOUND_DIREC

void sound_directivity(double *t,
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