#ifndef SOUND_DIREC // or just use #pragma once
#define SOUND_DIREC

void sound_directivity(double *t,
                       double *d_t,
                       double *d_xc,
                       double *d_yc,
                       double *d_zc,
                       double *d_uc,
                       double *d_vc,
                       double *d_wc,
                       double *d_p0c,
                       double *d_p1c,
                       double *d_NormalXc,
                       double *d_NormalYc,
                       double *d_NormalZc);

#endif