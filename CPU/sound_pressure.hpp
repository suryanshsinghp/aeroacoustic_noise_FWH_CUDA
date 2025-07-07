#ifndef SOUND_PRESSURE
#define SOUND_PRESSURE

void sound_pressure(int i,
                    double time,
                    double (&rr)[3],
                    double &pp,
                    int &flagRetarded,
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