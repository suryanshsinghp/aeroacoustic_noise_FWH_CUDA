#include "sound_pressure.hpp"
#include "calc_load.hpp"
#include "global.hpp"
#include <cmath>
#include <iostream>

// extern int flagRetarded;

void sound_pressure(int i,
                    double time,
                    double (&rr)[3],
                    double &pp, int &flagRetarded,
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
                    double *NormalZc)
{
#ifdef TIME_SOUND_PRESSURE
    Timer timer(__FUNCTION__, "ns");
#endif
    double term1 = 0.0;
    double term2 = 0.0;
    double term3 = 0.0;

    for (int n = 0; n < numElem; ++n)
    {
        double r0 = std::sqrt((rr[0] - xc[idx(n, i, numTstep)]) * (rr[0] - xc[idx(n, i, numTstep)]) +
                              (rr[1] - yc[idx(n, i, numTstep)]) * (rr[1] - yc[idx(n, i, numTstep)]) +
                              (rr[2] - zc[idx(n, i, numTstep)]) * (rr[2] - zc[idx(n, i, numTstep)]));

        if (time - r0 / cscale <= t[0])
        {
            flagRetarded = 1;
            return;
        }
        else
        {
            double term_temp[3] = {0.0, 0.0, 0.0};
            cal_load(time - r0 / cscale, n, rr, term_temp, t, xc, yc, zc, uc, vc, wc, p0c, p1c, NormalXc, NormalYc, NormalZc);
            term1 += term_temp[0];
            term2 += term_temp[1];
            term3 += term_temp[2];
        }
    }

    pp = (term1 + term2 + term3) / (4.0 * PI);
}