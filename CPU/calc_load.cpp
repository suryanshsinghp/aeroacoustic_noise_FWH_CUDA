#include <cmath>
#include <iostream>
#include "cal_direction_vec.hpp"
#include "global.hpp"
#include "calc_load.hpp"

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
              double *NormalZc)
{
#ifdef TIME_ALL_FUNCTIONS
    Timer timer(__FUNCTION__, "ns");
#endif
    double rx, ry, rz, r0;
    double rxp, ryp, rzp;
    double tnow = time;
    // std::cout << "Calculating load at time-> " << tnow << std::endl;
    //  return;

    if (tnow <= t[0])
    {
        std::cout << tnow << " Negative time should not be here, check code again" << std::endl;
        return;
    }

    int n = -1;
    for (int i = 0; i < numTstep; ++i) // this should be more efficient, fix this
    {
        if (tnow >= t[i] && tnow <= t[i + 1])
        {
            n = i;
            break;
        }
    }

    if (n == -1)
    {
        std::cerr << "Interpolation failed at:" << std::endl;
        std::cerr << "ti, tnow, ti+1 = " << t[n] << ", " << tnow << ", " << t[n + 1] << std::endl;
        return;
    }

    double dtc = t[n + 1] - t[n];

    double xcc = xc[idx(j, n, numTstep)] + (xc[idx(j, n + 1, numTstep)] - xc[idx(j, n, numTstep)]) * (tnow - t[n]) / dtc;
    double ycc = yc[idx(j, n, numTstep)] + (yc[idx(j, n + 1, numTstep)] - yc[idx(j, n, numTstep)]) * (tnow - t[n]) / dtc;
    double zcc = zc[idx(j, n, numTstep)] + (zc[idx(j, n + 1, numTstep)] - zc[idx(j, n, numTstep)]) * (tnow - t[n]) / dtc;

    double xcn = xc[idx(j, n, numTstep)], ycn = yc[idx(j, n, numTstep)], zcn = zc[idx(j, n, numTstep)];
    double xcp = xc[idx(j, n + 1, numTstep)], ycp = yc[idx(j, n + 1, numTstep)], zcp = zc[idx(j, n + 1, numTstep)];

    double ucc = uc[idx(j, n, numTstep)] + (uc[idx(j, n + 1, numTstep)] - uc[idx(j, n, numTstep)]) * (tnow - t[n]) / dtc;
    double vcc = vc[idx(j, n, numTstep)] + (vc[idx(j, n + 1, numTstep)] - vc[idx(j, n, numTstep)]) * (tnow - t[n]) / dtc;
    double wcc = wc[idx(j, n, numTstep)] + (wc[idx(j, n + 1, numTstep)] - wc[idx(j, n, numTstep)]) * (tnow - t[n]) / dtc;

    double ucn = uc[idx(j, n, numTstep)], vcn = vc[idx(j, n, numTstep)], wcn = wc[idx(j, n, numTstep)];
    double ucp = uc[idx(j, n + 1, numTstep)], vcp = vc[idx(j, n + 1, numTstep)], wcp = wc[idx(j, n + 1, numTstep)];

    double pcc = p0c[idx(j, n, numTstep)] + (p0c[idx(j, n + 1, numTstep)] - p0c[idx(j, n, numTstep)]) * (tnow - t[n]) / dtc - p1c[idx(j, n, numTstep)] - (p1c[idx(j, n + 1, numTstep)] - p1c[idx(j, n, numTstep)]) * (tnow - t[n]) / dtc;

    double pcn = p0c[idx(j, n, numTstep)] - p1c[idx(j, n, numTstep)];
    double pcp = p0c[idx(j, n + 1, numTstep)] - p1c[idx(j, n + 1, numTstep)];

    double vnx = NormalXc[idx(j, n, numTstep)], vny = NormalYc[idx(j, n, numTstep)], vnz = NormalZc[idx(j, n, numTstep)];
    double vpx = NormalXc[idx(j, n + 1, numTstep)], vpy = NormalYc[idx(j, n + 1, numTstep)], vpz = NormalZc[idx(j, n + 1, numTstep)];

    cal_direction_vec(rr, xcn, ycn, zcn, rx, ry, rz, r0);
    cal_direction_vec(rr, xcp, ycp, zcp, rxp, ryp, rzp, r0);

    double dLrdt = (pcp * (vpx * rxp + vpy * ryp + vpz * rzp) - pcn * (vnx * rx + vny * ry + vnz * rz)) / dtc;

    double dMrdt = ((ucp * rxp + vcp * ryp + wcp * rzp) - (ucn * rx + vcn * ry + wcn * rz)) / (dtc * cscale);

    double vcx = vnx + (vpx - vnx) * (tnow - t[n]) / dtc;
    double vcy = vny + (vpy - vny) * (tnow - t[n]) / dtc;
    double vcz = vnz + (vpz - vnz) * (tnow - t[n]) / dtc;

    cal_direction_vec(rr, xcc, ycc, zcc, rx, ry, rz, r0);

    double Mr = (rx * ucc + ry * vcc + rz * wcc) / cscale;

    // return;
    term[0] = (dLrdt / r0 / std::pow(1 - Mr, 2)) / cscale;

    term[1] = (pcc * (vcx * rx + vcy * ry + vcz * rz) - pcc * (vcx * ucc + vcy * vcc + vcz * wcc) / cscale) / std::pow(r0, 2) / std::pow(1 - Mr, 2);

    term[2] = (dMrdt * r0 + cscale * (Mr - (ucc * ucc + vcc * vcc + wcc * wcc) / (cscale * cscale))) * pcc * (vcx * rx + vcy * ry + vcz * rz) / std::pow(r0, 2) / std::pow(1 - Mr, 3) / cscale;
}