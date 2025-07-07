#include <cmath>
#include <iostream>
#include "cal_direction_vec.cuh"
#include "global.cuh"
#include "calc_load.cuh"

extern __constant__ SimConsts d_SimConsts;

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
                         double *NormalZc)
{
    double rx, ry, rz, inv_r0;
    double rxp, ryp, rzp;
    double tnow = time;

#ifdef OLD_INTER
    if (tnow <= t[0])
    {
        // std::cout << tnow << " Negative time should not be here, check code again" << std::endl;
        return;
    }

    int n = -1;
    for (int i = 0; i < d_SimConsts.numTstep; ++i)
    {
        if (tnow >= t[i] && tnow <= t[i + 1])
        {
            n = i;
            break;
        }
    }

    if (n == -1)
    {
        // std::cerr << "Interpolation failed at:" << std::endl;
        // std::cerr << "ti, tnow, ti+1 = " << t[n] << ", " << tnow << ", " << t[n + 1] << std::endl;
        return;
    }
#elseif BINARY_SEARCH

    int low = 0;
    int high = d_SimConsts.numTstep - 1;
    int n = -1;
    while (low < high - 1)
    {
        int mid = (low + high) / 2;
        if (tnow < t[mid])
        {
            high = mid;
        }
        else
        {
            low = mid;
        }
    }

    if (t[low] <= tnow && tnow <= t[low + 1])
    {
        n = low;
    }
    else
    {
        n = -INFINITY; // crash code if interpolation fails
    }
#else
    int n = (tnow - t[0]) / (t[1] - t[0]); // revert back to old option if dt is not constant
    if (n < 0 || n >= d_SimConsts.numTstep - 1)
    {
        n = -INFINITY; // crash code if interpolation fails
    }
#endif

    const double dtc = t[n + 1] - t[n];
    const int idx_jn = idx(j, n, d_SimConsts.numTstep);
    const int idx_jn_1 = idx(j, n + 1, d_SimConsts.numTstep);

    // Interpolated positions
    const double inv_dtc = 1.0 / dtc;
    const double dt_inv_dtc = (tnow - t[n]) * inv_dtc;

    double xcn = xc[idx_jn], ycn = yc[idx_jn], zcn = zc[idx_jn];
    double xcp = xc[idx_jn_1], ycp = yc[idx_jn_1], zcp = zc[idx_jn_1];
    double xcc = xcn + (xcp - xcn) * dt_inv_dtc;
    double ycc = ycn + (ycp - ycn) * dt_inv_dtc;
    double zcc = zcn + (zcp - zcn) * dt_inv_dtc;

    double ucn = uc[idx_jn], vcn = vc[idx_jn], wcn = wc[idx_jn];
    double ucp = uc[idx_jn_1], vcp = vc[idx_jn_1], wcp = wc[idx_jn_1];
    double ucc = ucn + (ucp - ucn) * dt_inv_dtc;
    double vcc = vcn + (vcp - vcn) * dt_inv_dtc;
    double wcc = wcn + (wcp - wcn) * dt_inv_dtc;

    double pcn = p0c[idx_jn] - p1c[idx_jn];
    double pcp = p0c[idx_jn_1] - p1c[idx_jn_1];
    double pcc = pcn + (pcp - pcn) * dt_inv_dtc;

    double vnx = NormalXc[idx_jn], vny = NormalYc[idx_jn], vnz = NormalZc[idx_jn];
    double vpx = NormalXc[idx_jn_1], vpy = NormalYc[idx_jn_1], vpz = NormalZc[idx_jn_1];

    cal_direction_vec(rr, xcn, ycn, zcn, rx, ry, rz, inv_r0);
    cal_direction_vec(rr, xcp, ycp, zcp, rxp, ryp, rzp, inv_r0);

    double dLrdt = (pcp * (vpx * rxp + vpy * ryp + vpz * rzp) - pcn * (vnx * rx + vny * ry + vnz * rz)) * inv_dtc;

    double inv_cscale = 1.0 / d_SimConsts.cscale;

    double dMrdt = ((ucp * rxp + vcp * ryp + wcp * rzp) - (ucn * rx + vcn * ry + wcn * rz)) * (inv_dtc * inv_cscale);

    double vcx = vnx + (vpx - vnx) * dt_inv_dtc;
    double vcy = vny + (vpy - vny) * dt_inv_dtc;
    double vcz = vnz + (vpz - vnz) * dt_inv_dtc;

    cal_direction_vec(rr, xcc, ycc, zcc, rx, ry, rz, inv_r0);

    double Mr = (rx * ucc + ry * vcc + rz * wcc) * inv_cscale;

    // return;
    double inv_ro_sq = inv_r0 * inv_r0;
    double inv_one_m_Mr = 1.0 / (1.0 - Mr);
    double inv_one_m_Mr_sq = inv_one_m_Mr * inv_one_m_Mr;

    pp = dLrdt * inv_r0 * inv_one_m_Mr_sq * inv_cscale;

    pp += (pcc * (vcx * rx + vcy * ry + vcz * rz) - pcc * (vcx * ucc + vcy * vcc + vcz * wcc) * inv_cscale) * inv_ro_sq * inv_one_m_Mr_sq;

    pp += (dMrdt / inv_r0 + d_SimConsts.cscale * (Mr - (ucc * ucc + vcc * vcc + wcc * wcc) * (inv_cscale * inv_cscale))) * pcc * (vcx * rx + vcy * ry + vcz * rz) * inv_ro_sq * (inv_one_m_Mr * inv_one_m_Mr_sq) * inv_cscale;

    pp /= 4 * d_SimConsts.PI;
}