#include "sound_pressure.cuh"
#include <math.h>
#include "calc_load.cuh"
#include "global.cuh"

extern __constant__ SimConsts d_SimConsts;

__global__ void sound_pressure(double *d_rr,
                               double *pp,
                               int *d_flagRetarded,
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
                               double *d_NormalZc)
{
    int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (thread_idy < d_SimConsts.numTstep)
    {
        double time = d_t[thread_idy];

        if (thread_idx < d_SimConsts.numElem)
        {
            double r0 = sqrt((d_rr[0] - d_xc[idx(thread_idx, thread_idy, d_SimConsts.numTstep)]) * (d_rr[0] - d_xc[idx(thread_idx, thread_idy, d_SimConsts.numTstep)]) +
                             (d_rr[1] - d_yc[idx(thread_idx, thread_idy, d_SimConsts.numTstep)]) * (d_rr[1] - d_yc[idx(thread_idx, thread_idy, d_SimConsts.numTstep)]) +
                             (d_rr[2] - d_zc[idx(thread_idx, thread_idy, d_SimConsts.numTstep)]) * (d_rr[2] - d_zc[idx(thread_idx, thread_idy, d_SimConsts.numTstep)]));

            if (time - r0 / d_SimConsts.cscale <= d_t[0])
            {
                d_flagRetarded[thread_idy] = 1; // several thread may try set this to 1 simentiously, but that is correct, since signal from any thread works.
                // return;
            }
            else
            {
                {
                    double pp_thread = 0.0;
                    cal_load(time - r0 / d_SimConsts.cscale, thread_idx, d_rr, pp_thread, d_t, d_xc, d_yc, d_zc, d_uc, d_vc, d_wc, d_p0c, d_p1c, d_NormalXc, d_NormalYc, d_NormalZc);

                    atomicAdd(&pp[thread_idy], pp_thread);
                }
            }
        }
        //__syncthreads();
    }
}