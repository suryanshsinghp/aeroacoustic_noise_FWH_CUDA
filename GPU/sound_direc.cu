#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <string>

#include "global.cuh"
#include "sound_pressure.cuh"
#include "sound_direc.cuh"
#include "timer.cuh"
extern __constant__ SimConsts d_SimConsts;

#define TIME_SOUND_PRESSURE_ALL

// from NVIDIA, for error checking
#define cudaCheckErrors(msg)                                   \
    do                                                         \
    {                                                          \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess)                              \
        {                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

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
                       double *d_NormalZc)
{
    Timer timer(__FUNCTION__, "s");
    double x0, y0, z0;
    double radius;
    int ntheta, nphi;
    double thetas, thetae, phis, phie;
    double *h_pp = new double[h_SimConsts.numTstep];
    double *d_pp;
    cudaMalloc(&d_pp, h_SimConsts.numTstep * sizeof(double));
    int *h_flagRetarded = new int[h_SimConsts.numTstep];
    int *d_flagRetarded;
    cudaMalloc(&d_flagRetarded, h_SimConsts.numTstep * sizeof(int));

    int threads_per_block_e = 32;
    int threads_per_block_t = 10;
    int blocks_per_grid_e = (h_SimConsts.numElem + threads_per_block_e - 1) / threads_per_block_e;
    int blocks_per_grid_t = (h_SimConsts.numTstep + threads_per_block_t - 1) / threads_per_block_t;
    dim3 blockDim(threads_per_block_e, threads_per_block_t, 1);
    dim3 gridDim(blocks_per_grid_e, blocks_per_grid_t, 1);

    std::ifstream inputFile("directivity_cordinates.dat");
    if (!inputFile)
    {
        std::cerr << "Error opening file to read directivity cordinates." << std::endl;
        return;
    }
    inputFile >> radius >> x0 >> y0 >> z0;
    inputFile >> ntheta >> thetas >> thetae;
    inputFile >> nphi >> phis >> phie;
    inputFile.close();

    phis = phis / 180.0 * h_SimConsts.PI;
    phie = phie / 180.0 * h_SimConsts.PI;
    thetas = thetas / 180.0 * h_SimConsts.PI;
    thetae = thetae / 180.0 * h_SimConsts.PI;

    double dtheta = (ntheta > 1) ? (thetae - thetas) / (ntheta - 1) : 0.0;
    double dphi = (nphi > 1) ? (phie - phis) / (nphi - 1) : 0.0;

    for (int l = 1; l <= nphi; ++l)
    {
        for (int j = 1; j <= ntheta; ++j)
        {
            double phi = phis + dphi * (l - 1);
            double theta = thetas + dtheta * (j - 1);
            double rr[3];
            rr[0] = x0 * h_SimConsts.lscale + radius * std::cos(phi) * std::cos(theta);
            rr[1] = y0 * h_SimConsts.lscale + radius * std::cos(phi) * std::sin(theta);
            rr[2] = z0 * h_SimConsts.lscale + radius * std::sin(phi);

            double *d_rr;
            cudaMalloc((void **)&d_rr, 3 * sizeof(double));
            cudaMemcpy(d_rr, rr, 3 * sizeof(double), cudaMemcpyHostToDevice);

            std::ostringstream fname;
            fname << "./monitor_out/monitor." << std::setw(3) << std::setfill('0') << h_SimConsts.bodyNum
                  << "." << std::setw(3) << std::setfill('0') << j
                  << "-" << std::setw(3) << std::setfill('0') << l << ".dat";

            std::ofstream outfile(fname.str());
            if (!outfile)
            {
                std::cerr << "Error opening output file: " << fname.str() << std::endl;
                return;
            }

#ifdef TIME_SOUND_PRESSURE_ALL
            {
                std::string timerName = "sound_pressure";
                std::string theta_iter = std::to_string(j);
                std::string phi_iter = std::to_string(l);
                Timer timer2(timerName + " (theta: " + theta_iter + ", phi: " + phi_iter + ")", "µs");
#endif
                cudaMemset(d_flagRetarded, 0, h_SimConsts.numTstep * sizeof(int)); // initialize to 0
                sound_pressure<<<gridDim, blockDim>>>(d_rr, d_pp, d_flagRetarded, d_t, d_xc, d_yc, d_zc, d_uc, d_vc, d_wc, d_p0c, d_p1c, d_NormalXc, d_NormalYc, d_NormalZc);
                cudaCheckErrors("sound_pressure kernel launch failed");
                cudaMemcpy(h_pp, d_pp, h_SimConsts.numTstep * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_flagRetarded, d_flagRetarded, h_SimConsts.numTstep * sizeof(int), cudaMemcpyDeviceToHost);
                for (int i = 0; i < h_SimConsts.numTstep; ++i)
                {
                    if (h_flagRetarded[i] == 0)
                    {
                        outfile << std::scientific << std::setprecision(5) << t[i] << "   " << h_pp[i] << "\n";
                    }
                }
#ifdef TIME_SOUND_PRESSURE_ALL
            }
#endif
            outfile.close();
        }
    }
}