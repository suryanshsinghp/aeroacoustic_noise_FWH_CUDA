#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <string>
#include "global.hpp"

#include "sound_pressure.hpp"
#include "sound_direc.hpp"

#define TIME_SOUND_PRESSURE_ALL

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
                       double *NormalZc)
{
#ifdef TIME_SOUND_DIRECTIVITY
    Timer timer(__FUNCTION__, "s");
#endif
    double pp, x0, y0, z0;
    double radius;
    int ntheta, nphi;
    double thetas, thetae, phis, phie;

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
    // std::cout << "Read input parameters:" << std::endl;
    // std::cout << "Radius: " << radius << ", x0: " << x0 << ", y0: " << y0 << ", z0: " << z0 << std::endl;
    // std::cout << "theta_start: " << thetas << ", thetaend: " << thetae << ", phistart: " << phis << ", phiend: " << phie << std::endl;

    phis = phis / 180.0 * PI;
    phie = phie / 180.0 * PI;
    thetas = thetas / 180.0 * PI;
    thetae = thetae / 180.0 * PI;

    double dtheta = (ntheta > 1) ? (thetae - thetas) / (ntheta - 1) : 0.0;
    double dphi = (nphi > 1) ? (phie - phis) / (nphi - 1) : 0.0;

    for (int l = 1; l <= nphi; ++l)
    {
        for (int j = 1; j <= ntheta; ++j)
        {
            // std::cout << "theta_start: " << thetas << ", thetaend: " << thetae << "theta step: " << dtheta << ", phistart: " << phis << ", phiend: " << phie << "phi step: " << dphi << std::endl;
            double phi = phis + dphi * (l - 1);
            double theta = thetas + dtheta * (j - 1);
            double rr[3];
            rr[0] = x0 * lscale + radius * std::cos(phi) * std::cos(theta);
            rr[1] = y0 * lscale + radius * std::cos(phi) * std::sin(theta);
            rr[2] = z0 * lscale + radius * std::sin(phi);

            std::ostringstream fname;
            fname << "./monitor_out/monitor." << std::setw(3) << std::setfill('0') << bodyNum
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
                for (int i = 0; i < numTstep; ++i)
                {
                    double time = t[i];
                    int flagRetarded = 0;
                    sound_pressure(i, time, rr, pp, flagRetarded, t, xc, yc, zc, uc, vc, wc, p0c, p1c, NormalXc, NormalYc, NormalZc);

                    if (flagRetarded == 0)
                    {
                        // std::cout << "Writing sound pressure at time " << time << std::endl;
                        outfile << std::scientific << std::setprecision(5) << time << "   " << pp << "\n";
                    }
#ifdef VERBOSE
                    else
                    {
                        std::cout << "Retarded time step required for sound pressure calculation at time "
                                  << time << " is negative. Skipping output for this time step." << std::endl;
                    }
#endif
                }
#ifdef TIME_SOUND_PRESSURE_ALL
            }
#endif
            outfile.close();
        }
    }
}