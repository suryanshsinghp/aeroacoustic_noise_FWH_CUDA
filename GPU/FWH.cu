// make run > log.txt && make clean
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include "global.cuh"
#include "sound_direc.cuh"
#include "timer.cuh"

__constant__ SimConsts d_SimConsts;
SimConsts h_SimConsts;

void GPU_Mem_Usage(const std::string &msg = "")
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << msg << "Free memory: " << free_mem / (1024.0 * 1024.0) << " MB " << "out of total "
              << total_mem / (1024.0 * 1024.0) << " MB\n";
}

int main()
{
    Timer timer_all("Entire code", "s");
    h_SimConsts.PI = std::acos(-1.0);
    h_SimConsts.lscale = 1.0;
    h_SimConsts.cscale = 340.0;
    h_SimConsts.numElem = 29080;
    h_SimConsts.numTstep = 2001;
    h_SimConsts.bodyNum = 1;
    cudaMemcpyToSymbol(d_SimConsts, &h_SimConsts, sizeof(SimConsts));

    double freq = 1.0;
    double tscle = 1.0 / freq;
    double vscale = h_SimConsts.lscale / tscle;

    double *t = new double[h_SimConsts.numTstep];
    double *xc = new double[h_SimConsts.numTstep * h_SimConsts.numElem]; // timestep will be contiguous, elements can be placed apart in memory
    double *yc = new double[h_SimConsts.numTstep * h_SimConsts.numElem];
    double *zc = new double[h_SimConsts.numTstep * h_SimConsts.numElem];
    double *uc = new double[h_SimConsts.numTstep * h_SimConsts.numElem];
    double *vc = new double[h_SimConsts.numTstep * h_SimConsts.numElem];
    double *wc = new double[h_SimConsts.numTstep * h_SimConsts.numElem];
    double *p0c = new double[h_SimConsts.numTstep * h_SimConsts.numElem];
    double *p1c = new double[h_SimConsts.numTstep * h_SimConsts.numElem];
    double *NormalXc = new double[h_SimConsts.numTstep * h_SimConsts.numElem];
    double *NormalYc = new double[h_SimConsts.numTstep * h_SimConsts.numElem];
    double *NormalZc = new double[h_SimConsts.numTstep * h_SimConsts.numElem];

    // device memory allocation
    double *d_t, *d_xc, *d_yc, *d_zc, *d_uc, *d_vc, *d_wc, *d_p0c, *d_p1c, *d_NormalXc, *d_NormalYc, *d_NormalZc;
    cudaMalloc((void **)&d_t, h_SimConsts.numTstep * sizeof(double));
    cudaMalloc((void **)&d_xc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double));
    cudaMalloc((void **)&d_yc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double));
    cudaMalloc((void **)&d_zc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double));
    cudaMalloc((void **)&d_uc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double));
    cudaMalloc((void **)&d_vc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double));
    cudaMalloc((void **)&d_wc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double));
    cudaMalloc((void **)&d_p0c, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double));
    cudaMalloc((void **)&d_p1c, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double));
    cudaMalloc((void **)&d_NormalXc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double));
    cudaMalloc((void **)&d_NormalYc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double));
    cudaMalloc((void **)&d_NormalZc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double));

    for (int i = 0; i < h_SimConsts.numElem; ++i)
    {
        std::ostringstream fname;
        fname << "../input_data/input." << std::setw(3) << std::setfill('0') << h_SimConsts.bodyNum << "." << std::setw(5) << std::setfill('0') << i + 1 << ".bin";
        std::ifstream infile(fname.str(), std::ios::binary);
        if (!infile)
        {
            std::cerr << "Error opening file: " << fname.str() << std::endl;
            return 1;
        }
        double buffer[12];
        for (int j = 0; j < h_SimConsts.numTstep; ++j)
        {
            infile.read(reinterpret_cast<char *>(buffer), sizeof(buffer));
            t[j] = buffer[0];
            xc[idx(i, j, h_SimConsts.numTstep)] = buffer[1];
            yc[idx(i, j, h_SimConsts.numTstep)] = buffer[2];
            zc[idx(i, j, h_SimConsts.numTstep)] = buffer[3];
            uc[idx(i, j, h_SimConsts.numTstep)] = buffer[4];
            vc[idx(i, j, h_SimConsts.numTstep)] = buffer[5];
            wc[idx(i, j, h_SimConsts.numTstep)] = buffer[6];
            p0c[idx(i, j, h_SimConsts.numTstep)] = buffer[7];
            p1c[idx(i, j, h_SimConsts.numTstep)] = buffer[8];
            NormalXc[idx(i, j, h_SimConsts.numTstep)] = buffer[9];
            NormalYc[idx(i, j, h_SimConsts.numTstep)] = buffer[10];
            NormalZc[idx(i, j, h_SimConsts.numTstep)] = buffer[11];
            // if (i== 0 && j == 0)
            //{
            // std::cout << j << "t: " << t[j] << ", xc: " << xc[idx(j, i, numTstep)] << ", yc: " << yc[idx(j, i, numTstep)] << ", zc: " << zc[idx(j, i, numTstep)] << p0c[idx(j, i, numTstep)] << ", p1c: " << p1c[idx(j, i, numTstep)] << ", NormalXc: " << NormalXc[idx(j, i, numTstep)] << ", NormalYc: " << NormalYc[idx(j, i, numTstep)] << ", NormalZc: " << NormalZc[idx(j, i, numTstep)] << std::endl;
            //}
        }
        infile.close();
    }

    cudaMemcpy(d_t, t, h_SimConsts.numTstep * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xc, xc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yc, yc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zc, zc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uc, uc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vc, vc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wc, wc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p0c, p0c, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1c, p1c, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_NormalXc, NormalXc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_NormalYc, NormalYc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_NormalZc, NormalZc, h_SimConsts.numTstep * h_SimConsts.numElem * sizeof(double), cudaMemcpyHostToDevice);

    // delete[] t; needed later
    delete[] xc;
    delete[] yc;
    delete[] zc;
    delete[] uc;
    delete[] vc;
    delete[] wc;
    delete[] p0c;
    delete[] p1c;
    delete[] NormalXc;
    delete[] NormalYc;
    delete[] NormalZc; // dont need all these

    GPU_Mem_Usage("After copying data to device: ");
    sound_directivity(t, d_t, d_xc, d_yc, d_zc, d_uc, d_vc, d_wc, d_p0c, d_p1c, d_NormalXc, d_NormalYc, d_NormalZc);

    cudaFree(d_t);
    cudaFree(d_xc);
    cudaFree(d_yc);
    cudaFree(d_zc);
    cudaFree(d_uc);
    cudaFree(d_vc);
    cudaFree(d_wc);
    cudaFree(d_p0c);
    cudaFree(d_p1c);
    cudaFree(d_NormalXc);
    cudaFree(d_NormalYc);
    cudaFree(d_NormalZc);

    return 0;
}