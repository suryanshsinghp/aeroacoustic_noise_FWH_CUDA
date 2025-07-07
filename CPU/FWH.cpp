// g++ -fsanitize=address -g main.cpp -o out && ./out
// g++ -std=c++17 -g -o out main.cpp && gdb ./out
// make run > log.txt && make clean
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include "global.hpp"
#include "sound_direc.hpp"

const double PI = std::acos(-1.0);
const double lscale = 1.0;
const double cscale = 340.0;
const int numElem = 29080;
const int numTstep = 2001;
int bodyNum = 1;

int main()
{
    Timer timer_all("Entire code", "s");

    double freq = 1.0;
    double tscle = 1.0 / freq;
    double vscale = lscale / tscle;

    double *t = new double[numTstep];
    double *xc = new double[numTstep * numElem]; // timestep will be contiguous, elements can be placed apart in memory
    double *yc = new double[numTstep * numElem];
    double *zc = new double[numTstep * numElem];
    double *uc = new double[numTstep * numElem];
    double *vc = new double[numTstep * numElem];
    double *wc = new double[numTstep * numElem];
    double *p0c = new double[numTstep * numElem];
    double *p1c = new double[numTstep * numElem];
    double *NormalXc = new double[numTstep * numElem];
    double *NormalYc = new double[numTstep * numElem];
    double *NormalZc = new double[numTstep * numElem];

    for (int i = 0; i < numElem; ++i)
    {
        // std::cout << "Reading element: " << i << " of " << numElem << std::endl;
        std::ostringstream fname;
        fname << "../input_data/input." << std::setw(3) << std::setfill('0') << bodyNum << "." << std::setw(5) << std::setfill('0') << i + 1 << ".bin";
        std::ifstream infile(fname.str(), std::ios::binary);
        if (!infile)
        {
            std::cerr << "Error opening file: " << fname.str() << std::endl;
            return 1;
        }
        double buffer[12];
        for (int j = 0; j < numTstep; ++j)
        {
            infile.read(reinterpret_cast<char *>(buffer), sizeof(buffer));
            t[j] = buffer[0];
            // std::cout << "idx(j, i, numTstep) = " << idx(j, i, numTstep) << std::endl;
            xc[idx(i, j, numTstep)] = buffer[1];
            yc[idx(i, j, numTstep)] = buffer[2];
            zc[idx(i, j, numTstep)] = buffer[3];
            uc[idx(i, j, numTstep)] = buffer[4];
            vc[idx(i, j, numTstep)] = buffer[5];
            wc[idx(i, j, numTstep)] = buffer[6];
            p0c[idx(i, j, numTstep)] = buffer[7];
            p1c[idx(i, j, numTstep)] = buffer[8];
            NormalXc[idx(i, j, numTstep)] = buffer[9];
            NormalYc[idx(i, j, numTstep)] = buffer[10];
            NormalZc[idx(i, j, numTstep)] = buffer[11];
            // if (i== 0 && j == 0)
            //{
            // std::cout << j << "t: " << t[j] << ", xc: " << xc[idx(j, i, numTstep)] << ", yc: " << yc[idx(j, i, numTstep)] << ", zc: " << zc[idx(j, i, numTstep)] << p0c[idx(j, i, numTstep)] << ", p1c: " << p1c[idx(j, i, numTstep)] << ", NormalXc: " << NormalXc[idx(j, i, numTstep)] << ", NormalYc: " << NormalYc[idx(j, i, numTstep)] << ", NormalZc: " << NormalZc[idx(j, i, numTstep)] << std::endl;
            //}
        }
        infile.close();
    }
    // std::cout << &t << std::endl;
    // Timer timer("sound directivity total calculation time","ms");
    sound_directivity(t, xc, yc, zc, uc, vc, wc, p0c, p1c, NormalXc, NormalYc, NormalZc);

    delete[] t;
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
    delete[] NormalZc;

    return 0;
}