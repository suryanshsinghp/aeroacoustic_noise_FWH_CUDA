# Aeroacoustic Noise calculation using Ffowcs Williams-Hawkings Equation (C++ & CUDA)

Ffowcs Williams-Hawkings equation (shown below) is used to calculate the aeroacoustic noise.

$4\pi p^{\prime}({\bf x},t) =  \frac{1}{c}\int \Big[\frac{\dot{L}_r}{r(1-M_r)^2} \Big]_{t-r/c} dS
    +\int \Big[\frac{L_r-L_M}{r^2(1-M_r)^2} \Big]_{t-r/c} dS
    +\frac{1}{c}\int \Big[\frac{L_r(r\dot{M_r}+c(M_r-M^2))}{r^2(1-M_r)^3} \Big]_{t-r/c} dS \,\, .$

`CPU` contains the serial version of the code written in C++ which was used as baseline and for debugging; `GPU` contains the CUDA version that parallelize the calculation in both the surface marker and the timesteps using thread x and y index respectively. The radius and number of $\theta$ and $\phi$ points needs to be specified as input and the sound pressure is computed on surface of cube with radius $r$ and all combination of $(\theta, \phi)$ .

The following benchmarking takes a single blade of 3 blade rotor that contains 2001 timesteps and 29080 surface marker points. The sound pressure is calculated at 10 $\theta$ and 10 $\phi$ locations (total 100 points). The speedup in initial GPU version of the code was ~250X but the current version uploaded is optimized for reduce memory access, multiple division operations, arithmetic operations where not needed etc. The most current GPU version of the code is uploaded and one sample input file and output sound pressure at one location is uploaded.

| 100 points | calculation run time (total run time) | scaled     |
| ---------- | ------------------------------------- | ---------- |
| CPU        | 4284 s (4351 s)                       | 714 (61.2) |
| GPU        | 6 s (71 s)                            | 1 (1)      |

For 1024 points, the GPU code takes 67 seconds for calculating and 131 s for total run time including file read. Since all the input data is read in the start, the file reading time is ~60s for both 100 and 1024 points. Therefore for large number of points (eg- 360 * 360 total points), the speed up should be ~700X compare to the serial version. However, that would take lot of time to run on my computer, so I did not large number of points.
