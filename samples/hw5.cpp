#include <hip/hip_runtime.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double G = 6.674e-11;
double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<std::string>& type) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    type.resize(n);
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

// GPU kernel to compute accelerations and update velocities/positions
__global__ void run_step_kernel(int n, double dt, double eps, double G,
                                const double* qx, const double* qy, const double* qz,
                                const double* vx, const double* vy, const double* vz,
                                const double* m, double* ax, double* ay, double* az,
                                double* qx_out, double* qy_out, double* qz_out,
                                double* vx_out, double* vy_out, double* vz_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Compute accelerations
    double acc_x = 0.0, acc_y = 0.0, acc_z = 0.0;
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        double dx = qx[j] - qx[i];
        double dy = qy[j] - qy[i];
        double dz = qz[j] - qz[i];
        double dist3 = pow(dx * dx + dy * dy + dz * dz + eps * eps, 1.5);
        acc_x += G * m[j] * dx / dist3;
        acc_y += G * m[j] * dy / dist3;
        acc_z += G * m[j] * dz / dist3;
    }

    // Update velocities and positions
    double vx_new = vx[i] + acc_x * dt;
    double vy_new = vy[i] + acc_y * dt;
    double vz_new = vz[i] + acc_z * dt;
    double qx_new = qx[i] + vx_new * dt;
    double qy_new = qy[i] + vy_new * dt;
    double qz_new = qz[i] + vz_new * dt;

    // Store results
    ax[i] = acc_x;
    ay[i] = acc_y;
    az[i] = acc_z;
    qx_out[i] = qx_new;
    qy_out[i] = qy_new;
    qz_out[i] = qz_new;
    vx_out[i] = vx_new;
    vy_out[i] = vy_new;
    vz_out[i] = vz_new;
}

void run_step_gpu(int step, int n, double* qx, double* qy, double* qz,
                  double* vx, double* vy, double* vz, double* m,
                  double* ax, double* ay, double* az, double* qx_out,
                  double* qy_out, double* qz_out, double* vx_out,
                  double* vy_out, double* vz_out) {
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(run_step_kernel, dim3(blocks), dim3(threads_per_block), 0, 0,
                       n, param::dt, param::eps, param::G,
                       qx, qy, qz, vx, vy, vz, m, ax, ay, az,
                       qx_out, qy_out, qz_out, vx_out, vy_out, vz_out);
    hipDeviceSynchronize();
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }

    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);

    // Allocate and copy data to the GPU
    double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_m;
    double *d_ax, *d_ay, *d_az, *d_qx_out, *d_qy_out, *d_qz_out, *d_vx_out, *d_vy_out, *d_vz_out;
    hipMalloc(&d_qx, n * sizeof(double));
    hipMalloc(&d_qy, n * sizeof(double));
    hipMalloc(&d_qz, n * sizeof(double));
    hipMalloc(&d_vx, n * sizeof(double));
    hipMalloc(&d_vy, n * sizeof(double));
    hipMalloc(&d_vz, n * sizeof(double));
    hipMalloc(&d_m, n * sizeof(double));
    hipMalloc(&d_ax, n * sizeof(double));
    hipMalloc(&d_ay, n * sizeof(double));
    hipMalloc(&d_az, n * sizeof(double));
    hipMalloc(&d_qx_out, n * sizeof(double));
    hipMalloc(&d_qy_out, n * sizeof(double));
    hipMalloc(&d_qz_out, n * sizeof(double));
    hipMalloc(&d_vx_out, n * sizeof(double));
    hipMalloc(&d_vy_out, n * sizeof(double));
    hipMalloc(&d_vz_out, n * sizeof(double));

    hipMemcpy(d_qx, qx.data(), n * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_qy, qy.data(), n * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_qz, qz.data(), n * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_vx, vx.data(), n * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_vy, vy.data(), n * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_vz, vz.data(), n * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_m, m.data(), n * sizeof(double), hipMemcpyHostToDevice);

    // Run simulation
    for (int step = 0; step < param::n_steps; ++step) {
        run_step_gpu(step, n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz,
                     d_m, d_ax, d_ay, d_az, d_qx_out, d_qy_out,
                     d_qz_out, d_vx_out, d_vy_out, d_vz_out);

        // Swap buffers (output becomes input for the next step)
        std::swap(d_qx, d_qx_out);
        std::swap(d_qy, d_qy_out);
        std::swap(d_qz, d_qz_out);
        std::swap(d_vx, d_vx_out);
        std::swap(d_vy, d_vy_out);
        std::swap(d_vz, d_vz_out);
    }

    // Copy results back to the CPU
    hipMemcpy(qx.data(), d_qx, n * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(qy.data(), d_qy, n * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(qz.data(), d_qz, n * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(vx.data(), d_vx, n * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(vy.data(), d_vy, n * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(vz.data(), d_vz, n * sizeof(double), hipMemcpyDeviceToHost);

    // Clean up
    hipFree(d_qx);
    hipFree(d_qy);
    hipFree(d_qz);
    hipFree(d_vx);
    hipFree(d_vy);
    hipFree(d_vz);
    hipFree(d_m);
    hipFree(d_ax);
    hipFree(d_ay);
    hipFree(d_az);
    hipFree(d_qx_out);
    hipFree(d_qy_out);
    hipFree(d_qz_out);
    hipFree(d_vx_out);
    hipFree(d_vy_out);
    hipFree(d_vz_out);

    // Output results
    double min_dist = 0.0;  // Compute as needed
    int hit_time_step = 0;  // Compute as needed
    int gravity_device_id = 0;  // Compute as needed
    double missile_cost = 0.0;  // Compute as needed
    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);

    return 0;
}
