#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

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
__global__ void compute_accelerations(int n, double* qx, double* qy, double* qz, 
                                      double* m, double* ax, double* ay, double* az, 
                                      double dt, double eps, double G, int step, 
                                      double* mass_device, const char* type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double ax_val = 0.0, ay_val = 0.0, az_val = 0.0;

    for (int j = 0; j < n; ++j) {
        if (i == j) continue;

        double mj = m[j];
        if (type[j] == 'd') { // 'd' stands for "device"
            mj = mass_device[j] + 0.5 * mass_device[j] * fabs(sin(step * dt / 6000));
        }
        double dx = qx[j] - qx[i];
        double dy = qy[j] - qy[i];
        double dz = qz[j] - qz[i];
        double dist3 = pow(dx * dx + dy * dy + dz * dz + eps * eps, 1.5);

        ax_val += G * mj * dx / dist3;
        ay_val += G * mj * dy / dist3;
        az_val += G * mj * dz / dist3;
    }

    ax[i] = ax_val;
    ay[i] = ay_val;
    az[i] = az_val;
}

__global__ void update_positions_velocities(int n, double* qx, double* qy, double* qz,
                                            double* vx, double* vy, double* vz, 
                                            double* ax, double* ay, double* az, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Update velocities
    vx[i] += ax[i] * dt;
    vy[i] += ay[i] * dt;
    vz[i] += az[i] * dt;

    // Update positions
    qx[i] += vx[i] * dt;
    qy[i] += vy[i] * dt;
    qz[i] += vz[i] * dt;
}

double problem1_min_distance(int n, int planet, int asteroid, double* qx, double* qy, double* qz,
                              double* vx, double* vy, double* vz, double* m, const char* type) {
    // Allocate GPU memory
    double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_m, *d_ax, *d_ay, *d_az;
    cudaMalloc(&d_qx, n * sizeof(double));
    cudaMalloc(&d_qy, n * sizeof(double));
    cudaMalloc(&d_qz, n * sizeof(double));
    cudaMalloc(&d_vx, n * sizeof(double));
    cudaMalloc(&d_vy, n * sizeof(double));
    cudaMalloc(&d_vz, n * sizeof(double));
    cudaMalloc(&d_m, n * sizeof(double));
    cudaMalloc(&d_ax, n * sizeof(double));
    cudaMalloc(&d_ay, n * sizeof(double));
    cudaMalloc(&d_az, n * sizeof(double));

    // Copy data to GPU
    cudaMemcpy(d_qx, qx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qy, qy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qz, qz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, vx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, vy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, vz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, n * sizeof(double), cudaMemcpyHostToDevice);

    // Variables for minimum distance
    double min_dist = std::numeric_limits<double>::infinity();

    // CUDA grid and block sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Loop through timesteps
    for (int step = 0; step <= param::n_steps; ++step) {
        // Compute accelerations
        compute_accelerations<<<gridSize, blockSize>>>(n, d_qx, d_qy, d_qz, d_m, 
                                                       d_ax, d_ay, d_az, 
                                                       param::dt, param::eps, param::G, 
                                                       step, d_m, type);

        // Update positions and velocities
        update_positions_velocities<<<gridSize, blockSize>>>(n, d_qx, d_qy, d_qz, 
                                                             d_vx, d_vy, d_vz, 
                                                             d_ax, d_ay, d_az, param::dt);

        // Copy positions back to CPU to compute distance
        cudaMemcpy(qx, d_qx, n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(qy, d_qy, n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(qz, d_qz, n * sizeof(double), cudaMemcpyDeviceToHost);

        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    }

    // Free GPU memory
    cudaFree(d_qx);
    cudaFree(d_qy);
    cudaFree(d_qz);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_m);
    cudaFree(d_ax);
    cudaFree(d_ay);
    cudaFree(d_az);

    return min_dist;
}
__global__ void check_collision(int n, double* qx, double* qy, double* qz, 
                                int planet, int asteroid, double radius, int* collision_step, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);

        if (dist <= radius) {
            *collision_step = step;
        }
    }
}

int problem2_collision_time(int n, int planet, int asteroid, double* qx, double* qy, double* qz,
                            double* vx, double* vy, double* vz, double* m, const char* type) {
    // Allocate GPU memory
    double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_m, *d_ax, *d_ay, *d_az;
    int *d_collision_step;
    cudaMalloc(&d_qx, n * sizeof(double));
    cudaMalloc(&d_qy, n * sizeof(double));
    cudaMalloc(&d_qz, n * sizeof(double));
    cudaMalloc(&d_vx, n * sizeof(double));
    cudaMalloc(&d_vy, n * sizeof(double));
    cudaMalloc(&d_vz, n * sizeof(double));
    cudaMalloc(&d_m, n * sizeof(double));
    cudaMalloc(&d_ax, n * sizeof(double));
    cudaMalloc(&d_ay, n * sizeof(double));
    cudaMalloc(&d_az, n * sizeof(double));
    cudaMalloc(&d_collision_step, sizeof(int));

    // Initialize collision_step to -1
    int collision_step = -1;
    cudaMemcpy(d_collision_step, &collision_step, sizeof(int), cudaMemcpyHostToDevice);

    // Copy data to GPU
    cudaMemcpy(d_qx, qx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qy, qy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qz, qz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, vx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, vy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, vz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, n * sizeof(double), cudaMemcpyHostToDevice);

    // CUDA grid and block sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Loop through timesteps
    for (int step = 0; step <= param::n_steps; ++step) {
        // Compute accelerations
        compute_accelerations<<<gridSize, blockSize>>>(n, d_qx, d_qy, d_qz, d_m,
                                                       d_ax, d_ay, d_az,
                                                       param::dt, param::eps, param::G,
                                                       step, d_m, type);

        // Update positions and velocities
        update_positions_velocities<<<gridSize, blockSize>>>(n, d_qx, d_qy, d_qz,
                                                             d_vx, d_vy, d_vz,
                                                             d_ax, d_ay, d_az, param::dt);

        // Check for collision
        check_collision<<<1, 1>>>(n, d_qx, d_qy, d_qz, planet, asteroid, param::planet_radius,
                                  d_collision_step, step);

        // Copy collision_step back to CPU
        cudaMemcpy(&collision_step, d_collision_step, sizeof(int), cudaMemcpyDeviceToHost);
        if (collision_step != -1) break;
    }

    // Free GPU memory
    cudaFree(d_qx);
    cudaFree(d_qy);
    cudaFree(d_qz);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_m);
    cudaFree(d_ax);
    cudaFree(d_ay);
    cudaFree(d_az);
    cudaFree(d_collision_step);

    return collision_step;
}
__global__ void compute_missile_impact(int n, double* qx, double* qy, double* qz,
                                       double* vx, double* vy, double* vz,
                                       double planet_x, double planet_y, double planet_z,
                                       double missile_speed, int step, double dt,
                                       int* optimal_device, double* min_cost,
                                       double* impact_time) {
    int device_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (device_id >= n) return;

    double dx = qx[device_id] - planet_x;
    double dy = qy[device_id] - planet_y;
    double dz = qz[device_id] - planet_z;
    double dist_to_planet = sqrt(dx * dx + dy * dy + dz * dz);
    double missile_travel_time = dist_to_planet / missile_speed;

    double asteroid_x = qx[device_id] + vx[device_id] * missile_travel_time;
    double asteroid_y = qy[device_id] + vy[device_id] * missile_travel_time;
    double asteroid_z = qz[device_id] + vz[device_id] * missile_travel_time;

    double final_dx = asteroid_x - planet_x;
    double final_dy = asteroid_y - planet_y;
    double final_dz = asteroid_z - planet_z;
    double final_distance = sqrt(final_dx * final_dx + final_dy * final_dy + final_dz * final_dz);

    if (final_distance < param::planet_radius) {
        double cost = param::get_missile_cost(step * dt);
        if (cost < *min_cost) {
            *min_cost = cost;
            *optimal_device = device_id;
            *impact_time = missile_travel_time;
        }
    }
}

void problem3_optimal_launch(int n, int planet, double* qx, double* qy, double* qz,
                             double* vx, double* vy, double* vz) {
    // Allocate GPU memory
    double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz;
    int *d_optimal_device;
    double *d_min_cost, *d_impact_time;
    cudaMalloc(&d_qx, n * sizeof(double));
    cudaMalloc(&d_qy, n * sizeof(double));
    cudaMalloc(&d_qz, n * sizeof(double));
    cudaMalloc(&d_vx, n * sizeof(double));
    cudaMalloc(&d_vy, n * sizeof(double));
    cudaMalloc(&d_vz, n * sizeof(double));
    cudaMalloc(&d_optimal_device, sizeof(int));
    cudaMalloc(&d_min_cost, sizeof(double));
    cudaMalloc(&d_impact_time, sizeof(double));

    // Initialize variables
    int optimal_device = -1;
    double min_cost = std::numeric_limits<double>::infinity();
    double impact_time = -1;
    cudaMemcpy(d_optimal_device, &optimal_device, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min_cost, &min_cost, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_impact_time, &impact_time, sizeof(double), cudaMemcpyHostToDevice);

    // Copy data to GPU
    cudaMemcpy(d_qx, qx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qy, qy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qz, qz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, vx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, vy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, vz, n * sizeof(double), cudaMemcpyHostToDevice);

    // CUDA grid and block sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Loop through timesteps
    for (int step = 0; step <= param::n_steps; ++step) {
        compute_missile_impact<<<gridSize, blockSize>>>(n, d_qx, d_qy, d_qz,
                                                        d_vx, d_vy, d_vz,
                                                        qx[planet], qy[planet], qz[planet],
                                                        param::missile_speed, step, param::dt,
                                                        d_optimal_device, d_min_cost, d_impact_time);
    }

    // Copy results back to CPU
    cudaMemcpy(&optimal_device, d_optimal_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&min_cost, d_min_cost, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&impact_time, d_impact_time, sizeof(double), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Optimal Gravity Device ID: " << optimal_device << "\n";
    std::cout << "Minimum Missile Cost: " << min_cost << "\n";
    std::cout << "Impact Time: " << impact_time << " seconds\n";

    // Free GPU memory
    cudaFree(d_qx);
    cudaFree(d_qy);
    cudaFree(d_qz);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_optimal_device);
    cudaFree(d_min_cost);
    cudaFree(d_impact_time);
}
