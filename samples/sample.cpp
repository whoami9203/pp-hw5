#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hip/hip_runtime.h"
namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double G = 6.674e-11;
__device__ double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000.0));
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;
__host__ __device__ double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

inline void hipCheck(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "HIP Error at: " << msg << " - " << hipGetErrorString(err) << std::endl;
        exit(1);
    }
}

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

// Convert type strings to int flags for device
// Let's say:
// planet -> 1
// device -> 2
// asteroid or other -> 0
int type_to_int(const std::string& t) {
    if (t == "planet")
        return 1;
    else if (t == "device")
        return 2;
    // asteroid or other is default 0 (assuming asteroid isn't planet or device)
    return 0;
}

__global__ void compute_acceleration_kernel(
    int n,
    double* qx, double* qy, double* qz,
    double* vx, double* vy, double* vz,
    const double* m,
    const int* type_arr, int step) {
    extern __shared__ double shared_data[];  // Shared memory for data
    double* shared_x = shared_data;
    double* shared_y = shared_data + n;
    double* shared_z = shared_data + 2 * n;

    int i = blockIdx.x;
    int j = threadIdx.x;
    if (i >= n) return;
    if (j >= n) return;

    shared_x[j] = 0.0;
    shared_y[j] = 0.0;
    shared_z[j] = 0.0;
    if (j != i) {
        double mj = m[j];
        if (type_arr[j] == 2) {
            mj = param::gravity_device_mass(mj, step * param::dt);
        }

        double dx = qx[j] - qx[i];
        double dy = qy[j] - qy[i];
        double dz = qz[j] - qz[i];
        double dist3 = pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);

        shared_x[j] = param::G * mj * dx / dist3;
        shared_y[j] = param::G * mj * dy / dist3;
        shared_z[j] = param::G * mj * dz / dist3;
    }
    __syncthreads();

    // if (j == 0) {
    //     for (int i = 1; i < n; i++) {
    //         shared_x[0] += shared_x[i];
    //         shared_y[0] += shared_y[i];
    //         shared_z[0] += shared_z[i];
    //     }
    // }
    // Parallel reduction over all threads in the block
    for (int stride = 512; stride > 0; stride >>= 1) {
        if (j + stride < n) {
            shared_x[j] += shared_x[j + stride];
            shared_y[j] += shared_y[j + stride];
            shared_z[j] += shared_z[j + stride];
        }
        __syncthreads();
    }

    // After reduction, thread 0 has the sum
    if (j == 0) {
        vx[i] += shared_x[0] * param::dt;
        vy[i] += shared_y[0] * param::dt;
        vz[i] += shared_z[0] * param::dt;
    }
}

__global__ void problem1_pos_kernel(double* qx, double* qy, double* qz,
                                    double* vx, double* vy, double* vz,
                                    int planet, int asteroid,
                                    double* min_dist) {
    int i = threadIdx.x;
    qx[i] += vx[i] * param::dt;
    qy[i] += vy[i] * param::dt;
    qz[i] += vz[i] * param::dt;
    __syncthreads();
    if (i == planet) {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        double dist = (dx * dx + dy * dy + dz * dz);
        if (dist < *min_dist) {
            *min_dist = dist;
        }
    }
}
__global__ void problem2_pos_kernel(double* qx, double* qy, double* qz,
                                    double* vx, double* vy, double* vz,
                                    int planet, int asteroid,
                                    int* hit_time_step, int step) {
    int i = threadIdx.x;
    qx[i] += vx[i] * param::dt;
    qy[i] += vy[i] * param::dt;
    qz[i] += vz[i] * param::dt;
    __syncthreads();
    if (i == planet) {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
            *hit_time_step = step;
        }
    }
}
__global__ void problem3_pos_kernel(double* qx, double* qy, double* qz,
                                    double* vx, double* vy, double* vz,
                                    double* m,
                                    int planet, int asteroid, int dev,
                                    int* device_destroy_step, int* prevent_collision_hit_step, int step) {
    int i = threadIdx.x;
    qx[i] += vx[i] * param::dt;
    qy[i] += vy[i] * param::dt;
    qz[i] += vz[i] * param::dt;
    __syncthreads();
    if (i == planet && m[dev] != 0.0) {
        double dx = qx[dev] - qx[planet];
        double dy = qy[dev] - qy[planet];
        double dz = qz[dev] - qz[planet];
        double dist_planet_device = sqrt(dx * dx + dy * dy + dz * dz);

        double missile_travel_distance = step * param::dt * param::missile_speed;
        if (missile_travel_distance > dist_planet_device) {
            *device_destroy_step = step;
            m[dev] = 0.0;  // set mass to zero
        }
    }
    if (i == asteroid) {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        double dist = dx * dx + dy * dy + dz * dz;
        if (dist < param::planet_radius * param::planet_radius) {
            *prevent_collision_hit_step = step;
        }
    }
}

// Function to handle computations on Device 1 (Problem 1 and Problem 2)
void handle_device1(int n, int planet, int asteroid,
                    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
                    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
                    std::vector<double>& m, std::vector<int>& type_arr,
                    double& h_min_dist, int& hit_time_step) {
    hipSetDevice(1);
    hipStream_t stream;
    hipStreamCreate(&stream);

    double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_m;
    int* d_type;

    // Allocate and copy data to Device 1
    hipMalloc((void**)&d_qx, n * sizeof(double));
    hipMalloc((void**)&d_qy, n * sizeof(double));
    hipMalloc((void**)&d_qz, n * sizeof(double));
    hipMalloc((void**)&d_vx, n * sizeof(double));
    hipMalloc((void**)&d_vy, n * sizeof(double));
    hipMalloc((void**)&d_vz, n * sizeof(double));
    hipMalloc((void**)&d_m, n * sizeof(double));
    hipMalloc((void**)&d_type, n * sizeof(int));

    hipMemcpyAsync(d_qx, qx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_qy, qy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_qz, qz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_vx, vx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_vy, vy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_vz, vz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_m, m.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_type, type_arr.data(), n * sizeof(int), hipMemcpyHostToDevice, stream);

    // Problem 2
    int* temp_hit;
    hipHostMalloc(&temp_hit, sizeof(int), hipHostMallocMapped);
    *temp_hit = -2;
    int* d_hit_time_step;
    hipHostGetDevicePointer(reinterpret_cast<void**>(&d_hit_time_step), temp_hit, 0);

    for (int i = 0; i < param::n_steps; i++) {
        if (i > 0) {
            compute_acceleration_kernel<<<n, n, 3 * n * sizeof(double), stream>>>(n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, d_type, i);
            hipStreamSynchronize(stream);
            problem2_pos_kernel<<<1, n, 0, stream>>>(d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, planet, asteroid, d_hit_time_step, i);
        }

        if (hit_time_step != -2) {
            break;
        }
    }

    hit_time_step = *temp_hit;

    // P1
    double* d_min_dist;
    h_min_dist = std::numeric_limits<double>::max();
    hipMalloc((void**)&d_min_dist, sizeof(double));
    hipMemcpyAsync(d_min_dist, &h_min_dist, sizeof(double), hipMemcpyHostToDevice, stream);
    std::vector<double>m1=m;
    for (int i = 0; i < n; i++) {
        if (type_arr[i] == 2) {
            m1[i] = 0;
        }
    }
    hipMemcpyAsync(d_qx, qx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_qy, qy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_qz, qz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_vx, vx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_vy, vy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_vz, vz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_m, m1.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_type, type_arr.data(), n * sizeof(int), hipMemcpyHostToDevice, stream);
    for (int i = 0; i < param::n_steps; i++) {
        if (i > 0) {
            compute_acceleration_kernel<<<n, n, 3 * n * sizeof(double), stream>>>(n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, d_type, i);
            hipStreamSynchronize(stream);
            problem1_pos_kernel<<<1, n, 0, stream>>>(d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, planet, asteroid, d_min_dist);
        }
    }
    hipMemcpy(&h_min_dist, d_min_dist, sizeof(double), hipMemcpyDeviceToHost);

    hipStreamDestroy(stream);
    hipFree(d_qx);
    hipFree(d_qy);
    hipFree(d_qz);
    hipFree(d_vx);
    hipFree(d_vy);
    hipFree(d_vz);
    hipFree(d_m);
    hipFree(d_type);
    hipFree(d_min_dist);
}

// Function to handle computations on Device 0 (Problem 3)
void handle_device0(int n, int planet, int asteroid, std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
                    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
                    std::vector<double>& m, std::vector<int>& type_arr,
                    int& best_device, double& best_cost) {
    hipSetDevice(0);
    hipStream_t stream;
    hipStreamCreate(&stream);

    double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_m;
    int* d_type;

    // Allocate and copy data to Device 0
    hipMalloc((void**)&d_qx, n * sizeof(double));
    hipMalloc((void**)&d_qy, n * sizeof(double));
    hipMalloc((void**)&d_qz, n * sizeof(double));
    hipMalloc((void**)&d_vx, n * sizeof(double));
    hipMalloc((void**)&d_vy, n * sizeof(double));
    hipMalloc((void**)&d_vz, n * sizeof(double));
    hipMalloc((void**)&d_m, n * sizeof(double));
    hipMalloc((void**)&d_type, n * sizeof(int));

    int* prevent_collision_hit_step;
    hipHostMalloc(&prevent_collision_hit_step, sizeof(int), hipHostMallocMapped);
    int* d_prevent_collision_hit_step;
    hipHostGetDevicePointer(reinterpret_cast<void**>(&d_prevent_collision_hit_step), prevent_collision_hit_step, 0);

    int* device_destroy_step;
    hipHostMalloc(&device_destroy_step, sizeof(int), hipHostMallocMapped);
    int* d_device_destroy_step;
    hipHostGetDevicePointer(reinterpret_cast<void**>(&d_device_destroy_step), device_destroy_step, 0);

    for (int dev = 0; dev < n; dev++) {
        if (type_arr[dev] != 2) continue;  // Only consider devices.

        hipMemcpyAsync(d_qx, qx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_qy, qy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_qz, qz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_vx, vx.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_vy, vy.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_vz, vz.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_m, m.data(), n * sizeof(double), hipMemcpyHostToDevice, stream);
        hipMemcpyAsync(d_type, type_arr.data(), n * sizeof(int), hipMemcpyHostToDevice, stream);

        *prevent_collision_hit_step = -1;
        *device_destroy_step = -1;

        for (int step = 0; step <= param::n_steps; step++) {
            if (step > 0) {
                compute_acceleration_kernel<<<n, n, 3 * n * sizeof(double), stream>>>(n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, d_type, step);
                hipStreamSynchronize(stream);
                problem3_pos_kernel<<<1, n, 0, stream>>>(d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, planet, asteroid, dev, d_device_destroy_step, d_prevent_collision_hit_step, step);
            }

            if (*prevent_collision_hit_step != -1) {
                break;
            }
        }
        // After simulation:
        if (*prevent_collision_hit_step == -1) {
            // Collision prevented
            // Compute missile cost
            double t_hit = *device_destroy_step * param::dt;
            double cost = param::get_missile_cost(t_hit);
            if (cost < best_cost) {
                best_cost = cost;
                best_device = dev;
            }
        }
    }

    hipStreamDestroy(stream);
    hipFree(d_qx);
    hipFree(d_qy);
    hipFree(d_qz);
    hipFree(d_vx);
    hipFree(d_vy);
    hipFree(d_vz);
    hipFree(d_m);
    hipFree(d_type);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }

    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;

    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);

    std::vector<int> type_arr(n, 0);
    for (int i = 0; i < n; i++) {
        type_arr[i] = type_to_int(type[i]);
    }

    // Shared results
    double h_min_dist = 0.0;
    int hit_time_step = -2;
    int best_device = -1;
    double best_cost = std::numeric_limits<double>::infinity();

    // Create threads
    std::thread thread1(handle_device1, n, planet, asteroid, std::ref(qx), std::ref(qy), std::ref(qz),
                        std::ref(vx), std::ref(vy), std::ref(vz), std::ref(m), std::ref(type_arr),
                        std::ref(h_min_dist), std::ref(hit_time_step));

    std::thread thread2(handle_device0, n, planet, asteroid, std::ref(qx), std::ref(qy), std::ref(qz),
                        std::ref(vx), std::ref(vy), std::ref(vz), std::ref(m), std::ref(type_arr),
                        std::ref(best_device), std::ref(best_cost));

    // Wait for threads to finish
    thread1.join();
    thread2.join();

    int gravity_device_id;
    double missile_cost;
    if (best_device != -1) {
        gravity_device_id = best_device;
        missile_cost = best_cost;
    } else {
        // Could not prevent collision or no need to
        gravity_device_id = -1;
        missile_cost = 0;
    }

    // Output results
    write_output(argv[2], sqrt(h_min_dist), hit_time_step, gravity_device_id, missile_cost);

    return 0;
}