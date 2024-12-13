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

#define BLOCK_SIZE 1024

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps2 = (1e-3) * (1e-3);
const double G = 6.674e-11;
__host__ __device__ double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius2 = 1e7 * 1e7;
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

__global__ void problem1(int *n, double4 *posw, double4 *vtype) {
    extern __shared__ double3 acceleration[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // calculate i-th body's acceleration
    acceleration[tid] = {0, 0, 0};
    if (tid < *n) {
        double4 pos_i = posw[bid];
        double4 pos_j = posw[tid];
        double3 d;
        d.x = pos_j.x - pos_i.x;
        d.y = pos_j.y - pos_i.y;
        d.z = pos_j.z - pos_i.z;
        double dist2 = (d.x * d.x + d.y * d.y + d.z * d.z + param::eps2);
        double dist6 = dist2 * dist2 * dist2;
        double dist3 = sqrt(dist6);
        double mass = vtype[tid].w == 1 ? 0 : pos_j.w;
        // double Gmd = param::G * mass;
        acceleration[tid] = {param::G * mass * d.x / dist3, 
                             param::G * mass * d.y / dist3, 
                             param::G * mass * d.z / dist3};
    }
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            acceleration[tid].x += acceleration[tid + s].x;
            acceleration[tid].y += acceleration[tid + s].y; 
            acceleration[tid].z += acceleration[tid + s].z;
        }
        __syncthreads();  // Synchronize to ensure all threads are done
    }
    if (tid == 0) {
        // acc[i] = acceleration[0];
        vtype[bid].x += acceleration[0].x * param::dt;
        vtype[bid].y += acceleration[0].y * param::dt;
        vtype[bid].z += acceleration[0].z * param::dt;
        // posw[bid].x += vtype[bid].x * param::dt;
        // posw[bid].y += vtype[bid].y * param::dt;
        // posw[bid].z += vtype[bid].z * param::dt;
    }
}
__global__ void update1(int *step, int *planet, int *asteroid, double4 *posw, double4 *vtype,
                        double *min_dist) {
    int tid = threadIdx.x;
    posw[tid].x += vtype[tid].x * param::dt;
    posw[tid].y += vtype[tid].y * param::dt;
    posw[tid].z += vtype[tid].z * param::dt;
    __syncthreads();
    
    if (tid == 0) {
        double4 dist;
        dist.x = posw[*planet].x - posw[*asteroid].x;
        dist.y = posw[*planet].y - posw[*asteroid].y;
        dist.z = posw[*planet].z - posw[*asteroid].z;
        dist.w = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
        if (dist.w < *min_dist) {
            *min_dist = dist.w;
        }
        *step += 1;
    }
}
__global__ void problem2(int *step, int *n, double4 *posw, double4 *vtype, int *hit_time_step) {
    extern __shared__ double3 acceleration[];

    if (*hit_time_step >= 0)
        return;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // calculate i-th body's acceleration
    acceleration[tid] = {0, 0, 0};
    if (tid < *n) {
        double4 pos_i = posw[bid];
        double4 pos_j = posw[tid];
        double3 d;
        d.x = pos_j.x - pos_i.x;
        d.y = pos_j.y - pos_i.y;
        d.z = pos_j.z - pos_i.z;
        double dist3 = pow(d.x * d.x + d.y * d.y + d.z * d.z + param::eps2, 1.5);
        double mass = vtype[tid].w == 1 ? 
                param::gravity_device_mass(pos_j.w, (*step)*param::dt) : pos_j.w;
        double Gmd = param::G * mass / dist3;
        acceleration[tid] = {d.x * Gmd, d.y * Gmd, d.z * Gmd};
    }
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            acceleration[tid].x += acceleration[tid + s].x;
            acceleration[tid].y += acceleration[tid + s].y; 
            acceleration[tid].z += acceleration[tid + s].z;
        }
        __syncthreads();  // Synchronize to ensure all threads are done
    }
    if (tid == 0) {
        // acc[i] = acceleration[0];
        vtype[bid].x += acceleration[0].x * param::dt;
        vtype[bid].y += acceleration[0].y * param::dt;
        vtype[bid].z += acceleration[0].z * param::dt;
        // posw[bid].x += vtype[bid].x * param::dt;
        // posw[bid].y += vtype[bid].y * param::dt;
        // posw[bid].z += vtype[bid].z * param::dt;
    }
    
}
__global__ void update2(int *step, int *planet, int *asteroid, double4 *posw, double4 *vtype, int *hit_time_step) {
    int tid = threadIdx.x;
    posw[tid].x += vtype[tid].x * param::dt;
    posw[tid].y += vtype[tid].y * param::dt;
    posw[tid].z += vtype[tid].z * param::dt;
    __syncthreads();

    if (tid == 0 && *hit_time_step < 0) {
        double4 dist;
        dist.x = posw[*planet].x - posw[*asteroid].x;
        dist.y = posw[*planet].y - posw[*asteroid].y;
        dist.z = posw[*planet].z - posw[*asteroid].z;
        dist.w = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
        if (dist.w < param::planet_radius2) {
            *hit_time_step = *step;
        }
        *step += 1;
    }
}
__global__ void problem3(int *step, int *n, double4 *posw, double4 *vtype, bool *collision_avoided) {
    extern __shared__ double3 acceleration[];

    if (!*collision_avoided)
        return;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // calculate i-th body's acceleration
    acceleration[tid] = {0, 0, 0};
    if (tid < *n) {
        double4 pos_i = posw[bid];
        double4 pos_j = posw[tid];
        double3 d;
        d.x = pos_j.x - pos_i.x;
        d.y = pos_j.y - pos_i.y;
        d.z = pos_j.z - pos_i.z;
        double dist3 = pow(d.x * d.x + d.y * d.y + d.z * d.z + param::eps2, 1.5);
        double mass = vtype[tid].w == 1 ? 
                param::gravity_device_mass(pos_j.w, (*step)*param::dt) : pos_j.w;
        double Gmd = param::G * mass / dist3;
        acceleration[tid] = {d.x * Gmd, d.y * Gmd, d.z * Gmd};
    }
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            acceleration[tid].x += acceleration[tid + s].x;
            acceleration[tid].y += acceleration[tid + s].y; 
            acceleration[tid].z += acceleration[tid + s].z;
        }
        __syncthreads();  // Synchronize to ensure all threads are done
    }
    if (tid == 0) {
        // acc[i] = acceleration[0];
        vtype[bid].x += acceleration[0].x * param::dt;
        vtype[bid].y += acceleration[0].y * param::dt;
        vtype[bid].z += acceleration[0].z * param::dt;
        // posw[bid].x += vtype[bid].x * param::dt;
        // posw[bid].y += vtype[bid].y * param::dt;
        // posw[bid].z += vtype[bid].z * param::dt;
    }
}
__global__ void update3(int *step, int *planet, int *asteroid, int *device, double4 *posw, double4 *vtype,
                        int *hit_time_step, bool *collision_avoided) {
    int tid = threadIdx.x;
    posw[tid].x += vtype[tid].x * param::dt;
    posw[tid].y += vtype[tid].y * param::dt;
    posw[tid].z += vtype[tid].z * param::dt;
    __syncthreads();
    
    if (tid == 0 && collision_avoided) {
        if (*hit_time_step < 0) {
            double4 dist;
            dist.x = posw[*planet].x - posw[*device].x;
            dist.y = posw[*planet].y - posw[*device].y;
            dist.z = posw[*planet].z - posw[*device].z;
            dist.w = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
            double missile_distance = (*step) * param::dt * param::missile_speed;
            if (dist.w < missile_distance * missile_distance) {
                *hit_time_step = *step;
                posw[*device].w = 0;
            }
        }

        double4 dist;
        dist.x = posw[*planet].x - posw[*asteroid].x;
        dist.y = posw[*planet].y - posw[*asteroid].y;
        dist.z = posw[*planet].z - posw[*asteroid].z;
        dist.w = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
        if (dist.w < param::planet_radius2) {
            *collision_avoided = false;
        }
        
        *step += 1;
    }
}

// Function to run problem2 on GPU 0
void run_problem12( double4 *h_posw, double4 *h_vtype, 
                    int h_step, int n, int gridSize, int blockSize, int shmem,
                    int planet, int asteroid, double &min_dist, int &hit_time_step) {
    hipSetDevice(0);  // Set GPU 0

    hipError_t err;
    double4 *d_posw, *d_vtype;
    int *d_step, *d_n, *d_planet, *d_asteroid;
    double *d_min_dist;

    hipMalloc(&d_posw, n * sizeof(double4));
    hipMalloc(&d_vtype, n * sizeof(double4));
    hipMalloc(&d_step, sizeof(int));
    hipMalloc(&d_n, sizeof(int));
    hipMalloc(&d_planet, sizeof(int));
    hipMalloc(&d_asteroid, sizeof(int));
    hipMalloc(&d_min_dist, sizeof(double));

    // Copy data from host to device
    hipMemcpy(d_posw, h_posw, n * sizeof(double4), hipMemcpyHostToDevice);
    hipMemcpy(d_vtype, h_vtype, n * sizeof(double4), hipMemcpyHostToDevice);
    hipMemcpy(d_step, &h_step, sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_n, &n, sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_planet, &planet, sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_asteroid, &asteroid, sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_min_dist, &min_dist, sizeof(double), hipMemcpyHostToDevice);
    err = hipGetLastError();
    if (err != hipSuccess){
        fprintf(stderr, "hipMalloc error: %s\n", hipGetErrorString(err));
    }

    for (int step = 1; step <= param::n_steps; step++) {
        problem1<<<gridSize, blockSize, shmem>>>(d_n, d_posw, d_vtype);
        update1<<<1, n>>>(d_step, d_planet, d_asteroid, d_posw, d_vtype,
                            d_min_dist);
    }
    err = hipGetLastError();
    if (err != hipSuccess){
        fprintf(stderr, "kernel1 error: %s\n", hipGetErrorString(err));
    }

    hipMemcpy(&min_dist, d_min_dist, sizeof(double), hipMemcpyDeviceToHost);

    

    // Problem 2
    int *d_hit_time_step;
    
    hipMalloc(&d_hit_time_step, sizeof(int));
    // Copy data from host to device
    hipMemcpy(d_hit_time_step, &hit_time_step, sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_posw, h_posw, n * sizeof(double4), hipMemcpyHostToDevice);
    hipMemcpy(d_vtype, h_vtype, n * sizeof(double4), hipMemcpyHostToDevice);
    hipMemcpy(d_step, &h_step, sizeof(int), hipMemcpyHostToDevice);

    for (int step = 1; step <= param::n_steps; step++) {
        problem2<<<gridSize, blockSize, shmem>>>(
            d_step, d_n, d_posw, d_vtype, d_hit_time_step);
        update2<<<1, n>>>(
            d_step, d_planet, d_asteroid, d_posw, d_vtype, d_hit_time_step);
    }

    err = hipGetLastError();
    if (err != hipSuccess){
        fprintf(stderr, "kernel2 error: %s\n", hipGetErrorString(err));
    }

    hipMemcpy(&hit_time_step, d_hit_time_step, sizeof(int), hipMemcpyDeviceToHost);

    // Free allocated memory
    hipFree(d_posw);
    hipFree(d_vtype);
    hipFree(d_step);
    hipFree(d_n);
    hipFree(d_planet);
    hipFree(d_asteroid);
    hipFree(d_min_dist);
    hipFree(d_hit_time_step);
}

// Function to run problem3 on GPU 1
void run_problem3(  double4 *h_posw, double4 *h_vtype, 
                    int h_step, int n, int gridSize, int blockSize, int shmem,
                    int planet, int asteroid, int &gravity_device_id, int &best_step) {
    hipSetDevice(1);  // Set GPU 1

    // Problem 3

    double4 *d_posw, *d_vtype;
    int *d_step, *d_n, *d_planet, *d_asteroid;
    bool *d_collision_avoided;
    int *d_step_missile_hits;
    int *d_device;

    hipMalloc(&d_posw, n * sizeof(double4));
    hipMalloc(&d_vtype, n * sizeof(double4));
    hipMalloc(&d_step, sizeof(int));
    hipMalloc(&d_n, sizeof(int));
    hipMalloc(&d_planet, sizeof(int));
    hipMalloc(&d_asteroid, sizeof(int));
    hipMalloc(&d_collision_avoided, sizeof(bool));
    hipMalloc(&d_step_missile_hits, sizeof(int));
    hipMalloc(&d_device, sizeof(int));

    hipMemcpy(d_n, &n, sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_planet, &planet, sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_asteroid, &asteroid, sizeof(int), hipMemcpyHostToDevice);

    for (int device_id = 0; device_id < n; ++device_id) {
        if (h_vtype[device_id].w == 0) continue;

        bool collision_avoided = true;
        int step_missile_hits = -1;

        hipMemcpy(d_posw, h_posw, n * sizeof(double4), hipMemcpyHostToDevice);
        hipMemcpy(d_vtype, h_vtype, n * sizeof(double4), hipMemcpyHostToDevice);
        hipMemcpy(d_step, &h_step, sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(d_device, &device_id, sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(d_collision_avoided, &collision_avoided, sizeof(bool), hipMemcpyHostToDevice);
        hipMemcpy(d_step_missile_hits, &step_missile_hits, sizeof(int), hipMemcpyHostToDevice);

        for (int step = 1; step <= param::n_steps; ++step) {
            problem3<<<gridSize, blockSize, shmem>>>(
                d_step, d_n, d_posw, d_vtype, d_collision_avoided);
            update3<<<1, n>>>(
                d_step, d_planet, d_asteroid, d_device, d_posw, d_vtype,
                d_step_missile_hits, d_collision_avoided);
        }
        hipMemcpy(&collision_avoided, d_collision_avoided, sizeof(bool), hipMemcpyDeviceToHost);
        hipMemcpy(&step_missile_hits, d_step_missile_hits, sizeof(int), hipMemcpyDeviceToHost);

        if (collision_avoided && step_missile_hits < best_step) {
            best_step = step_missile_hits;
            gravity_device_id = device_id;
        }
    }

    // Free allocated memory
    hipFree(d_collision_avoided);
    hipFree(d_step_missile_hits);
    hipFree(d_device);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;

    auto distance2 = [&](int i, int j) -> double {
        double dx = qx[i] - qx[j];
        double dy = qy[i] - qy[j];
        double dz = qz[i] - qz[j];
        return (dx * dx + dy * dy + dz * dz);
    };

    auto start_all = std::chrono::high_resolution_clock::now();

    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);

    // kernel configuration
    int blockSize = 1;
    for (int i = 1; i < n; i <<= 1) {
        blockSize <<= 1;
    }
    fprintf(stderr, "blockSize: %d\n", blockSize);
    int shmem = blockSize * sizeof(double3);
    int gridSize = n;

    // Host variables
    double4* h_posw = (double4*)malloc(n * sizeof(double4));
    double4* h_vtype = (double4*)malloc(n * sizeof(double4));
    int h_step = 1;

    // variables to output
    double min_dist = distance2(planet, asteroid);
    int hit_time_step = -2;
    int gravity_device_id = -1;
    int best_step = 400000;

    // Populate host data (e.g., position, velocity, mass, type)
    // Combine qx, qy, qz, m into h_posw
    // Combine vx, vy, vz, type into h_vtype
    for (int i = 0; i < n; i++) {
        h_posw[i] = {qx[i], qy[i], qz[i], m[i]};
        h_vtype[i] = {vx[i], vy[i], vz[i], (type[i] == "device") ? 1.0 : 0.0};
    }

    // Launch threads for both problems
    std::thread thread0(run_problem12,  h_posw, h_vtype,
                                        h_step, n, gridSize, blockSize, shmem, 
                                        planet, asteroid, std::ref(min_dist), std::ref(hit_time_step));
    std::thread thread1(run_problem3,   h_posw, h_vtype,
                                        h_step, n, gridSize, blockSize, shmem, 
                                        planet, asteroid, std::ref(gravity_device_id), std::ref(best_step));

    // Wait for threads to finish
    thread0.join();
    thread1.join();

    double missile_cost = gravity_device_id == -1 ? 0 : param::get_missile_cost(best_step * param::dt);;

    printf("step_missile_hits: %d\n", best_step);
    printf("missile_cost: %lf\n", missile_cost);

    auto end_all = std::chrono::high_resolution_clock::now();

    write_output(argv[2], sqrt(min_dist), hit_time_step, gravity_device_id, missile_cost);
    // write_output(argv[2], min_dist, hit_time_step, 0, 0);

    std::chrono::duration<double> p1_time = end_all - start_all;
    std::cout << " Program Time: " << p1_time.count() << " s" << std::endl;

    // Free allocated memory
    free(h_posw);
    free(h_vtype);
}
