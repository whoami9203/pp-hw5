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

void run_step(int step, int n, std::vector<double>& qx, std::vector<double>& qy,
    std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy,
    std::vector<double>& vz, const std::vector<double>& m,
    const std::vector<std::string>& type) {
    // compute accelerations
    std::vector<double> ax(n), ay(n), az(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double mj = m[j];
            if (type[j] == "device") {
                mj = param::gravity_device_mass(mj, step * param::dt);
            }
            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            double dist3 =
                pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);
            ax[i] += param::G * mj * dx / dist3;
            ay[i] += param::G * mj * dy / dist3;
            az[i] += param::G * mj * dz / dist3;
        }
    }

    // update velocities
    for (int i = 0; i < n; i++) {
        vx[i] += ax[i] * param::dt;
        vy[i] += ay[i] * param::dt;
        vz[i] += az[i] * param::dt;
    }

    // update positions
    for (int i = 0; i < n; i++) {
        qx[i] += vx[i] * param::dt;
        qy[i] += vy[i] * param::dt;
        qz[i] += vz[i] * param::dt;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;

    auto distance = [&](int i, int j) -> double {
        double dx = qx[i] - qx[j];
        double dy = qy[i] - qy[j];
        double dz = qz[i] - qz[j];
        return sqrt(dx * dx + dy * dy + dz * dz);
    };

    // Problem 1
    auto start_p1 = std::chrono::high_resolution_clock::now();

    double min_dist = std::numeric_limits<double>::infinity();
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    for (int i = 0; i < n; i++) {
        if (type[i] == "device") {
            m[i] = 0;
        }
    }
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
        }
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    }

    // Problem 2
    auto start_p2 = std::chrono::high_resolution_clock::now();

    int hit_time_step = -2;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
        }
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
            hit_time_step = step;
            break;
        }
    }

    // Problem 3
    auto start_p3 = std::chrono::high_resolution_clock::now();

    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    int best_step = 400000;
    int gravity_device_id = -1;

    // Iterate through all gravity devices
    for (int device_id = 0; device_id < n; ++device_id) {
        if (type[device_id] != "device") continue;

        // Backup initial state
        std::vector<double> qx_copy = qx, qy_copy = qy, qz_copy = qz;
        std::vector<double> vx_copy = vx, vy_copy = vy, vz_copy = vz;
        std::vector<double> m_copy = m;

        // Simulate with the device removed
        m[device_id] = 0; // Destroy the device
        bool collision_avoided = true;
        int step_missile_hits = -1;
        int step = 1;

        // Simulate till the device destroyed
        for (; step <= param::n_steps; ++step) {
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);

            // Calculate the distance between the missile and the gravity device
            double dist_planet_device = distance(planet, device_id);
            double missile_distance = step * param::dt * param::missile_speed;
            if (missile_distance > dist_planet_device && step_missile_hits == -1) {
                step_missile_hits = step;
                break;
            }

            // Check if asteroid hits the planet
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
                collision_avoided = false;
                break;
            }
        }

        // collision happens before destroying the device
        if (!collision_avoided)
            continue;
        m[device_id] = 0;

        // Simulate with the device destroyed
        for (step++ ; step <= param::n_steps; ++step) {
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);

            // Check if asteroid hits the planet
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
                collision_avoided = false;
                break;
            }
        }

        // Restore the initial state
        qx = qx_copy;
        qy = qy_copy;
        qz = qz_copy;
        vx = vx_copy;
        vy = vy_copy;
        vz = vz_copy;
        m = m_copy;

        if (collision_avoided && step_missile_hits != -1 && step_missile_hits < best_step) {
            best_step = step_missile_hits;
            gravity_device_id = device_id;
        }
    }

    double missile_cost = (gravity_device_id == -1) ? 0 : param::get_missile_cost(best_step * param::dt);

    auto end_p3 = std::chrono::high_resolution_clock::now();

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);

    std::chrono::duration<double> p1_time = start_p2 - start_p1;
    std::chrono::duration<double> p2_time = start_p3 - start_p2;
    std::chrono::duration<double> p3_time = end_p3 - start_p3;
    std::cout << " Problem 1 Time: " << p1_time.count() << " s" << std::endl;
    std::cout << " Problem 2 Time: " << p2_time.count() << " s" << std::endl;
    std::cout << " Problem 3 Time: " << p3_time.count() << " s" << std::endl;
}
