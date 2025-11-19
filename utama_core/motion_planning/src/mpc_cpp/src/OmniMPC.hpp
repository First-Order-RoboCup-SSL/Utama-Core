#pragma once
#include <Eigen/Dense>
#include <vector>
#include <tuple>

struct MPCConfig {
    int T = 5;
    double DT = 0.05;
    double max_vel = 2.0;
    double max_accel = 3.0;
    double Q_pos = 200.0;
    double Q_vel = 20.0;
    double R_accel = 0.5;
    double Q_slack = 5000000.0;
    double robot_radius = 0.09;
    double obstacle_buffer_ratio = 1.25;
    double safety_vel_coeff = 0.15;
};

class OmniMPC {
public:
    OmniMPC(MPCConfig config);
    ~OmniMPC(); // Keep destructor

    std::tuple<double, double, bool> get_control_velocities(
        Eigen::Vector4d current_state,
        Eigen::Vector2d goal_pos,
        std::vector<std::vector<double>> obstacles
    );

private:
    MPCConfig config;
};