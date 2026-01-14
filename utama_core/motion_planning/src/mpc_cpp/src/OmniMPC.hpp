#pragma once
#include <Eigen/Dense>
#include <vector>
#include <tuple>

// Empty because values are configured in mpc_cpp_controller.py
struct MPCConfig {
    int    T;
    double DT;
    double max_vel;
    double max_accel;
    double Q_pos;
    double Q_vel;
    double R_accel;
    double Q_slack;
    double robot_radius;
    double obstacle_buffer_ratio;
    double safety_vel_coeff;
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