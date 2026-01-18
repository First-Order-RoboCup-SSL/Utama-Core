#include "OmniMPC.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

OmniMPC::OmniMPC(MPCConfig config) : config(config) {
    // No OSQP setup needed for Heuristic Solver
}

OmniMPC::~OmniMPC() {
    // No OSQP cleanup needed
}

std::tuple<double, double, bool> OmniMPC::get_control_velocities(
    Eigen::Vector4d current_state,
    Eigen::Vector2d goal_pos,
    std::vector<std::vector<double>> obstacles
) {
    // --- FAST HEURISTIC SOLVER ---
    // (Same math as before, just no OSQP dependencies)

    double dt = config.DT;
    
    // 1. Distance Check
    double dist_to_goal = (goal_pos - current_state.head<2>()).norm();
    bool is_arriving = dist_to_goal < 0.40;

    // 2. Calculate Target Velocity
    Eigen::Vector2d ref_vel = Eigen::Vector2d::Zero();
    
    if (is_arriving || dist_to_goal < 0.15) {
        ref_vel.setZero();
    } else {
        ref_vel = (goal_pos - current_state.head<2>()).normalized() * config.max_vel;
    }

    // 3. Error and Gain
    Eigen::Vector2d velocity_error = ref_vel - current_state.tail<2>();
    
    double gain = 4.0; 
    if (is_arriving) gain = 2.0; 

    Eigen::Vector2d acc = velocity_error * gain;

    // 4. Dynamic Obstacle Avoidance
    Eigen::Vector2d repulsion = Eigen::Vector2d::Zero();
    double current_speed = current_state.tail<2>().norm();
    
    for (const auto& obs : obstacles) {
        Eigen::Vector2d obs_pos(obs[0], obs[1]);
        Eigen::Vector2d obs_vel(obs[2], obs[3]);
        double radius = obs[4];

        // Predict obstacle future pos (0.1s ahead)
        Eigen::Vector2d obs_future = obs_pos + obs_vel * 0.1;
        Eigen::Vector2d diff = current_state.head<2>() - obs_future;
        double dist = diff.norm();
        
        // --- CRASH FIX 1: Prevent Division by Zero ---
        // If robots overlap perfectly, 'dist' is 0. Normalizing (diff/dist) creates NaN.
        // We force a tiny distance to keep the math valid.
        if (dist < 1e-5) {
            dist = 1e-5;
            diff << 1.0, 0.0; // Arbitrary push direction
        }
        // ---------------------------------------------
        
        double safety = config.robot_radius * config.obstacle_buffer_ratio + radius;
        safety += current_speed * config.safety_vel_coeff;
        
        if (dist < safety * 1.2) {
            double violation = std::max(0.0, safety * 1.2 - dist);
            
            // Exponential force is good, but dangerous if violation is large
            double force_mag = 50.0 * std::exp(violation * 10.0); 
            
            // --- CRASH FIX 2: Clamp the Force ---
            // Never allow the force to exceed 2x the maximum acceleration.
            // This prevents the "Explosion" that kills your simulator.
            double max_allowed_force = config.max_accel * 2.0;
            if (std::isinf(force_mag) || std::isnan(force_mag) || force_mag > max_allowed_force) {
                force_mag = max_allowed_force;
            }
            // ------------------------------------
            
            repulsion += diff.normalized() * force_mag;
        }
    }
    
    acc += repulsion;

    // 5. Clamp Acceleration
    if (acc.norm() > config.max_accel) {
        acc = acc.normalized() * config.max_accel;
    }

    // 6. Integrate
    Eigen::Vector2d next_vel = current_state.tail<2>() + acc * dt;

    // 7. Clamp Velocity
    if (next_vel.norm() > config.max_vel) {
        next_vel = next_vel.normalized() * config.max_vel;
    }

    return {next_vel.x(), next_vel.y(), true};
}