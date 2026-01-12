#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/eigen/dense.h>
#include "OmniMPC.hpp"

namespace nb = nanobind;

NB_MODULE(mpc_cpp_extension, m) {
    nb::class_<MPCConfig>(m, "MPCConfig")
        .def(nb::init<>())
        .def_rw("T", &MPCConfig::T)
        .def_rw("DT", &MPCConfig::DT)
        .def_rw("max_vel", &MPCConfig::max_vel)
        .def_rw("max_accel", &MPCConfig::max_accel)
        .def_rw("Q_pos", &MPCConfig::Q_pos)
        .def_rw("Q_vel", &MPCConfig::Q_vel)
        .def_rw("R_accel", &MPCConfig::R_accel)
        .def_rw("obstacle_buffer_ratio", &MPCConfig::obstacle_buffer_ratio)
        // --- ADDED MISSING BINDINGS ---
        .def_rw("safety_vel_coeff", &MPCConfig::safety_vel_coeff)
        .def_rw("robot_radius", &MPCConfig::robot_radius); 

    nb::class_<OmniMPC>(m, "OmniMPC")
        .def(nb::init<MPCConfig>())
        .def("get_control_velocities", &OmniMPC::get_control_velocities);
}