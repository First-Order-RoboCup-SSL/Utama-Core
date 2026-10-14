#include <iostream>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "vssworld.h"
#include "sslworld.h"

namespace py = pybind11;
using vd = std::vector<double>;
using vvd = std::vector<std::vector<double>>;

struct VSS {
    VSS(int fieldType, int nRobotsBlue, int nRobotsYellow, int timeStep_ms,
        const vd &ballPos, const vvd &blueRobotsPos, const vvd &yellowRobotsPos) : m_fieldType(fieldType),
                                                                                   m_nRobotsBlue(nRobotsBlue),
                                                                                   m_nRobotsYellow(nRobotsYellow),
                                                                                   m_timeStep_ms(timeStep_ms) {
        m_world = new VSSWorld(m_fieldType, m_nRobotsBlue, m_nRobotsYellow, m_timeStep_ms / 1000.0,
                               ballPos, blueRobotsPos, yellowRobotsPos);
    }

    ~VSS() { delete m_world; }

    void step(vvd actions) const { m_world->step(std::move(actions)); }

    void step_fast(py::array_t<double, py::array::c_style | py::array::forcecast> actions_arr) const {
        auto buf = actions_arr.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("actions must be a 2D array");
        }
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        const double *ptr = static_cast<const double *>(buf.ptr);

        vvd actions(rows, vd(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                actions[i][j] = ptr[i * cols + j];
            }
        }
        py::gil_scoped_release release;
        m_world->step(std::move(actions));
    }

    vd getState() const { return m_world->getState(); }

    py::array_t<double> getStateArray() const {
        const vd &s = m_world->getState();
        py::array_t<double> res(s.size());
        auto buf = res.request();
        std::memcpy(buf.ptr, s.data(), s.size() * sizeof(double));
        return res;
    }

    void reset(const vd &ballPos, const vvd &blueRobotsPos, const vvd &yellowRobotsPos) {
        delete m_world;
        m_world = new VSSWorld(m_fieldType, m_nRobotsBlue, m_nRobotsYellow, m_timeStep_ms / 1000.0,
                               ballPos, blueRobotsPos, yellowRobotsPos);
    }

    std::unordered_map<std::string, double> getFieldParams() const { return m_world->getFieldParams(); }

    VSSWorld *m_world;
    int m_fieldType, m_nRobotsBlue, m_nRobotsYellow, m_timeStep_ms;
};

struct SSL {
    SSL(int fieldType, int nRobotsBlue, int nRobotsYellow, int timeStep_ms,
        const vd &ballPos, const vvd &blueRobotsPos, const vvd &yellowRobotsPos) : m_fieldType(fieldType),
                                                                                   m_nRobotsBlue(nRobotsBlue),
                                                                                   m_nRobotsYellow(nRobotsYellow),
                                                                                   m_timeStep_ms(timeStep_ms) {
        m_world = new SSLWorld(m_fieldType, m_nRobotsBlue, m_nRobotsYellow, m_timeStep_ms / 1000.0,
                               ballPos, blueRobotsPos, yellowRobotsPos);
    }

    ~SSL() { delete m_world; }

    void step(vvd actions) const { m_world->step(std::move(actions)); }

    void step_fast(py::array_t<double, py::array::c_style | py::array::forcecast> actions_arr) const {
        auto buf = actions_arr.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("actions must be a 2D array");
        }
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        const double *ptr = static_cast<const double *>(buf.ptr);

        vvd actions(rows, vd(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                actions[i][j] = ptr[i * cols + j];
            }
        }
        py::gil_scoped_release release;
        m_world->step(std::move(actions));
    }

    vd getState() const { return m_world->getState(); }

    py::array_t<double> getStateArray() const {
        const vd &s = m_world->getState();
        py::array_t<double> res(s.size());
        auto buf = res.request();
        std::memcpy(buf.ptr, s.data(), s.size() * sizeof(double));
        return res;
    }

    void reset(const vd &ballPos, const vvd &blueRobotsPos, const vvd &yellowRobotsPos) {
        delete m_world;
        m_world = new SSLWorld(m_fieldType, m_nRobotsBlue, m_nRobotsYellow, m_timeStep_ms / 1000.0,
                               ballPos, blueRobotsPos, yellowRobotsPos);
    }

    std::unordered_map<std::string, double> getFieldParams() const { return m_world->getFieldParams(); }

    SSLWorld *m_world;
    int m_fieldType, m_nRobotsBlue, m_nRobotsYellow, m_timeStep_ms;
};


PYBIND11_MODULE(_robosim, m) {
    py::class_<VSS>(m, "VSS")
            .def(py::init<int, int, int, int, vd, vvd, vvd>())
            .def("step", &VSS::step)
            .def("step_fast", &VSS::step_fast)
            .def("get_state", &VSS::getState)
            .def("get_state_array", &VSS::getStateArray)
            .def("reset", &VSS::reset)
            .def("get_field_params", &VSS::getFieldParams);

    py::class_<SSL>(m, "SSL")
            .def(py::init<int, int, int, int, vd, vvd, vvd>())
            .def("step", &SSL::step)
            .def("step_fast", &SSL::step_fast)
            .def("get_state", &SSL::getState)
            .def("get_state_array", &SSL::getStateArray)
            .def("reset", &SSL::reset)
            .def("get_field_params", &SSL::getFieldParams);
}


