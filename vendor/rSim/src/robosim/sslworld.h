/*
grSim - RoboCup Small Size Soccer Robots Simulator
Copyright (C) 2011, Parsian Robotic Center (eew.aut.ac.ir/~parsian/grsim)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef SSLWORLD_H
#define SSLWORLD_H


#include "physics/pworld.h"
#include "physics/pball.h"
#include "physics/pground.h"
#include "physics/pfixedbox.h"

#include "sslrobot.h"
#include "utils.h"
#include <unordered_map>

#define SSL_WALL_COUNT 10
#define SSL_MAX_ROBOT_COUNT 22 //don't change

class SSLWorld
{
private:
    double timeStep;
    int stateSize;
    std::vector<double> state;

public:
    bool fullSpeed = false;
    PWorld *physics;
    PBall *ball;
    PGround *ground;
    PFixedBox *walls[SSL_WALL_COUNT]{};
    SSLRobot *robots[SSL_MAX_ROBOT_COUNT * 2]{};
    SSLConfig::Field field = SSLConfig::Field();

    SSLWorld(int fieldType, int nRobotsBlue, int nRobotsYellow, double timeStep,
             std::vector<double> ballPos, std::vector<std::vector<double>> blueRobotsPos, std::vector<std::vector<double>> yellowRobotsPos);
    ~SSLWorld();
    void step(std::vector<std::vector<double>> actions);
    void replace(std::vector<double> ballPos, std::vector<std::vector<double>> blueRobotsPos, std::vector<std::vector<double>> yellowRobotsPos);
    void initWalls();
    int getNumRobotsBlue() { return this->field.getRobotsBlueCount(); }
    int getNumRobotsYellow() { return this->field.getRobotsYellowCount(); }
    const std::unordered_map<std::string, double> getFieldParams();
    const std::vector<double> &getState();
    void setActions(std::vector<std::vector<double>> actions);

    // Tunable physics (Utama fork, see FORK_NOTES.md). The defaults reproduce upstream rSim:
    // 5 substeps per step, and the ball slowed by a constant friction force
    // (ballFriction * g, about 0.65 m/s^2 in practice).
    // With ballRollDecel >= 0 the ball instead loses exactly ballSlideDecel m/s^2 after a kick
    // until its speed falls to ballSlideFraction of the kick speed, then ballRollDecel m/s^2.
    void setPhysicsParams(int substeps, double ballSlideDecel, double ballRollDecel, double ballSlideFraction);

private:
    int substeps = 5;
    double ballSlideDecel = -1;
    double ballRollDecel = -1;
    double ballSlideFraction = 5.0 / 7.0;
    bool ballSliding = false;
    double ballSlideEndSpeed = 0;
    void applyBallFriction(dReal dt);
};

#endif // SSLWorld_H
