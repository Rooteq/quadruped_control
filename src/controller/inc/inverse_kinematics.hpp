#pragma once
#include <chrono>
#include <memory>
#include <array>

#include <eigen3/Eigen/Dense>

#include <type_traits>

#include "model.hpp"

namespace quadro
{

struct LegJointPositions
{
    LegJointPositions() : q1(0), q2(0), q3(0) {}
    double q1,q2,q3;
};

class InverseKinematics
{

public:
    InverseKinematics() = default;

    void calcJointPositions(LegIdx leg, double x, double y, double z);

public:

    std::array<LegJointPositions, sizeof(LegIdx)> legs;
private:
    void basic_ik_calcs(LegIdx leg, double x, double y, double z);

    double normalize_angle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }

  double l1 = 0.065;
  double l2 = 0.16;
  double l3 = 0.16;

  double joint_offset_1 = 0.7854;
  double joint_offset_2 = 0.3491;
  double joint_offset_3 = 0.3491;
};
} // namespace quadro
