#pragma once
#include <chrono>
#include <memory>
#include <array>

#include <eigen3/Eigen/Dense>

#include <type_traits>
template < typename C, C beginVal, C endVal>
class Iterator {
  typedef typename std::underlying_type<C>::type val_t;
  int val;
public:
  Iterator(const C & f) : val(static_cast<val_t>(f)) {}
  Iterator() : val(static_cast<val_t>(beginVal)) {}
  Iterator operator++() {
    ++val;
    return *this;
  }
  C operator*() { return static_cast<C>(val); }
  Iterator begin() { return *this; } //default ctor is good
  Iterator end() {
      static const Iterator endIter=++Iterator(endVal); // cache it
      return endIter;
  }
  bool operator!=(const Iterator& i) { return val != i.val; }
};


namespace IK
{

  
  /*LEG ENUMERATION:
  1 2
  3 4*/
enum Leg {FL,FR,BL,BR};


typedef Iterator<Leg, Leg::FL, Leg::BR> legIterator;

struct LegJointPositions
{
    LegJointPositions() : q1(0), q2(0), q3(0) {}
    double q1,q2,q3;
};

class InverseKinematics
{

public:
    void calcJointPositions(Leg leg, double x, double y, double z);

public:

    std::array<LegJointPositions, sizeof(Leg)> legs;
private:
    void basic_ik_calcs(Leg leg, double x, double y, double z);

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
}
