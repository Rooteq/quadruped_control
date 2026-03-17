#include "../include/inverse_kinematics.hpp"

namespace IK
{
void InverseKinematics::calcJointPositions(Leg leg, double x, double y, double z)
{
    basic_ik_calcs(leg, x,y,z);

    if(leg == Leg::BL || leg == Leg::FL)
    {
        legs[leg].q1 += (joint_offset_1 - M_PI/2);
        legs[leg].q2 += (-joint_offset_2 + M_PI/2);
        legs[leg].q3 += (-joint_offset_3 - M_PI/2);
    }
    else
    {
        legs[leg].q1 += (-joint_offset_1 + M_PI/2);
        legs[leg].q2 += (-joint_offset_2 + M_PI/2);
        legs[leg].q3 += (-joint_offset_3 - M_PI/2);
    }

    // Apply motor reversals based on URDF axis directions
    // Reversed joints (z = -1): br_m1_s1, br_m2_s2, br_m3_s3, fr_m2_s2, fr_m3_s3
    // SIMULATION:
    if(leg == Leg::BR) {
        legs[leg].q1 = -legs[leg].q1;
        legs[leg].q2 = -legs[leg].q2;
        legs[leg].q3 = -legs[leg].q3;

        //VERY IMPORTANT - THE MOTOR IS OFFSET FOR SOME REASON
    }
    else if(leg == Leg::FR) {
        legs[leg].q2 = -legs[leg].q2;
        legs[leg].q3 = -legs[leg].q3;
    }
    else if(leg == Leg::BL) {
        legs[leg].q1 = -legs[leg].q1;
    }

    // if(leg == Leg::FR)
    // {
    //     legs[leg].q1 = -legs[leg].q1;
    // }
    // else if(leg == Leg::BL) {
    //     legs[leg].q2 = -legs[leg].q2;
    //     legs[leg].q3 = -legs[leg].q3;
    // }
    // else if(leg == Leg::BR) {
    //     legs[leg].q1 = -legs[leg].q1;
    //     legs[leg].q2 = -legs[leg].q2;
    //     legs[leg].q3 = -legs[leg].q3;
    // }
}

void InverseKinematics::basic_ik_calcs(Leg leg, double x, double y, double z)
{
    if(leg == Leg::BL || leg == Leg::FL)
        y = y+l1;
    else
        y = -y + l1;
    
    double alfa;

    double AG = std::sqrt(y*y + z*z - l1*l1);
    double OA = l1;
    double GC = x;
    // double GC = x;
    double AC = std::sqrt(AG*AG + GC*GC);
    
    double cos_alpha = -((AC*AC - l2*l2 - l3*l3)/(2*l2*l3));
    alfa = std::acos(cos_alpha);
    
    double angle1 = std::atan2(AG,OA);
    double angle2 = std::atan2(y,z);

    if(leg == Leg::BL || leg == Leg::FL)
        legs[leg].q1 = normalize_angle(-(angle1 - angle2));
    else
        legs[leg].q1 = normalize_angle(angle1 - angle2);

    legs[leg].q3 = M_PI - alfa;
    double q3 = legs[leg].q3;

    legs[leg].q2 = std::atan2(GC,AG) - std::atan2(l3*std::sin(q3), l2+l3*std::cos(q3));
}

}