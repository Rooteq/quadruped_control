#include <string>
#include <array>

namespace quadro
{
    
static constexpr size_t NUM_JOINTS = 12;
static constexpr size_t NUM_LEGS = 4;
static constexpr size_t JOINTS_PER_LEG = 3;

// ── Expected joint order ─────────────────────────────────────────
// Change these names to match your URDF.
// Internal layout: FL(hip,thigh,calf), FR(...), RL(...), RR(...)

enum LegIdx  { FL = 0, FR = 1, RL = 2, RR = 3 };
enum JointIdx {
    FL_HIP = 0, FL_KNEE, FL_ANKLE,
    FR_HIP,     FR_KNEE, FR_ANKLE,
    BL_HIP,     BL_KNEE, BL_ANKLE,
    BR_HIP,     BR_KNEE, BR_ANKLE
};

// Map our internal index → expected URDF joint name
static const std::array<std::string, NUM_JOINTS> EXPECTED_JOINT_NAMES = {
    "bl_m1_s1",   "br_m1_s1",   "fl_m1_s1",
    "fr_m1_s1",   "bl_m2_s2",   "br_m2_s2",
    "fl_m2_s2",   "fr_m2_s2",   "bl_l4_l3",
    "br_l4_l3",   "fl_l4_l3",   "fr_l4_l3",
};

class Controller
{

};

} // namespace quadro

