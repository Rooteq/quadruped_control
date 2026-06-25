#pragma once

#include <array>
#include <cmath>
#include <string>

#include "model.hpp"

namespace quadro
{

struct GaitDefinition
{
    std::string name;
    double period;                          // seconds, one full gait cycle
    double duty_cycle;                      // fraction of period spent in stance [0,1]
    std::array<double, NUM_LEGS> phase_offsets;  // per leg [FL, FR, BL, BR]
};

namespace gaits
{
    inline const GaitDefinition TROT  = {"trot",  1.0/3.0, 0.6, {0.5, 0.0, 0.0, 0.5}};
    inline const GaitDefinition WALK  = {"walk",  0.3,  0.6, {0.0, 0.5, 0.75, 0.25}};
    inline const GaitDefinition PACE  = {"pace",  0.33,  0.5,  {0.0, 0.5, 0.0, 0.5}};
    inline const GaitDefinition BOUND = {"bound", 0.33,  0.4,  {0.0, 0.0, 0.5, 0.5}};
    inline const GaitDefinition PRONK = {"pronk", 0.33,  0.3,  {0.0, 0.0, 0.0, 0.0}};
    inline const GaitDefinition STAND = {"stand", 1.0,  1.0,  {0.0, 0.0, 0.0, 0.0}};
}

class GaitScheduler
{
public:
    GaitScheduler() : gait_(gaits::WALK) {}
    explicit GaitScheduler(const GaitDefinition& gait) : gait_(gait) {}

    void advance(double dt);
    void setGait(const GaitDefinition& gait);

    bool inStance(int leg_idx) const;

    /// Normalized progress through swing [0, 1]. Returns -1 if leg is in stance.
    double swingPhase(int leg_idx) const;

    /// Normalized progress through stance [0, 1]. Returns -1 if leg is in swing.
    double stancePhase(int leg_idx) const;

    /// Contact schedule over a prediction horizon.
    /// contact_table[k][leg] = true if leg is in stance at horizon step k.
    /// Evaluated at the midpoint of each MPC interval (k + 0.5)*dt.
    template<int N>
    std::array<std::array<bool, NUM_LEGS>, N> contactTable(double mpc_dt) const
    {
        std::array<std::array<bool, NUM_LEGS>, N> table{};
        for (int k = 0; k < N; ++k)
        {
            const double phase_offset_k = (k + 0.5) * mpc_dt / gait_.period;

            for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
            {
                double future_phase = std::fmod(
                    phase_ + gait_.phase_offsets[leg] + phase_offset_k, 1.0);
                table[k][leg] = (future_phase < gait_.duty_cycle);
            }
        }
        return table;
    }

    /// Per-leg stance/swing mask at the future-time offset `t` (seconds) from
    /// the current phase_. Uncentered (no dt/2 shift) — used for transition
    /// detection in the per-horizon-step lever planner.
    std::array<bool, NUM_LEGS> contactMaskAt(double t) const;

    double stanceTime() const { return gait_.duty_cycle * gait_.period; }
    double swingTime()  const { return (1.0 - gait_.duty_cycle) * gait_.period; }

    double phase() const { return phase_; }
    const GaitDefinition& gait() const { return gait_; }

private:
    double legPhase(int leg_idx) const;

    GaitDefinition gait_;
    double phase_ = 0.0;
};

} // namespace quadro
