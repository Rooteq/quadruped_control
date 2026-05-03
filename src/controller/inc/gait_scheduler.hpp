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
    // Phase offsets [FL, FR, BL, BR] match Python PHASE_OFFSET = [0.5, 0.0, 0.0, 0.5]:
    // FR+BL (offset=0.0) are one diagonal pair; FL+BR (offset=0.5) are the other.
    // Period=1/3 s (3 Hz), duty=0.6 match the Python reference — the 10-step MPC
    // horizon at 33 ms/step covers one gait cycle, and 60% stance is more stable.
    inline const GaitDefinition TROT  = {"trot",  1.0/3.0, 0.6, {0.5, 0.0, 0.0, 0.5}};
    inline const GaitDefinition WALK  = {"walk",  0.8,  0.75, {0.0, 0.5, 0.75, 0.25}};
    inline const GaitDefinition PACE  = {"pace",  0.5,  0.5,  {0.0, 0.5, 0.0, 0.5}};
    inline const GaitDefinition BOUND = {"bound", 0.5,  0.4,  {0.0, 0.0, 0.5, 0.5}};
    inline const GaitDefinition PRONK = {"pronk", 0.5,  0.3,  {0.0, 0.0, 0.0, 0.0}};
    inline const GaitDefinition STAND = {"stand", 1.0,  1.0,  {0.0, 0.0, 0.0, 0.0}};
}

class GaitScheduler
{
public:
    GaitScheduler() : gait_(gaits::TROT) {}
    explicit GaitScheduler(const GaitDefinition& gait) : gait_(gait) {}

    void advance(double dt)
    {
        phase_ = std::fmod(phase_ + dt / gait_.period, 1.0);
    }

    void setGait(const GaitDefinition& gait)
    {
        gait_ = gait;
        // phase keeps running — new offsets/duty take effect immediately
    }

    bool inStance(int leg_idx) const
    {
        return legPhase(leg_idx) < gait_.duty_cycle;
    }

    /// Normalized progress through swing [0, 1]. Returns -1 if leg is in stance.
    double swingPhase(int leg_idx) const
    {
        double lp = legPhase(leg_idx);
        if (lp < gait_.duty_cycle) return -1.0;
        return (lp - gait_.duty_cycle) / (1.0 - gait_.duty_cycle);
    }

    /// Normalized progress through stance [0, 1]. Returns -1 if leg is in swing.
    double stancePhase(int leg_idx) const
    {
        double lp = legPhase(leg_idx);
        if (lp >= gait_.duty_cycle) return -1.0;
        return lp / gait_.duty_cycle;
    }

    /// Contact schedule over a prediction horizon.
    /// contact_table[k][leg] = true if leg is in stance at horizon step k.
    /// Evaluates at the midpoint of each MPC interval (k + 0.5)*dt, matching the
    /// Python reference: t = t0 + arange(N)*dt + dt/2. This means contactTable()[0]
    /// is evaluated half a step ahead of inStance(), which is intentional and mirrors
    /// the Python behavior (compute_current_mask uses t0, contact_table uses t0+dt/2).
    template<int N>
    std::array<std::array<bool, NUM_LEGS>, N> contactTable(double mpc_dt) const
    {
        std::array<std::array<bool, NUM_LEGS>, N> table{};
        for (int k = 0; k < N; ++k)
        {
            // Midpoint sampling: (k + 0.5) * dt / period — matches Python's dt/2 centering.
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

    double phase() const { return phase_; }
    const GaitDefinition& gait() const { return gait_; }

private:
    double legPhase(int leg_idx) const
    {
        return std::fmod(phase_ + gait_.phase_offsets[leg_idx], 1.0);
    }

    GaitDefinition gait_;
    double phase_ = 0.0;
};

} // namespace quadro
