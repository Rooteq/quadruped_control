#include "gait_scheduler.hpp"

namespace quadro
{

void GaitScheduler::advance(double dt)
{
    phase_ = std::fmod(phase_ + dt / gait_.period, 1.0);
}

void GaitScheduler::setGait(const GaitDefinition& gait)
{
    gait_ = gait;
}

bool GaitScheduler::inStance(int leg_idx) const
{
    return legPhase(leg_idx) < gait_.duty_cycle;
}

double GaitScheduler::swingPhase(int leg_idx) const
{
    double lp = legPhase(leg_idx);
    if (lp < gait_.duty_cycle) return -1.0;
    return (lp - gait_.duty_cycle) / (1.0 - gait_.duty_cycle);
}

double GaitScheduler::stancePhase(int leg_idx) const
{
    double lp = legPhase(leg_idx);
    if (lp >= gait_.duty_cycle) return -1.0;
    return lp / gait_.duty_cycle;
}

std::array<bool, NUM_LEGS> GaitScheduler::contactMaskAt(double t) const
{
    std::array<bool, NUM_LEGS> mask{};
    const double phase_offset_t = t / gait_.period;
    for (int leg = 0; leg < static_cast<int>(NUM_LEGS); ++leg)
    {
        const double future_phase = std::fmod(
            phase_ + gait_.phase_offsets[leg] + phase_offset_t, 1.0);
        mask[leg] = (future_phase < gait_.duty_cycle);
    }
    return mask;
}

double GaitScheduler::legPhase(int leg_idx) const
{
    return std::fmod(phase_ + gait_.phase_offsets[leg_idx], 1.0);
}

} // namespace quadro
