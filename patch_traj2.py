import sys

fname = 'src/controller/inc/trajectory_generator.hpp'
with open(fname, 'r') as f:
    content = f.read()

content = content.replace(
    "    Eigen::Vector3d nominalFootPosition(int leg) const\n    {\n        return {hipPos[leg].x(), hipPos[leg].y(), NOMINAL_HEIGHT};\n    }",
    "    Eigen::Vector3d nominalFootPosition(int leg) const\n    {\n        return {hipPos[leg].x(), hipPos[leg].y(), NOMINAL_HEIGHT};\n    }\n\n    Eigen::Vector3d getLandingPos(int leg) const\n    {\n        return swing_states_[leg].landing_pos;\n    }"
)

with open(fname, 'w') as f:
    f.write(content)
