import sys

fname = 'src/controller/inc/quadro_controller.hpp'
with open(fname, 'r') as f:
    content = f.read()

content = content.replace(
    "Eigen::Vector3d base_pos(x0[3], x0[4], 0.0);",
    "Eigen::Vector3d base_pos(x0[3], x0[4], x0[5]);"
)

with open(fname, 'w') as f:
    f.write(content)
