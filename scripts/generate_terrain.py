import random
import re

random.seed(42)

num_boxes = 50
boxes_xml = []

for i in range(num_boxes):
    x = random.uniform(-1.5, 1.5)
    y = random.uniform(-1.5, 1.5)
    # Exclude the center where the robot spawns (e.g. radius 0.4 around origin)
    if -0.4 < x < 0.4 and -0.4 < y < 0.4:
        continue
        
    dx = random.uniform(0.05, 0.15)
    dy = random.uniform(0.05, 0.15)
    
    # Height up to 7cm, so full height up to 0.07, half height up to 0.035
    full_height = random.uniform(0.01, 0.07)
    dz = full_height / 2.0
    z = dz
    
    boxes_xml.append(f'        <geom type="box" size="{dx:.3f} {dy:.3f} {dz:.3f}" pos="{x:.3f} {y:.3f} {z:.3f}" rgba="0.6 0.5 0.4 1" contype="1" conaffinity="1"/>')

boxes_xml_str = "\n".join(boxes_xml)

with open('description/scene.xml', 'r') as f:
    content = f.read()

replacement = f'<geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" contype="2" conaffinity="1"/>\n        <!-- Uneven Terrain -->\n{boxes_xml_str}'

content = re.sub(r'<geom name="floor"[^>]+/>', replacement, content)

with open('description/scene.xml', 'w') as f:
    f.write(content)

print("Added terrain to scene.xml")
