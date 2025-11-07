import re


def parse_xml(xml_string):
    # pattern = re.compile(
    #     r'<geom class="(.*?)" pos="(.*?)" size="(.*?)" type=".*?"\/>')
    # matches = re.findall(pattern, xml_string)
    # return matches

    pattern = re.compile(r'<geom pos="(.*?)" size="(.*?)" type=".*?"\/>')
    matches = re.findall(pattern, xml_string)
    return matches


def write_obj(geometries, output_file):
    with open(output_file, 'w') as f:
        vertices_offset = 0
        for geometry in geometries:
            vertices = [float(coord) for coord in geometry[1].split()]
            size = [float(val)*2 for val in geometry[2].split()]
            x, y, z = vertices
            dx, dy, dz = size
            vertices_list = [
                (x - dx/2, y + dy/2, z + dz/2),
                (x + dx/2, y + dy/2, z + dz/2),
                (x + dx/2, y - dy/2, z + dz/2),
                (x - dx/2, y - dy/2, z + dz/2),
                (x - dx/2, y + dy/2, z - dz/2),
                (x + dx/2, y + dy/2, z - dz/2),
                (x + dx/2, y - dy/2, z - dz/2),
                (x - dx/2, y - dy/2, z - dz/2)
            ]
            for vert in vertices_list:
                f.write(f'v {vert[0]} {vert[1]} {vert[2]}\n')
            vertices_offset += 8

        for i in range(0, vertices_offset, 8):
            f.write(f'f {i+1} {i+2} {i+3} {i+4}\n')
            f.write(f'f {i+5} {i+6} {i+7} {i+8}\n')
            f.write(f'f {i+1} {i+2} {i+6} {i+5}\n')
            f.write(f'f {i+2} {i+3} {i+7} {i+6}\n')
            f.write(f'f {i+3} {i+4} {i+8} {i+7}\n')
            f.write(f'f {i+4} {i+1} {i+5} {i+8}\n')


def write_box_obj(geometries, output_file):
    with open(output_file, 'w') as f:
        vertices_offset = 0
        for geometry in geometries:
            vertices = [float(coord) for coord in geometry[0].split()]
            size = [float(val) for val in geometry[1].split()]
            x, y, z = vertices
            dx, dy, dz = size
            vertices_list = [
                (x - dx/2, y + dy/2, z + dz/2),
                (x + dx/2, y + dy/2, z + dz/2),
                (x + dx/2, y - dy/2, z + dz/2),
                (x - dx/2, y - dy/2, z + dz/2),
                (x - dx/2, y + dy/2, z - dz/2),
                (x + dx/2, y + dy/2, z - dz/2),
                (x + dx/2, y - dy/2, z - dz/2),
                (x - dx/2, y - dy/2, z - dz/2)
            ]
            for vert in vertices_list:
                f.write(f'v {vert[0]} {vert[1]} {vert[2]}\n')
            vertices_offset += 8

        for i in range(0, vertices_offset, 8):
            f.write(f'f {i+1} {i+2} {i+3} {i+4}\n')
            f.write(f'f {i+5} {i+6} {i+7} {i+8}\n')
            f.write(f'f {i+1} {i+2} {i+6} {i+5}\n')
            f.write(f'f {i+2} {i+3} {i+7} {i+6}\n')
            f.write(f'f {i+3} {i+4} {i+8} {i+7}\n')
            f.write(f'f {i+4} {i+1} {i+5} {i+8}\n')


xml_string = \
    '''
<mujoco model="bluebox">
  <asset>
    <texture file="textures/boxes/blue.png" name="tex-bluebox" />
    <material name="bluebox" reflectance="0.5" texrepeat="5 5" texture="tex-bluebox" texuniform="true"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom pos="0 0 0" size="0.021 0.021 0.027" type="box" solimp="0.998 0.998 0.001" solref="0.001 1" density="1000" friction="0.95 0.3 0.1"  material="bluebox" group="1" condim="4"/>
      </body>
      
      <site rgba="1 0 0 0" size="0.005" pos="0 0 -0.08" name="bottom_site"/>
      <site rgba="0 1 0 0" size="0.005" pos="0 0 0.0" name="top_site"/>
      <site rgba="0 0 1 0" size="0.005" pos="0.025 0.025 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
'''

geometries = parse_xml(xml_string)
# write_obj(geometries, 'bluebox.obj')
write_box_obj(geometries, 'bluebox.obj')
