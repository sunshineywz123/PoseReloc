import blenderproc as bproc
import numpy as np

bproc.init()

objs = bproc.loader.load_obj("/data/ycb/models/002_master_chef_can/textured.obj")

# Find point of interest, all cam poses should look towards it
poi = bproc.object.compute_poi(objs)
# Sample five camera poses
for i in range(5):
    # Sample random camera location above objects
    location = np.random.uniform([-10, -10, 8], [10, 10, 12])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

# Add point light:
light = bproc.types.Light()
light.set_location([2,-2, 0])
light.set_energy(300)

render_results = bproc.renderer.render()

bproc.writer.write_hdf5("output/0.hdf5",render_results)