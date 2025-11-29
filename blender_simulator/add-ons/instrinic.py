import bpy
import json
import os

import numpy as np



scene = bpy.context.scene
cam = scene.camera.data

# Get render resolution and scale
res_x = scene.render.resolution_x
res_y = scene.render.resolution_y
scale = scene.render.resolution_percentage / 100

# Effective resolution in pixels
res_x_px = res_x * scale
res_y_px = res_y * scale

# Camera parameters
f_in_mm = cam.lens
f = f_in_mm * 0.001
sensor_width_in_mm = cam.sensor_width
sensor_height_in_mm = cam.sensor_height

# Pixel aspect ratio (can be ≠ 1 for anamorphic)
pixel_aspect = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

# Adjust sensor size for sensor fit
if cam.sensor_fit == 'VERTICAL':
    s_u = res_x_px / (sensor_width_in_mm * pixel_aspect)
    s_v = res_y_px / sensor_height_in_mm
else:  # 'HORIZONTAL' or 'AUTO'
    s_u = res_x_px / sensor_width_in_mm
    s_v = res_y_px * pixel_aspect / sensor_height_in_mm

# Focal length in pixels
f_x = f_in_mm * s_u
f_y = f_in_mm * s_v

# Principal point (usually center)
c_x = res_x_px / 2.0
c_y = res_y_px / 2.0

# Intrinsic matrix
K = np.array([[f_x, 0, c_x],
              [0, f_y, c_y],
              [0,   0,   1]])

print("Intrinsic matrix K:\n", K)

output_path = os.path.join(bpy.path.abspath("//"), "camera_intrinsics.json")

# Convert numpy array to plain Python list
def listify_matrix(matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list
    
data = {
    "K": K.tolist(),
    "resolution": [res_x_px, res_y_px],
    "focal_mm": f_in_mm,
    "scale": scale,
    "cam.sensor_fit": cam.sensor_fit,
    "sensro width": sensor_width_in_mm,
    "sensor height": sensor_height_in_mm,
    "pixel_aspect": pixel_aspect,
    "camera_tr": listify_matrix(scene.camera.matrix_world)
}

with open(output_path, "w") as f:
    json.dump(data, f, indent=4)

print(f"Saved intrinsics to {output_path}")