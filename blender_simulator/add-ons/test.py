import bpy, json, os
import numpy as np
from mathutils import Matrix

def listify_matrix(matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list

def get_camera_extrinsics(scene, camera):    
    frame_data = {
        'transform_matrix': listify_matrix(camera.matrix_world)
    }

    return frame_data

# ========== (2) SETTINGS ==========
scene = bpy.context.scene
frame_start = scene.frame_start
frame_end   = scene.frame_end
frame_step  = max(1, scene.frame_step)  # or set a custom step

# Output path (defaults to the .blend directory)
default_path = bpy.path.abspath("//camera_extrinsics.json")
OUTPUT_JSON = default_path  # set to e.g. "/tmp/camera_extrinsics.json"

# ========== (3) COLLECT EXTRINSICS ==========
init_frame = scene.frame_current
data = {"cameras": []}

# Get all camera OBJECTS in the file (not just active camera)
cam_objs = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']

for cam in cam_objs:
    cam_block = {"name": cam.name, "frames": []}
    for f in range(frame_start, frame_end + 1, frame_step):
        scene.frame_set(f)
        bpy.context.evaluated_depsgraph_get().update()

        cam_json = get_camera_extrinsics(scene, cam)
        cam_block["frames"] = cam_json

    data["cameras"].append(cam_block)

# Restore original frame
scene.frame_set(init_frame)

# ========== (4) WRITE JSON ==========
# Ensure output dir exists
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"[OK] Wrote extrinsics for {len(cam_objs)} camera(s) to: {OUTPUT_JSON}")