# ♻️ Simulating Platform

## 1. Contents

In this platform you will find:

- **add-ons/**: In this folder there is the list of needed blender add-ons to run the simuaton file. Inside you will find the add-on for **BlenderNeRF**. To understand how to use this pipeline resort to:
- **simulated_frames/**: In this folder tou will find various datasets of stump pictures that were generated with the blender simulating platform with varying number of degrees (cameras)
  - _dataset_d90.zip_ - Dataset of stump images taken with 90º degrees step (4 images)
  - _dataset_d60.zip_ - Dataset of stump images taken with 60º degrees step (6 images)
  - _dataset_d30.zip_ - Dataset of stump images taken with 30º degrees step (12 images)
- **3d_scanning_sim.blend**: In this file you will find the full enviroment used to generate the previous datasets

## 2. How Use?

To generate a new image first you will need to:

1. Load the BlenderNeRF add-on onto your blender enviroment: Edit > Preferences > Add-ons > Install from Disk > `BlenderNeRF.zip`.
2. Configure your camera radius: CameraPath (object) > Item (Panel) > Dimensions and change both x and y dimensions to best fit your needs.
3. Configure your AABB to fit everything (camera and stump): AABB_Wire (Object) > Item (Panel) > Dimensions (x,y and z)
4. Configure your dataset general settings: BlenderNeRF (Panel) > BlenderNeRF shared UI <br>
   4.1. There you adjust the AABB size <br>
   4.2. Also check Render Frames and Save Logs as you will need both for the 3D Reconstruction <br>
   4.3. Select NeRF file format <br>
   4.4.efine the Save path
5. Configure your dataset specific settings: BlenderNeRF (Panel) > Subset of Frames SOF <br>
   5.1. Select the step value (the inbetween angle of every frame) <br>
   5.2. Update the dataset name
6. Run the Simualtion: BlenderNeRF (Panel) > Subset of Frames SOF >PLAY SOF
7. Run the script: `intrinsic.py` to extract the matrix K
