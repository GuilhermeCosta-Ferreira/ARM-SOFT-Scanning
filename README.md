# 🔎 Scanning

## 📃 Description

A 3D Scanning Pipeline for the Amputee's Stump based on camera data to improve Robotic Arm socket confort.

## 🖇️ Requirements

- If exterior software is used, is must be open source — or at least easily accessible without constraints
- Simple scanning procedure, that doesn’t require very specific material
- The scan should be doable by a pair of two people (amputee and helper)
- End goal: Run the scanning algorithm on the mobile application

## ⏱️ Timeline

![alt text](figures/project_timeline_v1.png)

### <span style="color: darkred;"> ⚠️ Week 5 (Current Week):</span>

- [ ] To have a simulation platform that can generate stereo pictures (by defining a number of available angles)
  - [ ] Learn to script in Blender
  - [ ] Add the scene to Blender (with 3D stump)
  - [ ] Add the angle configuration to script
  - [ ] Generate a 4 angle run
  - [ ] Generate a 8 angle run
  - [ ] Generate a 16 angle run
  - [ ] Generate a 360 angle run

### <span style="color: gray;"> Week 6:</span>

- [ ] Build the first algorithm
- [ ] Have the break to support with this, maybe the learning curve could limit this

### <span style="color: gray;"> Week 7:</span>

- [ ] Have a second algorithm with a comparison to pick the best
  - [ ] Maybe the first algorithm works more as a proof of concept
- [ ] Or start on the first physical prototype
  - [ ] Need to define multiple cameras or a rotating system

### <span style="color: gray;"> Week 8:</span>

- [ ] If went with 2 model -> start to build prototype
  - [ ] Most likely
  - [ ] Have the 3d model of the cameras mount
  - [ ] Have the cameras requested by now (and decided on the schema)
- [ ] If not -> Finish and refine prototype for PDR
  - [ ] May need 3d printing training, may be to generous of a deadline
