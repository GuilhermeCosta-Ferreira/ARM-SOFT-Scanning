bl_info = {
    "name": "Intrinsic Camera Data",
    "author": "GuilhermeCF",
    "version": (1,0),
    "blender": (4,2,0),
    "location": "View3D > Sidebar > Stereo Camera",
    "description": "Tool to extract the intrinsic parameters in the Blender Camera for 3D Stereo Reconstruction",
    "warning": "",
    "wiki_url": "https://github.com/N-Pulse/ARM-SOFT-Scanning/tree/c6d7837bc42b13801c78d180df4a2a6d3ee14582",
    "category": "Add Mesh"
}

import sys, os
# Path that contains the package folder 'my_addon'
PKG_PARENT = "/Users/guilhermec.f/Documents/EPFL/N-Pulse/ARM-SOFT-Scanning/blender_simulator/add-ons/instrinsic_add_on/"
if PKG_PARENT not in sys.path:
    sys.path.append(PKG_PARENT)
import PanelManager


import importlib
import bpy

if "bpy" in locals():
    importlib.reload(PanelManager)

# Gather classes from submodules (keeps register() clean)
classes = (
    PanelManager.FocusPanel,
    PanelManager.StereoPanel
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
