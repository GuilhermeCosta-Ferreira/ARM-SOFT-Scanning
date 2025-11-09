# ================================================================
# 0. Section: Imports
# ================================================================
import cv2
import numpy as np

from matplotlib.colors import LinearSegmentedColormap

from .graphic_classes import AlphaColor
from .color_converter import hex2rgb



# ================================================================
# 1. Section: Background Removal
# ================================================================
def remove_color_for_background(image_path: str, output_path: str, color_threshold: int = 240):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # If the image doesn't have an alpha channel, add one
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Make white pixels transparent
    # Define white color threshold (accounting for near-white pixels)
    mask = np.all(img[:, :, :3] >= color_threshold, axis=2)
    img[mask, 3] = 0  # Set alpha channel to 0 (transparent) for white pixels

    # Save the image with transparency
    cv2.imwrite(output_path, img)



# ================================================================
# 2. Section: Color Manager
# ================================================================
def pick_colors(color_map: LinearSegmentedColormap, N: int) -> np.ndarray:
    return color_map(np.linspace(0, 1, N))

def build_colormap_transparent2color(ending_color: str, starting_color: str = '#FFFFFF', n_bins: int = 2) -> LinearSegmentedColormap:
    hex_ending_color = ending_color
    hex_starting_color = starting_color

    rgb_ending_color = hex2rgb(hex_ending_color)
    rgb_starting_color = hex2rgb(hex_starting_color)
    rgb_starting_color = (rgb_starting_color[0], rgb_starting_color[1], rgb_starting_color[2], 0)

    colors = [rgb_starting_color, rgb_ending_color]
    personal_cmap = LinearSegmentedColormap.from_list('transparent_to_red', colors, N=n_bins)

    return personal_cmap

def tri_colormap(cmap_name: str, color_1: str, color_2: str, color_3: str, **kwargs) -> LinearSegmentedColormap:
    n_bins = kwargs.get('n_bins', 256)

    color_1 = hex2rgb(color_1)
    color_2 = hex2rgb(color_2)
    color_3 = hex2rgb(color_3)

    colors = [color_1, color_2, color_3]
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    return cmap

def tri_alpha_colormap(color_1: AlphaColor, color_2: AlphaColor, color_3: AlphaColor, **kwargs) -> LinearSegmentedColormap:
    n_bins = kwargs.get('n_bins', 256)

    colors = [
        (color_1.r, color_1.g, color_1.b, color_1.a),
        (color_2.r, color_2.g, color_2.b, color_2.a),
        (color_3.r, color_3.g, color_3.b, color_3.a)
    ]
    cmap = LinearSegmentedColormap.from_list('tri_alpha_cmap', colors, N=n_bins)

    return cmap

def bi_alpha_colormap(color_1: AlphaColor, color_2: AlphaColor, **kwargs) -> LinearSegmentedColormap:
    n_bins = kwargs.get('n_bins', 256)

    colors = [
        (color_1.r, color_1.g, color_1.b, color_1.a),
        (color_2.r, color_2.g, color_2.b, color_2.a)
    ]
    cmap = LinearSegmentedColormap.from_list('alpha_cmap', colors, N=n_bins)

    return cmap