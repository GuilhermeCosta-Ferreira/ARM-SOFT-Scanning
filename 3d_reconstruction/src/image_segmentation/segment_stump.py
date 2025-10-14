# ================================================================
# 0. Section: Imports
# ================================================================
import os
import cv2
import numpy as np

from gradio_client import Client, handle_file
from time import time

from .utils.io_utils import clean_files



# ================================================================
# 1. Section: Deep Learning Segmentation
# ================================================================
def segment_scans(path: str, scans_names: np.ndarray, output_path: str = "figures/dataset/segmented", **kwargs) -> np.ndarray:
    """
        Segment arm stump images using BiRefNet model from Hugging Face.
        This function processes a collection of scan images by applying semantic segmentation
        using the BiRefNet model hosted on Hugging Face. It segments each image to isolate
        arm stump from the background and saves the processed results to a specified output directory.

        Parameters
        ----------
        path : str
            The directory path containing the input scan images to be segmented.
        scans_names : np.ndarray
            Array of filenames for the scan images to be processed.
        output_path : str, optional
            Directory path where segmented images will be saved. 
            Default is "figures/dataset/segmented".
        **kwargs : dict
            Additional keyword arguments:
            - verbose : int, optional
                Verbosity level for progress reporting. Default is 0.
                - 0: No progress output
                - 1: Summary completion message
                - 2: Individual file progress messages

        Returns
        -------
        np.ndarray
            Array of segmented images that were successfully processed.

        Notes
        -----
        - Uses the ZhengPeng7/BiRefNet_demo model from Hugging Face for segmentation
        - Creates the output directory if it doesn't exist
        - Processes images sequentially and handles API call failures gracefully
        - Cleans up temporary files after each segmentation
        - If an error occurs during processing, the pipeline stops but returns
          any successfully segmented images
        - Timing information is provided when verbose > 0
        """
    # Optional argument to control
    verbose = kwargs.get("verbose", 0)

    # Picks the model we want from Hugging Face
    client = Client("ZhengPeng7/BiRefNet_demo")

    # Initialized the segmentation path
    os.makedirs(output_path, exist_ok=True)

    # Start the segmentation process with all the API calls
    start_time = time()
    segmented_imgs = []
    for file in scans_names:
        try: 
            result = client.predict(
                images=handle_file(path + "/" + file),
                resolution="Hello!!",
                weights_file="General",
                api_name="/image")
        except Exception as e:
            print(f"\n❌ Error while segmenting {file}: {e}")
            print(f"⚠️ Pipeline aborted, but {len(segmented_imgs)} images were segmented.\n")
            verbose = 0
            break
        
        # Process the segmnted image and save it
        segmented_img = process_segmented_image(result[0], file, output_path, is_save=True)
        segmented_imgs.append(segmented_img)

        # Clean up temporary files
        clean_files(result)
        if(verbose > 1): print(f"    ✔ Segmented {file} ({len(segmented_imgs)}/{len(scans_names)})")
    segmented_imgs = np.array(segmented_imgs)

    if(verbose > 0): print(f"✅ Segmentation completed for {len(scans_names)} images in {time() - start_time:.2f} seconds.")

    return segmented_imgs


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Post-Processing of Segmented Images
# ──────────────────────────────────────────────────────
def process_segmented_image(temporary_path: str, original_path: str, segmented_path: str, **kwargs) -> np.ndarray:
    # Optional argument to control
    is_save = kwargs.get("is_save", True)

    # Gets the binarized image from the temporary path and scales it back to 0-255
    processed_img = cv2.imread(temporary_path, cv2.IMREAD_GRAYSCALE)
    processed_bin = np.where(processed_img > 0, 255, 0).astype(np.uint8)

    # Save the processed binary image
    if is_save: cv2.imwrite(f"{segmented_path}/segmented_{original_path}", processed_bin)

    return processed_bin