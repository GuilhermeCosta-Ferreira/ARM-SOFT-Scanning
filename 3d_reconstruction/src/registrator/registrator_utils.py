# ================================================================
# 0. Section: Imports
# ================================================================
import SimpleITK as sitk
import numpy as np

from .itk_utils import *



# ================================================================
# 1. Section: Input Conversion and Shape Application
# ================================================================
def convert_input(input: sitk.Image | np.ndarray) -> sitk.Image:
    """
    Convert input data to SimpleITK Image format.

    This function ensures that the input is in SimpleITK Image format, converting
    from NumPy array if necessary. If the input is already a SimpleITK Image,
    it is returned unchanged.

    Parameters
    ----------
    input : sitk.Image | np.ndarray
        The input data to convert. Can be either:
        - A SimpleITK Image object (returned unchanged)
        - A NumPy array (converted to SimpleITK Image with float32 dtype)

    Returns
    -------
    sitk.Image
        A SimpleITK Image object. If input was a NumPy array, it is converted
        to float32 dtype before creating the SimpleITK Image.

    Examples
    --------
    >>> import numpy as np
    >>> import SimpleITK as sitk
    >>> # Convert NumPy array
    >>> array = np.random.rand(10, 10, 10)
    >>> image = convert_input(array)
    >>> isinstance(image, sitk.Image)
    True
    >>> # Pass through existing SimpleITK Image
    >>> existing_image = sitk.GetImageFromArray(array)
    >>> result = convert_input(existing_image)
    >>> result is existing_image
    True

    Notes
    -----
    - NumPy arrays are automatically cast to float32 dtype before conversion
      to ensure compatibility with SimpleITK operations.
    - The function preserves the original SimpleITK Image without modification
      when the input is already in the correct format.
    """
    if isinstance(input, np.ndarray):
        return sitk.GetImageFromArray(input.astype(np.float32))
    return input
def apply_shape(fixed_image: sitk.Image, moving_image: sitk.Image) -> sitk.Image:
    """
    Resample a moving image to match the shape and spacing of a fixed image.

    This function ensures that two SimpleITK images have compatible dimensions
    for registration or comparison operations. If the moving image has a different
    size than the fixed image, it resamples the moving image using the fixed image
    as a reference while maintaining the moving image's original spacing and origin.

    Parameters
    ----------
    fixed_image : sitk.Image
        The reference image that defines the target size, spacing, and coordinate system.
        This image remains unchanged and serves as the template for resampling.
    moving_image : sitk.Image
        The image to be resampled if its size differs from the fixed image.
        If sizes match, this image is returned unchanged.

    Returns
    -------
    sitk.Image
        The moving image resampled to match the fixed image's size and coordinate system,
        or the original moving image if no resampling was needed.

    Notes
    -----
    - Uses linear interpolation (sitk.sitkLinear) for resampling to preserve image quality.
    - Only resamples when image sizes differ; identical sizes return the moving image unchanged.
    - Prints a warning message when resampling occurs to inform the user of the operation.
    - The resampling uses the fixed image as reference, meaning the output will have
      the same size, spacing, origin, and direction as the fixed image.

    Examples
    --------
    >>> import SimpleITK as sitk
    >>> fixed = sitk.ReadImage("reference.nii")
    >>> moving = sitk.ReadImage("input.nii") 
    >>> resampled = apply_shape(fixed, moving)
    >>> resampled.GetSize() == fixed.GetSize()
    True
    """
    if moving_image.GetSize() != fixed_image.GetSize():
        print("     Warning: The template and skull surface have different shapes. Resampling the fixed image to match the moving image.")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        moving_image = resampler.Execute(moving_image)
    return moving_image



# ================================================================
# 2. Section: Define Registration Components
# ================================================================
def define_loss(object, method: sitk.ImageRegistrationMethod) -> sitk.ImageRegistrationMethod:
    if(object.loss == 'MI'): method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=object.bin_size)
    elif(object.loss == 'LS'): method.SetMetricAsMeanSquares()
    else: print("Loss not supported yet, alter the class to add it.")

    return method
def define_reg_interpolator(object, method: sitk.ImageRegistrationMethod) -> sitk.ImageRegistrationMethod:
    if(object.reg_interpolator == 'linear'): method.SetInterpolator(sitk.sitkLinear)
    elif(object.reg_interpolator == 'nearest'): method.SetInterpolator(sitk.sitkNearestNeighbor)
    else: print("Interpolator not supported yet, alter the class to add it.")

    return method
def define_resample_interpolator(object, method: sitk.ResampleImageFilter) -> sitk.ResampleImageFilter:
    if(object.res_interpolator == 'linear'): method.SetInterpolator(sitk.sitkLinear)
    elif(object.res_interpolator == 'nearest'): method.SetInterpolator(sitk.sitkNearestNeighbor)
    else: print("Interpolator not supported yet, alter the class to add it.")

    return method
def define_optimizer(object, method: sitk.ImageRegistrationMethod) -> sitk.ImageRegistrationMethod:
    if(object.optimizer == 'LBFGS'): method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=object.gradientConvergenceTolerance,
                                                                numberOfIterations=object.numberOfIterations,
                                                                maximumNumberOfCorrections=object.maximumNumberOfCorrections)
    elif(object.optimizer == 'GD'): method.SetOptimizerAsGradientDescent(learningRate=object.learning_rate,
                                                                                    numberOfIterations=object.numberOfIterations,
                                                                                    convergenceMinimumValue=object.convergenceMinimumValue,
                                                                                    convergenceWindowSize=object.convergenceWindowSize,
                                                                                    estimateLearningRate=object.estimateLearningRate,
                                                                                    maximumStepSizeInPhysicalUnits=object.maximumStepSizeInPhysicalUnits
                                                                                    )
    elif(object.optimizer == 'Exhaustive'): method.SetOptimizerAsExhaustive(numberOfSteps=object.numberOfSteps,
                                                                            stepLength=object.stepLength)
    else: print("Optimizer not supported yet, alter the class to add it.")

    return method
def define_dimension_transform(object) -> sitk.ImageRegistrationMethod:
    if(object.dimension == '3d'): return sitk.VersorRigid3DTransform()
    elif(object.dimension == '2d'): return sitk.Euler2DTransform()
    else: print("Dimension not supported yet, alter the class to add it.")
def define_center_type(object) -> sitk.ImageRegistrationMethod:
    if(object.rigid_type == 'moments'): return sitk.CenteredTransformInitializerFilter.MOMENTS
    elif(object.rigid_type == 'geometric'): return sitk.CenteredTransformInitializerFilter.GEOMETRIC
    else: 
        print("Rigid type not supported yet, alter the class to add it.")
        return
def define_multiple_resolutions(object, method: sitk.ImageRegistrationMethod) -> sitk.ImageRegistrationMethod:
    if object.multiple_resolutions:
        method.SetShrinkFactorsPerLevel(shrinkFactors=object.shrinkFactors)
        method.SetSmoothingSigmasPerLevel(smoothingSigmas=object.smoothingSigmas)
        method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        return method
    else: return method



# ================================================================
# 2. Section: Registration Viewer
# ================================================================
def view_registration(registration_method: sitk.ImageRegistrationMethod) -> sitk.ImageRegistrationMethod:
    """
    Configure an image registration method with visualization callbacks for monitoring progress.
    This function adds event command callbacks to a SimpleITK ImageRegistrationMethod
    to enable real-time visualization and monitoring of the registration process.
    The callbacks track registration events including start, end, multi-resolution
    iterations, and per-iteration updates.

    Parameters
    ----------
    registration_method : sitk.ImageRegistrationMethod
        The SimpleITK image registration method object to be configured with
        visualization callbacks.

    Returns
    -------
    sitk.ImageRegistrationMethod
        The same registration method object with added event command callbacks
        for visualization and monitoring.

    Notes
    -----
    - Adds callbacks for the following SimpleITK events:
      - `sitkStartEvent`: Calls `start_plot()` when registration begins
      - `sitkEndEvent`: Calls `end_plot()` when registration completes  
      - `sitkMultiResolutionIterationEvent`: Calls `update_multires_iterations()`
        for multi-resolution level changes
      - `sitkIterationEvent`: Calls `plot_values(registration_method)` for
        each optimization iteration
    - The callback functions (`start_plot`, `end_plot`, `update_multires_iterations`,
      `plot_values`) must be defined elsewhere in the module.
    - This function modifies the input registration method in-place by adding
      command callbacks, then returns the same object.
      
    Examples
    --------
    >>> import SimpleITK as sitk
    >>> registration = sitk.ImageRegistrationMethod()
    >>> # Configure registration parameters...
    >>> registration_with_viewer = view_registration(registration)
    >>> # The registration method now has visualization callbacks attached
    """
    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    registration_method.AddCommand(
        sitk.sitkMultiResolutionIterationEvent, update_multires_iterations
    )
    registration_method.AddCommand(
        sitk.sitkIterationEvent, lambda: plot_values(registration_method)
    )

    print("     Registrator: Registration method with viewer created.")
    
    return registration_method