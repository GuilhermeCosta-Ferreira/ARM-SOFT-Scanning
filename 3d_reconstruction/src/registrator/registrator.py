# ================================================================
# 0. Section: Imports
# ================================================================
import time

import numpy as np
import SimpleITK as sitk

from SimpleITK import ImageRegistrationMethod
from .registrator_utils import *



# ================================================================
# 1. Section: Registrator Class
# ================================================================
class Registrator():
    """
    A configurable wrapper around SimpleITK for pairwise medical/volumetric image
    registration in 3D (and 2D), offering rigid, affine, and deformable pipelines
    with pluggable loss functions, optimizers, interpolators, multi-resolution
    strategies, and optional live view updates.

    Parameters
    ----------
    method : {'rigid', 'affine', 'deform'}, default='rigid'
        The transformation model to estimate.
        - 'rigid'  : rotation + translation
        - 'affine' : full linear transform (scale/shear/rotation/translation)
        - 'deform' : non-rigid (e.g., B-spline); requires helper setup
    loss : {'MI','MSE','NCC','MattesMI', ...}, default='MI'
        Similarity metric identifier consumed by `define_loss`.
        The exact mapping is implemented in `registrator_utils.define_loss`.
    optimizer : {'LBFGS','GD','Exhaustive', ...}, default='LBFGS'
        Optimizer identifier consumed by `define_optimizer`.
    dimension : {'2d','3d'}, default='3d'
        Registration dimensionality; forwarded to helper functions that choose
        appropriate transform types and parameters.
    view_update : bool, default=True
        If True, attaches observers (via `view_registration`) to visualize progress.
    check_shape : bool, default=False
        If True, resamples the moving image to match the fixed image size before
        registration (via `apply_shape`).
    verbose : int, default=1
        Verbosity level:
        - 0: silent
        - 1: high-level messages
        - 2: +timings
        - 3: +final metric and optimizer stop reason
    **kwargs
        Advanced configuration (optional). Recognized keys include:
        - numberOfIterations : int, default=500
        - sampling_percentage : float in (0,1], default=0.5
        - interpolator : {'nearest','linear','bspline', ...}, default='linear'
            (registration interpolator; used by `define_reg_interpolator`)
        - res_interpolator : {'nearest','linear','bspline', ...}, default='nearest'
            (resampling interpolator; used by `define_resample_interpolator`)
        - rigid_type : {'moments','geometry'}, default='moments'
            Initializer flavor for rigid transforms.
        - multiple_resolutions : bool, default=False
            If True, enables multi-resolution pyramid.
        - shrinkFactors : list[int], default=[4,2,1]
            Per-level shrink factors when multi-resolution is enabled.
        - smoothingSigmas : list[int|float], default=[2,1,0]
            Gaussian smoothing per level (in physical units unless otherwise set).
        - gradientConvergenceTolerance : float, default=1e-5
        - maximumNumberOfCorrections : int, default=5
        - learning_rate : float, default=0.1
        - convergenceMinimumValue : float, default=1e-6
        - convergenceWindowSize : int, default=10
        - estimateLearningRate : SimpleITK.ImageRegistrationMethod enum, default=Once
        - maximumStepSizeInPhysicalUnits : float, default=0.0
        - numberOfSteps : list[int], default=[0,1,1,0,0,0]
        - stepLength : float, default=1.0
        - grid_size : int, default=2
        - bin_size : int, default=50
        - isComposite : bool, default=False
        - composite : list[sitk.Transform], default=[]

    Attributes
    ----------
    method : str
        Selected transformation model.
    loss : str
        Selected similarity metric key.
    optimizer : str
        Selected optimizer key.
    dimension : str
        '2d' or '3d'.
    view_update : bool
        Whether progress observers are attached.
    check_shape : bool
        Whether to pre-match moving image size to fixed image.
    verbose : int
        Verbosity level.
    numberOfIterations : int
        Optimizer iteration cap (for relevant optimizers).
    sampling_percentage : float
        Fraction of voxels/pixels sampled for metric evaluation (if enabled).
    reg_interpolator : str
        Interpolator used during registration metric evaluation.
    res_interpolator : str
        Interpolator used when resampling the final registered image.
    rigid_type : str
        Rigid initialization type.
    multiple_resolutions : bool
        Whether a multi-resolution pyramid is used.
    shrinkFactors : list[int]
        Multi-resolution shrink factors.
    smoothingSigmas : list[int|float]
        Multi-resolution smoothing sigmas.
    gradientConvergenceTolerance : float
        LBFGS parameter.
    maximumNumberOfCorrections : int
        LBFGS parameter.
    learning_rate : float
        Gradient Descent parameter.
    convergenceMinimumValue : float
        GD convergence threshold.
    convergenceWindowSize : int
        GD window size for convergence.
    estimateLearningRate : SimpleITK.ImageRegistrationMethod enum
        GD learning rate policy.
    maximumStepSizeInPhysicalUnits : float
        GD step size cap in physical units.
    numberOfSteps : list[int]
        Exhaustive search steps per parameter.
    stepLength : float
        Exhaustive step length.
    grid_size : int
        Deformable grid size (e.g., for B-spline).
    bin_size : int
        Histogram bins for MI-like metrics (if applicable).
    isComposite : bool
        If True, use/return composite transforms.
    composite : list[sitk.Transform]
        Pre-existing transforms to compose or append.

    Methods
    -------
    register(fixed_image, moving_image, **kwargs) -> sitk.Image
        High-level entry point; routes to `rigid_transform`, `affine_transform`,
        or `deform_transform` based on `self.method`. Returns the registered image.
    rigid_transform(fixed_image, moving_image) -> (np.ndarray, sitk.Transform)
        Runs a rigid registration and returns the resampled moving image (NumPy)
        and the resulting rigid transform.
    setup_rigid(fixed_image, moving_image) -> sitk.ImageRegistrationMethod
        Creates and configures a SimpleITK registration method for rigid alignment.
    resample(fixed_image, moving_image, transform) -> sitk.Image
        Resamples `moving_image` onto the grid of `fixed_image` using `transform`.

    Raises
    ------
    ValueError
        May be raised by helper functions (e.g., invalid parameter values) or by
        SimpleITK when inputs are inconsistent (spacing/size/dimension).
    RuntimeError
        Propagated from SimpleITK if optimization fails to start or converge.

    Examples
    --------
    >>> import SimpleITK as sitk
    >>> import numpy as np
    >>> reg = Registrator(method='rigid', loss='MI', optimizer='LBFGS', verbose=2,
    ...                   multiple_resolutions=True, shrinkFactors=[4,2,1], smoothingSigmas=[2,1,0])
    >>> fixed = sitk.ReadImage("fixed.nii.gz")
    >>> moving = sitk.ReadImage("moving.nii.gz")
    >>> registered_np, T = reg.rigid_transform(fixed, moving)
    >>> isinstance(T, sitk.Transform)
    True
    >>> # High-level API:
    >>> registered_img = reg.register(fixed, moving)
    >>> registered_img.GetSize() == fixed.GetSize()
    True

    Notes
    -----
    - Helper functions referenced:
      `convert_input`, `apply_shape`, `define_loss`, `define_reg_interpolator`,
      `define_optimizer`, `view_registration`, `define_multiple_resolutions`,
      `define_dimension_transform`, `define_center_type`, `define_resample_interpolator`
      (all expected in `.registrator_utils`).
    - This class assumes SimpleITK images have correct metadata (spacing, origin,
      direction). If you pass NumPy arrays, they are converted with default metadata.
    - For reproducibility, consider fixing random seeds if stochastic sampling is used.
    """
    def __init__(self, method='rigid', loss='MI', optimizer='LBFGS', dimension='3d', view_update=True, check_shape=False, verbose=1, **kwargs):
        self.method = method
        self.loss = loss
        self.optimizer = optimizer
        self.dimension = dimension
        self.view_update = view_update
        self.check_shape = check_shape
        self.verbose = verbose

        # |----- Default Parameters -----|
        self.numberOfIterations = kwargs['numberOfIterations'] if 'numberOfIterations' in kwargs else 500
        self.sampling_percentage = kwargs['sampling_percentage'] if 'sampling_percentage' in kwargs else 0.5
        self.reg_interpolator = kwargs['interpolator'] if 'interpolator' in kwargs else 'linear'
        self.res_interpolator = kwargs['res_interpolator'] if 'res_interpolator' in kwargs else 'nearest'
        self.rigid_type = kwargs['rigid_type'] if 'rigid_type' in kwargs else 'moments'
        self.multiple_resolutions = kwargs['multiple_resolutions'] if 'multiple_resolutions' in kwargs else False
        self.shrinkFactors = kwargs['shrinkFactors'] if 'shrinkFactors' in kwargs else [4, 2, 1]
        self.smoothingSigmas = kwargs['smoothingSigmas'] if 'smoothingSigmas' in kwargs else [2, 1, 0]

        # |----- Default Parameters for LBFGS-----|
        self.gradientConvergenceTolerance = kwargs['gradientConvergenceTolerance'] if 'gradientConvergenceTolerance' in kwargs else 1e-5
        self.maximumNumberOfCorrections = kwargs['maximumNumberOfCorrections'] if 'maximumNumberOfCorrections' in kwargs else 5

        # |----- Default Parameters for Gradient Descent -----|
        self.learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else 0.1
        self.convergenceMinimumValue = kwargs['convergenceMinimumValue'] if 'convergenceMinimumValue' in kwargs else 1e-6
        self.convergenceWindowSize = kwargs['convergenceWindowSize'] if 'convergenceWindowSize' in kwargs else 10
        self.estimateLearningRate = kwargs['estimateLearningRate'] if 'estimateLearningRate' in kwargs else ImageRegistrationMethod.Once
        self.maximumStepSizeInPhysicalUnits = kwargs['maximumStepSizeInPhysicalUnits'] if 'maximumStepSizeInPhysicalUnits' in kwargs else 0.0
        
        # |----- Default Parameters for Exhaustive -----|
        self.numberOfSteps = kwargs['numberOfSteps'] if 'numberOfSteps' in kwargs else [0, 1, 1, 0 , 0, 0]
        self.stepLength = kwargs['stepLength'] if 'stepLength' in kwargs else 1.0

        ## |----- Default Parameters for Bspline -----|
        self.grid_size = kwargs['grid_size'] if 'grid_size' in kwargs else 2
        self.bin_size = kwargs['bin_size'] if 'bin_size' in kwargs else 50

        ## |----- Composite Transform -----|
        self.isComposite = kwargs['isComposite'] if 'isComposite' in kwargs else False
        self.composite = kwargs['composite'] if 'composite' in kwargs else []

    def register(self, fixed_image: sitk.Image | np.ndarray, moving_image: sitk.Image | np.ndarray, **kwargs) -> sitk.Image:
        """
        Register fixed and moving images using the specified transformation method.

        This method performs image registration by applying one of the supported
        transformation methods (rigid, deformable, or affine) to align the moving
        image with the fixed image.

        Parameters
        ----------
        fixed_image : sitk.Image | np.ndarray
            The reference image that remains stationary during registration.
        moving_image : sitk.Image | np.ndarray
            The image to be transformed and aligned with the fixed image.
        **kwargs : dict
            Additional keyword arguments, including:
            - composite : bool, optional
                Whether to use composite transformation. Overrides instance setting
                if provided.

        Returns
        -------
        sitk.Image
            The registered (transformed) moving image aligned with the fixed image.

        Notes
        -----
        - The registration method is determined by the instance's `method` attribute
        - Supported methods are: 'rigid', 'deform', and 'affine'
        - Progress information is printed if verbose level >= 1
        - If an unsupported method is specified, an error message is printed

        Examples
        --------
        >>> registrator = NR_Registrator(method='rigid')
        >>> registered_image = registrator.register(fixed_img, moving_img)
        >>> # With composite override
        >>> registered_image = registrator.register(fixed_img, moving_img, composite=True)
        """

        self.composite = kwargs['composite'] if 'composite' in kwargs else self.composite

        if(self.verbose >= 1): print(f"NR_Registrator: Registrator initialized with method: {self.method}, loss: {self.loss}, optimizer: {self.optimizer}, dimension: {self.dimension}")
        if(self.method == 'rigid'): results = self.rigid_transform(fixed_image, moving_image)
        elif(self.method == 'deform'): results = self.deform_transform(fixed_image, moving_image)
        elif(self.method == 'affine'): results = self.affine_transform(fixed_image, moving_image)
        else: print("❌ Method not supported yet, alter the class to add it.")

        if(self.verbose >= 1): print('\n')

        return results



    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Registration Methods
    # ──────────────────────────────────────────────────────
    def rigid_transform(self, fixed_image: sitk.Image | np.ndarray, moving_image: sitk.Image | np.ndarray) -> tuple[np.ndarray, sitk.Transform]:
        """
            Perform rigid image registration between fixed and moving images.
            
            This method aligns a moving image to a fixed reference image using rigid
            transformation (translation and rotation only). The registration process
            uses SimpleITK's optimization framework to find the optimal transformation
            parameters that maximize image similarity.

            Parameters
            ----------
            fixed_image : sitk.Image | np.ndarray
                The reference image that remains stationary during registration.
                Will be converted to SimpleITK format if provided as numpy array.
            moving_image : sitk.Image | np.ndarray
                The image to be transformed and aligned to the fixed image.
                Will be converted to SimpleITK format if provided as numpy array.

            Returns
            -------
            tuple[np.ndarray, sitk.Transform]
                A tuple containing:
                - registered_np : np.ndarray
                    The moving image after rigid transformation, resampled to match
                    the fixed image space and converted back to numpy array format.
                - transform : sitk.Transform
                    The computed rigid transformation object that maps points from
                    the moving image coordinate system to the fixed image coordinate
                    system.

            Notes
            -----
            - If `self.check_shape` is True, the moving image will be resampled to
              match the fixed image dimensions before registration.
            - The registration method and parameters are configured via `setup_rigid()`.
            - Verbose output levels control the amount of progress information printed:
              - Level 2+: Shows registration start/completion messages with timing
              - Level 3+: Shows final metric value and optimizer stopping condition
            - Registration timing is measured and reported when verbose >= 2.

            Examples
            --------
            >>> registrator = Registrator(verbose=2)
            >>> fixed = sitk.ReadImage("reference.nii")
            >>> moving = sitk.ReadImage("moving.nii")
            >>> registered_array, transform = registrator.rigid_transform(fixed, moving)
            >>> print(f"Registered image shape: {registered_array.shape}")
            >>> print(f"Transform type: {transform.GetName()}")
            """
        # Properly convert the images to SimpleITK format
        fixed_image = convert_input(fixed_image)
        moving_image = convert_input(moving_image)

        # Check if the fixed and moving images have the same size, if not resamples the moving image
        if(self.check_shape): moving_image = apply_shape(fixed_image, moving_image)

        # Initialize the rigid registration
        registration_method = self.setup_rigid(fixed_image, moving_image)

        # Execute registration
        if(self.verbose >= 2): print("    🔄 Beginning Rigid Registration...")
        start_time = time.time()
        transform = registration_method.Execute(fixed_image, moving_image)
        registration_time = time.time() - start_time

        # Resample moving image
        resampled_image = self.resample(fixed_image, moving_image, transform)

        # Convert back to numpy
        registered_np = sitk.GetArrayFromImage(resampled_image)

        if(self.verbose >= 3): print(f"        ✔︎ Final metric value: {registration_method.GetMetricValue()}")
        if(self.verbose >= 3): print(f"        ✔︎ Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}")
        if(self.verbose >= 2): print(f"    ✅ Rigid Registration completed ({registration_time:.2f} seconds)\n")

        return registered_np, transform
    

    # ››››››››››››››››››››››››››››››››››››››››››››››››
    # 1.1.1 Sub-subsection: Rigid Registration Helpers
    # ›››››››››››››››››››››››››››››››››››››››››››››››››
    def setup_rigid(self, fixed_image: sitk.Image, moving_image: sitk.Image) -> sitk.ImageRegistrationMethod:
        """
        Set up a rigid image registration method for 3D reconstruction.
        This method configures a SimpleITK ImageRegistrationMethod with all necessary
        components for rigid registration including loss function, interpolator,
        optimizer, sampling strategy, multi-resolution pyramid, and initial transform.

        Parameters
        ----------
        fixed_image : sitk.Image
            The reference image that remains stationary during registration.
        moving_image : sitk.Image
            The image to be transformed and aligned to the fixed image.

        Returns
        -------
        sitk.ImageRegistrationMethod
            A fully configured registration method ready for execution with:
            - Configured metric, interpolator, and optimizer
            - Sampling percentage and optimizer scales set
            - Optional visualization observers attached
            - Multi-resolution registration enabled
            - Initial centered transform computed and set

        Notes
        -----
        - The initial transform is computed using centered transform initialization
          to provide a good starting point for optimization.
        - Multi-resolution registration is used to improve convergence and avoid
          local minima.
        - The registration method uses physical shift-based optimizer scaling
          for better numerical stability.
        - If view_update is enabled, visualization observers are attached to
          monitor registration progress.

        Examples
        --------
        >>> registrator = Registrator()
        >>> fixed_img = sitk.ReadImage("fixed.nii")
        >>> moving_img = sitk.ReadImage("moving.nii") 
        >>> reg_method = registrator.setup_rigid(fixed_img, moving_img)
        >>> final_transform = reg_method.Execute()
        """
        # Initialize the registration method
        registration_method = sitk.ImageRegistrationMethod()

        # Set the metric, interpolator, and optimizer
        registration_method = define_loss(self, registration_method)
        registration_method = define_reg_interpolator(self, registration_method)
        registration_method = define_optimizer(self, registration_method)
        registration_method.SetMetricSamplingPercentage(self.sampling_percentage)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Connect all of the observers so that we can perform plotting during registration.
        if(self.view_update): registration_method = view_registration(registration_method)

        # Allow for multi-resolution registration
        registration_method = define_multiple_resolutions(self, registration_method)
        
        # Create the transform
        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                            moving_image,
                                                            define_dimension_transform(self),
                                                            define_center_type(self))
        
        # Set the initial transform
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        return registration_method
    


    # ──────────────────────────────────────────────────────
    # 1.2 Subsection: Resampling Method
    # ──────────────────────────────────────────────────────
    def resample(self, fixed_image: sitk.Image, moving_image: sitk.Image, transform: sitk.Transform) -> sitk.Image:
        """
        Resample a moving image to align with a fixed image using a given transform.

        This method applies spatial transformation to the moving image so that it
        aligns with the coordinate space and grid of the fixed image. The resampling
        uses the reference image properties (spacing, size, origin, direction) and
        applies the specified transform with interpolation.

        Parameters
        ----------
        fixed_image : sitk.Image
            The reference image that defines the output coordinate space, grid size,
            spacing, origin, and direction for the resampled result.
        moving_image : sitk.Image
            The image to be transformed and resampled to match the fixed image's
            coordinate space.
        transform : sitk.Transform
            The spatial transformation to apply to the moving image. This defines
            how points in the fixed image space map to the moving image space.

        Returns
        -------
        sitk.Image
            The resampled moving image in the coordinate space of the fixed image.
            The output image will have the same dimensions, spacing, origin, and
            direction as the fixed image.

        Notes
        -----
        - The interpolation method is set by the `define_resample_interpolator` 
          method which should be defined elsewhere in the class.
        - The transform maps from the fixed image coordinate system to the moving
          image coordinate system.
        - The resampling process may introduce interpolation artifacts depending
          on the chosen interpolation method.

        Examples
        --------
        >>> fixed_img = sitk.ReadImage("reference.nii")
        >>> moving_img = sitk.ReadImage("moving.nii") 
        >>> transform = sitk.AffineTransform(3)
        >>> resampled = registrator.resample(fixed_img, moving_img, transform)
        >>> resampled.GetSize() == fixed_img.GetSize()
        True
        """
        # Create a resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler = define_resample_interpolator(self, resampler)
        resampler.SetTransform(transform)

        # Resample the image
        resampled_image = resampler.Execute(moving_image)

        return resampled_image