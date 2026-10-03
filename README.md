Kernel Reconstruction in Convolutional Neural Networks

University of Trier · Research project · March 2024

An interactive PyTorch project exploring image filtering and the recovery of a 3×3 convolutional kernel from a known original image and its filtered counterpart.
The project combines predefined and custom image filters, notebook-based visualization, and gradient-based kernel estimation. It was completed by a four-member research group under the supervision of PD Dr. Stephan Schmidt.

**Overview**

Applying a kernel to an image is a forward operation: the image and kernel determine the filtered output. This project also investigates the inverse problem: given the original image and a filtered result, can a model estimate the kernel that relates them?
The implementation fits the nine weights of a single convolutional layer by minimizing the difference between its predicted output and the target filtered image. It is a compact experiment in inverse problems and differentiable image processing.

**Features**

- **Predefined filters:** Gaussian blur, sharpening, ridge detection, edge detection, and identity.
- **Custom kernels:** An editable 3×3 grid built with ipywidgets.
- **Interactive visualization:** Original image, filtered image, and kernel heatmap displayed side by side.
- **Kernel estimation:** A trainable PyTorch convolutional layer optimized using mean squared error.
- **Result inspection:** Recovered kernel coefficients, a heatmap, and the final image-output MSE.

**Methodology**

**1. Apply a known or custom kernel**

An RGB image is loaded with Pillow, converted into a tensor, and given a batch dimension. The same fixed 3×3 kernel is applied independently to each color channel through separate nn.Conv2d layers.
The forward-filtering layers use no bias or padding. An image of size H × W therefore produces an output of size (H - 2) × (W - 2).

**2. Prepare the image pair**

The original and saved filtered images are converted to grayscale tensors. Kernel recovery uses both images: the original is the model input, and the filtered image is the target.

**3. Learn the kernel**

The recovery model contains one linear convolutional layer:
class KernelRecoveryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, bias=False)

    def forward(self, x):
        return self.conv1(x)
        
Training adjusts the kernel weights to minimize pixelwise mean squared error between the model output and the target image.


Input and output channels	: 1, grayscale

Kernel size	: 3×3

Trainable parameters :	9 weights

Bias :	Disabled

Stride / padding	: 1 / 0

Activation function :	None

Loss :	nn.MSELoss()

Optimizer	: torch.optim.SGD

Learning rate :	0.1

Training iterations	: 5,000

The optimizer processes the same full image pair at every iteration, so it behaves as full-image gradient descent rather than stochastic mini-batch training.

**Implementation detail:** PyTorch's Conv2d applies cross-correlation without flipping the kernel. Both the filtering and recovery stages use this convention.

**Recorded Results**

The saved output in RCSP12.ipynb reports:

Metric : Image-output mean squared error	

Recorded value :	0.00010579529771348462


This is the error between the learned model's output and the target filtered image, evaluated on the **same image pair used for fitting**. It is not kernel-coefficient error, classification accuracy, or a held-out generalization result. The recorded output is historical; it has not been independently rerun for this README.

The report describes reconstructed coefficients that differ from the original kernel while producing visually similar filtering behavior. A low image-output error alone does not establish exact or unique recovery of the original kernel.

**Experiment versions:** The report describes a 10,000-iteration experiment, while the supplied notebook and kernelrecovery.py specify 5,000 iterations. Keep this distinction when reproducing or comparing results.

Repository Contents
| File	| Purpose |
| --- | --- |
| RCSP12.ipynb |	Main notebook containing filtering, custom-kernel, and recovery experiments with saved outputs |
| knownkernel.py	| Notebook-oriented code for predefined filters and visualization |
| customkernel.py	| Notebook-oriented code for editing and applying custom kernels |
| kernelrecovery.py |	Notebook-oriented code for fitting and displaying a recovered kernel |
| Kernel Reconstruction in Convolutional Neural.pdf |	Research report covering background, methodology, results, and future extensions |


**Getting Started**

**Recommended environment**

Use Google Colab or Jupyter Notebook/JupyterLab with widget support. The code uses interactive widgets and Colab-style /content/ paths.
Install the required packages in a notebook cell:

%pip install torch torchvision numpy matplotlib scipy pillow ipywidgets

The saved notebook installation output shows Python 3.10, PyTorch 2.2.1, torchvision 0.17.1, and Matplotlib 3.7.1. The repository does not provide a complete pinned environment.

**Run the notebook**

1. Open RCSP12.ipynb in Colab or Jupyter.
2. Supply an RGB image. The code expects /content/stones.jpg; this image is not included in the supplied project files. Upload a suitable image or update every input-image path consistently.
3. Run the import and predefined-filter cells. Select a filter, or use the custom-kernel cell to enter your own coefficients.
4. Apply the chosen kernel and click Save Convoluted Image. The default output is /content/convoluted_image1.jpg.
5. Run the Kernel Recovery section after saving the target. It reads the original image and the saved filtered image, fits the kernel, and displays the coefficients and MSE.
6. Repeat with another kernel by generating and saving a new target before rerunning recovery. Saving overwrites the previous target image.
For local Jupyter use, replace both input and output /content/ paths with valid local paths. Preserve the filtered image dimensions; resizing the saved target will break the expected shape relationship.

**Limitations and Reproducibility Notes**

- **One image pair per fit:** The implementation does not include a multi-image training dataset or a held-out evaluation split.
- **Clipping and JPEG export:** Filtered values are clipped to [0, 1], converted to 8-bit pixels, and saved as JPEG before recovery. This changes the target and can prevent exact recovery, especially for filters producing negative or out-of-range values.
- **Random initialization:** No random seed is fixed, so recovered weights and MSE can vary between runs.
- **Fixed model:** Recovery supports a single grayscale 3×3 kernel with no bias, padding, or nonlinear activation.
- **Constant-kernel visualization:** The heatmap normalization divides by the kernel's value range; an all-equal kernel needs a guard against division by zero.
- **Scope:** The supplied implementation does not contain a ResNet, an optical-flow estimator, or a completed deconvolution model, despite the broader report subtitle and discussion.
  
**Future Work**

- Evaluate reconstruction across multiple images and unseen image pairs.
- Compare recovered coefficients directly with the known generating kernel.
- Preserve floating-point filtered targets to avoid clipping and JPEG artifacts.
- Explore larger kernels, alternative optimizers such as Adam, and convergence tracking.
- Add fixed seeds, dependency versions, sample assets, and reproducible experiment configurations.
- Extend the work to image deconvolution: estimating an unknown original image from a filtered image and a known kernel.
These are proposed extensions, not completed features of the supplied implementation.
