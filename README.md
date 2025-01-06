# Image Noise Analysis Tool

A Python-based tool for analyzing different types of image noise distributions and their effects on images. This project implements various noise models including Gaussian, Salt & Pepper, Rayleigh, and Gamma noise, along with visualization tools for analyzing their distributions.

## Features

- Support for multiple noise types:
  - Salt & Pepper noise
  - Gaussian noise
  - Rayleigh noise
  - Gamma noise
- Custom histogram generation
- Kernel Density Estimation (KDE) for probability density estimation
- Comparative visualization of noise effects
- Empirical Probability Density Function (PDF) plotting
- Automated output directory management

## Sample Outputs

[Place screenshots of your output images here]

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/noise-analysis.git
cd noise-analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

4. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your input image in the project directory

2. Update the image path in `noise_analysis.py`:
```python
image = load_and_prepare_image("path/to/your/image.png")
```

3. Run the script:
```bash
python noise_analysis.py
```

4. Check the `output_images` directory for results:
- Original and noisy images
- Histogram comparisons
- Empirical PDF plots

## Output Directory Structure

```
output_images/
├── original.png
├── noisy_salt_and_pepper.png
├── noisy_gaussian.png
├── noisy_rayleigh.png
├── noisy_gamma.png
├── comparison_salt_and_pepper_noise.png
├── comparison_gaussian_noise.png
├── comparison_rayleigh_noise.png
├── comparison_gamma_noise.png
└── empirical_pdfs.png
```

## Project Structure

```
noise-analysis/
├── src/
│   └── noise_analysis.py
├── requirements.txt
├── README.md
└── output_images/
```

## Dependencies

- NumPy
- Matplotlib
- scikit-image

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- scikit-image for image processing capabilities
- NumPy for numerical computations
- Matplotlib for visualization

## Author

Ameen Malik - ameenmalik167@gmail.com

## Support

For support, please open an issue in the GitHub repository.