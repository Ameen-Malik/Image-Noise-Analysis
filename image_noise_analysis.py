import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.util import random_noise
import os

def create_output_directory():
    """Create output directory if it doesn't exist."""
    output_dir = 'output_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def load_and_prepare_image(path):
    """Load and convert image to grayscale if needed."""
    img = io.imread(path)
    if len(img.shape) > 2:
        img = np.mean(img, axis=2)  # Convert to grayscale
    return img / 255.0  # Normalize to [0,1]

def custom_histogram(image, num_bins=256):
    """
    Calculate histogram manually without using built-in functions.
    Returns bin edges and counts.
    """
    # Flatten the image
    flat_img = image.ravel()
    
    # Create bins
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Initialize counts
    counts = np.zeros(num_bins)
    
    # Count pixels in each bin
    for pixel in flat_img:
        bin_idx = int(pixel * num_bins)
        if bin_idx == num_bins:  # Handle edge case for pixel value 1.0
            bin_idx -= 1
        counts[bin_idx] += 1
    
    # Normalize to get probability density
    counts = counts / (len(flat_img) * bin_width)
    
    return bin_edges[:-1], counts

def kernel_density_estimation(data, points, bandwidth=0.01):
    """
    Compute PDF using Kernel Density Estimation with Gaussian kernel.
    """
    density = np.zeros_like(points)
    for point in data:
        density += np.exp(-0.5 * ((points - point) / bandwidth)**2)
    density = density / (len(data) * bandwidth * np.sqrt(2 * np.pi))
    return density

def add_salt_and_pepper(image, amount=0.05):
    """Add salt and pepper noise to image."""
    return random_noise(image, mode='s&p', amount=amount)

def add_gaussian_noise(image, mean=0, var=0.01):
    """Add Gaussian noise to image."""
    return random_noise(image, mode='gaussian', mean=mean, var=var)

def add_rayleigh_noise(image, scale=0.1):
    """Add Rayleigh noise to image."""
    noise = np.random.rayleigh(scale, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def add_gamma_noise(image, shape=1.0, scale=0.1):
    """Add Gamma noise to image."""
    noise = np.random.gamma(shape, scale, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def save_noisy_image(image, output_dir, filename):
    """Save a noisy image to file."""
    plt.imsave(os.path.join(output_dir, filename), image, cmap='gray')

def plot_images_and_histograms(original, noisy, title, output_dir):
    """Plot and save original and noisy images with their custom histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Noisy image
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title(f'Image with {title}')
    axes[0, 1].axis('off')
    
    # Custom histogram for original image
    bin_edges_orig, counts_orig = custom_histogram(original)
    axes[1, 0].bar(bin_edges_orig, counts_orig, width=1/256, alpha=0.7)
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Density')
    
    # Custom histogram for noisy image
    bin_edges_noisy, counts_noisy = custom_histogram(noisy)
    axes[1, 1].bar(bin_edges_noisy, counts_noisy, width=1/256, alpha=0.7)
    axes[1, 1].set_title(f'Histogram with {title}')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Density')
    
    plt.tight_layout()
    plot_filename = f'comparison_{title.lower().replace(" ", "_")}.png'
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_empirical_pdfs(images_dict, output_dir):
    """Plot and save empirical PDFs based on actual image data."""
    plt.figure(figsize=(10, 6))
    
    # Generate points for PDF estimation
    x_points = np.linspace(0, 1, 1000)
    
    # Calculate and plot PDF for each image using KDE
    for name, image in images_dict.items():
        data = image.ravel()
        density = kernel_density_estimation(data, x_points)
        plt.plot(x_points, density, label=name)
    
    plt.title('Empirical Probability Density Functions (Based on Image Data)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    # Save the PDF plot
    plt.savefig(os.path.join(output_dir, 'empirical_pdfs.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create output directory
    output_dir = create_output_directory()
    
    # Load image
    image = load_and_prepare_image("D:/7th_sem/DIP_Project/Restore_img.png")
    
    # Save original image
    save_noisy_image(image, output_dir, 'original.png')
    
    # Dictionary to store all images for PDF comparison
    images_dict = {
        'Original': image
    }
    
    # Add different types of noise, plot results, and save images
    # Salt and Pepper
    noisy_sp = add_salt_and_pepper(image)
    save_noisy_image(noisy_sp, output_dir, 'noisy_salt_and_pepper.png')
    plot_images_and_histograms(image, noisy_sp, 'Salt & Pepper Noise', output_dir)
    images_dict['Salt & Pepper'] = noisy_sp
    
    # Gaussian
    noisy_gaussian = add_gaussian_noise(image)
    save_noisy_image(noisy_gaussian, output_dir, 'noisy_gaussian.png')
    plot_images_and_histograms(image, noisy_gaussian, 'Gaussian Noise', output_dir)
    images_dict['Gaussian'] = noisy_gaussian
    
    # Rayleigh
    noisy_rayleigh = add_rayleigh_noise(image)
    save_noisy_image(noisy_rayleigh, output_dir, 'noisy_rayleigh.png')
    plot_images_and_histograms(image, noisy_rayleigh, 'Rayleigh Noise', output_dir)
    images_dict['Rayleigh'] = noisy_rayleigh
    
    # Gamma
    noisy_gamma = add_gamma_noise(image)
    save_noisy_image(noisy_gamma, output_dir, 'noisy_gamma.png')
    plot_images_and_histograms(image, noisy_gamma, 'Gamma Noise', output_dir)
    images_dict['Gamma'] = noisy_gamma
    
    # Plot empirical PDFs based on actual image data
    plot_empirical_pdfs(images_dict, output_dir)

if __name__ == "__main__":
    main()