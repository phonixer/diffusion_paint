import numpy as np
import cv2
import os

def generate_noise_image(width, height, noise_level=0.5, seed=None, output_path='noise_image.png', horizontal_blocks=10, vertical_blocks=10):
    """
    Generate a noise image with a blocky effect.
    
    Parameters:
    - width: int, the width of the image
    - height: int, the height of the image
    - noise_level: float, the level of noise (0 to 1)
    - seed: int, the seed for random number generator
    - output_path: str, the path to save the generated image
    - horizontal_blocks: int, the number of blocks horizontally
    - vertical_blocks: int, the number of blocks vertically
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate block size
    block_width = width // horizontal_blocks
    block_height = height // vertical_blocks
    
    # Generate random noise
    noise = np.random.rand(vertical_blocks, horizontal_blocks) * 255 * noise_level
    noise = noise.astype(np.uint8)
    
    # Repeat the noise to create a blocky effect
    noise = np.repeat(np.repeat(noise, block_height, axis=0), block_width, axis=1)
    
    # Resize the image if necessary
    if noise.shape[0] != height or noise.shape[1] != width:
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Save the image
    cv2.imwrite(output_path, noise)
    print(f"Noise image saved to {output_path}")


width = 1
height = 16
# Example usage
generate_noise_image(32*width, 32*height, noise_level=0.5, seed=42, output_path='noise_image.png', horizontal_blocks=width, vertical_blocks=height)