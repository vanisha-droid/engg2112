import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt 

def load_image(image_path):
    """Load image as TensorFlow tensor."""
    img = tf.io.read_file(image_path)

    if image_path.lower().endswith(('.jpg', '.jpeg')):
        img = tf.image.decode_jpeg(img, channels=3)
    elif image_path.lower().endswith('.png'):
        img = tf.image.decode_png(img, channels=3)
    else:
        img = tf.image.decode_image(img, channels=3)  # fallback only
    img = tf.expand_dims(img, axis=0)  # Add batch dimension -> [1, H, W, 3]
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
    return img[0]  # Remove batch dim -> [H, W, 3]

def save_image(image_tensor, output_path):
    """Save TF tensor back to image file."""
    img = tf.cast(image_tensor * 255.0, tf.uint8)
    encoded = tf.image.encode_png(img) if output_path.lower().endswith('.png') else tf.image.encode_jpeg(img)
    tf.io.write_file(output_path, encoded)

def flip_image(image_tensor):
    """Horizontal flip using NumPy style or tf."""
    return tf.image.flip_left_right(image_tensor)

def add_noise(image_tensor, noise_factor=0.1):
    """Add random noise (salt & pepper like or gaussian)."""
    noise = tf.random.uniform(shape=tf.shape(image_tensor), minval=-noise_factor, maxval=noise_factor)
    noisy_img = tf.clip_by_value(image_tensor + noise, 0.0, 1.0)
    return noisy_img

def adjust_saturation(image_tensor, saturation_factor=2.0):
    """Saturate image using tf.image."""
    return tf.image.adjust_saturation(image_tensor, saturation_factor)

def adjust_brightness(image_tensor, brightness_delta=0.3):
    """Change brightness using tf.image."""
    return tf.image.adjust_brightness(image_tensor, brightness_delta)

def central_crop(image_tensor):    
    """Crop the central part of the image."""
    return tf.image.central_crop(image_tensor, central_fraction=0.5)


def random_augment(image_tensor):
    """
    Randomly apply a combination of augmentations.
    """
    aug_img = image_tensor
    
    # Randomly decide which augmentations to apply
    if random.random() < 0.9:
        aug_img = flip_image(aug_img)
    
    if random.random() < 0.1:
        aug_img = add_noise(aug_img, noise_factor=random.uniform(0.05, 0.15))
    
    if random.random() < 0.3:
        sat_factor = random.uniform(1.5, 3.0)
        aug_img = adjust_saturation(aug_img, sat_factor)
    
    if random.random() < 0.75:
        bright_delta = random.uniform(-0.3, 0.4)
        aug_img = adjust_brightness(aug_img, bright_delta)
    
    return aug_img

def has_color_artifacts(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
    # Check channel imbalance
    rg_diff = tf.reduce_mean(tf.abs(r - g))
    gb_diff = tf.reduce_mean(tf.abs(g - b))
    rb_diff = tf.reduce_mean(tf.abs(r - b))
    
    # If channels differ too much → likely corrupted
    if rg_diff > 0.4 or gb_diff > 0.4 or rb_diff > 0.4:
        return True
    
    return False
    
def augment_folder(input_folder, fraction=1/3, output_subfolder="augmented"):
    """
    Main function to perform data augmentation.
    
    Args:
        input_folder (str): Path to folder containing images.
        fraction (float): Fraction of images to sample and augment (default 1/3).
        output_subfolder (str): Name of the output folder.
    """
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    # Get all image files
    images = [f for f in os.listdir(input_folder) 
              if f.lower().endswith(valid_extensions)]
    
    if not images:
        print("No images found in the folder.")
        return
    
    # Sample images
    num_to_sample = max(1, int(len(images) * fraction))
    sampled_images = random.sample(images, num_to_sample)
    
    # Create output folder
    output_folder = os.path.join(os.path.dirname(input_folder) if os.path.dirname(input_folder) else '.', 
                                 output_subfolder)
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Found {len(images)} images. Sampling {num_to_sample} for augmentation.")
    print(f"Augmented images will be saved to: {output_folder}")
    
    for img_name in sampled_images:
        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, f"aug_{img_name}")
        
        try:
            # Load image
            img_tensor = load_image(input_path)
            
            # Apply random augmentations
            augmented = random_augment(img_tensor)
            
            # Save
            save_image(augmented, output_path)
            print(f"✓ Augmented and saved: {img_name}")
            
        except Exception as e:
            print(f"✗ Error processing {img_name}: {e}")
    
    print("~ Data augmentation done!")


if __name__ == "__main__":
    folder_path = "/Users/vanisha/Documents/coding/2112/Jaywalking" 
    augment_folder(
        input_folder=folder_path,
        fraction=1/3,           #no. of samples; this takes in 1/3rd of total and augments it
        output_subfolder="jaywalking_augmented_images"
    )