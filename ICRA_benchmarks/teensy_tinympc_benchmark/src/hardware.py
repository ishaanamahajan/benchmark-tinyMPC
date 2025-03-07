import cv2
import numpy as np
import os

# Define the base directory and file paths
base_dir = '/Users/ishaanmahajan/Downloads/iros-25/'
image_paths = [
    'View recent photos-1.jpeg',
    'View recent photos-2.jpeg',
    'View recent photos-3.jpeg',
    'View recent photos-4.jpeg',
    'View recent photos-5.jpeg',
    'View recent photos.jpeg'
]

# Full paths
full_paths = [os.path.join(base_dir, path) for path in image_paths]

# Load all the images
images = []
for path in full_paths:
    img = cv2.imread(path)
    if img is not None:
        images.append(img)
    else:
        print(f"Warning: Could not load {path}")

if len(images) == 0:
    print("No images loaded!")
else:
    print(f"Loaded {len(images)} images. Creating composite...")
    
    # First, resize all images to the same height
    # Find the smallest height from all images
    min_height = min(img.shape[0] for img in images)
    
    # Resize all images to have the same height while maintaining aspect ratio
    resized_images = []
    for img in images:
        aspect = img.shape[1] / img.shape[0]  # width/height
        new_width = int(min_height * aspect)
        resized = cv2.resize(img, (new_width, min_height))
        resized_images.append(resized)
    
    # Method 1: Horizontal concatenation (side by side)
    horizontal_img = np.hstack(resized_images)
    
    # If the image is too large to display, resize it
    display_width = 1200  # Maximum display width
    if horizontal_img.shape[1] > display_width:
        display_height = int(horizontal_img.shape[0] * (display_width / horizontal_img.shape[1]))
        display_img = cv2.resize(horizontal_img, (display_width, display_height))
    else:
        display_img = horizontal_img
    
    # Show the horizontal composite
    cv2.imshow('Horizontal Composite (Press any key to continue)', display_img)
    cv2.waitKey(0)
    
    # Ask for confirmation before saving
    print("Do you want to save the horizontal composite? (y/n)")
    choice = input().lower()
    if choice == 'y':
        cv2.imwrite(os.path.join(base_dir, 'horizontal_composite.jpg'), horizontal_img)
        print(f"Saved horizontal composite at {os.path.join(base_dir, 'horizontal_composite.jpg')}")
    
    # Method 2: Grid layout (if there are enough images)
    num_images = len(resized_images)
    if num_images >= 4:
        # Determine grid size
        grid_size = int(np.ceil(np.sqrt(num_images)))
        
        # Create a blank canvas
        max_width = max(img.shape[1] for img in resized_images)
        max_height = max(img.shape[0] for img in resized_images)
        
        # Create a grid layout
        grid_img = np.zeros((grid_size * max_height, grid_size * max_width, 3), dtype=np.uint8)
        
        # Place images in grid
        for idx, img in enumerate(resized_images):
            row = idx // grid_size
            col = idx % grid_size
            
            y_offset = row * max_height
            x_offset = col * max_width
            
            # Center the image in its grid cell
            y_padding = (max_height - img.shape[0]) // 2
            x_padding = (max_width - img.shape[1]) // 2
            
            grid_img[y_offset + y_padding:y_offset + y_padding + img.shape[0], 
                    x_offset + x_padding:x_offset + x_padding + img.shape[1]] = img
        
        # If the image is too large to display, resize it
        if grid_img.shape[1] > display_width:
            display_height = int(grid_img.shape[0] * (display_width / grid_img.shape[1]))
            display_grid = cv2.resize(grid_img, (display_width, display_height))
        else:
            display_grid = grid_img
        
        # Show the grid composite
        cv2.imshow('Grid Composite (Press any key to continue)', display_grid)
        cv2.waitKey(0)
        
        # Ask for confirmation before saving
        print("Do you want to save the grid composite? (y/n)")
        choice = input().lower()
        if choice == 'y':
            cv2.imwrite(os.path.join(base_dir, 'grid_composite.jpg'), grid_img)
            print(f"Saved grid composite at {os.path.join(base_dir, 'grid_composite.jpg')}")
    
    # Method 3: Overlay with transparency (showing "ghost" images of the drone movement)
    if len(images) > 1:
        # Use the first image as base
        base_img = images[0].copy()
        
        # Overlay subsequent images with transparency
        alpha = 0.6  # Transparency factor
        for img in images[1:]:
            # Resize overlay image to match base image dimensions
            overlay = cv2.resize(img, (base_img.shape[1], base_img.shape[0]))
            
            # Blend the images
            cv2.addWeighted(overlay, alpha, base_img, 1 - alpha, 0, base_img)
            
            # Reduce alpha for subsequent overlays
            alpha *= 0.8
        
        # Show the overlay composite
        cv2.imshow('Overlay Composite (Press any key to continue)', base_img)
        cv2.waitKey(0)
        
        # Ask for confirmation before saving
        print("Do you want to save the overlay composite? (y/n)")
        choice = input().lower()
        if choice == 'y':
            cv2.imwrite(os.path.join(base_dir, 'overlay_composite.jpg'), base_img)
            print(f"Saved overlay composite at {os.path.join(base_dir, 'overlay_composite.jpg')}")
    
    # Close all opencv windows
    cv2.destroyAllWindows()