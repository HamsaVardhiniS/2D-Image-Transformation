import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Stacks for undo and redo operations
undo_stack = []
redo_stack = []

# Load the image from file
def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image. Please check the file path or file format.")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Function to save the current state of the image to the undo stack
def save_state(image):
    undo_stack.append(image.copy())  # Save a copy of the current image
    redo_stack.clear()  # Clear the redo stack because new operations invalidate the redo history

# Function to undo the last operation
def undo_last_operation():
    if undo_stack:
        redo_stack.append(modified_image.copy())  # Save current state to redo stack
        return undo_stack.pop()  # Restore the previous state from the undo stack
    else:
        print("No more actions to undo.")
        return None

# Function to redo the last undone operation
def redo_last_operation():
    if redo_stack:
        undo_stack.append(modified_image.copy())  # Save current state to undo stack
        return redo_stack.pop()  # Restore the last undone state from the redo stack
    else:
        print("No more actions to redo.")
        return None

# Function to scale the image
def scale_image(image, scale_factor):
    rows, cols, ch = image.shape
    new_size = (int(cols * scale_factor), int(rows * scale_factor))
    scaled_image = np.zeros((new_size[1], new_size[0], ch), dtype=image.dtype)

    for i in range(new_size[1]):
        for j in range(new_size[0]):
            orig_x = int(j / scale_factor)
            orig_y = int(i / scale_factor)
            if orig_x < cols and orig_y < rows:
                scaled_image[i, j] = image[orig_y, orig_x]

    return scaled_image

# Function to flip the image
def flip_image(image, direction):
    rows, cols, ch = image.shape
    flipped_image = np.zeros_like(image)

    if direction == 'horizontal':
        for i in range(rows):
            for j in range(cols):
                flipped_image[i, j] = image[i, cols - 1 - j]

    elif direction == 'vertical':
        for i in range(rows):
            for j in range(cols):
                flipped_image[i, j] = image[rows - 1 - i, j]

    return flipped_image

# Function to convert to grayscale
def convert_to_grayscale(image):
    rows, cols, ch = image.shape
    gray_image = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            R = image[i, j, 0]
            G = image[i, j, 1]
            B = image[i, j, 2]
            gray_image[i, j] = int(0.299 * R + 0.587 * G + 0.114 * B)

    return gray_image

# Function to translate the image
def translate_image(image, tx, ty):
    rows, cols, ch = image.shape
    translated_image = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            new_x = j + tx
            new_y = i + ty
            if 0 <= new_x < cols and 0 <= new_y < rows:
                translated_image[new_y, new_x] = image[i, j]

    return translated_image

# Function to rotate the image
def rotate_image(image, angle_z):
    import math
    rows, cols, ch = image.shape
    rad = math.radians(angle_z)
    cos_val = math.cos(rad)
    sin_val = math.sin(rad)

    new_rows = int(abs(rows * cos_val) + abs(cols * sin_val))
    new_cols = int(abs(cols * cos_val) + abs(rows * sin_val))
    rotated_image = np.zeros((new_rows, new_cols, ch), dtype=image.dtype)
    #Centre Calculation
    center_x, center_y = cols / 2, rows / 2
    new_center_x, new_center_y = new_cols / 2, new_rows / 2
    #Pixel Rotation
    for i in range(rows):
        for j in range(cols):
            x = j - center_x
            y = i - center_y

            new_x = int(cos_val * x - sin_val * y + new_center_x)
            new_y = int(sin_val * x + cos_val * y + new_center_y)

            if 0 <= new_x < new_cols and 0 <= new_y < new_rows:
                rotated_image[new_y, new_x] = image[i, j]

    return rotated_image

# Function to shear the image
def shear_image(image, shear_factor):
    rows, cols, ch = image.shape
    sheared_image = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            new_x = j + int(shear_factor * i)
            if 0 <= new_x < cols:
                sheared_image[i, new_x] = image[i, j]

    return sheared_image

# Function to adjust brightness
def adjust_brightness(image, value):
    adjusted_image = np.empty_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):
                # Adjust the pixel value
                adjusted_value = int(image[i, j, c].astype(np.int32) + value)
                # Clamp the value to the range [0, 255]
                adjusted_image[i, j, c] = np.clip(adjusted_value, 0, 255)

    return adjusted_image

def adjust_contrast(image, factor):
    rows, cols, ch = image.shape
    adjusted_image = np.zeros_like(image)
    mean = np.mean(image)
    for i in range(rows):
        for j in range(cols):
            for c in range(ch):
                adjusted_value = int((image[i, j, c] - mean) * factor + mean)
                adjusted_image[i, j, c] = np.clip(adjusted_value, 0, 255)

    return adjusted_image



# Function for histogram equalization
def histogram_equalization(image):
    rows, cols, ch = image.shape
    histogram = [0] * 256

    for i in range(rows):
        for j in range(cols):
            for c in range(ch):
                histogram[image[i, j, c]] += 1

    cdf = [0] * 256
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histogram[i]

    cdf_min = min(filter(lambda x: x > 0, cdf))  # Avoid zero values
    cdf_max = cdf[-1]
    normalized_cdf = [(cdf[i] - cdf_min) * 255 / (cdf_max - cdf_min) for i in range(256)]

    equalized_image = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            for c in range(ch):
                equalized_value = int(normalized_cdf[image[i, j, c]])
                equalized_image[i, j, c] = equalized_value

    return equalized_image

# Function to display the original and processed images side by side
def display_images(original, processed, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed)
    plt.title(title)
    plt.axis('off')

    plt.show()

# Function to save the modified image
def save_image(image, save_path):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR before saving with OpenCV
    cv2.imwrite(save_path, image_bgr)
    print(f"Image saved at: {save_path}")

# Main menu-driven program with undo and redo functionality
def main():
    image_path = ''
    global modified_image  # Make modified_image global so it can be used across functions
    modified_image = None
    image = None

    while True:
        print("\nMain Menu:")
        print("1. Load Image")
        print("2. Scale Image")
        print("3. Flip Image")
        print("4. Convert to Grayscale")
        print("5. Translate Image")
        print("6. Rotate Image")
        print("7. Shear Image")
        print("8. Adjust Brightness")
        print("9. Adjust Contrast")
        print("10. Apply Histogram Equalization")
        print("11. Save Image")
        print("12. Undo Last Operation")
        print("13. Redo Last Operation")
        print("14. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            image_path = input("Enter image path: ")
            image = load_image(image_path)
            if image is not None:
                modified_image = image
                undo_stack.clear()  # Clear the undo stack when a new image is loaded
                redo_stack.clear()  # Clear the redo stack when a new image is loaded
            continue

        if modified_image is None:
            print("No image loaded. Please load an image first.")
            continue

        if choice == '2':
            save_state(modified_image)
            scale_factor = float(input("Enter scale factor(Range:0.1 to 0.3): "))
            modified_image = scale_image(modified_image, scale_factor)

        elif choice == '3':
            save_state(modified_image)
            direction = input("Enter direction (horizontal/vertical): ")
            modified_image = flip_image(modified_image, direction)

        elif choice == '4':
            save_state(modified_image)
            modified_image = convert_to_grayscale(modified_image)

        elif choice == '5':
            save_state(modified_image)
            tx = int(input("Enter translation along x-axis(+ve for right, -ve for left): "))
            ty = int(input("Enter translation along y-axis(+ve for down, -ve for up): "))
            modified_image = translate_image(modified_image, tx, ty)

        elif choice == '6':
            save_state(modified_image)
            angle_z = float(input("Enter rotation angle (in degrees(Range: 0-360)): "))
            modified_image = rotate_image(modified_image, angle_z)

        elif choice == '7':
            save_state(modified_image)
            shear_factor = float(input("Enter shear factor(Range: -1.0 to 1.0): "))
            modified_image = shear_image(modified_image, shear_factor)

        elif choice == '8':
            save_state(modified_image)
            brightness_value = int(input("Enter brightness adjustment value (Range: -100 to 100): "))
            modified_image = adjust_brightness(modified_image, brightness_value)

        elif choice == '9':
            save_state(modified_image)
            contrast_factor = float(input("Enter contrast factor (Range: 0.0 to 3.0): "))
            modified_image = adjust_contrast(modified_image, contrast_factor)

        elif choice == '10':
            save_state(modified_image)
            modified_image = histogram_equalization(modified_image)

        elif choice == '11':
            save_path = input("Enter the save path (including filename): ")
            save_image(modified_image, save_path)

        elif choice == '12':
            previous_image = undo_last_operation()
            if previous_image is not None:
                modified_image = previous_image

        elif choice == '13':
            redone_image = redo_last_operation()
            if redone_image is not None:
                modified_image = redone_image

        elif choice == '14':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

        # Display the original and modified images side by side
        display_images(image, modified_image, 'Processed Image')

if __name__ == "__main__":
    main()