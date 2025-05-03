import tkinter as tk
from tkinter import filedialog
from PIL import Image
import os
import math

root = tk.Tk()
root.withdraw()

print("Opening file dialog to select images...")
file_paths = filedialog.askopenfilenames(
    title="Select images in order (up to 6 recommended for 2x3)",
    filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")],
)

if not file_paths:
    print("No files selected. Exiting.")
    exit()

selected_paths = file_paths
num_images = len(selected_paths)
print(f"Selected {num_images} images:")
for i, path in enumerate(selected_paths):
    print(f"  {i + 1}: {os.path.basename(path)}")

images = []
max_width = 0
max_height = 0

print("Loading images and determining maximum dimensions...")
try:
    for i, path in enumerate(selected_paths):
        img = Image.open(path)
        images.append(img)
        width, height = img.size
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height
        print(f"  Image {i + 1}: {width}x{height}")
except Exception as e:
    print(f"Error loading images: {e}")
    exit()

if num_images == 0:
    print("No images loaded successfully. Exiting.")
    exit()

print(f"Using maximum cell dimensions: {max_width}x{max_height}")

max_cols_per_row = 3
cols = min(num_images, max_cols_per_row)
rows = math.ceil(num_images / cols)
print(f"Calculated grid dimensions: {rows} rows x {cols} columns")

canvas_width = cols * max_width
canvas_height = rows * max_height

combined_image = Image.new("RGB", (canvas_width, canvas_height), color="white")
print(f"Creating a {canvas_width}x{canvas_height} canvas...")

current_image_index = 0
for r in range(rows):
    for c in range(cols):
        if current_image_index < num_images:
            x_offset = c * max_width
            y_offset = r * max_height

            img_to_paste = images[current_image_index]
            combined_image.paste(img_to_paste, (x_offset, y_offset))
            print(
                f"  Pasting image {current_image_index + 1} (size: {img_to_paste.size[0]}x{img_to_paste.size[1]}) at ({x_offset}, {y_offset})"
            )
            current_image_index += 1
        else:
            break
    if current_image_index >= num_images:
        break

output_filename = r"C:\Users\xjwei\VScodeProjects\PyCalib\pycalib\results\debugging\height_map_debugging\combined_image_auto_grid.png"
try:
    combined_image.save(output_filename)
    print(f"Combined image saved successfully as '{os.path.abspath(output_filename)}'")
except Exception as e:
    print(f"Error saving the combined image: {e}")

combined_image.show()
