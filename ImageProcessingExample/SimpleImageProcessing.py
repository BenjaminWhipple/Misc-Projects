"""
A small script using pillow to fix colors in an input image.
1. We want to replace the purple with transluscence.
2. We want to scale up the files.
"""

from PIL import Image

IN_FILE = "FontsOG.png"
OUT_FILE_1 = "FONTS_8.png"
OUT_FILE_2 = "FONTS_16.png"

# Load files, make conversions as necessary

with Image.open(IN_FILE) as img:
    new_img = img.convert("RGBA")
    
    # The new_img_map is necessary in order to access pixel level data.
    new_img_map = new_img.load()
    HEIGHT, WIDTH = img.size

# Replace the purple with transluscence.

for i in range(HEIGHT):
    for j in range(WIDTH):
        print(new_img_map[i,j])
        
        # For sake of simplicity, we hardcode this.
        if new_img_map[i,j]!=(0,0,0,255):
            new_img_map[i,j]=(0,0,0,0)

# Export at original and 2x scales.

new_img.save(OUT_FILE_1)

new_img = new_img.resize((2*HEIGHT,2*WIDTH))
new_img.save(OUT_FILE_2)


