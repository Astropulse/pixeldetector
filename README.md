# Pixel Detector
Downscale and restore pixel art images that have been resized or corrupted.

This is primarily created to fix compression issues from saving pixel art in jpg format.

Exaggerated example:

![Example](https://github.com/Astropulse/pixeldetector/assets/61034487/f8ae2802-42c1-4dba-af56-fe849ac8915c)

# Usage
Requirements: Pillow, Numpy, Scipy

Run pixeldetector.py

Arguments:

-i --input: Expects string, Path to input image

-o --output: Expects string, Path to save output image, default = output.png

-m --max: Expects int, Max colors for computation, more = slower, default = 128

-p --palette: Expects none, Automatically reduce the image to predicted color palette

# Credits
Big thanks to https://github.com/paultron for numpy-ifying the downscale calculation and making it tons faster.

Test image by Skeddles https://lospec.com/gallery/skeddles/rock-and-grass
