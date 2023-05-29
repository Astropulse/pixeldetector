import os, argparse, time
from PIL import Image
import numpy as np
import scipy
from itertools import product

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "Path to input image")
ap.add_argument("-o", "--output", required = False, default="output.png", help = "Path to save output image")
ap.add_argument("-m", "--max", required = False, type=int, default=128, help = "Max colors for computation, more = slower")
ap.add_argument("-p", "--palette", required = False, action="store_true", help = "Automatically reduce the image to predicted color palette")
args = vars(ap.parse_args())

def kCentroid(image: Image, width: int, height: int, centroids: int):
    image = image.convert("RGB")

    # Create an empty array for the downscaled image
    downscaled = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the scaling factors
    wFactor = image.width/width
    hFactor = image.height/height

    # Iterate over each tile in the downscaled image
    for x, y in product(range(width), range(height)):
            # Crop the tile from the original image
            tile = image.crop((x*wFactor, y*hFactor, (x*wFactor)+wFactor, (y*hFactor)+hFactor))

            # Quantize the colors of the tile using k-means clustering
            tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert("RGB")

            # Get the color counts and find the most common color
            color_counts = tile.getcolors()
            most_common_color = max(color_counts, key=lambda x: x[0])[1]

            # Assign the most common color to the corresponding pixel in the downscaled image
            downscaled[y, x, :] = most_common_color

    return Image.fromarray(downscaled, mode='RGB')

def pixel_detect(image: Image):
    # Thanks to https://github.com/paultron for optimizing my garbage code 
    # I swapped the axis so they accurately reflect the horizontal and vertical scaling factor for images with uneven ratios

    # Convert the image to a NumPy array
    npim = np.array(image)[..., :3]

    # Compute horizontal differences between pixels
    hdiff = np.sqrt(np.sum((npim[:, :-1, :] - npim[:, 1:, :])**2, axis=2))
    hsum = np.sum(hdiff, 0)

    # Compute vertical differences between pixels
    vdiff = np.sqrt(np.sum((npim[:-1, :, :] - npim[1:, :, :])**2, axis=2))
    vsum = np.sum(vdiff, 1)

    # Find peaks in the horizontal and vertical sums
    hpeaks, _ = scipy.signal.find_peaks(hsum, distance=1, height=0.0)
    vpeaks, _ = scipy.signal.find_peaks(vsum, distance=1, height=0.0)
    
    # Compute spacing between the peaks
    hspacing = np.diff(hpeaks)
    vspacing = np.diff(vpeaks)

    # Resize input image using kCentroid with the calculated horizontal and vertical factors
    return kCentroid(image, round(image.width/np.median(hspacing)), round(image.height/np.median(vspacing)), 2), np.median(hspacing), np.median(vspacing)

def determine_best_k(image: Image, max_k: int):
    # Convert the image to RGB mode
    image = image.convert("RGB")

    # Prepare arrays for distortion calculation
    pixels = np.array(image)
    pixel_indices = np.reshape(pixels, (-1, 3))

    # Calculate distortion for different values of k
    distortions = []
    for k in range(1, max_k + 1):
        quantized_image = image.quantize(colors=k, method=0, kmeans=k, dither=0)
        centroids = np.array(quantized_image.getpalette()[:k * 3]).reshape(-1, 3)
        
        # Calculate distortions
        distances = np.linalg.norm(pixel_indices[:, np.newaxis] - centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        distortions.append(np.sum(min_distances ** 2))

    # Calculate the rate of change of distortions
    rate_of_change = np.diff(distortions) / np.array(distortions[:-1])
    
    # Find the elbow point (best k value)
    if len(rate_of_change) == 0:
        best_k = 2
    else:
        elbow_index = np.argmax(rate_of_change) + 1
        best_k = elbow_index + 2

    return best_k

if os.path.isfile(args["input"]):
    # Open input image
    image = Image.open(args["input"]).convert('RGB')

    # Start timer
    start = round(time.time()*1000)

    # Find 1:1 pixel scale
    downscale, hf, vf = pixel_detect(image)

    print(f"Size detected and reduced from {image.width}x{image.height} to {downscale.width}x{downscale.height} in {round(time.time()*1000)-start} milliseconds")

    scale = max(hf, vf)
    output = downscale

    if args["palette"]:
        # Start timer
        start = round(time.time()*1000)

        # Reduce color palette using elbow method
        best_k = determine_best_k(downscale, args["max"])
        output = downscale.quantize(colors=best_k, method=1, kmeans=best_k, dither=0).convert('RGB')

        print(f"Palette reduced to {best_k} colors in {round(time.time()*1000)-start} milliseconds")
    
    output.save(args["output"])