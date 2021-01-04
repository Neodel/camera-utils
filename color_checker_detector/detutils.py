# detutils.py
from __future__ import division
import cv2
import numpy as np
from collections import namedtuple

WORKING_WIDTH = 1440
SWATCHES_HORIZONTAL = 6
SWATCHES_VERTICAL = 4
SWATCHES = SWATCHES_HORIZONTAL * SWATCHES_VERTICAL
SWATCH_MINIMUM_AREA_FACTOR = 8000
ASPECT_RATIO = 1.5
ERODE_FACTOR = 3


class ColourCheckerSwatchesData(namedtuple('ColourCheckerSwatchesData',
                                           ('swatch_colours', 'colour_checker_image', 'swatch_masks'))):
    __slots__ = ()


def colour_checkers_coordinates_segmentation(image, additional_data=False, verbose=False):
    """
    This function permit to identify in a picture all color checkers
    :param verbose: show steps
    :param image: working image
    :param additional_data: permit to define if return also clusters, swatches, and the reworked image
    :return:
    """
    image = as_8_bit_BGR_image(adjust_image(image, WORKING_WIDTH))

    width, height = image.shape[1], image.shape[0]
    maximum_area = width * height / SWATCHES
    minimum_area = width * height / SWATCHES / SWATCH_MINIMUM_AREA_FACTOR

    block_size = int(WORKING_WIDTH * 0.015)
    block_size = block_size - block_size % 2 + 1

    # Thresholding/Segmentation
    image_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #image_g = cv2.fastNlMeansDenoising(image_g, None, 10, 7, 21)
    # Show steps
    if verbose:
        img = adjust_image(image_g)
        cv2.imshow('image', img)
        cv2.waitKey(0)

    image_s = cv2.adaptiveThreshold(image_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 3)
    # Show steps
    if verbose:
        img = adjust_image(image_s)
        cv2.imshow('image', img)
        cv2.waitKey(0)

    # Cleanup
    kernel = np.ones((ERODE_FACTOR, ERODE_FACTOR), np.uint8)
    image_c = cv2.erode(image_s, kernel, iterations=1)
    image_c = cv2.dilate(image_c, kernel, iterations=1)
    # Show steps
    if verbose:
        img = adjust_image(image_c)
        cv2.imshow('image', img)
        cv2.waitKey(0)

    # Detecting contours
    contours, _hierarchy = cv2.findContours(image_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Show steps
    if verbose:
        img = image_c.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        img = adjust_image(img)
        cv2.imshow('image', img)
        cv2.waitKey(0)

    # Filtering squares/swatches contours
    swatches = []
    curves = []
    for contour in contours:
        curve = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        curves.append(curve)
        if minimum_area < cv2.contourArea(curve) < maximum_area and is_square(curve):
            swatches.append(np.asarray(cv2.boxPoints(cv2.minAreaRect(curve)), dtype=np.int64))
    # Show steps
    if verbose:
        img = image_c.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.drawContours(img, curves, -1, (0, 0, 255), 3)
        img = adjust_image(img)
        cv2.imshow('image', img)
        cv2.waitKey(0)

        img = image_c.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.drawContours(img, swatches, -1, (0, 0, 255), 3)
        img = adjust_image(img)
        cv2.imshow('image', img)
        cv2.waitKey(0)

    # Clustering squares/swatches
    clusters = np.zeros(image.shape, dtype=np.uint8)
    for swatch in [np.asarray(scale_contour(swatch, 1 + 1 / 3), dtype=np.int64) for swatch in swatches]:
        cv2.drawContours(clusters, [swatch], -1, [255] * 3, -1)
    clusters = cv2.cvtColor(clusters, cv2.COLOR_RGB2GRAY)
    # Show steps
    if verbose:
        img = adjust_image(clusters)
        copy = img.copy()
        cv2.imshow('image', img)
        cv2.waitKey(0)
    clusters, _hierarchy = cv2.findContours(clusters, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    clusters = [np.asarray(scale_contour(cv2.boxPoints(cv2.minAreaRect(cluster)), 0.975), dtype=np.int64)
                for cluster in clusters]

    if verbose:
        contours_img = cv2.cvtColor(copy, cv2.COLOR_GRAY2RGB)
        contours_img = cv2.drawContours(contours_img, clusters, -1, (0, 255, 0), 3)
        contours_img = adjust_image(contours_img)
        cv2.imshow('image', contours_img)
        cv2.waitKey(0)

        # Filtering clusters using their aspect ratio
    filtered_clusters = []
    for cluster in clusters[:]:
        rectangle = cv2.minAreaRect(cluster)
        width = max(rectangle[1][0], rectangle[1][1])
        height = min(rectangle[1][0], rectangle[1][1])
        ratio = width / height
        if ASPECT_RATIO * 0.9 < ratio < ASPECT_RATIO * 1.1:
            filtered_clusters.append(cluster)
    clusters = filtered_clusters

    # Filtering swatches within cluster
    counts = []
    for cluster in clusters:
        count = 0
        for swatch in swatches:
            if cv2.pointPolygonTest(cluster, contour_centroid(swatch), False) == 1:
                count += 1
        counts.append(count)
    counts = np.array(counts)
    indexes = np.where(np.logical_and(counts >= SWATCHES * 0.5, counts <= SWATCHES * 1.25))[0].tolist()

    colour_checkers = [clusters[i] for i in indexes]

    if additional_data:
        return ColourCheckersDetectionData(colour_checkers, clusters, swatches, image_c)
    else:
        return colour_checkers


def detect_colour_checkers_segmentation(image, size=8, additional_data=False, verbose=False):
    image = adjust_image(image, WORKING_WIDTH)
    swatches_h, swatches_v = SWATCHES_HORIZONTAL, SWATCHES_VERTICAL
    colour_checkers_colours = []
    colour_checkers_data = []
    for colour_checker in extract_colour_checkers_segmentation(image, verbose=verbose):
        colour_checker = np.asarray(colour_checker[..., ::-1], dtype=np.float32) / 255
        width, height = (colour_checker.shape[1], colour_checker.shape[0])
        masks = swatch_masks(width, height, swatches_h, swatches_v, size)

        swatch_colours = []
        for i, mask in enumerate(masks):
            swatch_colours.append(np.mean(colour_checker[mask[0]:mask[1], mask[2]:mask[3], ...], axis=(0, 1)))

        # Colour checker could be in reverse order.
        swatch_neutral_colours = swatch_colours[18:23]
        is_reversed = False
        for i, swatch, in enumerate(swatch_neutral_colours[:-1]):
            if np.mean(swatch) < np.mean(swatch_neutral_colours[i + 1]):
                is_reversed = True
                break

        if is_reversed:
            swatch_colours = swatch_colours[::-1]

        swatch_colours = np.asarray(swatch_colours)

        colour_checkers_colours.append(swatch_colours)
        print_colour_checker_colours(colour_checkers_colours)
        colour_checkers_data.append((colour_checker, masks))

    if additional_data:
        return [ColourCheckerSwatchesData(colour_checkers_colours[i], *colour_checkers_data[i])
                for i, colour_checker_colours in enumerate(colour_checkers_colours)]
    else:
        return colour_checkers_colours


def print_colour_checker_colours(colour_checkers_colours):
    for colour_checker in colour_checkers_colours:
        print("Color checker detected:")
        for swatch in colour_checker:
            print("r: " + str(round(swatch[0]*255/(swatch[0]*255 + swatch[1]*255 + swatch[2]*255), 2)) +
                  " g: " + str(round(swatch[1]*255/(swatch[0]*255 + swatch[1]*255 + swatch[2]*255), 2)) +
                  " b: " + str(round(swatch[2]*255/(swatch[0]*255 + swatch[1]*255 + swatch[2]*255), 2)))


def adjust_image(image, target_width=WORKING_WIDTH):
    """
    This function permit to resize an image from the original size to the target width (and do proportional)
    :param image: the image to resize
    :param target_width: width needed for returned image
    :return: return the resized image
    """
    width, height = image.shape[1], image.shape[0]
    if width < height:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        height, width = width, height

    ratio = width / target_width

    if np.allclose(ratio, 1):
        return image
    else:
        return cv2.resize(image, (int(target_width), int(height / ratio)), interpolation=cv2.INTER_CUBIC)

def as_8_bit_BGR_image(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    return cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def extract_colour_checkers_segmentation(image, verbose=False):
    image = as_8_bit_BGR_image(adjust_image(image))
    colour_checkers = []
    for colour_checker in colour_checkers_coordinates_segmentation(image, verbose=verbose):
        colour_checker = crop_and_level_image_with_rectangle(image, cv2.minAreaRect(colour_checker))
        width, height = (colour_checker.shape[1], colour_checker.shape[0])
        if width < height:
            colour_checker = cv2.rotate(colour_checker, cv2.ROTATE_90_CLOCKWISE)
        colour_checkers.append(colour_checker)
    return colour_checkers


def is_square(contour, tolerance=0.015):
    return cv2.matchShapes(contour, np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), cv2.CONTOURS_MATCH_I2, 0.0) < tolerance


def scale_contour(contour, factor):
    centroid = np.asarray(contour_centroid(contour), dtype=np.int64)
    scaled_contour = (np.asarray(contour, dtype=np.float32) - centroid) * factor + centroid
    return scaled_contour


def contour_centroid(contour):
    moments = cv2.moments(contour)
    centroid = np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])
    return centroid[0], centroid[1]


def crop_and_level_image_with_rectangle(image, rectangle):
    width, height = image.shape[1], image.shape[0]
    width_r, height_r = rectangle[1]
    centroid = np.asarray(contour_centroid(cv2.boxPoints(rectangle)), dtype=np.int64)
    centroid = centroid[0], centroid[1]
    angle = rectangle[-1]

    if angle < -45:
        angle += 90
        width_r, height_r = height_r, width_r

    width_r, height_r = np.asarray([width_r, height_r], dtype=np.int64)
    M_r = cv2.getRotationMatrix2D(centroid, angle, 1)
    image_r = cv2.warpAffine(image, M_r, (width, height), cv2.INTER_CUBIC)
    image_c = cv2.getRectSubPix(image_r, (width_r, height_r), (centroid[0], centroid[1]))
    return image_c


def swatch_masks(width, height, swatches_h, swatches_v, size):
    size = int(size / 2)
    masks = []
    offset_h = width / swatches_h / 2
    offset_v = height / swatches_v / 2
    for j in np.linspace(offset_v, height - offset_v, swatches_v):
        for i in np.linspace(offset_h, width - offset_h, swatches_h):
            masks.append(np.asarray([j - size, j + size, i - size, i + size], dtype=np.int64))
    return masks

