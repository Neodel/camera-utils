# TEST COMMAND
# python color_checker_detection.py -i test/color_checkers/ -v

# import the necessary packages
import os
import argparse
import cv2
import numpy as np
import pandas as pd

from detutils import adjust_image, detect_colour_checkers_segmentation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input image")
    ap.add_argument("-o", "--output", required=False, help="path to output csv file")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Display resulting images and wait for user validation")
    ap.add_argument("-s", "--steps", action="store_true", help="Show iteration of detection")
    ap.add_argument("-t", "--thresholds", required=False, help="Threshold file (csv)")
    args = vars(ap.parse_args())

    # Load image
    img = cv2.imread(args["input"])
    img = adjust_image(img)

    # Print images in the first time in order to see
    if args["verbose"]:
        cv2.imshow('image', img)
        cv2.waitKey(0)

    Swatches = []

    # Apply algorithm and print color_checkers images
    for swatches, colour_checker, masks in detect_colour_checkers_segmentation(img, additional_data=True,
                                                                               verbose=args["steps"]):
        Swatches.append(swatches)
        # Using the additional data to plot the colour checker and masks.
        masks_i = np.zeros(colour_checker.shape)
        for i, mask in enumerate(masks):
            masks_i[mask[0]:mask[1], mask[2]:mask[3], ...] = 1
        if args["verbose"]:
            colour_checker = cv2.cvtColor(colour_checker, cv2.COLOR_BGR2RGB)
            cv2.imshow('image', (np.clip(colour_checker + masks_i * 0.25, 0, 1)))
            cv2.waitKey(0)

    try:
        swatches
    except NameError:
        swatches = None

    if swatches is None:
        print("Color checker not detected.")
        exit(-1)

    # Save color of the last colour checker detected
    output = {}
    i = 0
    if args["thresholds"]:
        thresholds = pd.read_csv(args["thresholds"])

    error = 0
    count_error = 0
    for line in swatches:
        output["r_Color_" + str(i)] = round(line[0] * 255 / (line[0] * 255 + line[1] * 255 + line[2] * 255), 3)
        output["g_Color_" + str(i)] = round(line[1] * 255 / (line[0] * 255 + line[1] * 255 + line[2] * 255), 3)
        output["b_Color_" + str(i)] = round(line[2] * 255 / (line[0] * 255 + line[1] * 255 + line[2] * 255), 3)
        if args["thresholds"]:
            if not thresholds["r_Color_" + str(i)][0] - 3 * thresholds["r_Color_" + str(i)][1] < \
                   output["r_Color_" + str(i)] < \
                   thresholds["r_Color_" + str(i)][0] + 3 * thresholds["r_Color_" + str(i)][1]:
                error = i
                count_error += 1
                print("ERROR: red color on color " + str(i))
            if not thresholds["g_Color_" + str(i)][0] - 3 * thresholds["g_Color_" + str(i)][1] < \
                   output["g_Color_" + str(i)] < \
                   thresholds["g_Color_" + str(i)][0] + 3 * thresholds["g_Color_" + str(i)][1]:
                error = i + 24
                count_error += 1
                print("ERROR: green color on color " + str(i))
            if not thresholds["b_Color_" + str(i)][0] - 3 * thresholds["b_Color_" + str(i)][1] < \
                   output["b_Color_" + str(i)] < \
                   thresholds["b_Color_" + str(i)][0] + 3 * thresholds["b_Color_" + str(i)][1]:
                error = i + 2 * 24
                count_error += 1
                print("ERROR: blue color on color " + str(i))
        i += 1
    flag = False
    try:
        f = open(args["output"])
    except IOError:
        flag = True

    if flag:
        print("CSV file not created, creation of the file")
        df = pd.DataFrame.from_records([output])
    else:
        df = pd.read_csv(args["output"])
        df = df.append(output, ignore_index=True)

    df.to_csv(args["output"], index=False)

    if error != 0:
        print("Result : FAIL, error:" + str(error))
        exit(-error)
    else:
        print("Result : PASS")
        exit(0)


if __name__ == "__main__":
    main()
