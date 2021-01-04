# TEST COMMAND
# python detect_blur.py -i test/blur/ -v

# import the necessary packages
import argparse
import cv2
import pandas as pd


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input image")
    ap.add_argument("-o", "--output", required=False, help="Path to output csv file")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Display resulting images and wait for user validation")
    ap.add_argument("-t", "--thresholds", required=False, help="Threshold file (csv)")
    args = vars(ap.parse_args())

    # load the image, clone it, and setup the mouse callback function
    img = cv2.imread(args["input"])
    clone = img.copy()

    gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm_original = 10000 * variance_of_laplacian(gray_original / gray_original.mean())

    text = "Score: "
    print("Original image score: " + str(fm_original) + ".")

    if args["verbose"]:
        # show the image
        cv2.putText(img, "{}: {:.2f}".format(text, fm_original), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        cv2.imshow("Original image", img)
        cv2.waitKey(0)

    flag = False
    try:
        f = open(args["output"])
    # Do something with the file
    except IOError:
        flag = True
    # Store results
    if flag:
        print("CSV file not created, creation of the file")
        output = {"Blur original": fm_original}
        df = pd.DataFrame.from_records([output])
    else:
        df = pd.read_csv(args["output"])
        df["Blur original"] = fm_original

    # Register results in csv format
    df.to_csv(args["output"], index=False)


    error = 0
    # Check Thresholds
    if args["thresholds"]:
        thresholds = pd.read_csv(args["thresholds"])
        if thresholds["Blur cropped"][0] - 3 * thresholds["Blur cropped"][1] > df["Blur cropped"][0]:
            print("ERROR: blur cropped")
            error = 1
        if thresholds["Blur original"][0] - 3 * thresholds["Blur original"][1] > df["Blur original"][0]:
            print("ERROR: blur original")
            error = error + 2

    # Finish, exit code with error; printing fail or pass
    if error != 0:
        print("Result : FAIL, error:" + str(error))
        exit(-error)
    else:
        print("Result : PASS")
        exit(0)


if __name__ == "__main__":
    main()
