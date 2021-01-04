# Color checker detector

The color checker detector aims to detect a color checker in an image and save the parameters given by the camera 
in a csv file.

The parameters saved aim to do some statistics on camera color, in order to define good parameters on the camera to have
the best image as possible.

## Installation

The Color checker detector uses Python3, OpenCV and Pandas. The guideline in order to install it are summarized at root.

## Usage

### Standalone

It is possible to launch the program standalone using bash command:

```bash
python color_checker_detection.py -i data/color_checkers/image.png
```

#### Required arguments
- -i stands for the input image. (path)
  
#### Not required arguments
- -o stands for output, the output is a csv file. If it does not exist, it will automatically create one with the name 
  given. If it already exists, it will add new lines to the csv. (path)
- -v stands for verbose, if you want to see resulted image. (boolean)
- -s stands for steps. It will show every steps on the color checker detection, enabling easy debugging. (boolean)
- -t stands for thresholds. It might be a csv file where all values of each color admitted are stored. The program 
return an error exit if at least one value is wrong on the image. (path)

## Examples

## Author

Loïc Kerboriou