# Blur detection

The blur detector aims to give a score at an image depending on the blur in it. The highest the score is, the cleaner 
the image is.

## Installation

The Blur detector uses Python3, OpenCV and Pandas. The guideline in order to install it are summarized at root.

## Usage

### Standalone

It is possible to launch the program standalone using bash command:

```bash
python detect_blur.py -i data/blur/image.png 
```

#### Required arguments
- -i stands for the input image. (path)
  
#### Not required arguments
- -o stands for output, the output is a csv file. If it does not exist, it will automatically create one with the name 
  given. If it already exists, it will add new lines to the csv. (path)
- -v stands for verbose, if you want to see resulted image. (boolean)
- -t stands for thresholds. It might be a csv file where all values of each color admitted are stored. The program 
return an error exit if at least one value is wrong on the image. (path)

## Examples

## Author

Loïc Kerboriou