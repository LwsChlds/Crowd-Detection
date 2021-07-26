import numpy as np
from PIL import Image
import configparser

def preprocess(inputIMG, config):
  
    MEAN = [0.43476477, 0.44504763, 0.43252817]
    STD = [0.20490805, 0.19712372, 0.20312176]

    Length = 960 # the amount of pixels the input/output length contains
    Height = 540 # the amount of pixels the input/output height contains

    # if the values are present in the config file read from file
    # if they are not present it uses the system default values instead
    if config.get('properties', 'Length', fallback=0) != 0:
        Length = int(config.get('properties', 'Length'))
    if config.get('properties', 'Height', fallback=0) != 0:
        Height = int(config.get('properties', 'Height'))

    normalised = np.full(shape=(3, Height, Length), fill_value=128.0)

    # resizing the image whilst keeping original proportions
    basewidth = Length
    img = Image.open(inputIMG)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)

    array = np.array(img)

    # preprocessing data
    for height in range(540):
        for width in range(960):
            for RGB in range(3):
                value = array[height][width][RGB]
                value = value + 3.75
                value = value/255
                value = value - MEAN[RGB]
                normalised[RGB][height][width] = (value/STD[RGB])

    # getting the array into the (1, 3, Height, Length) shape
    normalised = normalised[np.newaxis, ...]

    return normalised
