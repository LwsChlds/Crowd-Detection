import numpy as np
import matplotlib.pyplot as plt
import configparser


def postprocess(prediction, IMG_name, Length, Height, Box_Size, numBoxes, detection_value, coverage, maxGroups):

    # Set variables
    LOG_PARA = 2550.0
    groupingNum = 0
    RGB_white = (1., 1., 1.)
    RGB_black = (0., 0., 0.)
    RGB_red = (1., 0., 0.)


    # Arrays used in the file:

    # stores the RGB values of each pixel in the image
    pixels = [[RGB_white for y in range(Length)] for x in range(Height)] 
    
    # stores what group each box is in
    boxes = [[-1 for y in range(int(Length / Box_Size))] for x in range(int(Height / Box_Size))] 

    # stores the estimated number of pedestrians in each box 
    values = [[-1 for y in range(int(Length / Box_Size))] for x in range(int(Height / Box_Size))] 

    # stores the number of boxes in each grouping of boxes
    groups = [0 for rows in range(maxGroups)]

    # sotres the estimated number of pedestrians in each grouping
    groupVal = [0 for rows in range(maxGroups)]


    # a recursive function used to chain together any groups of boxes that are next to eachother
    def checkADJ(i, j, numCells):
        if j > 0:
            if boxes[i][j - 1] == 0:
                boxes[i][j - 1] = boxes[i][j]
                numCells += 1
                numCells += checkADJ(i, j - 1, numCells) - numCells
        if i > 0 and j > 0:
            if boxes[i - 1][j - 1] == 0:
                boxes[i - 1][j - 1] = boxes[i][j]
                numCells += 1
                numCells += checkADJ(i - 1, j - 1, numCells) - numCells
        if i < int(Height / Box_Size) - 1 and j > 0:
            if boxes[i + 1][j - 1] == 0:
                boxes[i + 1][j - 1] = boxes[i][j]
                numCells += 1
                numCells += checkADJ(i + 1, j - 1, numCells) - numCells
        if j < int(Length / Box_Size) - 1:
            if boxes[i][j + 1] == 0:
                boxes[i][j + 1] = boxes[i][j]
                numCells += 1
                numCells += checkADJ(i, j + 1, numCells) - numCells
        if i > 0 and j < int(Length / Box_Size) - 1:
            if boxes[i - 1][j + 1] == 0:
                boxes[i - 1][j + 1] = boxes[i][j]
                numCells += 1
                numCells += checkADJ(i - 1, j + 1, numCells) - numCells
        if i < int(Height / Box_Size) - 1 and j < int(Length / Box_Size) - 1:
            if boxes[i + 1][j + 1] == 0:
                boxes[i + 1][j + 1] = boxes[i][j]
                numCells += 1
                numCells += checkADJ(i + 1, j + 1, numCells) - numCells
        if i < int(Height / Box_Size) - 1:
            if boxes[i + 1][j] == 0:
                boxes[i + 1][j] = boxes[i][j]
                numCells += 1
                numCells += checkADJ(i + 1, j, numCells) - numCells
        if i > 0:
            if boxes[i - 1][j] == 0:
                boxes[i - 1][j] = boxes[i][j]
                numCells += 1
                numCells += checkADJ(i - 1, j, numCells) - numCells
        return numCells

    # looping over each box's individal pixels to see if it contains enough certanty of pedestrians to be recognised
    for j in range(int(Length / Box_Size)):
        for i in range(int(Height / Box_Size)):
            sumPedestrians = 0
            numDetectedPixels = 0
            for n in range(Box_Size):
                for m in range(Box_Size):
                    sumPedestrians += (prediction.squeeze()[i * Box_Size + n][j * Box_Size + m] / LOG_PARA)
                    if detection_value < (prediction.squeeze()[i * Box_Size + n][j * Box_Size + m] / LOG_PARA):
                        numDetectedPixels = numDetectedPixels + 1
            values[i][j] = sumPedestrians
            if numDetectedPixels / (Box_Size * Box_Size) > coverage:
                boxes[i][j] = 0

    # looping over each box to chain adjacent boxes of pedestrians together 
    for b in range(int(Length / Box_Size)):
        for a in range(int(Height / Box_Size)):
            if boxes[a][b] == 0:
                groupingNum += 1
                boxes[a][b] = groupingNum
                groups[groupingNum] = checkADJ(a, b, 0)

            # if there are enough boxes together it is recognised as a crowd and drawn onto the image

            # detects smaller groups in red
            if groups[boxes[a][b]] > numBoxes/4:
                groupVal[boxes[a][b]] += values[a][b]
                for c in range(Box_Size):
                    if a < int(Height / Box_Size) - 1:
                        if boxes[a + 1][b] != boxes[a][b]:
                            pixels[a * Box_Size + Box_Size - 1][b * Box_Size + c] = RGB_red
                    if a > 0:
                        if boxes[a - 1][b] != boxes[a][b]:
                            pixels[a * Box_Size][b * Box_Size + c] = RGB_red
                    if b < int(Length / Box_Size) - 1:
                        if boxes[a][b + 1] != boxes[a][b]:
                            pixels[a * Box_Size + c][b * Box_Size + Box_Size - 1] = RGB_red
                    if b > 0:
                        if boxes[a][b - 1] != boxes[a][b]:
                            pixels[a * Box_Size + c][b * Box_Size] = RGB_red

            # detects larger groups in black
            if groups[boxes[a][b]] > numBoxes:
                groupVal[boxes[a][b]] += values[a][b]
                for c in range(Box_Size):
                    if a < int(Height / Box_Size) - 1:
                        if boxes[a + 1][b] != boxes[a][b]:
                            pixels[a * Box_Size + Box_Size - 1][b * Box_Size + c] = RGB_black
                    if a > 0:
                        if boxes[a - 1][b] != boxes[a][b]:
                            pixels[a * Box_Size][b * Box_Size + c] = RGB_black
                    if b < int(Length / Box_Size) - 1:
                        if boxes[a][b + 1] != boxes[a][b]:
                            pixels[a * Box_Size + c][b * Box_Size + Box_Size - 1] = RGB_black
                    if b > 0:
                        if boxes[a][b - 1] != boxes[a][b]:
                            pixels[a * Box_Size + c][b * Box_Size] = RGB_black



    # any large enough group has it's estimated pedestrian count printed
    for g in range(maxGroups):
        if groupVal[g] > 0:
            print("Group " + str(g) + " = " + str(groupVal[g]))

    # print the pixels RBJ values onto an image 
    outputIMG = np.asarray(pixels)
    plt.figure(figsize=(int(Length/100),int(Height/100)))
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.imshow(outputIMG)
    plt.savefig(IMG_name)


def main():
    config = configparser.ConfigParser()
    config.read('config.txt')
    # Default values
    Length = 960 # the amount of pixels the input/output length contains
    Height = 540 # the amount of pixels the input/output height contains
    Box_Size = 5 # the amount of pixels each area the image is split into will contain
    numBoxes = 100 # the number of boxes that a group has to contain to be recognised as a crowd
    detection_value = 0.001 # the certanty each pixel has to be to seen as a person
    coverage = 0.4 # the % area of a box that has to be detected as a person 
    maxGroups = 200 # the max number of groups the system can handle

    # if the values are present in the config file read from file
    # if they are not present it uses the system default values instead
    if config.get('properties', 'Length', fallback=0) != 0:
        Length = int(config.get('properties', 'Length'))
    if config.get('properties', 'Height', fallback=0) != 0:
        Height = int(config.get('properties', 'Height'))
    if config.get('properties', 'Box_Size', fallback=0) != 0:
        Box_Size = int(config.get('properties', 'Box_Size'))
    if config.get('properties', 'numBoxes', fallback=0) != 0:
        numBoxes = int(config.get('properties', 'numBoxes'))
    if config.get('properties', 'detection_value', fallback=0) != 0:
        detection_value = float(config.get('properties', 'detection_value'))
    if config.get('properties', 'coverage', fallback=0) != 0:
        coverage = float(config.get('properties', 'coverage'))
    if config.get('properties', 'maxGroups', fallback=0) != 0:
        maxGroups = int(config.get('properties', 'maxGroups'))

    # essential values that cannot be defaulted will crash the program if not present
    if config.get('properties', 'outputIMG', fallback=0) != 0:
        IMG_name = str(config.get('properties', 'outputIMG'))
    else:
        print("No outputIMG was found in the spec file")
    if config.get('properties', 'inputFile', fallback=0) != 0:
        inputFile = str(config.get('properties', 'inputFile'))
    else:
        print("No inputFile was found in the spec file")

    prediction = (np.loadtxt(inputFile, dtype=float))
    postprocess(prediction, IMG_name, Length, Height, Box_Size, numBoxes, detection_value, coverage, maxGroups)


if __name__ == "__main__":
    main()
