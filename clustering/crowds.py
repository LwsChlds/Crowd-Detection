import numpy as np
import matplotlib.pyplot as plt
import configparser




# Default values
Length = 960
Height = 540
Box_Size = 5
numBoxes = 100
detection_value = 0.001
coverage = 0.4
maxGroups = 200
dataFile = 'onnxData_105.txt'
IMG_name = "image.jpg"

# Reading from a config file
config = configparser.ConfigParser()

config.read('config.txt')

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
if config.get('properties', 'dataFile', fallback=0) != 0:
    dataFile = config.get('properties', 'dataFile')
if config.get('properties', 'IMG_name', fallback=0) != 0:
    IMG_name = config.get('properties', 'IMG_name')

# Set variables
LOG_PARA = 2550.0
groupingNum = 0
RGB_white = (1., 1., 1.)
RGB_black = (0., 0., 0.)

prediction = (np.loadtxt(dataFile, dtype=float))

# Arrays used in the file
groups = [0 for rows in range(maxGroups)]
groupVal = [0 for rows in range(maxGroups)]
pixels = [[RGB_white for y in range(Length)] for x in range(Height)]
boxes = [[-1 for y in range(int(Length / Box_Size))] for x in range(int(Height / Box_Size))]
values = [[-1 for y in range(int(Length / Box_Size))] for x in range(int(Height / Box_Size))]




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

for b in range(int(Length / Box_Size)):
    for a in range(int(Height / Box_Size)):
        if boxes[a][b] == 0:
            groupingNum += 1
            boxes[a][b] = groupingNum
            groups[groupingNum] = checkADJ(a, b, 0)

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

for g in range(maxGroups):
    if groupVal[g] > 0:
        print("Group " + str(g) + " = " + str(groupVal[g]))

outputIMG = np.asarray(pixels)

plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
plt.margins(0, 0)

plt.imshow(outputIMG)
plt.savefig(IMG_name)
