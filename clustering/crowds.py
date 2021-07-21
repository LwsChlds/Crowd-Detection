from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

groups = [ 0 for rows in range(200)]
groupVal = [ 0 for rows in range(200)]
LOG_PARA = 2550.0

Box_Size = 5
Length = 960
Height = 540

prediction = (np.loadtxt('onnxData_105.txt', dtype=float))
pixels = [ [ (1., 1., 1.) for y in range( Length ) ] for x in range( Height ) ]
boxes = [ [ -1 for y in range( int(Length/Box_Size) ) ] for x in range( int(Height/Box_Size) ) ]
values = [ [ -1 for y in range( int(Length/Box_Size) ) ] for x in range( int(Height/Box_Size) ) ]

groupingNum = 0

def checkADJ (i, j, numCells):
    if j > 0:
        if boxes[i][j-1] == 0:
            boxes[i][j-1] = boxes[i][j]
            numCells += 1
            numCells += checkADJ (i, j-1, numCells) - numCells
    if i > 0 and j > 0:
        if boxes[i-1][j-1] == 0:
            boxes[i-1][j-1] = boxes[i][j]
            numCells += 1
            numCells += checkADJ (i-1, j-1, numCells) - numCells
    if i < int(Height/Box_Size)-1 and j > 0:
        if boxes[i+1][j-1] == 0:
            boxes[i+1][j-1] = boxes[i][j]
            numCells += 1
            numCells += checkADJ (i+1, j-1, numCells) - numCells
    if j < int(Length/Box_Size)-1:
        if boxes[i][j+1] == 0:
            boxes[i][j+1] = boxes[i][j]
            numCells += 1
            numCells += checkADJ (i, j+1, numCells) - numCells
    if i > 0 and j < int(Length/Box_Size)-1:
        if boxes[i-1][j+1] == 0:
            boxes[i-1][j+1] = boxes[i][j]
            numCells += 1
            numCells += checkADJ (i-1, j+1, numCells) - numCells
    if i < int(Height/Box_Size)-1 and j < int(Length/Box_Size)-1:
        if boxes[i+1][j+1] == 0:
            boxes[i+1][j+1] = boxes[i][j]
            numCells += 1
            numCells += checkADJ (i+1, j+1, numCells) - numCells
    if i < int(Height/Box_Size)-1:
        if boxes[i+1][j] == 0:
            boxes[i+1][j] = boxes[i][j]
            numCells += 1
            numCells += checkADJ (i+1, j, numCells) - numCells
    if i > 0:
        if boxes[i-1][j] == 0:
            boxes[i-1][j] = boxes[i][j]
            numCells += 1
            numCells += checkADJ (i-1, j, numCells) - numCells
    return numCells



for j in range(int(Length/Box_Size)):
    for i in range(int(Height/Box_Size)):
        y = 0
        z = 0
        for n in range(Box_Size):
            for m in range(Box_Size):
                y = y + (prediction.squeeze()[i*Box_Size+n][j*Box_Size+m] / LOG_PARA)
                if 0.001 < (prediction.squeeze()[i*Box_Size+n][j*Box_Size+m] / LOG_PARA):
                    z = z + 1
        values[i][j] = y
        if z/(Box_Size*Box_Size) > 0.4:
            boxes[i][j] = 0



for b in range(int(Length/Box_Size)):
    for a in range(int(Height/Box_Size)):
        if boxes[a][b] == 0:
            groupingNum += 1
            boxes[a][b] = groupingNum
            groups[groupingNum] = checkADJ(a, b, 0)

        if groups[boxes[a][b]] > 100:
            groupVal[boxes[a][b]] += values[a][b]
            for c in range(Box_Size):
                if a < int(Height/Box_Size)-1:
                    if boxes[a+1][b] != boxes[a][b]:
                        pixels[a*Box_Size + Box_Size - 1][b*Box_Size+c] = (0., 0., 0.)   
                if a > 0:                  
                    if boxes[a-1][b] != boxes[a][b]:
                        pixels[a*Box_Size][b*Box_Size+c] = (0., 0., 0.)  
                if b < int(Length/Box_Size)-1:
                    if boxes[a][b+1] != boxes[a][b]:
                        pixels[a*Box_Size+c][b*Box_Size + Box_Size - 1] = (0., 0., 0.)
                if b > 0:  
                    if boxes[a][b-1] != boxes[a][b]:
                        pixels[a*Box_Size+c][b*Box_Size] = (0., 0., 0.)  


for g in range(200):
    if groupVal[g] > 0:
        print("Group " + str(g) + " = " + str(groupVal[g]))

outputIMG = np.asarray(pixels)

plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
plt.margins(0, 0)

plt.imshow(outputIMG)
plt.savefig("image.jpg")

