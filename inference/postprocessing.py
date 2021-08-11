from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt


def postprocess(prediction, epsilon, min_samples):
    LOG_PARA = 2550.0
    # store the x and y values of the pixels where a person is detected
    y_Axis, x_Axis = (np.where((prediction / LOG_PARA) > 0.001))
    detection = np.array(list(map(lambda i: [y_Axis[i],x_Axis[i]], range(len(y_Axis)))))
    
    # if no detection is found exit with None values
    if str(np.shape(detection)) == '(0,)':
        print("No detection was found in the image")
        return None, None

    # Compute DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(detection)
    labels = db.labels_

    # Remove the noise
    range_max = len(detection)
    detection = np.array([detection[i] for i in range(0, range_max) if labels[i] != -1])
    labels = np.array([labels[i] for i in range(0, range_max) if labels[i] != -1])

    # code used for printing the number of groups and corresponding detections from each image/frame:
    
    # no_clusters = len(np.unique(labels))    
    # no_noise = np.sum(np.array(labels) == -1, axis=0)

    # print the values for each grouping
    # for i in range(no_clusters):
    #     print(str(i) + " = " + str(np.sum(np.array(labels) == i, axis=0)))
    # print('Estimated no. of clusters: %d' % no_clusters)
    
    # if no clusters are found found exit with None values
    if str(np.shape(detection)) == '(0,)':
        print("No clusters were found in the detection")
        return None, None
    
    return detection, labels

def saveIMG(detection, labels, outputIMG, Length, Height, input, overlay):
    # Generate scatter plot
    # colour_list = np.array(['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple', 'black'])
    # if overlay is set to 1 print the detections onto the original frame
    if overlay == 1:
        plt.imshow(input)
        detection = detection * 4
        plt.scatter(detection[:,1], detection[:,0], c='red', marker="o", picker=True, s=.01, alpha=0.5)
        plt.xlim(0, Length*4)
        plt.ylim(Height*4, 0)
    # if overlay is not set to 1 print the detection individually with larger markers
    else:
        plt.scatter(detection[:,1], detection[:,0], c='red', marker="o", picker=True)
        plt.xlim(0, Length)
        plt.ylim(Height, 0)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.savefig(outputIMG)
    plt.clf()

