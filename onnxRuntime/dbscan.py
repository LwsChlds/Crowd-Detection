from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt


def postprocess(prediction, epsilon, min_samples, outputIMG):
    LOG_PARA = 2550.0
    # store the x and y values of the pixels where a person is detected
    y_Axis, x_Axis = (np.where((prediction / LOG_PARA) > 0.001))
    detection = np.array(list(map(lambda i: [y_Axis[i],x_Axis[i]], range(len(y_Axis)))))


    # Compute DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(detection)
    labels = db.labels_

    # Generate scatter plot for training data
    colour_list = np.array(['green', 'blue', 'red', 'yellow',  'orange',  'magenta', 'cyan', 'purple', 'black'])

    # Remove the noise
    range_max = len(detection)
    detection = np.array([detection[i] for i in range(0, range_max) if labels[i] != -1])
    labels = np.array([labels[i] for i in range(0, range_max) if labels[i] != -1])

    no_clusters = len(np.unique(labels) )
    no_noise = np.sum(np.array(labels) == -1, axis=0)

    # print the values for each grouping
    for i in range(no_clusters):
        print(colour_list[i] + " = " + str(np.sum(np.array(labels) == i, axis=0)))

    print('Estimated no. of clusters: %d' % no_clusters)
    print('Estimated no. of noise points: %d' % no_noise)
    
    # Generate scatter plot for training data
    colors = list(map(lambda x: '#000000' if x == -1 else '#b40426', labels))
    plt.scatter(detection[:,1], detection[:,0], c=colour_list[labels], marker="o", picker=True)
    plt.title(f'Crowd detection')
    plt.xlabel('Axis X[0]')
    plt.ylabel('Axis X[1]')
    plt.ylim(540, 0)

    plt.savefig(outputIMG)

def main():
    prediction = (np.loadtxt("img105001.txt", dtype=float))
    outputIMG = "crowdClusters.jpg"
    postprocess(prediction, epsilon=20, min_samples=500, outputIMG="crowdClusters.jpg")


if __name__ == "__main__":
    main()
