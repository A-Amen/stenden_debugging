import csv

import matplotlib.pyplot as plt
import numpy as np

NoneType = type(None)

# You can copy this code to your personal pipeline project or execute it here.
def plot_data(csv_file_path: str):
    """
    This code plots the precision-recall curve based on data from a .csv file,
    where precision is on the x-axis and recall is on the y-axis.
    It it not so important right now what precision and recall means.

    :param csv_file_path: The CSV file containing the data to plot.


    **This method is part of a series of debugging exercises.**
    **Each Python method of this series contains bug that needs to be found.**

    | ``1   For some reason the plot is not showing correctly, can you find out what is going wrong?``
    | ``2   How could this be fixed?``

    This example demonstrates the issue.
    It first generates some data in a csv file format and the plots it using the ``plot_data`` method.
    If you manually check the coordinates and then check the plot, they do not correspond.

    >>> f = open("data_file.csv", "w")
    >>> w = csv.writer(f)
    >>> _ = w.writerow(["precision", "recall"])
    >>> w.writerows([[0.013,0.951],
    ...              [0.376,0.851],
    ...              [0.441,0.839],
    ...              [0.570,0.758],
    ...              [0.635,0.674],
    ...              [0.721,0.604],
    ...              [0.837,0.531],
    ...              [0.860,0.453],
    ...              [0.962,0.348],
    ...              [0.982,0.273],
    ...              [1.0,0.0]])
    >>> f.close()
    >>> plot_data('data_file.csv')

    Answer:
    | ``1. First, there is an error when reading the file, at least on windows. w.writerow() appends a new newline after each entry,
            - This causes results to have empty arrays like [[], [0.013, 0.951], ...]
            - The values are read in as strings, not floats. So matplotlib has issues when ordering these plots(results.append([float(element) for element in row]))

        The second issue is that the plt.plot(...) call uses the x-axis for Y-axis in its plot and vice-versa. This does not make a change in the curve,
        but the norm should be [x,y] when passing coordinates to .plot(x-points, y-points). The changes are as follows
            - It should be `plt.plot(results[:, 0], results[:, 1])` instead of `plt.plot(results[:, 1], results[:, 0])`
            - Additionally, the xlabel and ylabel should be swapped.
        
    """
    # load data
    results = []
    with open(csv_file_path, newline='\n') as result_csv:
        csv_reader = csv.reader(result_csv, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            results.append([float(element) for element in row])
            
        results = np.stack(results)

    # plot precision-recall curve
    plt.plot(results[:, 0], results[:, 1])
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.show()
    plt.savefig("fig.png")

f = open("data_file.csv", "w")
w = csv.writer(f)
_ = w.writerow(["precision", "recall"])
w.writerows([[0.013,0.951],
             [0.376,0.851],
             [0.441,0.839],
             [0.570,0.758],
             [0.635,0.674],
             [0.721,0.604],
             [0.837,0.531],
             [0.860,0.453],
             [0.962,0.348],
             [0.982,0.273],
             [1.0,0.0]])
f.close()
plot_data('data_file.csv')
