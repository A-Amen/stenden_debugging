import numpy as np
from copy import deepcopy

NoneType = type(None)

# You can copy this code to your personal pipeline project or execute it here.

def swap(coords: np.ndarray):
    """
    This method will flip the x and y coordinates in the coords array.

    :param coords: A numpy array of bounding box coordinates with shape [n,5] in format:
        ::

            [[x11, y11, x12, y12, classid1],
             [x21, y21, x22, y22, classid2],
             ...
             [xn1, yn1, xn2, yn2, classid3]]

    :return: The new numpy array where the x and y coordinates are flipped.

    **This method is part of a series of debugging exercises.**
    **Each Python method of this series contains bug that needs to be found.**

    | ``1   Can you spot the obvious error?``
    | ``2   After fixing the obvious error it is still wrong, how can this be fixed?``

    >>> import numpy as np
    >>> coords = np.array([[10, 5, 15, 6, 0],
    ...                    [11, 3, 13, 6, 0],
    ...                    [5, 3, 13, 6, 1],
    ...                    [4, 4, 13, 6, 1],
    ...                    [6, 5, 13, 16, 1]])
    >>> swapped_coords = swap(coords)

    The example demonstrates the issue. The returned swapped_coords are expected to have swapped
    x and y coordinates in each of the rows.

    Answer:
    | ``1. The first error is the swapping of the first two cells, i.e. x11 <-> y11 , x21 <-> y21 etc.
            - Change the second coords[:, 1] to coords[:, 0]
    | ``2. Even after the error, the function is still wrong, as the assignment happens sequentially, based on the same input. 
            - To simplify, coords[:, 0] = coords[:, 1] executes first, so coords[:,0] is now [5,3,3,4,5].
            - Then coords[:, 1] = coords[:, 0] now reads from this updated coords[:, 0], thus duplicating itself. 
            - This current form of swapping is highly unreadable.
            - To fix this, a deep copy is an obvious workaround with the current list slicing method, with import copy. 
            - An alternate way would be to utilize for loops and would probably be frowned upon by some.
    """
#     List slicing method
    x1_coords = deepcopy(coords[:, 0])
    y1_coords = deepcopy(coords[:, 1])
    x2_coords = deepcopy(coords[:, 2])
    y2_coords = deepcopy(coords[:, 3])

    coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3], = y1_coords, x1_coords, y2_coords, x2_coords
    #     For loop method(uncomment below code + comment above code to check.)
    # for index in range(coords.shape[1]):
    #     # Swap x11/y11, x21/y21 etc.
    #     coords[index][0] = coords[index][0] + coords[index][1]
    #     coords[index][1] = coords[index][0] - coords[index][1]
    #     coords[index][0] = coords[index][0] - coords[index][1]
    #     # Swap x11/y11, x21/y21 etc.
    #     coords[index][2] = coords[index][2] + coords[index][3]
    #     coords[index][3] = coords[index][2] - coords[index][3]
    #     coords[index][2] = coords[index][2] - coords[index][3]
    return coords


coords = np.array([[10, 5, 15, 6, 0],
                   [11, 3, 13, 6, 0],
                   [5, 3, 13, 6, 1],
                   [4, 4, 13, 6, 1],
                   [6, 5, 13, 16, 1]])
print(coords)
swapped_coords = swap(coords)
print(swapped_coords)