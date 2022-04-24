import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

"""
Returns the lower left bound of a pair of points

@param start: The first point of the pair of points
@param end: The second point of the pair of points
@return: Returns the lower left point of the passed points
"""
def get_lower_left_bound(start, end):
    if (len(start) != 2 or len(end) != 2):
        print("Invalid coordinate format")
        return

    min_x = min(start[0], end[0])
    min_y = min(start[1], end[1])

    return [min_x, min_y];

"""
Returns the upper right bound of a pair of points

@param start: The first point of the pair of points
@param end: The second point of the pair of points
@return: Returns the upper right point of the passed points
"""
def get_upper_right_bound(start, end):
    if (len(start) != 2 or len(end) != 2):
        print("Invalid coordinate format")
        return

    max_x = max(start[0], end[0])
    max_y = max(start[1], end[1])

    return [max_x, max_y];

"""
Generates a heatmap with the dwelling times at each point in the plot

Dwelling times are calculated by the time difference between the fastest
an entity can reach the point and the time gap between anchors

@param start: Starting position for an entity with a time gap
@param end: Ending position for an entity with a time gap
@param extent: How far from the passed points to calculate values for
@param grain_size: Granularity of the plot and calculations
@param time: Time gap between the start and end points
@param speed: Maximum speed that an entity can move
"""
def generate_dwelling(start, end, extent, grain_size, time, speed):
    lower_left = get_lower_left_bound(start, end)
    upper_right = get_upper_right_bound(start, end)

    adjusted_lower_left = list(map(lambda x: x - extent, lower_left))
    adjusted_upper_right = list(map(lambda x: x + extent, upper_right))

    length_of_horizontal = adjusted_upper_right[0] - adjusted_lower_left[0]
    length_of_vertical = adjusted_upper_right[1] - adjusted_lower_left[1]

    horizontal_axis_min = adjusted_lower_left[0]
    horizontal_axis_max = adjusted_upper_right[0] 
    horizontal_range = np.arange(horizontal_axis_min, horizontal_axis_max + 0.5, grain_size)

    vertical_axis_min = adjusted_lower_left[1] 
    vertical_axis_max = adjusted_upper_right[1]
    vertical_range = np.arange(vertical_axis_max, vertical_axis_min - 0.5, -grain_size)

    dwellings = []
    labels = []
    max_dwelling = -np.inf

    for i in range(0, len(vertical_range)):
        dwelling_row = []
        label_row = []

        for j in range(0, len(horizontal_range)):
            triangle_length1 = abs(start[0] - horizontal_range[j])
            triangle_length2 = abs(start[1] - vertical_range[i])
            triangle_length3 = abs(end[0] - horizontal_range[j])
            triangle_length4 = abs(end[1] - vertical_range[i])
            
            hypotenuse1 = np.hypot(triangle_length1, triangle_length2)
            hypotenuse2 = np.hypot(triangle_length3, triangle_length4)
            hypotenuse_calculation = time - (hypotenuse1 + hypotenuse2) / speed

            condition = (triangle_length1 == 0 and triangle_length2 == 0) or (triangle_length3 == 0 and triangle_length4 == 0)
            dwelling_row.append(hypotenuse_calculation)
            # label_row.append(f"{hypotenuse_calculation:.1f}" if (hypotenuse_calculation >= 0 and not condition) else "")

            if (hypotenuse_calculation > max_dwelling):
                max_dwelling = hypotenuse_calculation

        dwellings = [dwelling_row] + dwellings
        # labels = [label_row] + labels

    dwellings = np.array(dwellings)
    # labels = np.array(labels)

    
    ax = sns.heatmap(dwellings, annot=False, fmt="", vmin=0.0, vmax=max_dwelling, cmap="Blues")

    ax.set_title("Dwelling Times for a Ship")
    ax.invert_yaxis()
    plt.show()

    return

"""
FLAWED

Generates a contour map without an intersection the dwelling times at each point in the plot

@param start: Starting position for an entity with a time gap
@param end: Ending position for an entity with a time gap
@param extent: How far from the passed points to calculate values for
@param grain_size: Granularity of the plot and calculations
@param time: Time gap between the start and end points
@param speed: Maximum speed that an entity can move
"""
def generate_double_dwelling(start1, end1, start2, end2, extent, grain_size, time, speed):
    lower_left = get_lower_left_bound(start1, end1)
    upper_right = get_upper_right_bound(start1, end1)

    lower_left2 = get_lower_left_bound(start2, end2)
    upper_right2 = get_upper_right_bound(start2, end2)

    lower_left = get_lower_left_bound(lower_left, lower_left2)
    upper_right = get_upper_right_bound(upper_right, upper_right2)


    adjusted_lower_left = list(map(lambda x: x - extent, lower_left))
    adjusted_upper_right = list(map(lambda x: x + extent, upper_right))

    length_of_horizontal = adjusted_upper_right[0] - adjusted_lower_left[0]
    length_of_vertical = adjusted_upper_right[1] - adjusted_lower_left[1]

    horizontal_axis_min = adjusted_lower_left[0]
    horizontal_axis_max = adjusted_upper_right[0] 
    horizontal_range = np.arange(horizontal_axis_min, horizontal_axis_max + 0.5, grain_size)

    vertical_axis_min = adjusted_lower_left[1] 
    vertical_axis_max = adjusted_upper_right[1]
    vertical_range = np.arange(vertical_axis_max, vertical_axis_min - 0.5, -grain_size)

    dwellings = []
    dwellings2 = []
    doverlay = []
    labels = []
    max_dwelling = -np.inf

    for i in range(0, len(vertical_range)):
        dwelling_row = []
        dwelling_row2 = []
        doverlay_row = []
        label_row = []

        for j in range(0, len(horizontal_range)):
            triangle_length1 = abs(start1[0] - horizontal_range[j])
            triangle_length2 = abs(start1[1] - vertical_range[i])
            triangle_length3 = abs(end1[0] - horizontal_range[j])
            triangle_length4 = abs(end1[1] - vertical_range[i])
            
            hypotenuse1 = np.hypot(triangle_length1, triangle_length2)
            hypotenuse2 = np.hypot(triangle_length3, triangle_length4)
            hypotenuse_calculation = time - (hypotenuse1 + hypotenuse2) / speed

            condition = (triangle_length1 == 0 and triangle_length2 == 0) or (triangle_length3 == 0 and triangle_length4 == 0)
            if (hypotenuse_calculation <= 0):
                dwelling_row.append(-np.inf)
            else:
                dwelling_row.append(-hypotenuse_calculation)
            # label_row.append(f"{hypotenuse_calculation:.1f}" if (hypotenuse_calculation >= 0 and not condition) else "")


            triangle_length1 = abs(start2[0] - horizontal_range[j])
            triangle_length2 = abs(start2[1] - vertical_range[i])
            triangle_length3 = abs(end2[0] - horizontal_range[j])
            triangle_length4 = abs(end2[1] - vertical_range[i])
            
            hypotenuse1 = np.hypot(triangle_length1, triangle_length2)
            hypotenuse2 = np.hypot(triangle_length3, triangle_length4)
            hypotenuse_calculation2 = time - (hypotenuse1 + hypotenuse2) / speed
            
            if (hypotenuse_calculation2 <= 0):
                dwelling_row2.append(-np.inf)
            else:
                dwelling_row2.append(hypotenuse_calculation2)


            if (hypotenuse_calculation2 > 0 and hypotenuse_calculation > 0):
                doverlay_row.append(-(hypotenuse_calculation2 + hypotenuse_calculation) / 2)
            else:
                if (hypotenuse_calculation > 0):
                    doverlay_row.append(hypotenuse_calculation)
                elif (hypotenuse_calculation2 > 0):
                    doverlay_row.append(hypotenuse_calculation2) # change this for split colors
                else:
                    doverlay_row.append(-np.inf)

            if (hypotenuse_calculation > max_dwelling):
                max_dwelling = hypotenuse_calculation

            if (hypotenuse_calculation2 > max_dwelling):
                max_dwelling = hypotenuse_calculation2



        dwellings = [dwelling_row] + dwellings
        dwellings2 = [dwelling_row2] + dwellings2
        doverlay = [doverlay_row] + doverlay

    dwellings = np.array(dwellings)
    dwellings2 = np.array(dwellings2)
    doverlay = np.array(doverlay)

    levels = np.arange(-max_dwelling, max_dwelling + 0.0001, grain_size)  
  
    fig, ax = plt.subplots()
    g = plt.contourf(horizontal_range, vertical_range, doverlay, np.arange(-max_dwelling, max_dwelling + grain_size, grain_size), vmin=-max_dwelling, vmax=max_dwelling, cmap="twilight_shifted", alpha=0.8)

    plt.colorbar(g)

    ax.set_title("Dwelling Times for a Ship")

    plt.show()

    return

"""
Generates a contour map with intersection of dwelling times. Intersection is
calculated as an average the the component trajectories' dwelling times.

@param start1: Starting position for an entity with a time gap
@param end1: Ending position for an entity with a time gap
@param start2: Starting position for a second entity with a time gap
@param end2: Ending position for a second entity with a time gap
@param extent: How far from the passed points to calculate values for
@param grain_size: Granularity of the plot and calculations
@param time: Time gap between the start and end points
@param speed: Maximum speed that an entity can move
"""
def generate_double_dwelling_with_intersection(start1, end1, start2, end2, extent, grain_size, time, speed):
    # Calculating bounds for the plot
    lower_left = get_lower_left_bound(start1, end1)
    upper_right = get_upper_right_bound(start1, end1)

    lower_left2 = get_lower_left_bound(start2, end2)
    upper_right2 = get_upper_right_bound(start2, end2)

    lower_left = get_lower_left_bound(lower_left, lower_left2)
    upper_right = get_upper_right_bound(upper_right, upper_right2)

    adjusted_lower_left = list(map(lambda x: x - extent, lower_left))
    adjusted_upper_right = list(map(lambda x: x + extent, upper_right))

    length_of_horizontal = adjusted_upper_right[0] - adjusted_lower_left[0]
    length_of_vertical = adjusted_upper_right[1] - adjusted_lower_left[1]

    horizontal_axis_min = adjusted_lower_left[0]
    horizontal_axis_max = adjusted_upper_right[0] 
    horizontal_range = np.arange(horizontal_axis_min, horizontal_axis_max + 0.5, grain_size)

    vertical_axis_min = adjusted_lower_left[1] 
    vertical_axis_max = adjusted_upper_right[1]
    vertical_range = np.arange(vertical_axis_max, vertical_axis_min - 0.5, -grain_size)

    # Storing values for the contour maps
    dwellings = []
    dwellings2 = []
    doverlay = []
    labels = []
    max_dwelling = -np.inf

    # Loops through all points in the graph to calculate the dwelling time
    for i in range(0, len(vertical_range)):
        dwelling_row = []
        dwelling_row2 = []
        doverlay_row = []
        label_row = []

        for j in range(0, len(horizontal_range)):
            # Calculating the dwelling time for the first gap
            triangle_length1 = abs(start1[0] - horizontal_range[j])
            triangle_length2 = abs(start1[1] - vertical_range[i])
            triangle_length3 = abs(end1[0] - horizontal_range[j])
            triangle_length4 = abs(end1[1] - vertical_range[i])
            
            hypotenuse1 = np.hypot(triangle_length1, triangle_length2)
            hypotenuse2 = np.hypot(triangle_length3, triangle_length4)
            hypotenuse_calculation = time - (hypotenuse1 + hypotenuse2) / speed

            # For graphing style
            if (hypotenuse_calculation <= 0):
                dwelling_row.append(-np.inf)
            else:
                dwelling_row.append(-hypotenuse_calculation)

            # Calculating the dwelling time for the second gap
            triangle_length1 = abs(start2[0] - horizontal_range[j])
            triangle_length2 = abs(start2[1] - vertical_range[i])
            triangle_length3 = abs(end2[0] - horizontal_range[j])
            triangle_length4 = abs(end2[1] - vertical_range[i])
            
            hypotenuse1 = np.hypot(triangle_length1, triangle_length2)
            hypotenuse2 = np.hypot(triangle_length3, triangle_length4)
            hypotenuse_calculation2 = time - (hypotenuse1 + hypotenuse2) / speed
            
            # For graphing style
            if (hypotenuse_calculation2 <= 0):
                dwelling_row2.append(-np.inf)
            else:
                dwelling_row2.append(hypotenuse_calculation2)

            # Data for the intersection between the dwelling times
            if (hypotenuse_calculation2 > 0 and hypotenuse_calculation > 0):
                doverlay_row.append((hypotenuse_calculation2 + hypotenuse_calculation) / 2)
            else:
                doverlay_row.append(-np.inf)

            # Determine max for the colorbar
            if (hypotenuse_calculation > max_dwelling):
                max_dwelling = hypotenuse_calculation

            if (hypotenuse_calculation2 > max_dwelling):
                max_dwelling = hypotenuse_calculation2

        dwellings = [dwelling_row] + dwellings
        dwellings2 = [dwelling_row2] + dwellings2
        doverlay = [doverlay_row] + doverlay

    dwellings = np.array(dwellings)
    dwellings2 = np.array(dwellings2)
    doverlay = np.array(doverlay)

    # Plotting the data points
    fig, ax = plt.subplots(figsize=(12,8))
    levels = np.arange(-max_dwelling, max_dwelling + 0.11, grain_size)  
    
    r = plt.contourf(horizontal_range, vertical_range, dwellings, levels, vmin=-max_dwelling, vmax=max_dwelling + 0.0001, cmap="RdBu", alpha=1)
    b = plt.contourf(horizontal_range, vertical_range, dwellings2, levels, vmin=-max_dwelling, vmax=max_dwelling + 0.0001, cmap="RdBu", alpha=1)
    g = plt.contourf(horizontal_range, vertical_range, doverlay, np.arange(0, max_dwelling + 0.0001, grain_size), vmin=0, vmax=max_dwelling, cmap="Purples", alpha=0.75)

    #cl = plt.clabel(g, inline=False, fontsize=3, colors='black')
    x=[start1[0], start2[0]]
    x1=[end1[0], end2[0]]
    y=[start1[1], start2[1]]
    y1=[end1[1], end2[1]]

    #plt.scatter(x=[start1[0], start2[0], end1[0], end2[0]], y=[start1[1], start2[1], end1[1], end2[1]])
    #plt.plot(x, y, '-o')
    #plt.plot(x1, y1, '-o')


    plt.colorbar(g)
    ax.set_title("Intersection of Dwelling Times")
    plt.savefig('./intersection2.png', format='png', dpi=1200)
    #plt.show()

"""
Generates a contour map with intersection of dwelling times. Intersection is
calculated as an average the the component trajectories' dwelling times.

@param start1: Starting position for an entity with a time gap
@param end1: Ending position for an entity with a time gap
@param start2: Starting position for a second entity with a time gap
@param end2: Ending position for a second entity with a time gap
@param extent: How far from the passed points to calculate values for
@param grain_size: Granularity of the plot and calculations
@param start_t1: Start time for the first trajectory gap
@param start_t2: Start time for the second trajectory gap
@param time: Time gap between the start and end points
@param time2: Time gap between the start and end points
@param speed: Maximum speed that the first entity can move
@param speed2: Maximum speed that the second entity can move
"""
def generate_informative(start1, end1, start2, end2, extent, grain_size, start_t1, start_t2, time, time2, speed, speed2):
    # Calculating bounds for the plot
    lower_left = get_lower_left_bound(start1, end1)
    upper_right = get_upper_right_bound(start1, end1)

    lower_left2 = get_lower_left_bound(start2, end2)
    upper_right2 = get_upper_right_bound(start2, end2)

    lower_left = get_lower_left_bound(lower_left, lower_left2)
    upper_right = get_upper_right_bound(upper_right, upper_right2)

    adjusted_lower_left = list(map(lambda x: x - extent, lower_left))
    adjusted_upper_right = list(map(lambda x: x + extent, upper_right))

    length_of_horizontal = adjusted_upper_right[0] - adjusted_lower_left[0]
    length_of_vertical = adjusted_upper_right[1] - adjusted_lower_left[1]

    horizontal_axis_min = adjusted_lower_left[0]
    horizontal_axis_max = adjusted_upper_right[0] 
    horizontal_range = np.arange(horizontal_axis_min, horizontal_axis_max + 0.5, grain_size)

    vertical_axis_min = adjusted_lower_left[1] 
    vertical_axis_max = adjusted_upper_right[1]
    vertical_range = np.arange(vertical_axis_max, vertical_axis_min - 0.5, -grain_size)


    gap_size = abs(time - time2)
    if (start_t2 > start_t1 + time or start_t1 > start_t2 + time2):
        print("Not in range.")
        return



    gap_start = max(start_t1, start_t2)



    # Storing values for the contour maps
    dwellings = []
    dwellings2 = []
    doverlay = []
    labels = []
    max_dwelling = -np.inf

    # Loops through all points in the graph to calculate the dwelling time
    for i in range(0, len(vertical_range)):
        dwelling_row = []
        dwelling_row2 = []
        doverlay_row = []
        label_row = []

        for j in range(0, len(horizontal_range)):
            # Calculating the dwelling time for the first gap
            triangle_length1 = abs(start1[0] - horizontal_range[j])
            triangle_length2 = abs(start1[1] - vertical_range[i])
            triangle_length3 = abs(end1[0] - horizontal_range[j])
            triangle_length4 = abs(end1[1] - vertical_range[i])
            
            hypotenuse1 = np.hypot(triangle_length1, triangle_length2)
            hypotenuse2 = np.hypot(triangle_length3, triangle_length4)
            hypotenuse_calculation = time - (hypotenuse1 + hypotenuse2) / speed

            t_to = hypotenuse1 / speed
            t_from = hypotenuse2 / speed

            # For graphing style
            if (hypotenuse_calculation <= 0):
                dwelling_row.append(-np.inf)
            else:
                dwelling_row.append(-hypotenuse_calculation)

            # Calculating the dwelling time for the second gap
            triangle_length1 = abs(start2[0] - horizontal_range[j])
            triangle_length2 = abs(start2[1] - vertical_range[i])
            triangle_length3 = abs(end2[0] - horizontal_range[j])
            triangle_length4 = abs(end2[1] - vertical_range[i])
            
            hypotenuse1 = np.hypot(triangle_length1, triangle_length2)
            hypotenuse2 = np.hypot(triangle_length3, triangle_length4)
            hypotenuse_calculation2 = time2 - (hypotenuse1 + hypotenuse2) / speed2
            
            t_to2 = hypotenuse1 / speed2
            t_from2 = hypotenuse2 / speed2 

            # For graphing style
            if (hypotenuse_calculation2 <= 0):
                dwelling_row2.append(-np.inf)
            else:
                dwelling_row2.append(hypotenuse_calculation2)

            # Data for the intersection between the dwelling times
            # Last part of condition checks for whether the other entity has to leave before one has time to get to the location
            # As a result, the only intersections that are kept are plausible meeting locations
            if (hypotenuse_calculation2 > 0 and hypotenuse_calculation > 0 and t_to2 + start_t2 < start_t1 + time - t_from and t_to + start_t1 < start_t2 + time2 - t_from2):
                doverlay_row.append((hypotenuse_calculation2 + hypotenuse_calculation) / 2)
            else:
                doverlay_row.append(-np.inf)

            # Determine max for the colorbar
            if (hypotenuse_calculation > max_dwelling):
                max_dwelling = hypotenuse_calculation

            if (hypotenuse_calculation2 > max_dwelling):
                max_dwelling = hypotenuse_calculation2

        dwellings = [dwelling_row] + dwellings
        dwellings2 = [dwelling_row2] + dwellings2
        doverlay = [doverlay_row] + doverlay

    dwellings = np.array(dwellings)
    dwellings2 = np.array(dwellings2)
    doverlay = np.array(doverlay)

    # Plotting the data points
    fig, ax = plt.subplots(figsize=(12,8))
    levels = np.arange(-max_dwelling, max_dwelling + 0.11, grain_size)  
    
    r = plt.contourf(horizontal_range, vertical_range, dwellings, levels, vmin=-max_dwelling, vmax=max_dwelling + 0.0001, cmap="RdBu", alpha=1)
    b = plt.contourf(horizontal_range, vertical_range, dwellings2, levels, vmin=-max_dwelling, vmax=max_dwelling + 0.0001, cmap="RdBu", alpha=1)
    g = plt.contourf(horizontal_range, vertical_range, doverlay, np.arange(0, max_dwelling + 0.0001, grain_size), vmin=0, vmax=max_dwelling, cmap="Purples", alpha=0.75)

    #cl = plt.clabel(g, inline=False, fontsize=3, colors='black')
    x=[start1[0], start2[0]]
    x1=[end1[0], end2[0]]
    y=[start1[1], start2[1]]
    y1=[end1[1], end2[1]]

    #plt.scatter(x=[start1[0], start2[0], end1[0], end2[0]], y=[start1[1], start2[1], end1[1], end2[1]])
    #plt.plot(x, y, '-o')
    #plt.plot(x1, y1, '-o')


    plt.colorbar(g)
    ax.set_title("Intersection of Dwelling Times")
    plt.savefig('./intersection2.png', format='png', dpi=1200)
    #plt.show()


"""
Generates a contour map with intersection of dwelling times. Intersection is
calculated as an average the the component trajectories' dwelling times.

@param start1: Starting position for an entity with a time gap
@param end1: Ending position for an entity with a time gap
@param start2: Starting position for a second entity with a time gap
@param end2: Ending position for a second entity with a time gap
@param extent: How far from the passed points to calculate values for
@param grain_size: Granularity of the plot and calculations
@param start_t1: Start time for the first trajectory gap
@param start_t2: Start time for the second trajectory gap
@param time: Time gap between the start and end points
@param time2: Time gap between the start and end points
@param speed: Maximum speed that the first entity can move
@param speed2: Maximum speed that the second entity can move
"""
def generate_1(start1, end1, start2, end2, extent, grain_size, start_t1, start_t2, time, time2, speed, speed2):
    # Calculating bounds for the plot
    lower_left = get_lower_left_bound(start1, end1)
    upper_right = get_upper_right_bound(start1, end1)

    lower_left2 = get_lower_left_bound(start2, end2)
    upper_right2 = get_upper_right_bound(start2, end2)

    lower_left = get_lower_left_bound(lower_left, lower_left2)
    upper_right = get_upper_right_bound(upper_right, upper_right2)

    adjusted_lower_left = list(map(lambda x: x - extent, lower_left))
    adjusted_upper_right = list(map(lambda x: x + extent, upper_right))

    length_of_horizontal = adjusted_upper_right[0] - adjusted_lower_left[0]
    length_of_vertical = adjusted_upper_right[1] - adjusted_lower_left[1]

    horizontal_axis_min = adjusted_lower_left[0]
    horizontal_axis_max = adjusted_upper_right[0] 
    horizontal_range = np.arange(horizontal_axis_min, horizontal_axis_max + 0.5, grain_size)

    vertical_axis_min = adjusted_lower_left[1] 
    vertical_axis_max = adjusted_upper_right[1]
    vertical_range = np.arange(vertical_axis_max, vertical_axis_min - 0.5, -grain_size)


    gap_size = abs(time - time2)
    if (start_t2 > start_t1 + time or start_t1 > start_t2 + time2):
        print("Not in range.")
        return



    gap_start = max(start_t1, start_t2)



    # Storing values for the contour maps
    dwellings = []
    dwellings2 = []
    doverlay = []
    labels = []
    max_dwelling = -np.inf
    max_to = -np.inf
    min_to = np.inf

    # Loops through all points in the graph to calculate the dwelling time
    for i in range(0, len(vertical_range)):
        dwelling_row = []
        dwelling_row2 = []
        doverlay_row = []
        label_row = []

        for j in range(0, len(horizontal_range)):
            # Calculating the dwelling time for the first gap
            triangle_length1 = abs(start1[0] - horizontal_range[j])
            triangle_length2 = abs(start1[1] - vertical_range[i])
            triangle_length3 = abs(end1[0] - horizontal_range[j])
            triangle_length4 = abs(end1[1] - vertical_range[i])
            
            hypotenuse1 = np.hypot(triangle_length1, triangle_length2)
            hypotenuse2 = np.hypot(triangle_length3, triangle_length4)
            hypotenuse_calculation = time - (hypotenuse1 + hypotenuse2) / speed

            t_to = hypotenuse1 / speed
            t_from = hypotenuse2 / speed

            # For graphing style
            if (hypotenuse_calculation <= 0):
                dwelling_row.append(-np.inf)
            else:
                dwelling_row.append(-t_to)

            # Calculating the dwelling time for the second gap
            triangle_length1 = abs(start2[0] - horizontal_range[j])
            triangle_length2 = abs(start2[1] - vertical_range[i])
            triangle_length3 = abs(end2[0] - horizontal_range[j])
            triangle_length4 = abs(end2[1] - vertical_range[i])
            
            hypotenuse1 = np.hypot(triangle_length1, triangle_length2)
            hypotenuse2 = np.hypot(triangle_length3, triangle_length4)
            hypotenuse_calculation2 = time2 - (hypotenuse1 + hypotenuse2) / speed2
            
            t_to2 = hypotenuse1 / speed2
            t_from2 = hypotenuse2 / speed2 

            # For graphing style
            if (hypotenuse_calculation2 <= 0):
                dwelling_row2.append(-np.inf)
            else:
                dwelling_row2.append(t_to2)

            # Data for the intersection between the dwelling times
            # Last part of condition checks for whether the other entity has to leave before one has time to get to the location
            # As a result, the only intersections that are kept are plausible meeting locations
            diff_to = max(t_to2 + start_t2, t_to + start_t1) - min(t_to2 + start_t2, t_to + start_t1)
            if (hypotenuse_calculation2 > 0 and hypotenuse_calculation > 0 and t_to2 + start_t2 < start_t1 + time - t_from and t_to + start_t1 < start_t2 + time2 - t_from2):
                doverlay_row.append(diff_to)
                if (diff_to > max_to):
                    max_to = diff_to

                if (diff_to < min_to):
                    min_to = diff_to
            else:
                doverlay_row.append(-np.inf)

            # Determine max for the colorbar
            if (hypotenuse_calculation > max_dwelling):
                max_dwelling = hypotenuse_calculation

            if (hypotenuse_calculation2 > max_dwelling):
                max_dwelling = hypotenuse_calculation2




        dwellings = [dwelling_row] + dwellings
        dwellings2 = [dwelling_row2] + dwellings2
        doverlay = [doverlay_row] + doverlay

    dwellings = np.array(dwellings)
    dwellings2 = np.array(dwellings2)
    doverlay = np.array(doverlay)

    # Plotting the data points
    fig, ax = plt.subplots(figsize=(12,8))
    levels = np.arange(-max_dwelling, max_dwelling + 0.11, grain_size)  
    levels2 = np.arange(min_to, max_to + 0.11, grain_size)  
    
    r = plt.contourf(horizontal_range, vertical_range, dwellings, levels, vmin=-max_dwelling, vmax=max_dwelling + 0.0001, cmap="RdBu", alpha=.75)
    b = plt.contourf(horizontal_range, vertical_range, dwellings2, levels, vmin=-max_dwelling, vmax=max_dwelling + 0.0001, cmap="RdBu", alpha=.75)
    g = plt.contourf(horizontal_range, vertical_range, doverlay, levels2, vmin=min_to, vmax=max_to, cmap="Purples", alpha=.75)

    #cl = plt.clabel(g, inline=False, fontsize=3, colors='black')
    x=[start1[0], start2[0]]
    x1=[end1[0], end2[0]]
    y=[start1[1], start2[1]]
    y1=[end1[1], end2[1]]

    #plt.scatter(x=[start1[0], start2[0], end1[0], end2[0]], y=[start1[1], start2[1], end1[1], end2[1]])
    #plt.plot(x, y, '-o')
    #plt.plot(x1, y1, '-o')


    plt.colorbar(g)
    ax.set_title("Intersection of Dwelling Times")
    plt.savefig('./intersection2.png', format='png', dpi=1200)
    #plt.show()

"""
Generates a contour map with intersection of dwelling times. Intersection is
calculated as the length of the  dwelling times.

@param start1: Starting position for an entity with a time gap
@param end1: Ending position for an entity with a time gap
@param start2: Starting position for a second entity with a time gap
@param end2: Ending position for a second entity with a time gap
@param extent: How far from the passed points to calculate values for
@param grain_size: Granularity of the plot and calculations
@param start_t1: Start time for the first trajectory gap
@param start_t2: Start time for the second trajectory gap
@param time: Time gap between the start and end points
@param time2: Time gap between the start and end points
@param speed: Maximum speed that the first entity can move
@param speed2: Maximum speed that the second entity can move
"""
def generate_2(start1, end1, start2, end2, extent, grain_size, start_t1, start_t2, time, time2, speed, speed2):
    # Calculating bounds for the plot
    lower_left = get_lower_left_bound(start1, end1)
    upper_right = get_upper_right_bound(start1, end1)

    lower_left2 = get_lower_left_bound(start2, end2)
    upper_right2 = get_upper_right_bound(start2, end2)

    lower_left = get_lower_left_bound(lower_left, lower_left2)
    upper_right = get_upper_right_bound(upper_right, upper_right2)

    adjusted_lower_left = list(map(lambda x: x - extent, lower_left))
    adjusted_upper_right = list(map(lambda x: x + extent, upper_right))

    length_of_horizontal = adjusted_upper_right[0] - adjusted_lower_left[0]
    length_of_vertical = adjusted_upper_right[1] - adjusted_lower_left[1]

    horizontal_axis_min = adjusted_lower_left[0]
    horizontal_axis_max = adjusted_upper_right[0] 
    horizontal_range = np.arange(horizontal_axis_min, horizontal_axis_max + 0.5, grain_size)

    vertical_axis_min = adjusted_lower_left[1] 
    vertical_axis_max = adjusted_upper_right[1]
    vertical_range = np.arange(vertical_axis_max, vertical_axis_min - 0.5, -grain_size)


    gap_size = abs(time - time2)
    if (start_t2 > start_t1 + time or start_t1 > start_t2 + time2):
        print("Not in range.")
        return




    # Storing values for the contour maps
    dwellings = []
    dwellings2 = []
    doverlay = []
    labels = []
    max_dwelling = -np.inf
    max_to = -np.inf
    min_to = np.inf
    max_meeting_length = -np.inf

    # Loops through all points in the graph to calculate the dwelling time
    for i in range(0, len(vertical_range)):
        dwelling_row = []
        dwelling_row2 = []
        doverlay_row = []
        label_row = []

        for j in range(0, len(horizontal_range)):
            # Calculating the dwelling time for the first gap
            triangle_length1 = abs(start1[0] - horizontal_range[j])
            triangle_length2 = abs(start1[1] - vertical_range[i])
            triangle_length3 = abs(end1[0] - horizontal_range[j])
            triangle_length4 = abs(end1[1] - vertical_range[i])
            
            hypotenuse1 = np.hypot(triangle_length1, triangle_length2)
            hypotenuse2 = np.hypot(triangle_length3, triangle_length4)
            hypotenuse_calculation = time - (hypotenuse1 + hypotenuse2) / speed

            t_to = hypotenuse1 / speed
            t_from = hypotenuse2 / speed

            dwelling_start_1 = start_t1 + t_to
            dwelling_end_1 = start_t1 + time - t_from

            # For graphing style
            if (hypotenuse_calculation <= 0):
                dwelling_row.append(-np.inf)
            else:
                dwelling_row.append(-t_to)

            # Calculating the dwelling time for the second gap
            triangle_length1 = abs(start2[0] - horizontal_range[j])
            triangle_length2 = abs(start2[1] - vertical_range[i])
            triangle_length3 = abs(end2[0] - horizontal_range[j])
            triangle_length4 = abs(end2[1] - vertical_range[i])
            
            hypotenuse1 = np.hypot(triangle_length1, triangle_length2)
            hypotenuse2 = np.hypot(triangle_length3, triangle_length4)
            hypotenuse_calculation2 = time2 - (hypotenuse1 + hypotenuse2) / speed2
            
            t_to2 = hypotenuse1 / speed2
            t_from2 = hypotenuse2 / speed2 

            dwelling_start_2 = start_t2 + t_to2
            dwelling_end_2 = start_t2 + time2 - t_from2

            meeting_start = max(dwelling_start_1, dwelling_start_2)
            meeting_end = min(dwelling_end_1, dwelling_end_2)

            meeting_length = meeting_end - meeting_start

            if (meeting_length > max_meeting_length):
                max_meeting_length = meeting_length

            # For graphing style
            if (hypotenuse_calculation2 <= 0):
                dwelling_row2.append(-np.inf)
            else:
                dwelling_row2.append(t_to2)

            # Data for the intersection between the dwelling times
            # Last part of condition checks for whether the other entity has to leave before one has time to get to the location
            # As a result, the only intersections that are kept are plausible meeting locations
            diff_to = max(t_to2 + start_t2, t_to + start_t1) - min(t_to2 + start_t2, t_to + start_t1)
            if (hypotenuse_calculation2 > 0 and hypotenuse_calculation > 0 and t_to2 + start_t2 < start_t1 + time - t_from and t_to + start_t1 < start_t2 + time2 - t_from2):
                doverlay_row.append(meeting_length)
                if (diff_to > max_to):
                    max_to = diff_to

                if (diff_to < min_to):
                    min_to = diff_to
            else:
                doverlay_row.append(-np.inf)

            # Determine max for the colorbar
            if (hypotenuse_calculation > max_dwelling):
                max_dwelling = hypotenuse_calculation

            if (hypotenuse_calculation2 > max_dwelling):
                max_dwelling = hypotenuse_calculation2




        dwellings = [dwelling_row] + dwellings
        dwellings2 = [dwelling_row2] + dwellings2
        doverlay = [doverlay_row] + doverlay

    dwellings = np.array(dwellings)
    dwellings2 = np.array(dwellings2)
    doverlay = np.array(doverlay)

    # Plotting the data points
    fig, ax = plt.subplots(figsize=(12,8))
    levels = np.arange(-max_dwelling, max_dwelling + 0.11, grain_size)  
    levels2 = np.arange(0, max_meeting_length + 0.11, grain_size)  
    
    r = plt.contourf(horizontal_range, vertical_range, dwellings, levels, vmin=-max_dwelling, vmax=max_dwelling + 0.0001, cmap="RdBu", alpha=.75)
    b = plt.contourf(horizontal_range, vertical_range, dwellings2, levels, vmin=-max_dwelling, vmax=max_dwelling + 0.0001, cmap="RdBu", alpha=.75)
    g = plt.contourf(horizontal_range, vertical_range, doverlay, levels2, vmin=0, vmax=max_meeting_length, cmap="Purples", alpha=.75)

    #cl = plt.clabel(g, inline=False, fontsize=3, colors='black')
    x=[start1[0], start2[0]]
    x1=[end1[0], end2[0]]
    y=[start1[1], start2[1]]
    y1=[end1[1], end2[1]]

    #plt.scatter(x=[start1[0], start2[0], end1[0], end2[0]], y=[start1[1], start2[1], end1[1], end2[1]])
    #plt.plot(x, y, '-o')
    #plt.plot(x1, y1, '-o')


    plt.colorbar(g)
    ax.set_title("Intersection of Dwelling Times")
    plt.savefig('./intersection2.png', format='png', dpi=1200)
    #plt.show()

def main():
    # Anchor points for the trajectory's gap
    ll = [-124.05, 20.54]
    rr = [-124.132, 22.98]

    ll2 = [-115.05, 25.54]
    rr2 = [-124.132, 20.98]

    ll = [-124.05, 15]
    rr = [-116.05, 15]

    ll2 = [-124.05, 30]
    rr2 = [-116.05, 30]
                       # px  py  px2  py2  e   g s1 s2 tg tg2 s s2
    #generate_informative(ll, rr, ll2, rr2, 8, .1, 0, 0, 8, 10, 3, 2)
    #generate_2(ll, rr, ll2, rr2, 8, .1, 0, 0, 8, 8, 3, 3)
    #generate_2(ll, rr, ll2, rr2, 8, .1, 0, 1, 8, 8, 3, 3)
    generate_2(ll, rr, ll2, rr2, 8, .01, 0, 1, 8, 10, 3, 2)


if __name__ == "__main__":
    main()


