#Importing all the necessary libraries
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backend_bases import MouseButton
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to stretch colors to improve image visualization
def color_stretch(image, index, minmax=(0, 10000)):
    colors = image[:, :, index].astype(np.float64)  # Convert selected bands to float
    
    max_val, min_val = minmax[1], minmax[0]  # Define max and min values

    # Enforce minimum and maximum value constraints
    colors[colors[:, :, :] > max_val] = max_val
    colors[colors[:, :, :] < min_val] = min_val

    # Normalize each band within the specified range
    for b in range(colors.shape[2]):
        colors[:, :, b] = colors[:, :, b] * 1 / (max_val - min_val)

    return colors

# Function to create a custom colormap based on class predictions
def get_cmap(class_prediction):
    n = class_prediction.max()  # Find the max class label
    
    # Define RGB color codes for each land use type
    colors = dict((
(0, (0, 0, 0, 255)), #No Data	
(1, (0, 0, 205, 255)), #Water
(2, (26, 101, 26, 255)),#(50, 205, 50, 255)), #Agriculture
(3, (210, 180, 140, 255)),
(4,  (220, 20, 60, 255)), # Urban
(5, (28, 107, 160, 255)), #Inland Water	
    ))
    
    # Convert colors from 0-255 to 0-1 for Matplotlib
    for k, v in colors.items():
        colors[k] = [_v / 255.0 for _v in v]

    # Create a list of colors indexed by class, filling gaps with transparent
    index_colors = [colors[key] if key in colors else (255, 255, 255, 0) for key in range(1, n + 1)]
    cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', n)

    return cmap

# Load the Sentinel imagery data from the .npz file
inp = np.load('P:/ai assignment/data.npz')
dat = inp['arr_0']
img = color_stretch(dat, [3, 2, 1], (0, 0.3))  # Apply color stretch to RGB bands
plt.imshow(img)
plt.title("Satellite Image")
plt.show()


coordinates = []

# Set up global variables for zoom level and panning
zoom_level = 1.0
zoom_step = 0.1
pan_start = None  # Store the starting position for panning

fig, ax = plt.subplots(figsize=(12, 8), dpi=400)  # Adjust the size as needed
im = ax.imshow(img)
plt.title("Scroll to Zoom, Right Click to Select Points, Left Click and Drag to Pan")

# Function to handle zooming
def on_scroll(event):
    global zoom_level
    if event.button == 'up':
        zoom_level = min(zoom_level + zoom_step, 2.0)
    elif event.button == 'down':
        zoom_level = max(zoom_level - zoom_step, 0.5)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    center_x = (xlim[0] + xlim[1]) / 2
    center_y = (ylim[0] + ylim[1]) / 2
    width = (xlim[1] - xlim[0]) / 2 / zoom_level
    height = (ylim[1] - ylim[0]) / 2 / zoom_level
    ax.set_xlim(center_x - width, center_x + width)
    ax.set_ylim(center_y - height, center_y + height)
    plt.draw()

# Function to handle point selection by right clicking
def on_click(event):
    if event.inaxes == ax:  # Only register clicks within the axes
        if event.button == 3:  # Right click for point selection
            x, y = int(event.xdata), int(event.ydata)
            click_number = len(coordinates) + 1  # Determine the click number
            print(f"Selected point {click_number}: ({x}, {y})")  # Include click number
            plt.scatter(x, y, color='red', s=1)
            plt.draw()
            coordinates.append((x, y))  # Store the coordinates

# Function to handle panning
def on_mouse_drag(event):
    global pan_start
    if event.inaxes == ax and event.button == 1:  # Left click for panning
        if pan_start is None:  # Start panning
            pan_start = (event.xdata, event.ydata)
        else:  # Update the view limits
            dx = pan_start[0] - event.xdata
            dy = pan_start[1] - event.ydata
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
            ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
            plt.draw()
    
# Function to handle mouse release
def on_mouse_release(event):
    global pan_start
    pan_start = None  # Reset panning

# Connect the events
fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_drag)
fig.canvas.mpl_connect('button_release_event', on_mouse_release)

# Output collected coordinates
print("Collected coordinates:", coordinates)
plt.show()