import pandas as pd
import os
import math as math
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt


pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 99999999999)  # Set width to avoid wrapping


os.system("cls")
os.chdir("C:\\Users\\cloud\\Desktop\\Offset_calculator\\Odometry_offset_calculator")

data_set_raw = pd.read_csv("Data_set_11.csv")
# print("raw data")
# print(data_set_raw)
# print("--------------------------------------------------")

delta_data_set = data_set_raw.diff()
delta_data_set = delta_data_set.rename(columns={'lat':'delta_lat','vert':'delta_vert','heading':'delta_heading'})
delta_data_set = delta_data_set.drop(index = 0)
# print ("delta data set")
# print(delta_data_set)
# print("--------------------------------------------------")

#robot characteristics
# lat (inches) | vert (inches) | heading (radians)
vert_encoder_diameter = (3.29/2) * (3.0/4.0)
lat_encoder_diameter =  (2.0/2)

delta_data_set['delta_lat_converted'] = delta_data_set['delta_lat'].transform(lambda x: math.radians(x) * lat_encoder_diameter)
delta_data_set['delta_vert_converted'] = delta_data_set['delta_vert'].transform(lambda x:math.radians(x) * vert_encoder_diameter)
delta_data_set['delta_heading_rads'] = delta_data_set['delta_heading'].transform(lambda x: math.radians(x))

delta_data_set = delta_data_set.drop(columns = ['delta_lat','delta_vert','delta_heading'])

# print ("converted delta data set")
# print(delta_data_set)
# print("--------------------------------------------------")

#############################################################################################
#can doulble check, theoretical optimal offset calculations
# delta_data_set['optimal_lat_offset'] = -1 * delta_data_set['delta_heading_rads'] / delta_data_set['delta_lat_converted']
# delta_data_set['optimal_vert_offset'] = -1 * delta_data_set['delta_heading_rads'] / delta_data_set['delta_vert_converted']

# print ("theoretical optimal from delta x/y = 0")
# print(delta_data_set)
# print("--------------------------------------------------")
#################################################################################################################
print ("Clean data set for simulation")
Frame_for_simulation = pd.DataFrame()
Frame_for_simulation = delta_data_set[['delta_lat_converted','delta_vert_converted','delta_heading_rads']].join(Frame_for_simulation)
Frame_for_simulation = Frame_for_simulation.reset_index(drop=True)
Frame_for_simulation = data_set_raw.iloc[0:-1][['heading']].reset_index(drop=True).join(Frame_for_simulation)

Frame_for_simulation = Frame_for_simulation.rename(columns={'heading':'prev_heading','delta_lat_converted':'delta_lat','delta_vert_converted':'delta_vert','delta_heading_rads':'delta_heading'})
Frame_for_simulation['prev_heading'] = Frame_for_simulation['prev_heading'].transform(lambda x: math.radians(x))

#print(Frame_for_simulation)
# print("--------------------------------------------------")



position_frame = pd.DataFrame()
#make sure prev_theta is in radians
def calculate_odom_pos(delta_lat_inches,delta_vert_inches,delta_heading_rads,lat_offset,vert_offset,prev_theta):
    if(delta_heading_rads == 0):
        local_delta_y = delta_vert_inches
        local_delta_x = delta_lat_inches
    else:
        arc_radius = delta_vert_inches / delta_heading_rads + vert_offset
        lateral_arc_radius = delta_lat_inches / delta_heading_rads + lat_offset

        if(delta_vert_inches == 0):
            arc_radius = 0
        if(delta_lat_inches == 0):
            lateral_arc_radius = 0
        
        local_delta_y = 2 * arc_radius * math.sin(delta_heading_rads/2)
        local_delta_x = 2 * lateral_arc_radius * math.sin(delta_heading_rads/2)

    transform_theta = -1 * (prev_theta + (delta_heading_rads/2))  
    global_delta_x = local_delta_x * math.cos(transform_theta) - local_delta_y * math.sin(transform_theta)
    global_delta_y = local_delta_x * math.sin(transform_theta) + local_delta_y * math.cos(transform_theta)

    return[global_delta_x,global_delta_y]

for i in np.arange(-1,4,0.1): #lat then vert #-3,3
    for j in np.arange(-10,-4,0.1):         #-14,-9
        position_frame['lat:',i,'|vert:',j] = Frame_for_simulation.apply(lambda row: calculate_odom_pos(row['delta_lat'],row['delta_vert'],row['delta_heading'],i,j,row['prev_heading']), axis = 1)


#print(tabulate(position_frame, headers='keys', tablefmt='pretty'))
#print("Position Frame:")
#print(position_frame)
#print(tabulate(position_frame, headers='keys', tablefmt='psql'))
# print("--------------------------------------------------\n")
# print(tabulate(position_frame, headers='keys', tablefmt='psql'))
# print("--------------------------------------------------")
##################################################################
distance_Frame = pd.DataFrame()
for col in position_frame.columns:
    distance_Frame[col] = position_frame[col].apply(lambda v: np.linalg.norm(v))

#print(distance_Frame)
##################################################################
# print("RMS_Frame")

RMS_Frame = pd.DataFrame([{
    col: np.sqrt(np.mean(distance_Frame[col]**2)) for col in distance_Frame.columns
}])
#print(RMS_Frame)



#####################################################################import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Assume RMS_Frame is your 1-row DataFrame with tuple column names, e.g., ('lat:', -2, '|vert:', -2)

# Convert RMS_Frame to a Series
rms_series = pd.Series({col: RMS_Frame[col].iloc[0] for col in RMS_Frame.columns})

# Create a MultiIndex from the tuple columns by extracting the lat and vert offsets.
# Here, col[1] is lat_offset and col[3] is vert_offset.
rms_series.index = pd.MultiIndex.from_tuples(
    [(col[1], col[3]) for col in RMS_Frame.columns],
    names=["lat_offset", "vert_offset"]
)

# Identify the tile (i.e., the combination of offsets) with the lowest RMS value.
min_tile = rms_series.idxmin()    # Returns a tuple (lat_offset, vert_offset)
min_value = rms_series.min()

print(f"Tile with lowest RMS value: {min_tile} with RMS value: {min_value:.3f}")

# --- Prepare data for heatmap plotting ---
# Reshape the Series into a 2D DataFrame with rows as lat_offset and columns as vert_offset.
rms_grid = rms_series.unstack("vert_offset")
rms_grid = rms_grid.sort_index(ascending=False)  # Optional: sort rows descending for clarity

# Create the heatmap.
plt.figure(figsize=(8, 6))
im = plt.imshow(rms_grid, cmap='viridis', interpolation='nearest')
plt.colorbar(im, label="RMS Value")
plt.title("RMS Heatmap")
plt.xlabel("Vert Offset")
plt.ylabel("Lat Offset")

# Set tick labels using the actual offsets.
plt.xticks(ticks=np.arange(len(rms_grid.columns)), labels=rms_grid.columns)
plt.yticks(ticks=np.arange(len(rms_grid.index)), labels=rms_grid.index)

# --- Optional: Mark the tile with the lowest RMS value on the heatmap ---
# Determine the position of the minimum tile in the grid.
row_idx = list(rms_grid.index).index(min_tile[0])
col_idx = list(rms_grid.columns).index(min_tile[1])

# Plot a red circle marker at the tile's location.
plt.scatter(col_idx, row_idx, color='red', marker='o', s=150, edgecolor='black')

plt.show()
