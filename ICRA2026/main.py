#!/usr/bin/env python3
import bagpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math
from matplotlib.patches import Polygon

def quaternion_to_yaw(x, y, z, w):
    """Convert quaternion to yaw angle in radians"""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def create_vessel_shape(x, y, heading, length=5, width=2.8):
    """Create vessel shape (rectangle + triangle) at given position and heading"""
    # Create vessel outline points (rectangle + triangle)
    # Start with vessel pointing "up" (heading = 0)
    vessel_points = [
        [-width/2, -length/2],    # Back left
        [width/2, -length/2],     # Back right  
        [width/2, length/4],      # Front right of rectangle
        [width/4, length/4],      # Triangle base right
        [0, length/2],            # Triangle tip (bow)
        [-width/4, length/4],     # Triangle base left
        [-width/2, length/4],     # Front left of rectangle
        [-width/2, -length/2]     # Back to start
    ]
    
    vessel_points = [
        [-width/2, -length/2],    # Back left
        [width/2, -length/2],     # Back right  
        [width/2, length/2],      # Front right of rectangle
        [width/2-1, length/2+1],            # Triangle tip (bow)
        [-(width/2-1), length/2+1],            # Triangle tip (bow)
        # [width/2-0.5, length/2+0.5],            # Triangle tip (bow)
        [-width/2, length/2],     # Front left of rectangle
        [-width/2, -length/2]     # Back to start
    ]
    
    # Fix heading (subtract 90 degrees to align with NED frame)
    heading_corrected = heading + np.deg2rad(180)
    
    # Rotate points by corrected heading angle
    cos_h = math.cos(heading_corrected)
    sin_h = math.sin(heading_corrected)
    
    rotated_points = []
    for px, py in vessel_points:
        rx = px * cos_h - py * sin_h + x
        ry = px * sin_h + py * cos_h + y
        rotated_points.append([rx, ry])
    
    return rotated_points



# Data extraction functions
def get_manager_state_data(bag_path):
    """Extract manager state data from bag"""
    bag = bagpy.bagreader(bag_path)
    csv_file = bag.message_by_topic("/manager_state")
    df = pd.read_csv(csv_file)
    return df

def get_pose_data(bag_path):
    """Extract pose data from bag"""
    bag = bagpy.bagreader(bag_path)
    csv_file = bag.message_by_topic("/navigation/pose")
    df = pd.read_csv(csv_file)
    return df

# Plotting functions
def plot_manager_state(df):
    """Plot manager state over time"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    unique_states = df['data'].unique()
    state_to_y = {state: i for i, state in enumerate(unique_states)}
    df['y_value'] = df['data'].map(state_to_y)
    
    ax.step(df['Time'], df['y_value'], where='post', linewidth=2)
    ax.set_yticks(range(len(unique_states)))
    ax.set_yticklabels(unique_states)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Manager State')
    ax.set_title('Docking Manager State Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.savefig('manager_state_plot.png', dpi=300, bbox_inches='tight')
    return fig



def get_reference_pose_data(bag_path):
    """Extract guidance reference pose data from bag"""
    bag = bagpy.bagreader(bag_path)
    csv_file = bag.message_by_topic("/guidance/reference/pose")
    df = pd.read_csv(csv_file)
    return df



def get_dock_pose_data(bag_path):
    """Extract dock pose data from bag"""
    bag = bagpy.bagreader(bag_path)
    csv_file = bag.message_by_topic("/manager/dock_pose")
    df = pd.read_csv(csv_file)
    return df



def plot_vessel_trajectory(pose_df, reference_df=None, dock_df=None):
    """Plot vessel trajectory with vessel shapes at key positions"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # NED frame: X = North (vertical), Y = East (horizontal)
    x_pos = pose_df['pose.position.x']  # North
    y_pos = pose_df['pose.position.y']  # East
    time = pose_df['Time']
    
    # Extract heading from quaternion
    qx = pose_df['pose.orientation.x']
    qy = pose_df['pose.orientation.y'] 
    qz = pose_df['pose.orientation.z']
    qw = pose_df['pose.orientation.w']
    
    headings = [quaternion_to_yaw(x, y, z, w) for x, y, z, w in zip(qx, qy, qz, qw)]
    
    # Plot reference trajectory first (if available)
    if reference_df is not None and not reference_df.empty:
        ref_cols = reference_df.columns.tolist()
        
        ref_x_col = None
        ref_y_col = None
        for col in ref_cols:
            if 'north' in col.lower():
                ref_x_col = col
            elif 'east' in col.lower():
                ref_y_col = col
        
        if ref_x_col and ref_y_col:
            # Plot with Y (East) on horizontal axis, X (North) on vertical axis
            ax.plot(reference_df[ref_y_col], reference_df[ref_x_col], 'g--', 
                   linewidth=2, alpha=0.8, label='Reference Path')
    
    # Plot trajectory line (Y on horizontal axis, X on vertical axis)
    ax.plot(y_pos, x_pos, 'b-', linewidth=1, alpha=0.5, label='Trajectory')
    
    # Highlight docking phase
    docking_mask = (time >= 100) & (time <= 175)
    if docking_mask.any():
        ax.plot(y_pos[docking_mask], x_pos[docking_mask], 'r-', linewidth=2, alpha=0.7, label='Docking Phase')
    
    # Draw vessel shapes at key positions only
    # Start position
    vessel_start = create_vessel_shape(y_pos.iloc[0], x_pos.iloc[0], headings[0])
    ax.add_patch(Polygon(vessel_start, facecolor='green', edgecolor='darkgreen', alpha=0.8, label='Start'))
    
    # End position
    vessel_end = create_vessel_shape(y_pos.iloc[-1], x_pos.iloc[-1], headings[-1])
    ax.add_patch(Polygon(vessel_end, facecolor='red', edgecolor='darkred', alpha=0.8, label='End'))
    
    # Dock pose (if available)
    if dock_df is not None and not dock_df.empty:
        dock_cols = dock_df.columns.tolist()
        
        dock_x_col = None
        dock_y_col = None
        dock_heading_col = None
        for col in dock_cols:
            if 'north' in col.lower():
                dock_x_col = col
            elif 'east' in col.lower():
                dock_y_col = col
            elif 'heading' in col.lower():
                dock_heading_col = col
        
        if dock_x_col and dock_y_col:
            dock_north = dock_df[dock_x_col].iloc[0]  # North
            dock_east = dock_df[dock_y_col].iloc[0]   # East
            dock_heading = dock_df[dock_heading_col].iloc[0] if dock_heading_col else 0
            
            # Coordinates: (East, North)
            vessel_dock = create_vessel_shape(dock_east, dock_north, dock_heading)
            ax.add_patch(Polygon(vessel_dock, facecolor='purple', edgecolor='purple', alpha=0.8, label='Dock Pose'))
    
    ax.set_xlabel('East (m) →')  # Y axis, increasing right
    ax.set_ylabel('North (m) →')  # X axis, increasing up
    ax.set_title('Vessel Trajectory with Orientation (NED Frame)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.savefig('vessel_trajectory.png', dpi=300, bbox_inches='tight')
    return fig



# Main plotting function
def generate_all_plots(bag_path):
    """Generate all docking plots"""
    print("Extracting data...")
    manager_data = get_manager_state_data(bag_path)
    pose_data = get_pose_data(bag_path)
    reference_data = get_reference_pose_data(bag_path)
    dock_data = get_dock_pose_data(bag_path)
    
    print("Generating plots...")
    plot_manager_state(manager_data)
    plot_vessel_trajectory(pose_data, reference_data, dock_data)
    
    print("All plots saved!")
    plt.show()



if __name__ == "__main__":
    generate_all_plots("data/exp2_aug_simon_2025-06-06-09-26-37.bag")