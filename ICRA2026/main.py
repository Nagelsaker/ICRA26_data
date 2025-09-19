#!/usr/bin/env python3
import bagpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import re
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
        [width/2, length/2-1],      # Front right of rectangle
        [width/2-1, length/2],            # Triangle tip (bow)
        [-(width/2-1), length/2],            # Triangle tip (bow)
        # [width/2-0.5, length/2+0.5],            # Triangle tip (bow)
        [-width/2, length/2-1],     # Front left of rectangle
        [-width/2, -length/2]     # Back to start
    ]
    
    # Fix heading (subtract 90 degrees to align with NED frame)
    heading_corrected = -heading #+ np.deg2rad(180)
    
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



def get_tracks_list_data(bag_path):
    """Extract tracks list data from bag"""
    bag = bagpy.bagreader(bag_path)
    csv_file = bag.message_by_topic("/tracks_list")
    df = pd.read_csv(csv_file)
    return df


def parse_tracks_string(tracks_str):
    """Parse the tracks string to extract track positions and footprints"""
    tracks = []
    if pd.isna(tracks_str) or tracks_str == '[]':
        return tracks
    
    # Split by track entries (each starts with "id:")
    track_entries = re.split(r'(?=id:)', str(tracks_str))
    
    for entry in track_entries:
        if 'id:' not in entry:
            continue
            
        # Extract ID
        id_match = re.search(r'id:\s*([\d.]+)', entry)
        if not id_match:
            continue
        track_id = float(id_match.group(1))
        
        # Extract position x, y
        x_match = re.search(r'position:.*?x:\s*([-\d.]+)', entry, re.DOTALL)
        y_match = re.search(r'position:.*?y:\s*([-\d.]+)', entry, re.DOTALL)
        
        position_x = float(x_match.group(1)) if x_match else None
        position_y = float(y_match.group(1)) if y_match else None
        
        # Extract footprint points
        footprint_points = []
        if 'footprint' in entry:
            # Find all x,y pairs in the footprint section
            footprint_section = re.search(r'footprint:(.*?)(?=velocity:|$)', entry, re.DOTALL)
            if footprint_section:
                footprint_text = footprint_section.group(1)
                # Extract all x,y coordinate pairs
                footprint_x_vals = re.findall(r'x:\s*([-\d.]+)', footprint_text)
                footprint_y_vals = re.findall(r'y:\s*([-\d.]+)', footprint_text)
                
                # Combine x,y pairs
                for fx, fy in zip(footprint_x_vals, footprint_y_vals):
                    footprint_points.append({'x': float(fx), 'y': float(fy)})
        
        if position_x is not None and position_y is not None:
            tracks.append({
                'id': track_id, 
                'x': position_x, 
                'y': position_y,
                'footprint': footprint_points
            })
    
    return tracks


def get_dockable_area_data(bag_path):
    """Extract dockable area data from bag"""
    bag = bagpy.bagreader(bag_path)
    csv_file = bag.message_by_topic("/manager/dockable_area")
    df = pd.read_csv(csv_file)
    return df


def parse_dockable_area_string(points_str):
    """Parse the dockable area points string to extract polygon coordinates"""
    points = []
    if pd.isna(points_str) or points_str == '[]':
        return points
    
    # Extract x, y coordinates using regex
    # Look for patterns like "x: 1031.839599609375\ny: 276.451416015625"
    x_matches = re.findall(r'x:\s*([-\d.]+)', str(points_str))
    y_matches = re.findall(r'y:\s*([-\d.]+)', str(points_str))
    
    # Combine x,y pairs
    for x_val, y_val in zip(x_matches, y_matches):
        points.append({'x': float(x_val), 'y': float(y_val)})
    
    return points



def plot_vessel_trajectory(pose_df, reference_df=None, dock_df=None, tracks_df=None, dockable_area_df=None, show_tracks=False, show_footprints=False, offset_x=0.0, offset_y=0.0):
    """Plot vessel trajectory with vessel shapes at key positions, tracks, dockable area, and dock contours"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Nyhavna floating dock contour (original coordinates)
    CONTOUR1 = [[7035581.26, 7035521.52, 7035521.81, 7035581.6, 7035581.26], 
               [570828.72, 570822.49, 570818.11, 570824.44, 570828.72]]
    
    # Piren reference coordinates
    piren_east = 569796.90517957
    piren_north = 7035254.30142759
    
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
            ax.plot(reference_df[ref_y_col], reference_df[ref_x_col], 'g--', 
                   linewidth=2, alpha=0.8, label='Reference Path')
    
    # Transform and plot Nyhavna floating dock contour
    print(f"Transforming Nyhavna dock contour with offset: East={offset_x}, North={offset_y}")
    
    # Transform contour: subtract piren coordinates and add offset
    # contour_north = [c - piren_north + offset_y for c in CONTOUR1[0]]  # North coordinates
    # contour_east = [c - piren_east + offset_x for c in CONTOUR1[1]]    # East coordinates
    contour_north = [c - piren_north - 23.5 for c in CONTOUR1[0]]  # North coordinates
    contour_east = [c - piren_east + 7.2 for c in CONTOUR1[1]]    # East coordinates
    
    ax.plot(contour_east, contour_north, 'k-', linewidth=3, alpha=0.9, label='Nyhavna Dock')
    ax.fill(contour_east, contour_north, color='gray', alpha=0.3)
    
    # Plot dockable area (if available) - WITH OFFSET
    if dockable_area_df is not None and not dockable_area_df.empty:
        print(f"Processing {len(dockable_area_df)} dockable area messages...")
        print(f"Applying offset: East={offset_x}, North={offset_y}")
        
        for i, row in dockable_area_df.iterrows():
            points = parse_dockable_area_string(row['points'])
            
            if points:
                polygon_x = []
                polygon_y = []
                
                for point in points:
                    # Flip coordinates and apply offset
                    polygon_x.append(point['x'] + offset_x)  # East (horizontal) + offset
                    polygon_y.append(point['y'] + offset_y)  # North (vertical) + offset
                
                if polygon_x and polygon_y:
                    # Close the polygon by adding first point at the end
                    polygon_x.append(polygon_x[0])
                    polygon_y.append(polygon_y[0])
                    
                    ax.plot(polygon_x, polygon_y, 'c-', linewidth=2, alpha=0.8, 
                           label='Dockable Area' if i == 0 else None)
                    ax.fill(polygon_x, polygon_y, color='cyan', alpha=0.2)
                    
                    print(f"Plotted dockable area {i} with {len(points)} points")
    
    # Plot tracks with footprints (if available and enabled) - WITH OFFSET
    if show_tracks and tracks_df is not None and not tracks_df.empty:
        print(f"Processing {len(tracks_df)} track messages...")
        print(f"Applying offset to tracks: East={offset_x}, North={offset_y}")
        plotted_tracks = False
        
        for i, row in tracks_df.iterrows():
            tracks = parse_tracks_string(row['tracks'])
            
            for track in tracks:
                track_east = track['x'] + offset_x   # East (horizontal axis) + offset
                track_north = track['y'] + offset_y  # North (vertical axis) + offset
                
                # Plot footprint if available
                if track['footprint'] and len(track['footprint']) > 2 and show_footprints:
                    footprint_x = []
                    footprint_y = []
                    
                    for fp_point in track['footprint']:
                        footprint_x.append(fp_point['x'] + offset_x)  # East + offset
                        footprint_y.append(fp_point['y'] + offset_y)  # North + offset
                    
                    # Close the footprint polygon
                    if len(footprint_x) > 0:
                        footprint_x.append(footprint_x[0])
                        footprint_y.append(footprint_y[0])
                        
                        label = 'Track Footprints' if not plotted_tracks else None
                        if label:
                            plotted_tracks = True
                        
                        ax.plot(footprint_x, footprint_y, 'orange', linewidth=1, alpha=0.7, label=label)
                        ax.fill(footprint_x, footprint_y, color='orange', alpha=0.3)
                        
                        # Add track ID at center
                        center_x = sum(footprint_x[:-1]) / len(footprint_x[:-1])  # Exclude duplicate last point
                        center_y = sum(footprint_y[:-1]) / len(footprint_y[:-1])
                        
                        if i % 10 == 0:  # Only label every 10th message
                            ax.annotate(f'{int(track["id"])}', (center_x, center_y), 
                                      ha='center', va='center', fontsize=8, alpha=0.8, color='darkred')
                else:
                    # Fallback to point if no footprint
                    label = 'Track Points' if not plotted_tracks else None
                    if label:
                        plotted_tracks = True
                    
                    ax.scatter(track_east, track_north, c='orange', s=30, alpha=0.6, 
                             marker='o', label=label)
        
        print(f"Plotted tracks with footprints: {plotted_tracks}")
    
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
    ax.set_ylabel('North (m) ↑')  # X axis, increasing up
    ax.set_title('Vessel Trajectory with Orientation (NED Frame)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.savefig('vessel_trajectory.png', dpi=300, bbox_inches='tight')
    return fig




def debug_tracks_data(bag_path):
    """Debug what's in the tracks list topic"""
    bag = bagpy.bagreader(bag_path)
    csv_file = bag.message_by_topic("/tracks_list")
    df = pd.read_csv(csv_file)
    
    print(f"Tracks list has {len(df)} messages")
    print("Columns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    
    return df



# Main plotting function
def generate_all_plots(bag_path):
    """Generate all docking plots"""
    print("Extracting data...")
    manager_data = get_manager_state_data(bag_path)
    pose_data = get_pose_data(bag_path)
    reference_data = get_reference_pose_data(bag_path)
    dock_data = get_dock_pose_data(bag_path)
    tracks_data = get_tracks_list_data(bag_path)
    dockable_area_data = get_dockable_area_data(bag_path)
    
    print("Generating plots...")
    plot_manager_state(manager_data)
    
    # ADJUST THESE OFFSET VALUES AS NEEDED:
    offset_east = 0.0 #-23.5  # Positive = move right, Negative = move left
    offset_north = 0.0 # 7.2 # Positive = move up, Negative = move down
    
    plot_vessel_trajectory(pose_data, reference_data, dock_data, tracks_data, dockable_area_data, 
                          show_tracks=True, show_footprints=False, offset_x=offset_east, offset_y=offset_north)
    
    print("All plots saved!")
    plt.show()


if __name__ == "__main__":
    generate_all_plots("data/exp2_aug_simon_2025-06-06-09-26-37.bag")