#!/usr/bin/env python3
import re
import bagpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

# Import functions from main.py
from main import (
    get_pose_data,
    get_dock_pose_data,
    get_manager_state_data, 
    get_dockable_area_data,
    parse_dockable_area_string,
    get_manager_path_data,
    get_tracks_list_data,
    parse_tracks_string, 
    create_vessel_shape,
    quaternion_to_yaw
)



def parse_manager_path_poses(poses_str):
    """Parse the poses string to extract x,y coordinates from position only"""
    # Find all position blocks and extract x,y from them
    position_blocks = re.findall(r'position:\s*\n\s*x:\s*([\d.-]+)\s*\n\s*y:\s*([\d.-]+)', poses_str)
    
    if position_blocks:
        x_coords = [float(match[0]) for match in position_blocks]
        y_coords = [float(match[1]) for match in position_blocks]  
        return x_coords, y_coords
    else:
        return [], []



def extract_dock_pose_data(bag_path, start_time):
    """Extract and parse dock pose data from bag file"""
    print("Extracting dock pose data for animation...")
    dock_data = get_dock_pose_data(bag_path)
    
    # Parse dock pose data
    dock_pose = None
    if not dock_data.empty:
        dock_cols = dock_data.columns.tolist()
        
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
            dock_pose = {
                'north': dock_data[dock_x_col].iloc[0],
                'east': dock_data[dock_y_col].iloc[0], 
                'heading': dock_data[dock_heading_col].iloc[0] if dock_heading_col else 0,
                'time': dock_data['Time'].iloc[0] - start_time
            }
            print(f"Dock pose will appear at time: {dock_pose['time']:.1f}s")
    
    return dock_pose



def extract_dockable_area_data(bag_path, start_time):
    """Extract and parse dockable area data from bag file"""
    print("Extracting dockable area data for animation...")
    dockable_area_data = get_dockable_area_data(bag_path)
    
    dockable_areas = []
    if not dockable_area_data.empty:
        print(f"Processing {len(dockable_area_data)} dockable area messages...")
        
        for i, row in dockable_area_data.iterrows():
            points = parse_dockable_area_string(row['points'])
            
            if points:
                polygon_x = []
                polygon_y = []
                
                for point in points:
                    polygon_x.append(point['x'])  # East (horizontal)
                    polygon_y.append(point['y'])  # North (vertical)
                
                if polygon_x and polygon_y:
                    # Close the polygon
                    polygon_x.append(polygon_x[0])
                    polygon_y.append(polygon_y[0])
                    
                    dockable_areas.append({
                        'x': polygon_x,
                        'y': polygon_y,
                        'time': row['Time'] - start_time
                    })
                    
        print(f"Found {len(dockable_areas)} dockable area updates")
    
    return dockable_areas




def extract_tracks_data(bag_path, start_time):
    """Extract and parse tracks data from bag file"""
    print("Extracting tracks data for animation...")
    tracks_data = get_tracks_list_data(bag_path)
    
    tracks_by_time = []
    if not tracks_data.empty:
        print(f"Processing {len(tracks_data)} track messages...")
        
        for i, row in tracks_data.iterrows():
            tracks = parse_tracks_string(row['tracks'])
            
            if tracks:
                tracks_by_time.append({
                    'tracks': tracks,
                    'time': row['Time'] - start_time
                })
                    
        print(f"Found {len(tracks_by_time)} track updates")
    
    return tracks_by_time



def extract_thruster_data(bag_path, start_time):
    """Extract throttle reference data for all 4 thrusters"""
    print("Extracting thruster data for animation...")
    
    thruster_data = {}
    for i in range(1, 5):
        topic = f"/actuator_ref_{i}"
        try:
            bag = bagpy.bagreader(bag_path)
            csv_file = bag.message_by_topic(topic)
            df = pd.read_csv(csv_file)
            df['rel_time'] = df['Time'] - start_time
            thruster_data[f'thruster_{i}'] = df
            print(f"Loaded {len(df)} messages for thruster {i}")
        except Exception as e:
            print(f"Could not load data for {topic}: {e}")
            thruster_data[f'thruster_{i}'] = None
    
    return thruster_data



def animate_docking_experiment(bag_path):
    """Create real-time animated visualization of the docking experiment"""
    
    print("Extracting pose data for animation...")
    pose_data = get_pose_data(bag_path)
    
    print("Extracting manager path data for animation...")
    manager_path_data = get_manager_path_data(bag_path)
    
    # Convert timestamp to relative time (start from 0)
    start_time = pose_data['Time'].iloc[0]
    pose_data['rel_time'] = pose_data['Time'] - start_time
    
    # Extract positions and orientations
    x_pos = pose_data['pose.position.x'].values  # North
    y_pos = pose_data['pose.position.y'].values  # East
    times = pose_data['rel_time'].values
    
    # Extract headings from quaternions
    qx = pose_data['pose.orientation.x'].values
    qy = pose_data['pose.orientation.y'].values
    qz = pose_data['pose.orientation.z'].values
    qw = pose_data['pose.orientation.w'].values
    headings = [quaternion_to_yaw(x, y, z, w) for x, y, z, w in zip(qx, qy, qz, qw)]
    
    # Parse manager path data
    path_lines = []
    if not manager_path_data.empty:
        print(f"Processing {len(manager_path_data)} manager path messages...")
        for idx, row in manager_path_data.iterrows():
            x_coords, y_coords = parse_manager_path_poses(row['poses'])
            if x_coords and y_coords:
                path_lines.append({
                    'x': y_coords,  # East coordinates for plotting (horizontal axis)
                    'y': x_coords,  # North coordinates for plotting (vertical axis)
                    'time': row['Time'] - start_time
                })
    
    # Extract dock pose data  
    dock_pose = extract_dock_pose_data(bag_path, start_time)
    
    # Extract dockable area data
    dockable_areas = extract_dockable_area_data(bag_path, start_time)
    
    # Extract tracks data
    tracks_by_time = extract_tracks_data(bag_path, start_time)
    
    # Extract thruster data
    thruster_data = extract_thruster_data(bag_path, start_time)
    
    # Set up the figure with dark theme and subplots
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 14))  # Taller figure for subplots
    fig.patch.set_facecolor('#0a0a0a')
    
    # Create subplot layout: map on top, thrusters below
    ax_map = plt.subplot2grid((3, 1), (0, 0), rowspan=2)  # Map takes 2/3 of height
    ax_thrusters = plt.subplot2grid((3, 1), (2, 0))       # Thrusters take 1/3 of height
    
    # Set up map subplot
    ax_map.set_facecolor('#1a1a1a')
    ax_map.set_xlabel('East (m) →', fontsize=14, color='white', fontweight='bold')
    ax_map.set_ylabel('North (m) ↑', fontsize=14, color='white', fontweight='bold')
    ax_map.set_title('Autonomous Docking Experiment - Real-time Visualization', 
                    fontsize=16, color='white', fontweight='bold', pad=20)
    
    # Set fixed axis limits for map
    ax_map.axis('equal')
    ax_map.set_xlim(980, 1040)  # East (m)
    ax_map.set_ylim(240, 300)   # North (m)
    
    # Grid styling for map
    ax_map.grid(True, alpha=0.3, color='#404040', linestyle='-', linewidth=0.5)
    ax_map.tick_params(colors='white', labelsize=12)
    
    # Set up thruster subplot
    ax_thrusters.set_facecolor('#1a1a1a')
    ax_thrusters.set_xlabel('Thruster RPM', fontsize=12, color='white', fontweight='bold')
    ax_thrusters.set_ylabel('Thrusters', fontsize=12, color='white', fontweight='bold')
    ax_thrusters.set_xlim(0, 1000)  # Changed to 0-1000
    ax_thrusters.set_ylim(-0.5, 3.5)
    ax_thrusters.grid(True, alpha=0.3, color='#404040', linestyle='-', linewidth=0.5)
    ax_thrusters.tick_params(colors='white', labelsize=10)
    
    # Add zero line for thrusters
    ax_thrusters.axvline(x=0, color='white', linewidth=2, alpha=0.8)
    
    # Set thruster labels
    thruster_labels = ['Thruster 1', 'Thruster 2', 'Thruster 3', 'Thruster 4']
    ax_thrusters.set_yticks(range(4))
    ax_thrusters.set_yticklabels(thruster_labels)
    
    # Draw static floating dock contour
    # Nyhavna floating dock contour (original coordinates)
    CONTOUR1 = [[7035581.26, 7035521.52, 7035521.81, 7035581.6, 7035581.26], 
               [570828.72, 570822.49, 570818.11, 570824.44, 570828.72]]
    
    # Piren reference coordinates
    piren_east = 569796.90517957
    piren_north = 7035254.30142759
    
    # Transform contour: subtract piren coordinates and add offset
    contour_north = [c - piren_north - 23.5 for c in CONTOUR1[0]]  # North coordinates
    contour_east = [c - piren_east + 7.2 for c in CONTOUR1[1]]    # East coordinates
    
    ax_map.plot(contour_east, contour_north, '#4a90e2', linewidth=3, alpha=0.9, label='Nyhavna Dock')
    ax_map.fill(contour_east, contour_north, color='#37474f', alpha=0.6)
    
    # Initialize animation elements
    vessel_patch = None
    trajectory_line = None
    path_plots = []
    drawn_paths = set()
    dock_pose_patch = None
    dock_pose_drawn = False
    dockable_area_plots = []
    drawn_dockable_areas = set()
    track_plots = []
    current_tracks_index = -1
    
    # Initialize thruster bars
    thruster_bars = []
    thruster_angles = []  # Store angle indicators
    thruster_colors = ['#00ff88', '#0088ff', '#ff6b35', '#ffa726']
    for i in range(4):
        bar = ax_thrusters.barh(i, 0, height=0.3, color=thruster_colors[i], alpha=0.8)  # Reduced height from 0.6 to 0.3
        thruster_bars.append(bar)
        
        # Initialize angle indicators (small circles with arrows)
        angle_circle = plt.Circle((950, i), 15, fill=False, color=thruster_colors[i], linewidth=2)
        ax_thrusters.add_patch(angle_circle)
        
        # Arrow for angle direction (initially pointing right)
        angle_arrow = ax_thrusters.annotate('', xy=(965, i), xytext=(935, i),
                                        arrowprops=dict(arrowstyle='->', color=thruster_colors[i], lw=2))
        
        thruster_angles.append({'circle': angle_circle, 'arrow': angle_arrow})
    
    time_text = ax_map.text(0.02, 0.98, '', transform=ax_map.transAxes, 
                           fontsize=14, color='#00ff88', fontweight='bold',
                           verticalalignment='top')
    
    def animate(frame):
        nonlocal vessel_patch, trajectory_line, path_plots, drawn_paths, dock_pose_patch, dock_pose_drawn, dockable_area_plots, drawn_dockable_areas, track_plots, current_tracks_index
        
        current_time = times[frame]
        
        # Update trajectory
        trajectory_line = update_trajectory(ax_map, y_pos, x_pos, frame, trajectory_line)
        
        # Update vessel
        vessel_patch = update_vessel(ax_map, y_pos, x_pos, headings, frame, vessel_patch)
        
        # Update manager paths (only draw new ones)
        path_plots = update_manager_paths(ax_map, path_lines, current_time, path_plots, drawn_paths)
        
        # Update dock pose (draw once when it appears)
        dock_pose_patch, dock_pose_drawn = update_dock_pose(ax_map, dock_pose, current_time, dock_pose_patch, dock_pose_drawn)
        
        # Update dockable areas (draw new ones when they appear)
        dockable_area_plots = update_dockable_areas(ax_map, dockable_areas, current_time, dockable_area_plots, drawn_dockable_areas)
        
        # Update tracks (replace old with new)
        track_plots, current_tracks_index = update_tracks(ax_map, tracks_by_time, current_time, track_plots, current_tracks_index)
        
        # Update thrusters
        update_thrusters(ax_thrusters, thruster_data, thruster_bars, thruster_angles, current_time)
        
        # Update time display
        time_text.set_text(f'Time: {current_time:.1f}s')
        
        return_list = [vessel_patch, time_text] + ([trajectory_line] if trajectory_line else []) + path_plots + dockable_area_plots + track_plots + [bar[0] for bar in thruster_bars] + [angle['arrow'] for angle in thruster_angles]
        if dock_pose_patch:
            return_list.append(dock_pose_patch)
        
        return return_list
    
    # Create animation
    print(f"Creating animation with {len(pose_data)} frames...")
    print(f"Experiment duration: {times[-1]:.1f} seconds")
    print(f"Found {len(path_lines)} manager path updates")
    
    # Animation timing
    avg_dt = np.mean(np.diff(times)) * 1000  # Convert to milliseconds
    interval = max(3, min(100, avg_dt)) // 3  # 3x speed for debugging
    
    anim = FuncAnimation(fig, animate, frames=len(pose_data), 
                        interval=interval, blit=False, repeat=False)
    
    print(f"Animation ready! Playback speed: {1000/interval:.1f} FPS")
    plt.tight_layout()
    plt.show()
    
    return anim


def update_trajectory(ax, y_pos, x_pos, frame, trajectory_line):
    """Update trajectory trail"""
    if trajectory_line is not None:
        trajectory_line.remove()
    
    trajectory_line = None
    if frame > 0:
        trajectory_line, = ax.plot(y_pos[:frame+1], x_pos[:frame+1], 
                                 color='#0088ff', linewidth=2, alpha=0.7)
    return trajectory_line


def update_vessel(ax, y_pos, x_pos, headings, frame, vessel_patch):
    """Update vessel position and orientation"""
    if vessel_patch is not None:
        vessel_patch.remove()
    
    current_x = x_pos[frame]  # North
    current_y = y_pos[frame]  # East
    current_heading = headings[frame]
    
    vessel_points = create_vessel_shape(current_y, current_x, current_heading)
    vessel_patch = Polygon(vessel_points, facecolor='#00ff88', edgecolor='#00cc66', 
                          alpha=0.9, linewidth=2)
    ax.add_patch(vessel_patch)
    return vessel_patch


def update_manager_paths(ax, path_lines, current_time, path_plots, drawn_paths):
    """Update manager path display - only draw new paths when they appear"""
    
    # Check for new paths that should appear at current time
    for i, path_data in enumerate(path_lines):
        if current_time >= path_data['time'] and i not in drawn_paths:
            # This path should be visible and hasn't been drawn yet
            path_plot, = ax.plot(path_data['x'], path_data['y'], 
                               color='#ffaa00', linewidth=2, alpha=0.8, 
                               linestyle='--', label='Manager Path' if not path_plots else None)
            path_plots.append(path_plot)
            drawn_paths.add(i)  # Mark this path as drawn
    
    return path_plots



def update_dock_pose(ax, dock_pose, current_time, dock_pose_patch, dock_pose_drawn):
    """Update dock pose display - draw once when it appears"""
    if dock_pose and current_time >= dock_pose['time'] and not dock_pose_drawn:
        # Draw the dock pose with target/goal styling
        vessel_dock = create_vessel_shape(dock_pose['east'], dock_pose['north'], dock_pose['heading'])
        dock_pose_patch = Polygon(vessel_dock, 
                                facecolor='#ffa726',  # Professional amber/gold
                                edgecolor='#ff8f00',  # Darker amber outline  
                                alpha=0.8, linewidth=2.5,
                                linestyle='--',  # Dashed outline to show it's planned
                                label='Target Dock Pose')
        ax.add_patch(dock_pose_patch)
        dock_pose_drawn = True
    
    return dock_pose_patch, dock_pose_drawn



def update_dockable_areas(ax, dockable_areas, current_time, dockable_area_plots, drawn_dockable_areas):
    """Update dockable area display - only draw new areas when they appear"""
    
    # Check for new dockable areas that should appear at current time
    for i, area_data in enumerate(dockable_areas):
        if current_time >= area_data['time'] and i not in drawn_dockable_areas:
            # This area should be visible and hasn't been drawn yet
            area_line, = ax.plot(area_data['x'], area_data['y'], 
                               color='#00e5ff', linewidth=2.5, alpha=0.9, 
                               label='Dockable Area' if not dockable_area_plots else None)
            area_fill = ax.fill(area_data['x'], area_data['y'], 
                              color='#00e5ff', alpha=0.25)
            
            dockable_area_plots.extend([area_line] + area_fill)
            drawn_dockable_areas.add(i)  # Mark this area as drawn
    
    return dockable_area_plots



def update_tracks(ax, tracks_by_time, current_time, track_plots, current_tracks_index):
    """Update tracks display - remove old tracks and draw current ones"""
    
    # Find the most recent tracks that should be visible
    latest_tracks_index = -1
    for i, track_data in enumerate(tracks_by_time):
        if current_time >= track_data['time']:
            latest_tracks_index = i
        else:
            break
    
    # If we have a new tracks update, redraw everything
    if latest_tracks_index != current_tracks_index and latest_tracks_index >= 0:
        # Remove all old track plots
        for plot in track_plots:
            if hasattr(plot, 'remove'):
                plot.remove()
        track_plots.clear()
        
        # Draw the latest tracks
        track_data = tracks_by_time[latest_tracks_index]
        
        for track in track_data['tracks']:
            # Plot footprint if available
            if track['footprint'] and len(track['footprint']) > 2:
                footprint_x = []
                footprint_y = []
                
                for fp_point in track['footprint']:
                    footprint_x.append(fp_point['x'])  # East
                    footprint_y.append(fp_point['y'])  # North
                
                # Close the footprint polygon
                if len(footprint_x) > 0:
                    footprint_x.append(footprint_x[0])
                    footprint_y.append(footprint_y[0])
                    
                    # Draw track footprint
                    track_line, = ax.plot(footprint_x, footprint_y, 
                                       color='#ff6b35', linewidth=1.5, alpha=0.8,
                                       label='Track Footprints' if not track_plots else None)
                    track_fill = ax.fill(footprint_x, footprint_y, 
                                       color='#ff6b35', alpha=0.3)
                    
                    track_plots.extend([track_line] + track_fill)
                    
                    # Add track ID at center
                    center_x = sum(footprint_x[:-1]) / len(footprint_x[:-1])
                    center_y = sum(footprint_y[:-1]) / len(footprint_y[:-1])
                    
                    track_text = ax.annotate(f'{int(track["id"])}', (center_x, center_y), 
                                           ha='center', va='center', fontsize=9, 
                                           alpha=0.9, color='#ff4500', fontweight='bold')
                    track_plots.append(track_text)
        
        current_tracks_index = latest_tracks_index
    
    return track_plots, current_tracks_index



def update_thrusters(ax_thrusters, thruster_data, thruster_bars, thruster_angles, current_time):
    """Update thruster bar plots and angle indicators with current RPM and angle values"""
    
    for i in range(4):
        thruster_key = f'thruster_{i+1}'
        if thruster_data[thruster_key] is not None:
            df = thruster_data[thruster_key]
            
            # Find the most recent throttle and angle values before current time
            valid_data = df[df['rel_time'] <= current_time]
            if not valid_data.empty:
                latest_throttle = valid_data['throttle_reference'].iloc[-1]
                latest_angle = valid_data['angle_reference'].iloc[-1]
                
                # Update bar width (ensure positive values only)
                bar_width = max(0, abs(latest_throttle))  # Take absolute value and ensure >= 0
                thruster_bars[i][0].set_width(bar_width)
                
                # Color coding based on original sign
                if latest_throttle >= 0:
                    thruster_bars[i][0].set_color('#00ff88')
                else:
                    thruster_bars[i][0].set_color('#ff4444')  # Red for negative (reverse)
                
                # Update angle indicator
                # Convert angle to radians and create arrow direction
                angle_rad = np.deg2rad(latest_angle)
                center_x, center_y = 950, i
                arrow_length = 12
                
                # Calculate arrow end point
                end_x = center_x + arrow_length * np.cos(angle_rad)
                end_y = center_y + arrow_length * np.sin(angle_rad)
                start_x = center_x - arrow_length * np.cos(angle_rad)
                start_y = center_y - arrow_length * np.sin(angle_rad)
                
                # Update arrow position
                thruster_angles[i]['arrow'].set_position((end_x, end_y))
                thruster_angles[i]['arrow'].xy = (end_x, end_y)
                thruster_angles[i]['arrow'].xytext = (start_x, start_y)



if __name__ == "__main__":
    animate_docking_experiment("data/exp2_aug_simon_2025-06-06-09-26-37.bag")
