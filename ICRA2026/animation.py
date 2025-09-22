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



def animate_docking_experiment(bag_path, save_video=False):
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
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor('#0a0a0a')

    # Create subplot layout: video left (16:9), map right (smaller), bottom plots  
    ax_video = plt.subplot2grid((4, 10), (0, 0), rowspan=3, colspan=7)     # Video: 3 rows, 7 cols (more width)
    ax_map = plt.subplot2grid((4, 10), (0, 7), rowspan=3, colspan=3)       # Map: 3 rows, 3 cols (smaller)
    ax_vessel = plt.subplot2grid((4, 10), (3, 0), colspan=4)               # Vessel: row 3
    ax_manager = plt.subplot2grid((4, 10), (3, 4), colspan=6)              # Manager: row 3

    # Set up video placeholder
    ax_video.set_facecolor('#000000')
    # ax_video.set_title('Experiment Video Feed', fontsize=14, color='white', fontweight='bold')
    ax_video.text(0.5, 0.5, 'Video Feed\n(16:9)', ha='center', va='center', 
                transform=ax_video.transAxes, fontsize=16, color='#666666', fontweight='bold')
    ax_video.set_xticks([])
    ax_video.set_yticks([])
    
    # Set up map subplot
    ax_map.set_facecolor('#1a1a1a')
    ax_map.set_xlabel('East (m) →', fontsize=14, color='white', fontweight='bold')
    ax_map.set_ylabel('North (m) →', fontsize=14, color='white', fontweight='bold')
    ax_map.set_title('', 
                    fontsize=16, color='white', fontweight='bold', pad=20)
    
    # Set fixed axis limits for map
    ax_map.axis('equal')
    ax_map.set_xlim(980, 1040)  # East (m)
    ax_map.set_ylim(240, 300)   # North (m)
    
    # Grid styling for map
    ax_map.grid(True, alpha=0.3, color='#404040', linestyle='-', linewidth=0.5)
    ax_map.tick_params(colors='white', labelsize=12)
    
    # Set up vessel diagram subplot
    ax_vessel.set_facecolor('#1a1a1a')
    ax_vessel.set_title('Thruster Configuration', fontsize=12, color='white', fontweight='bold')
    ax_vessel.set_xlim(-4, 4)
    ax_vessel.set_ylim(-3, 3)
    ax_vessel.axis('equal')
    ax_vessel.axis('off')  # Remove axes

    # Draw vessel using existing function (nose pointing right: heading = 0)
    vessel_shape = create_vessel_shape(0, 0, np.pi/2)  # Center at origin, nose right
    vessel_patch_static = Polygon(vessel_shape, facecolor='#4a90e2', 
                                edgecolor='#37474f', alpha=0.6, linewidth=2)
    ax_vessel.add_patch(vessel_patch_static)

    # Initialize thruster arrows at correct positions (ship pointing right)
    thruster_positions = [
        [2.0, 1.0],   # T1: Bow port (front-left)
        [2.0, -1.0],  # T2: Bow starboard (front-right)
        [-2.0, -1.0], # T3: Stern port (back-right as you specified)
        [-2.0, 1.0]   # T4: Stern starboard (back-left as you specified)
    ]

    # Set up manager state subplot
    ax_manager.set_facecolor('#1a1a1a')
    ax_manager.set_title('Manager State', fontsize=12, color='white', fontweight='bold')
    ax_manager.tick_params(colors='white', labelsize=10)
    ax_manager.grid(True, alpha=0.3, color='#404040', linestyle='-', linewidth=0.5)


    # Extract manager state data
    manager_state_data = get_manager_state_data(bag_path)
    manager_state_data['rel_time'] = manager_state_data['Time'] - start_time

    # Process manager state for animation
    unique_states = manager_state_data['data'].unique()
    state_to_y = {state: i for i, state in enumerate(unique_states)}
    manager_state_data['y_value'] = manager_state_data['data'].map(state_to_y)

    # Draw static floating dock contour on map
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
    
    # Initialize animation elements for map
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
    manager_state_line = None
    manager_state_current_point = None


    thruster_arrows = []
    thruster_color = '#ffaa00'
    for i, pos in enumerate(thruster_positions):
        # Create thruster arrows
        arrow = ax_vessel.annotate('', xy=(pos[0]+0.8, pos[1]), xytext=(pos[0], pos[1]),
                                arrowprops=dict(arrowstyle='->', color=thruster_color, 
                                                lw=4, alpha=0.9, shrinkA=0, shrinkB=0,
                                                mutation_scale=15))
        thruster_arrows.append(arrow)
        
        # Add thruster label (positioned clearly outside the vessel)
        if pos[0] > 0:  # Front thrusters
            label_x = pos[0] + 1.2
        else:  # Stern thrusters  
            label_x = pos[0] - 1.2
            
        ax_vessel.text(label_x, pos[1], f'T{i+1}', ha='center', va='center', 
                    color=thruster_color, fontsize=12, fontweight='bold')
    
    time_text = ax_map.text(0.02, 0.98, '', transform=ax_map.transAxes, 
                           fontsize=14, color='#00ff88', fontweight='bold',
                           verticalalignment='top')
    
    
    def animate(frame):
        nonlocal vessel_patch, trajectory_line, path_plots, drawn_paths, dock_pose_patch, dock_pose_drawn, dockable_area_plots, drawn_dockable_areas, track_plots, current_tracks_index, manager_state_line, manager_state_current_point
        
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
        
        # Update vessel thruster diagram
        update_vessel_thrusters(ax_vessel, thruster_data, thruster_arrows, thruster_positions, current_time)
        
        # Update time display
        time_text.set_text(f'Time: {current_time:.1f}s')
        
        # Update manager state
        manager_state_line, manager_state_current_point = update_manager_state(
            ax_manager, manager_state_data, current_time, manager_state_line, 
            manager_state_current_point, unique_states)

        return_list = [vessel_patch, time_text] + ([trajectory_line] if trajectory_line else []) + path_plots + dockable_area_plots + track_plots + [arrow for arrow in thruster_arrows]
        if dock_pose_patch:
            return_list.append(dock_pose_patch)
        if manager_state_line:
            return_list.append(manager_state_line)
        if manager_state_current_point:
            return_list.append(manager_state_current_point)

        return return_list
    
    # Create animation
    print(f"Creating animation with {len(pose_data)} frames...")
    print(f"Experiment duration: {times[-1]:.1f} seconds")
    print(f"Found {len(path_lines)} manager path updates")

    # Animation timing
    avg_dt = np.mean(np.diff(times)) * 1000  # Convert to milliseconds
    interval = max(10, min(100, avg_dt))  # Real-time playback
    
    anim = FuncAnimation(fig, animate, frames=len(pose_data), 
                        interval=interval, blit=False, repeat=False)
    
    print(f"Animation ready! Playback speed: {1000/interval:.1f} FPS")
    plt.tight_layout()

    if save_video:
        print("Saving animation as MP4...")
        save_fps = int(1000/interval)  # Match the playback speed
        anim.save('docking_experiment_animation.mp4', writer='ffmpeg', fps=save_fps, bitrate=1800)
        print(f"Animation saved as 'docking_experiment_animation.mp4' at {save_fps} FPS")
        plt.close(fig)  # Close figure to free memory
    else:
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



def update_vessel_thrusters(ax_vessel, thruster_data, thruster_arrows, thruster_positions, current_time):
    """Update thruster arrows on vessel diagram with current RPM and angle values"""
    
    # Remove all existing arrows
    for arrow in thruster_arrows:
        arrow.remove()
    thruster_arrows.clear()
    
    angle_offsets = [90, -90, 90, -90]
    # Recreate arrows with current data
    for i in range(4):
        thruster_key = f'thruster_{i+1}'
        if thruster_data[thruster_key] is not None:
            df = thruster_data[thruster_key]
            
            # Find the most recent throttle and angle values before current time
            valid_data = df[df['rel_time'] <= current_time]
            if not valid_data.empty:
                latest_throttle = valid_data['throttle_reference'].iloc[-1]
                latest_angle = valid_data['angle_reference'].iloc[-1]
                
                # Calculate arrow length based on throttle magnitude
                max_arrow_length = 2.0
                min_arrow_length = 0.3
                arrow_length = min_arrow_length + (abs(latest_throttle) / 1000.0) * (max_arrow_length - min_arrow_length)
                
                # Determine arrow direction and color based on throttle sign
                if latest_throttle < 0:
                    angle_rad = np.deg2rad(latest_angle + angle_offsets[i] + 180)  # Reverse direction
                    color = '#ff4444'  # Red for reverse thrust
                else:
                    angle_rad = np.deg2rad(latest_angle + angle_offsets[i])
                    color = '#ffaa00'  # Blue for forward thrust
                
                # Calculate arrow end point relative to thruster position
                pos = thruster_positions[i]
                end_x = pos[0] + arrow_length * np.cos(angle_rad)
                end_y = pos[1] + arrow_length * np.sin(angle_rad)
                
                # Create new arrow
                arrow = ax_vessel.annotate('', xy=(end_x, end_y), xytext=(pos[0], pos[1]),
                                        arrowprops=dict(arrowstyle='->', color=color, 
                                                        lw=4, alpha=0.9, shrinkA=0, shrinkB=0,
                                                        mutation_scale=15))
                thruster_arrows.append(arrow)
            else:
                # No data yet, create minimal default arrow
                pos = thruster_positions[i]
                arrow = ax_vessel.annotate('', xy=(pos[0]+0.3, pos[1]), xytext=(pos[0], pos[1]),
                                        arrowprops=dict(arrowstyle='->', color='#ffaa00', 
                                                        lw=4, alpha=0.3, shrinkA=0, shrinkB=0,
                                                        mutation_scale=15))
                thruster_arrows.append(arrow)



def update_manager_state(ax, manager_data, current_time, state_line, current_point, unique_states):
    """Update manager state display"""
    # Remove previous plots
    if state_line is not None:
        state_line.remove()
    if current_point is not None:
        current_point.remove()
    
    # Get data up to current time
    current_data = manager_data[manager_data['rel_time'] <= current_time]
    
    if not current_data.empty:
        # Plot the state history
        state_line = ax.step(current_data['rel_time'], current_data['y_value'], 
                           where='post', linewidth=2, color='#00ff88')[0]
        
        # Highlight current state
        latest_state = current_data.iloc[-1]
        current_point = ax.scatter(latest_state['rel_time'], latest_state['y_value'], 
                                 color='#ff4444', s=100, zorder=5)
        
        # Set up y-axis labels
        ax.set_yticks(range(len(unique_states)))
        ax.set_yticklabels(unique_states)
        ax.set_ylabel('State', color='white')
        ax.set_xlabel('Time (s)', color='white')
        
        # Auto-adjust x-axis to show some context
        ax.set_xlim(max(0, current_time - 30), current_time + 5)
    
    return state_line, current_point






def debug_thruster_data(bag_path):
    """Debug function to check actual thruster RPM ranges in the bag file"""
    print("=== THRUSTER DATA DEBUG ===")
    
    for i in range(1, 5):
        topic = f"/actuator_ref_{i}"
        try:
            bag = bagpy.bagreader(bag_path)
            csv_file = bag.message_by_topic(topic)
            df = pd.read_csv(csv_file)
            
            # Check RPM values
            rpm_values = df['throttle_reference']
            angle_values = df['angle_reference']
            
            print(f"\nThruster {i} ({topic}):")
            print(f"  Total messages: {len(df)}")
            print(f"  RPM range: {rpm_values.min()} to {rpm_values.max()}")
            print(f"  Angle range: {angle_values.min()}° to {angle_values.max()}°")
            print(f"  Negative RPM count: {(rpm_values < 0).sum()}")
            print(f"  Zero RPM count: {(rpm_values == 0).sum()}")
            print(f"  Sample RPM values: {rpm_values.head(10).tolist()}")
            
        except Exception as e:
            print(f"Error loading {topic}: {e}")
    
    print("\n=== END DEBUG ===")

if __name__ == "__main__":
    bag_file = "data/exp2_aug_simon_2025-06-06-09-26-37.bag"
    animate_docking_experiment(bag_file, save_video=True)
    # debug_thruster_data(bag_file)
