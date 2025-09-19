#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

# Import functions from main.py
from main import (
    get_pose_data,
    get_manager_state_data, 
    get_manager_path_data,
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




def animate_docking_experiment(bag_path):
    """Create real-time animated visualization of the docking experiment"""
    
    print("Extracting pose data for animation...")
    pose_data = get_pose_data(bag_path)
    
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
    
    # Set up the figure with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#1a1a1a')
    
    # Initial plot setup
    ax.set_xlabel('East (m) →', fontsize=14, color='white', fontweight='bold')
    ax.set_ylabel('North (m) ↑', fontsize=14, color='white', fontweight='bold')
    ax.set_title('Autonomous Docking Experiment - Real-time Visualization', 
                fontsize=16, color='white', fontweight='bold', pad=20)
    
    # Set fixed axis limits
    ax.axis('equal')
    ax.set_xlim(980, 1040)  # East (m)
    ax.set_ylim(240, 300)   # North (m)
    
    # Grid styling
    ax.grid(True, alpha=0.3, color='#404040', linestyle='-', linewidth=0.5)
    ax.tick_params(colors='white', labelsize=12)
    
    # Initialize animation elements
    vessel_patch = None
    trajectory_line = None
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=14, color='#00ff88', fontweight='bold',
                       verticalalignment='top')
    
    def animate(frame):
        nonlocal vessel_patch, trajectory_line
        
        current_time = times[frame]
        
        # Update trajectory
        trajectory_line = update_trajectory(ax, y_pos, x_pos, frame, trajectory_line)
        
        # Update vessel
        vessel_patch = update_vessel(ax, y_pos, x_pos, headings, frame, vessel_patch)
        
        # Update time display
        time_text.set_text(f'Time: {current_time:.1f}s')
        
        return [vessel_patch, time_text] + ([trajectory_line] if trajectory_line else [])
    
    # Create animation
    print(f"Creating animation with {len(pose_data)} frames...")
    print(f"Experiment duration: {times[-1]:.1f} seconds")
    
    # Animation timing
    avg_dt = np.mean(np.diff(times)) * 1000  # Convert to milliseconds
    interval = max(10, min(100, avg_dt))  # Clamp between 10-100ms
    
    anim = FuncAnimation(fig, animate, frames=len(pose_data), 
                        interval=interval, blit=False, repeat=True)
    
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


if __name__ == "__main__":
    # animate_docking_experiment("data/exp2_aug_simon_2025-06-06-09-26-37.bag")
    # Test the fixed parser
    from main import get_manager_path_data

    bag_path = "data/exp2_aug_simon_2025-06-06-09-26-37.bag"
    manager_path_data = get_manager_path_data(bag_path)

    print("Parsing first path with fixed parser:")
    first_poses = manager_path_data['poses'].iloc[0]
    x_coords, y_coords = parse_manager_path_poses(first_poses)

    print(f"Found {len(x_coords)} path points")
    print(f"X range (North): {min(x_coords):.1f} to {max(x_coords):.1f}")  
    print(f"Y range (East): {min(y_coords):.1f} to {max(y_coords):.1f}")
    print(f"First few points: {list(zip(x_coords[:5], y_coords[:5]))}")
