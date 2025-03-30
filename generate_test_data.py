import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math

def generate_trajectory_with_loop_closure(num_keyframes=12, noise_level=0.1):
    """
    Generate a 2D trajectory with a loop closure.
    Returns ground truth, noisy odometry, and loop closure constraint.
    """
    # Create ground truth trajectory (circular path)
    theta = np.linspace(0, 2.5 * np.pi, num_keyframes)
    radius = 5.0
    x_gt = radius * np.cos(theta)
    y_gt = radius * np.sin(theta)

    # Add heading (theta) at each pose
    # Heading points tangent to the circle
    heading_gt = theta + np.pi/2

    # Combine into ground truth poses [x, y, theta]
    gt_poses = np.column_stack((x_gt, y_gt, heading_gt))

    # Create noisy odometry measurements
    odometry = []
    for i in range(1, num_keyframes):
        dx = x_gt[i] - x_gt[i-1]
        dy = y_gt[i] - y_gt[i-1]
        dtheta = heading_gt[i] - heading_gt[i-1]

        # Normalize angle difference
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

        # Add noise to odometry
        noise_x = np.random.normal(0, noise_level)
        noise_y = np.random.normal(0, noise_level)
        noise_theta = np.random.normal(0, noise_level * 2)

        odometry.append((dx + noise_x, dy + noise_y, dtheta + noise_theta))

    # Find loop closure
    # Check for keyframes that are close to each other
    loop_closures = []
    for i in range(num_keyframes - 4):  # Exclude adjacent poses
        for j in range(i + 3, num_keyframes):
            dist = np.sqrt((x_gt[i] - x_gt[j])**2 + (y_gt[i] - y_gt[j])**2)
            if dist < 1.5:  # If they're close, add a loop closure
                dx = x_gt[j] - x_gt[i]
                dy = y_gt[j] - y_gt[i]
                dtheta = heading_gt[j] - heading_gt[i]
                dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

                # Add small noise to loop closure measurement
                noise_x = np.random.normal(0, noise_level * 0.3)
                noise_y = np.random.normal(0, noise_level * 0.3)
                noise_theta = np.random.normal(0, noise_level * 0.5)

                loop_closures.append((i, j, dx + noise_x, dy + noise_y, dtheta + noise_theta))

    return gt_poses, odometry, loop_closures

def reconstruct_from_odometry(odometry, initial_pose=None):
    """
    Reconstruct a trajectory from odometry measurements.
    First pose is at origin with zero heading.
    """
    num_poses = len(odometry) + 1
    poses = np.zeros((num_poses, 3))
    if initial_pose is not None:
        poses[0] = initial_pose
    else:
        poses[0] = [0, 0, 0]

    for i in range(1, num_poses):
        dx, dy, dtheta = odometry[i-1]

        # Current pose heading
        theta = poses[i-1, 2]

        # TODO: revisit frame conversions (odometry was calculated in the global frame, but will be used in local frame)
        dx_global = dx
        dy_global = dy
        dtheta = dtheta

        # Update pose
        poses[i, 0] = poses[i-1, 0] + dx_global
        poses[i, 1] = poses[i-1, 1] + dy_global
        poses[i, 2] = poses[i-1, 2] + dtheta

        # Normalize angle
        poses[i, 2] = (poses[i, 2] + np.pi) % (2 * np.pi) - np.pi

    return poses

def plot_results(gt_poses, reconstructed, optimized=None, loop_closures=None):
    """
    Plot ground truth, reconstructed, and optimized trajectories.
    """
    plt.figure(figsize=(10, 8))

    # Plot ground truth
    plt.plot(gt_poses[:, 0], gt_poses[:, 1], 'g-', label='Ground Truth')
    plt.scatter(gt_poses[:, 0], gt_poses[:, 1], c='g', s=50)

    # Plot arrows for orientation
    for i in range(len(gt_poses)):
        x, y, theta = gt_poses[i]
        dx, dy = 0.3 * np.cos(theta), 0.3 * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.2, fc='g', ec='g', alpha=0.5)

    # Plot reconstructed
    plt.plot(reconstructed[:, 0], reconstructed[:, 1], 'r-', label='Odometry Only')
    plt.scatter(reconstructed[:, 0], reconstructed[:, 1], c='r', s=50)

    # Plot arrows for orientation
    for i in range(len(reconstructed)):
        x, y, theta = reconstructed[i]
        dx, dy = 0.3 * np.cos(theta), 0.3 * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.2, fc='r', ec='r', alpha=0.5)

    # Plot optimized if provided
    if optimized is not None:
        plt.plot(optimized[:, 0], optimized[:, 1], 'b-', label='Optimized')
        plt.scatter(optimized[:, 0], optimized[:, 1], c='b', s=50)

        # Plot arrows for orientation
        for i in range(len(optimized)):
            x, y, theta = optimized[i]
            dx, dy = 0.3 * np.cos(theta), 0.3 * np.sin(theta)
            plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.2, fc='b', ec='b', alpha=0.5)

    # Plot loop closures if provided
    if loop_closures is not None:
        # Plot loop closures for reconstructed trajectory
        for i, j, _, _, _ in loop_closures:
            # Convert indices to integers for array indexing
            i_idx = int(i)
            j_idx = int(j)

            plt.plot([reconstructed[i_idx, 0], reconstructed[j_idx, 0]], 
                     [reconstructed[i_idx, 1], reconstructed[j_idx, 1]], 
                     'm--', linewidth=1.5, alpha=0.7, label='_Loop Closure (Odom)')

            # Add markers to highlight the points involved in loop closures
            plt.scatter(reconstructed[i_idx, 0], reconstructed[i_idx, 1], c='m', s=100, alpha=0.7, marker='o', edgecolors='k')
            plt.scatter(reconstructed[j_idx, 0], reconstructed[j_idx, 1], c='m', s=100, alpha=0.7, marker='o', edgecolors='k')

        # If optimized trajectory is available, plot loop closures there too
        if optimized is not None:
            for i, j, _, _, _ in loop_closures:
                # Convert indices to integers for array indexing
                i_idx = int(i)
                j_idx = int(j)

                plt.plot([optimized[i_idx, 0], optimized[j_idx, 0]], 
                         [optimized[i_idx, 1], optimized[j_idx, 1]], 
                         'c--', linewidth=1.5, alpha=0.7, label='_Loop Closure (Opt)')

                # Add markers to highlight the points involved in loop closures
                plt.scatter(optimized[i_idx, 0], optimized[i_idx, 1], c='c', s=100, alpha=0.7, marker='o', edgecolors='k')
                plt.scatter(optimized[j_idx, 0], optimized[j_idx, 1], c='c', s=100, alpha=0.7, marker='o', edgecolors='k')

            # Add custom legend entries for loop closures
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color='m', lw=1.5, linestyle='--'),
                Line2D([0], [0], color='c', lw=1.5, linestyle='--')
            ]
            plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + custom_lines,
                      labels=plt.gca().get_legend_handles_labels()[1] + ['Loop Closures (Odom)', 'Loop Closures (Opt)'])
        else:
            # Add custom legend entry for loop closures if only reconstructed trajectory
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='m', lw=1.5, linestyle='--')]
            plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + custom_lines,
                      labels=plt.gca().get_legend_handles_labels()[1] + ['Loop Closures'])
    else:
        plt.legend()

    plt.grid(True)
    plt.axis('equal')
    plt.title('SLAM Trajectory Optimization')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()

if __name__ == "__main__":
    # Generate test data
    np.random.seed(42)  # For reproducibility
    gt_poses, odometry, loop_closures = generate_trajectory_with_loop_closure(12, 0.2)

    # Reconstruct trajectory from odometry
    reconstructed = reconstruct_from_odometry(odometry, initial_pose=gt_poses[0])

    # Plot
    plot_results(gt_poses, reconstructed, loop_closures=loop_closures)

    print(f"Number of poses: {len(gt_poses)}")
    print(f"Number of odometry measurements: {len(odometry)}")
    print(f"Number of loop closures: {len(loop_closures)}")

    # Save data for optimization examples
    np.savez('trajectory_data.npz', 
             gt_poses=gt_poses, 
             odometry=odometry, 
             loop_closures=loop_closures)

    print("Data saved to trajectory_data.npz") 