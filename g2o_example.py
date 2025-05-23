import numpy as np
import matplotlib.pyplot as plt
import g2o
from generate_test_data import plot_results

def optimize_with_g2o(
        odometry,
        loop_closures,
        prior_noise=0.1,
        odom_noise=0.2,
        loop_noise=0.1,
        initial_pose=None,
        fill_debug_results=False
    ):
    """
    Optimize a pose graph using g2o.

    Args:
        odometry: List of (dx, dy, dtheta) relative pose measurements in global frame
        loop_closures: List of (i, j, dx, dy, dtheta) loop closure constraints
        prior_noise: Noise for the prior on the first pose
        odom_noise: Noise for odometry measurements
        loop_noise: Noise for loop closure measurements
        initial_pose: Initial pose [x, y, theta], default is [0, 0, 0]
        fill_debug_results: Whether to return debug information

    Returns:
        optimized_poses: Numpy array of optimized poses
        debug_results: Debug information if requested
    """
    # Create a g2o optimizer
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE2(g2o.LinearSolverDenseSE2())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)

    # Set initial pose
    if initial_pose is None:
        initial_pose = [0, 0, 0]
    x0, y0, theta0 = initial_pose

    # Add the first vertex (fixed)
    v0 = g2o.VertexSE2()
    v0.set_id(0)
    v0.set_estimate(g2o.SE2(x0, y0, theta0))
    v0.set_fixed(True)
    optimizer.add_vertex(v0)

    # Reconstruct poses for initial estimates
    poses = [g2o.SE2(x0, y0, theta0)]
    vertices = [v0]

    debug_results = []

    # Add vertices and odometry edges
    for i, (dx_global, dy_global, dtheta) in enumerate(odometry):
        prev_pose = poses[-1]
        prev_theta = prev_pose.rotation().angle()

        # Convert global odometry to local frame
        c = np.cos(prev_theta)
        s = np.sin(prev_theta)

        # Apply inverse rotation to get measurements in local frame
        dx_local = c * dx_global + s * dy_global
        dy_local = -s * dx_global + c * dy_global

        # Create relative transform in local frame
        delta = g2o.SE2(dx_local, dy_local, dtheta)

        # Compute global pose
        next_pose = prev_pose * delta
        poses.append(next_pose)

        # Add vertex for this pose
        v = g2o.VertexSE2()
        v.set_id(i + 1)
        v.set_estimate(next_pose)
        optimizer.add_vertex(v)
        vertices.append(v)

        # Add edge (odometry constraint)
        edge = g2o.EdgeSE2()
        edge.set_vertex(0, vertices[i])
        edge.set_vertex(1, vertices[i + 1])
        edge.set_measurement(delta)

        # Set information matrix (inverse of covariance)
        information = np.eye(3)
        information[0, 0] = 1.0 / (odom_noise ** 2)
        information[1, 1] = 1.0 / (odom_noise ** 2)
        information[2, 2] = 1.0 / ((odom_noise * 0.1) ** 2)  # Less weight on rotation
        edge.set_information(information)

        optimizer.add_edge(edge)

        if fill_debug_results:
            debug_results.append({
                "step": i+1,
                "global_odometry": (dx_global, dy_global, dtheta),
                "converted_to_local": (dx_local, dy_local),
                "previous_pose": prev_pose,
                "next_pose": next_pose
            })

    # Add loop closure edges
    for i, j, dx_global, dy_global, dtheta in loop_closures:
        # Ensure indices are integers
        i_idx = int(i)
        j_idx = int(j)

        # Get the source pose for this loop closure
        source_pose = poses[i_idx]
        source_theta = source_pose.rotation().angle()

        # Convert global displacement to local frame of source pose
        c = np.cos(source_theta)
        s = np.sin(source_theta)

        dx_local = c * dx_global + s * dy_global
        dy_local = -s * dx_global + c * dy_global

        # Create relative transform for loop closure
        delta = g2o.SE2(dx_local, dy_local, dtheta)

        # Add edge (loop closure constraint)
        edge = g2o.EdgeSE2()
        edge.set_vertex(0, vertices[i_idx])
        edge.set_vertex(1, vertices[j_idx])
        edge.set_measurement(delta)

        # Set information matrix (inverse of covariance)
        information = np.eye(3)
        information[0, 0] = 1.0 / (loop_noise ** 2)
        information[1, 1] = 1.0 / (loop_noise ** 2)
        information[2, 2] = 1.0 / ((loop_noise * 0.1) ** 2)  # Less weight on rotation
        edge.set_information(information)

        optimizer.add_edge(edge)

    # Optimize
    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(20)

    # Extract results
    optimized_poses = np.zeros((len(poses), 3))
    for i in range(len(poses)):
        v = optimizer.vertex(i)
        pose = v.estimate()
        xy = pose.translation()
        optimized_poses[i] = [xy[0], xy[1], pose.rotation().angle()]

    return optimized_poses, debug_results


def show_debug_results(debug_results, gt_poses):
    for i, entry in enumerate(debug_results):
            dx_global, dy_global, dtheta = entry["global_odometry"]
            dx_local, dy_local = entry["converted_to_local"]
            prev_pose = entry["previous_pose"]
            next_pose = entry["next_pose"]
            print(f"Step {i+1}:")
            print(f"  Global odometry: dx={dx_global:.3f}, dy={dy_global:.3f}, dtheta={dtheta:.3f}")
            print(f"  Converted to local: dx={dx_local:.3f}, dy={dy_local:.3f}")
            print(f"  Previous pose: x={prev_pose.translation()[0]:.3f}, y={prev_pose.translation()[1]:.3f}, θ={prev_pose.rotation().angle():.3f}")
            print(f"  GT pose: x={gt_poses[i+1][0]:.3f}, y={gt_poses[i+1][1]:.3f}, θ={gt_poses[i+1][2]:.3f}")
            print(f"  Next pose: x={next_pose.translation()[0]:.3f}, y={next_pose.translation()[1]:.3f}, θ={next_pose.rotation().angle():.3f}")
            print()

if __name__ == "__main__":
    # Load the data
    data = np.load('trajectory_data.npz', allow_pickle=True)
    gt_poses = data['gt_poses']
    odometry = data['odometry']
    loop_closures = data['loop_closures']

    # Get initial pose from ground truth
    initial_pose = gt_poses[0]

    # Reconstruct trajectory from odometry
    from generate_test_data import reconstruct_from_odometry
    reconstructed = reconstruct_from_odometry(odometry, initial_pose=initial_pose)

    # Optimize using g2o
    fill_debug_results = False
    optimized, debug_results = optimize_with_g2o(
        odometry, loop_closures, initial_pose=initial_pose, fill_debug_results=fill_debug_results
    )
    if fill_debug_results:
        show_debug_results(debug_results, gt_poses=gt_poses)

    # Calculate error metrics
    odom_error = np.mean(np.sqrt(np.sum((reconstructed[:, :2] - gt_poses[:, :2])**2, axis=1)))
    g2o_error = np.mean(np.sqrt(np.sum((optimized[:, :2] - gt_poses[:, :2])**2, axis=1)))

    print(f"Mean position error (odometry only): {odom_error:.4f} m")
    print(f"Mean position error (g2o optimized): {g2o_error:.4f} m")
    print(f"Improvement: {(1 - g2o_error/odom_error) * 100:.2f}%")

    # Plot results
    plot_results(gt_poses, reconstructed, optimized, loop_closures)
