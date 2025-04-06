import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam.symbol_shorthand import X
from generate_test_data import plot_results

def optimize_with_gtsam(
        odometry,
        loop_closures,
        prior_noise=0.1,
        odom_noise=0.2,
        loop_noise=0.1,
        initial_pose=None,
        fill_debug_results=False
    ):
    """
    Optimize a pose graph using GTSAM.

    Args:
        odometry: List of (dx, dy, dtheta) relative pose measurements in global frame
        loop_closures: List of (i, j, dx, dy, dtheta) loop closure constraints
        prior_noise: Noise for the prior on the first pose
        odom_noise: Noise for odometry measurements
        loop_noise: Noise for loop closure measurements
        initial_pose: Initial pose [x, y, theta], default is [0, 0, 0]
        gt_poses: Optional ground truth poses for debugging

    Returns:
        optimized_poses: Numpy array of optimized poses
    """
    # Create a factor graph
    graph = gtsam.NonlinearFactorGraph()

    # Create the initial estimate
    initial_estimate = gtsam.Values()

    # Set initial pose
    if initial_pose is None:
        initial_pose = [0, 0, 0]
    x0, y0, theta0 = initial_pose

    # Add a prior on the first pose
    prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_noise, prior_noise, prior_noise * 0.1]))
    graph.add(gtsam.PriorFactorPose2(X(0), gtsam.Pose2(x0, y0, theta0), prior_noise_model))

    # Set initial estimate for first pose
    initial_estimate.insert(X(0), gtsam.Pose2(x0, y0, theta0))

    # Add odometry factors
    odom_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([odom_noise, odom_noise, odom_noise * 0.1]))

    # Generate poses and add factors
    poses = [gtsam.Pose2(x0, y0, theta0)]

    debug_results = []

    for i, (dx_global, dy_global, dtheta) in enumerate(odometry):
        prev_pose = poses[-1]
        prev_theta = prev_pose.theta()

        # Convert global odometry to local frame
        # Rotation matrix from global to local frame
        c = np.cos(prev_theta)
        s = np.sin(prev_theta)

        # Apply inverse rotation to get measurements in local frame
        dx_local = c * dx_global + s * dy_global
        dy_local = -s * dx_global + c * dy_global

        # Create odometry measurement as relative pose in local frame
        delta = gtsam.Pose2(dx_local, dy_local, dtheta)

        # Calculate next pose
        next_pose = prev_pose.compose(delta)
        poses.append(next_pose)

        # Add odometry factor
        graph.add(gtsam.BetweenFactorPose2(X(i), X(i+1), delta, odom_noise_model))

        # Add to initial estimate
        initial_estimate.insert(X(i+1), next_pose)

        if fill_debug_results:
            debug_results.append({
                "step": i+1,
                "global_odometry": (dx_global, dy_global, dtheta),
                "converted_to_local": (dx_local, dy_local),
                "previous_pose": prev_pose,
                "next_pose": next_pose
            })

    # Add loop closure factors - these also need frame conversion
    loop_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([loop_noise, loop_noise, loop_noise * 0.1]))

    for i, j, dx_global, dy_global, dtheta in loop_closures:
        # Ensure indices are integers
        i_idx = int(i)
        j_idx = int(j)

        # Get the source pose for this loop closure
        source_pose = poses[i_idx]
        source_theta = source_pose.theta()

        # Convert global displacement to local frame of the source pose
        c = np.cos(source_theta)
        s = np.sin(source_theta)

        dx_local = c * dx_global + s * dy_global
        dy_local = -s * dx_global + c * dy_global

        # Create loop closure measurement as relative pose
        delta = gtsam.Pose2(dx_local, dy_local, dtheta)

        # Add the factor
        graph.add(gtsam.BetweenFactorPose2(X(i_idx), X(j_idx), delta, loop_noise_model))

    # Set up the optimizer
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setRelativeErrorTol(1e-5)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # Optimize
    result = optimizer.optimize()

    # Extract results
    optimized_poses = np.zeros((len(poses), 3))
    for i in range(len(poses)):
        pose = result.atPose2(X(i))
        optimized_poses[i] = [pose.x(), pose.y(), pose.theta()]

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
            print(f"  Previous pose: x={prev_pose.x():.3f}, y={prev_pose.y():.3f}, θ={prev_pose.theta():.3f}")
            print(f"  GT pose: x={gt_poses[i+1][0]:.3f}, y={gt_poses[i+1][1]:.3f}, θ={gt_poses[i+1][2]:.3f}")
            print(f"  Next pose: x={next_pose.x():.3f}, y={next_pose.y():.3f}, θ={next_pose.theta():.3f}")
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

    # Optimize using GTSAM
    fill_debug_results = False
    optimized, debug_results = optimize_with_gtsam(
        odometry, loop_closures, initial_pose=initial_pose, fill_debug_results=fill_debug_results
    )

    if fill_debug_results:
        show_debug_results(debug_results, gt_poses=gt_poses)

    # Calculate error metrics
    odom_error = np.mean(np.sqrt(np.sum((reconstructed[:, :2] - gt_poses[:, :2])**2, axis=1)))
    gtsam_error = np.mean(np.sqrt(np.sum((optimized[:, :2] - gt_poses[:, :2])**2, axis=1)))

    print(f"Mean position error (odometry only): {odom_error:.4f} m")
    print(f"Mean position error (GTSAM optimized): {gtsam_error:.4f} m")
    print(f"Improvement: {(1 - gtsam_error/odom_error) * 100:.2f}%")

    # Plot results
    plot_results(gt_poses, reconstructed, optimized, loop_closures)
