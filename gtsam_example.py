import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam.symbol_shorthand import X
from generate_test_data import plot_results

def optimize_with_gtsam(odometry, loop_closures, prior_noise=0.1, odom_noise=0.2, loop_noise=0.1):
    """
    Optimize a pose graph using GTSAM.
    
    Args:
        odometry: List of (dx, dy, dtheta) relative pose measurements
        loop_closures: List of (i, j, dx, dy, dtheta) loop closure constraints
        prior_noise: Noise for the prior on the first pose
        odom_noise: Noise for odometry measurements
        loop_noise: Noise for loop closure measurements
        
    Returns:
        optimized_poses: Numpy array of optimized poses
    """
    # Create a factor graph
    graph = gtsam.NonlinearFactorGraph()
    
    # Create the initial estimate
    initial_estimate = gtsam.Values()
    
    # Add a prior on the first pose
    prior_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_noise, prior_noise, prior_noise * 0.1]))
    graph.add(gtsam.PriorFactorPose2(X(0), gtsam.Pose2(0, 0, 0), prior_noise_model))
    
    # Set initial estimate for first pose
    initial_estimate.insert(X(0), gtsam.Pose2(0, 0, 0))
    
    # Add odometry factors
    odom_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([odom_noise, odom_noise, odom_noise * 0.1]))
    
    # Reconstruct poses based on odometry to get initial estimates
    poses = [gtsam.Pose2(0, 0, 0)]
    
    for i, (dx, dy, dtheta) in enumerate(odometry):
        prev_pose = poses[-1]
        
        # Create odometry measurement as relative pose
        delta = gtsam.Pose2(dx, dy, dtheta)
        
        # In GTSAM, we need to convert local odometry to global frame
        # However, the Pose2 class handles this when using .compose()
        next_pose = prev_pose.compose(delta)
        poses.append(next_pose)
        
        # Add odometry factor
        graph.add(gtsam.BetweenFactorPose2(X(i), X(i+1), delta, odom_noise_model))
        
        # Add to initial estimate
        initial_estimate.insert(X(i+1), next_pose)
    
    # Add loop closure factors
    loop_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([loop_noise, loop_noise, loop_noise * 0.1]))
    
    for i, j, dx, dy, dtheta in loop_closures:
        # Create loop closure measurement as relative pose
        delta = gtsam.Pose2(dx, dy, dtheta)
        graph.add(gtsam.BetweenFactorPose2(X(i), X(j), delta, loop_noise_model))
    
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
    
    return optimized_poses

if __name__ == "__main__":
    # Load the data
    data = np.load('trajectory_data.npz', allow_pickle=True)
    gt_poses = data['gt_poses']
    odometry = data['odometry']
    loop_closures = data['loop_closures']
    
    # Reconstruct trajectory from odometry
    from generate_test_data import reconstruct_from_odometry
    reconstructed = reconstruct_from_odometry(odometry)
    
    # Optimize using GTSAM
    optimized = optimize_with_gtsam(odometry, loop_closures)
    
    # Plot results
    plot_results(gt_poses, reconstructed, optimized, loop_closures)
    
    # Calculate error metrics
    odom_error = np.mean(np.sqrt(np.sum((reconstructed[:, :2] - gt_poses[:, :2])**2, axis=1)))
    gtsam_error = np.mean(np.sqrt(np.sum((optimized[:, :2] - gt_poses[:, :2])**2, axis=1)))
    
    print(f"Mean position error (odometry only): {odom_error:.4f} m")
    print(f"Mean position error (GTSAM optimized): {gtsam_error:.4f} m")
    print(f"Improvement: {(1 - gtsam_error/odom_error) * 100:.2f}%") 