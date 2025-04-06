import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from generate_test_data import plot_results, reconstruct_from_odometry

# Try to import the optimization frameworks
frameworks = []
try:
    from gtsam_example import optimize_with_gtsam
    frameworks.append("GTSAM")
except ImportError:
    print("GTSAM not available")

try:
    from g2o_example import optimize_with_g2o
    frameworks.append("g2o")
except ImportError:
    print("g2o not available")

def compare_frameworks(gt_poses, odometry, loop_closures, num_runs=10):
    """Compare the performance of different optimization frameworks."""
    # Get initial pose
    initial_pose = gt_poses[0]

    # Reconstruct trajectory from odometry
    reconstructed = reconstruct_from_odometry(odometry, initial_pose=initial_pose)

    results = {}
    timing = {}
    errors = {}

    # Add odometry-only result
    results["Odometry"] = reconstructed
    odom_error = np.mean(np.sqrt(np.sum((reconstructed[:, :2] - gt_poses[:, :2])**2, axis=1)))
    errors["Odometry"] = odom_error

    # Run each framework multiple times and average results
    if "GTSAM" in frameworks:
        gtsam_times = []
        for _ in range(num_runs):
            start_time = time.time()
            gtsam_result, _ = optimize_with_gtsam(odometry, loop_closures, initial_pose=initial_pose)
            gtsam_times.append(time.time() - start_time)
        
        timing["GTSAM"] = np.mean(gtsam_times)
        results["GTSAM"] = gtsam_result  # Keep the last result for visualization

        gtsam_error = np.mean(np.sqrt(np.sum((gtsam_result[:, :2] - gt_poses[:, :2])**2, axis=1)))
        errors["GTSAM"] = gtsam_error

    if "g2o" in frameworks:
        g2o_times = []
        for _ in range(num_runs):
            start_time = time.time()
            g2o_result, _ = optimize_with_g2o(odometry, loop_closures, initial_pose=initial_pose)
            g2o_times.append(time.time() - start_time)
        
        timing["g2o"] = np.mean(g2o_times)
        results["g2o"] = g2o_result  # Keep the last result for visualization

        g2o_error = np.mean(np.sqrt(np.sum((g2o_result[:, :2] - gt_poses[:, :2])**2, axis=1)))
        errors["g2o"] = g2o_error

    # Print results
    print("\n--- Performance Comparison ---")
    print(f"{'Framework':<10} | {'Error (m)':<10} | {'Time (s)':<10} | {'Improvement':<10}")
    print("-" * 45)

    for fw in ["Odometry"] + frameworks:
        if fw == "Odometry":
            print(f"{fw:<10} | {errors[fw]:.4f} | {'N/A':<10} | {'0.00%':<10}")
        else:
            improvement = (1 - errors[fw]/errors["Odometry"]) * 100
            print(f"{fw:<10} | {errors[fw]:.4f} | {timing[fw]:.4f} | {improvement:.2f}%")

    # Plot comparison
    plt.figure(figsize=(12, 8))

    # Plot ground truth
    plt.plot(gt_poses[:, 0], gt_poses[:, 1], 'g-', linewidth=2, label='Ground Truth')

    # Plot odometry
    plt.plot(reconstructed[:, 0], reconstructed[:, 1], 'r--', linewidth=2, label='Odometry Only')

    # Plot optimized results
    colors = ['b-', 'm-', 'c-']
    for i, fw in enumerate(frameworks):
        plt.plot(results[fw][:, 0], results[fw][:, 1], colors[i], linewidth=2, label=f'{fw} Optimized')

    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.title('Framework Comparison for SLAM Pose Graph Optimization')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.savefig('framework_comparison.png', dpi=300)
    plt.show()

    return results, timing, errors

if __name__ == "__main__":
    if len(frameworks) == 0:
        print("No optimization frameworks available. Please install GTSAM or g2o.")
        sys.exit(1)

    # Load the data
    data = np.load('trajectory_data.npz', allow_pickle=True)
    gt_poses = data['gt_poses']
    odometry = data['odometry']
    loop_closures = data['loop_closures']

    # Run comparison with 10 runs for timing
    results, timing, errors = compare_frameworks(gt_poses, odometry, loop_closures, num_runs=10)

    # Save results
    np.savez('optimization_results.npz',
             gt_poses=gt_poses,
             odometry_poses=results["Odometry"],
             framework_results=results,
             timing=timing,
             errors=errors)

    print("\nResults saved to optimization_results.npz")
    print("Comparison plot saved to framework_comparison.png") 