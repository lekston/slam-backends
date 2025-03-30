Below is an overview of four popular optimization back-ends (Ceres, g2o, MRPT, GTSAM) for pose-graph SLAM, alongside a minimal “dummy” 2D example in Python. This example helps illustrate how one might construct and solve a simple pose-graph (i.e., incremental odometry constraints plus a loop closure) using each library. The goal is to highlight (1) how each library could be integrated in Python, (2) how we handle 2D poses with uncertainties, and (3) how to configure early exit criteria for efficiency. Although we focus on 2D (x, y, θ) for simplicity, the same approaches extend to 6D (x, y, z, roll, pitch, yaw).

──────────────────────────────────────────────────────────────────────────
1. SHORTLIST OF FRAMEWORKS
──────────────────────────────────────────────────────────────────────────

Below is a quick summary of each toolkit’s suitability given your criteria:

• Ceres Solver (Google)
  – Very flexible non-linear least squares solver.  
  – Written in C++, with a handful of Python bindings available (though many are third-party).  
  – Can handle large-scale problems efficiently. Minimal “built-in” pose-graph structures, but can be built up from scratch using residual blocks.  
  – Early-exit criteria can be set (max iterations, function tolerance, gradient tolerance).  
  – Cross-compilation for embedded Arm platforms is straightforward, and it has no mandatory CUDA dependency.  
  – Drawback: Setting up uncertain 6D poses (covariances, etc.) more manual than in specialized robotics libraries.

• g2o
  – A widely-used optimization engine in SLAM.  
  – Open-source, C++ library with an active community.  
  – Good performance, used by many SLAM systems (ORB-SLAM, LSD-SLAM, etc.).  
  – Python integration is not official: possible through third-party Python bindings (pyg2o, etc.) or manual SWIG wrappers.  
  – Handles full 6D or 2D pose-graph out-of-the-box with well-defined edges.  
  – Early-exit criteria can be tuned in solver parameters.

• GTSAM (Georgia Tech Smoothing And Mapping)
  – C++ library from Frank Dellaert’s group, with official Python wrappers (pip install gtsam).  
  – Specifically designed for robotics factor graphs (including SLAM).  
  – Good built-in types for 2D/3D poses and their covariance/information matrices, plus “Between” factors, “Prior” factors, etc.  
  – Well-documented, easy to get started for 2D or 3D pose-graph.  
  – Early-exit criteria via Levenberg-Marquardt or Gauss-Newton parameter settings (maxIterations, relativeErrorTol, etc.).  
  – Runs well on embedded platforms (pure CPU).

• MRPT (Mobile Robot Programming Toolkit)
  – C++ library that includes localization, mapping, and some optimization back-ends.  
  – Some Python bindings exist, but coverage might be incomplete.  
  – Provides pose-graph classes (e.g., mrpt::graphs) for 2D or 3D, loop closure detection, and built-in solvers (e.g., LevMarq).  
  – Light on direct advanced factor-graph functionality compared to GTSAM/g2o, but offers convenience for many other robotic tasks.  
  – Early-exit/maximum iteration controls are possible, though less configurable than GTSAM or g2o.

In practice, GTSAM is often chosen for Python-friendly factor-graph SLAM out-of-the-box. If you prefer building up from more generic tooling and do not mind bridging or writing custom residuals in C++ for maximum speed, Ceres or g2o could be strong choices. MRPT can be convenient if you want a broader robotics toolkit and are comfortable with partial Python coverage.

──────────────────────────────────────────────────────────────────────────
2. A SIMPLE DUMMY 2D EXAMPLE
──────────────────────────────────────────────────────────────────────────

Shown below is a toy example using GTSAM in Python—the library that most directly fits a typical robotics/SLAM approach with official Python binding. We will (a) create a simple 2D pose graph of 10 keyframes, (b) define odometry factors linking consecutive poses, (c) insert one loop closure, and (d) solve the pose graph. After the GTSAM example, you will see (much shorter) pseudo-snippets indicating how to achieve something similar in g2o, Ceres, and MRPT with Python.

──────────────────────────────────────────────────────────────────────────
2.1. GTSAM EXAMPLE IN PYTHON
──────────────────────────────────────────────────────────────────────────

Install:
  • pip install gtsam

Code:

-------------------------------------------------------------------------------
import math
import numpy as np

import gtsam
from gtsam.symbol_shorthand import X  # X(i) will represent the ith pose
# Alternatively: from gtsam import symbol; symbol('x', i) to create unique keys

def create_2d_pose_graph_example():
    """
    Creates a small 2D pose graph with 10 poses (keyframes).
    - Poses X0..X9
    - Odometry constraints between consecutive poses
    - One loop closure: X9 close to X0
    """
    # 1) Create an empty factor graph
    graph = gtsam.NonlinearFactorGraph()
    
    # 2) Define noise models
    #    We'll define a prior noise on X0 and an odometry noise for consecutive poses
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))  # x,y,theta
    odom_noise  = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    
    # 3) Add a prior on the first pose (X0 at [0,0,0])
    prior_mean = gtsam.Pose2(0.0, 0.0, 0.0)
    graph.add(gtsam.PriorFactorPose2(X(0), prior_mean, prior_noise))

    # 4) Create initial estimates and odometry constraints
    initial_estimate = gtsam.Values()
    current_pose = gtsam.Pose2(0.0, 0.0, 0.0)
    initial_estimate.insert(X(0), current_pose)
    
    # Each step: move 1m forward in X, with a small rotation
    for i in range(1, 10):
        # "True" odometry
        move_forward = gtsam.Pose2(1.0, 0.0, math.radians(5))
        # Add an odometry factor between X(i-1) and X(i)
        graph.add(gtsam.BetweenFactorPose2(
            X(i-1),
            X(i),
            move_forward,  # This is the measured relative pose
            odom_noise
        ))

        # Insert an initial guess (just compounding the forward poses)
        current_pose = current_pose.compose(move_forward)
        initial_estimate.insert(X(i), current_pose)
    
    # 5) Loop closure: let's say X9 is roughly near X0
    #    We'll measure that X9 is "close" to X0 with some small offset
    loop_closure_measure = gtsam.Pose2(-0.2, 0.1, math.radians(-5))
    # Add the factor
    graph.add(gtsam.BetweenFactorPose2(X(9), X(0), loop_closure_measure, odom_noise))
    
    return graph, initial_estimate


def solve_pose_graph(graph, initial_estimate):
    """
    Solve the factor graph using Levenberg-Marquardt or Gauss-Newton
    with an early-exit criterion.
    """
    # Specify optimization parameters
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(50)                 # Early exit criterion #1
    params.setRelativeErrorTol(1e-5)            # Early exit criterion #2
    params.setAbsoluteErrorTol(1e-5)            # Early exit criterion #3
    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    return result


if __name__ == "__main__":
    graph, initial_estimate = create_2d_pose_graph_example()
    result = solve_pose_graph(graph, initial_estimate)
    
    # Print final poses
    for i in range(10):
        print(f"X{i} final: {result.atPose2(X(i))}")
-------------------------------------------------------------------------------

Notes & highlights (GTSAM):
1. We created a factor graph with a prior on X0 and consecutive “BetweenFactorPose2” constraints.  
2. The loop closure is simply another BetweenFactorPose2 that connects X9 and X0.  
3. We specify diagonal covariances for simplicity but GTSAM also supports full 3×3 (2D) or 6×6 (3D) covariance/information matrices.  
4. Early-exit criteria is controlled in params.setMaxIterations(), setRelativeErrorTol(), setAbsoluteErrorTol(), etc.

Extending to 6D poses (Pose3) is straightforward in GTSAM—replace Pose2 with Pose3, BetweenFactorPose2 with BetweenFactorPose3, etc.

──────────────────────────────────────────────────────────────────────────
2.2. g2o IN PYTHON (HIGH-LEVEL OVERVIEW)
──────────────────────────────────────────────────────────────────────────

g2o is widely used for SLAM backend optimization but has no official Python binding. You can try pyg2o (third-party), or generate your own wrappers. The approach is conceptually the same:

Pseudo-code:

-------------------------------------------------------------------------------
import pyg2o
import numpy as np

def g2o_pose_graph_example():
    # Create the optimizer
    optimizer = pyg2o.SparseOptimizer()
    
    # Create the block solver
    solver = pyg2o.BlockSolverSE2(pyg2o.LinearSolverCSparseSE2())
    solver = pyg2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)
    
    # Add first vertex (X0) with a prior
    v0 = pyg2o.VertexSE2()
    v0.set_estimate(pyg2o.SE2(0,0,0))
    v0.set_id(0)
    v0.set_fixed(True)  # Fix the first pose as the origin
    optimizer.add_vertex(v0)
    
    # Add subsequent vertices, edges for odometry
    num_poses = 10
    for i in range(1, num_poses):
        v = pyg2o.VertexSE2()
        v.set_id(i)
        # Rough initial guess
        v.set_estimate(pyg2o.SE2(i*1.0, 0, np.radians(5*i)))
        optimizer.add_vertex(v)
        
        # Add edge from X(i-1) to X(i)
        e = pyg2o.EdgeSE2()
        e.set_vertex(0, optimizer.vertex(i-1))
        e.set_vertex(1, optimizer.vertex(i))
        # Relative pose + noise
        e.set_measurement(pyg2o.SE2(1.0, 0.0, np.radians(5.0)))
        # Info matrix (inverse covariance)
        info_matrix = np.diag([1.0/(0.2**2), 1.0/(0.2**2), 1.0/(0.1**2)])
        e.set_information(info_matrix)
        optimizer.add_edge(e)
    
    # Add loop closure from X9 to X0
    loop = pyg2o.EdgeSE2()
    loop.set_vertex(0, optimizer.vertex(num_poses-1))
    loop.set_vertex(1, optimizer.vertex(0))
    # Slight offset measurement
    loop.set_measurement(pyg2o.SE2(-0.2, 0.1, np.radians(-5)))
    loop.set_information(info_matrix)
    optimizer.add_edge(loop)
    
    # Configure early-exit or iteration limit (for example):
    optimizer.set_verbose(False)
    optimizer.initialize_optimization()
    optimizer.optimize(50)  # up to 50 iterations
    
    # Retrieve results:
    for i in range(num_poses):
        v = optimizer.vertex(i)
        est = v.estimate()
        print(f"X{i} = ({est.translation()[0]}, {est.translation()[1]}, {est.rotation().angle()})")
-------------------------------------------------------------------------------

• The pattern is the same: create vertices for poses (SE2), add edges with relative transforms, set the information matrix (inverse covariance) for each.  
• Loop closures are just additional edges.  
• For 6D optimization, use VertexSE3, EdgeSE3, etc. in g2o.  
• For Python usage you might rely on the pyg2o package or other community wrappers.

──────────────────────────────────────────────────────────────────────────
2.3. CERES SOLVER IN PYTHON (HIGH-LEVEL OVERVIEW)
──────────────────────────────────────────────────────────────────────────

Ceres is primarily a generic non-linear least squares library. To do a pose graph, you typically define your own parameter blocks (the poses) and residual blocks (the “Between” constraints). You can call it from Python via ceres-python-bindings or other third-party efforts. The flow is more manual than GTSAM/g2o:

Pseudo-code (“bare-bones” approach):

-------------------------------------------------------------------------------
import ceres

def residual_2d_pose(params_i, params_j, measured_relative_pose, sqrt_info):
    """
    Compute difference between the predicted relative pose (params_j wrt params_i)
    and measured_relative_pose, multiply by sqrt_info, output the residuals.
    """
    # params_i, params_j are [x, y, theta]
    # Construct transformations and compute difference
    # ...
    # Return 3 residuals (dx, dy, dtheta) in a list
    pass


def build_ceres_problem(poses, constraints):
    problem = ceres.Problem()
    for constraint in constraints:
        i, j = constraint['indices']        # e.g. (i=0, j=1)
        measurement = constraint['pose']   # e.g. [1.0, 0.0, 5 deg]
        sqrt_info = constraint['sqrt_info'] 
        
        # Create a CostFunction wrapper
        cost_function = ceres.NumericDiffCostFunction(
            lambda x,y: residual_2d_pose(x,y,measurement,sqrt_info),
            # dims: residual dim=3, param dim=3 for each pose
            ...
        )
        
        # Add the residual block
        problem.AddResidualBlock(cost_function,
                                 None,         # or LossFunction
                                 poses[i],     # pointer to parameter array
                                 poses[j])
    
    # Optionally fix the first pose
    problem.SetParameterBlockConstant(poses[0])
    
    # Set solver options
    options = ceres.SolverOptions()
    options.max_num_iterations = 50    # early exit
    options.function_tolerance  = 1e-6
    options.gradient_tolerance  = 1e-6
    
    # Solve
    summary = ceres.Solve(options, problem)
    return summary

-------------------------------------------------------------------------------

• You have full flexibility but must handle transformations, Jacobians (or use numeric differentiation), etc.  
• For a big project with changing requirements (2D → 6D, changing covariance forms), you might need more scaffolding. This is precisely what GTSAM/g2o provides directly.  
• Early-exit criteria are set in ceres.SolverOptions (max_num_iterations, function_tolerance, etc.).

──────────────────────────────────────────────────────────────────────────
2.4. MRPT (HIGH-LEVEL OVERVIEW)
──────────────────────────────────────────────────────────────────────────

MRPT has some Python bindings (pip install mrpt, though coverage is partial). In C++ it’s straightforward to build a 2D or 3D graph using mrpt::graphs::TGraphSaveable and call its optimization method. From Python, you might do something like:

Pseudo-code:

-------------------------------------------------------------------------------
import mrpt.graphs as mg
import mrpt.poses as mp

graph = mg.TNodeProps2D()
# The actual usage from Python might be limited
# but conceptually you'd do:
# 1) Create nodes
# 2) Create edges
# 3) Provide constraints (poses + information matrices)
# 4) Call an optimizer: e.g. "graph.getOptimizer()->optimizeGraphLEVMARQ()"

# Then retrieve the result
-------------------------------------------------------------------------------

As of now, MRPT’s Python bindings are far less documented for pose-graph optimization than GTSAM’s or even g2o’s. If you are comfortable in C++, MRPT can handle the entire pipeline (loop closures, etc.), but in Python you may find yourself bridging more code from the C++ side.

──────────────────────────────────────────────────────────────────────────
3. SUMMARY & NEXT STEPS
──────────────────────────────────────────────────────────────────────────

• For a Python-centric project that needs robust factor-graph SLAM capability (2D → 6D, loop closures, configurable covariance/information matrices, early-exit criteria), GTSAM is typically the path of least friction. Its official Python API matches robotics best practices, and performance is still very good.  
• If you prefer maximum control at the cost of writing additional plumbing, Ceres or g2o (and their Python wrappers) are strong. g2o is widely recognized for SLAM. Ceres is extremely flexible for all sorts of non-linear least squares. Both allow controlling iteration limits and tolerances for early exit.  
• MRPT is more of a complete robotics library (maps, sensors, etc.). Its optimization routines are less direct from Python but can be quite handy if you combine it with other MRPT features.  

Given your requirements (embedded Unix on Arm, no CUDA, potential 6D expansions, and a Python codebase), GTSAM is often the most straightforward. However, it is best to run a minimal test with each library (as shown above) on your target environment for final performance checks. The simple code snippets or pseudo-code examples above should help demonstrate how to set up a small “dummy” trajectory with loop closure in each framework. Then you can benchmark runtime, memory usage, code complexity, etc., to choose the one that best fits your development style and constraints.