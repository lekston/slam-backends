# Overview

Auto-generated (mostly) SLAM backend comparison framework focused on experimenting with various graph optimization frameworks.

Prompt and original answer are provided in the docs subdirectory.

# Results:

The 2 available examples were compared on x86_64 architecture (Intel Core i7-8565U CPU @ 1.80GHz).
Library versions"
```
g2o-python==0.0.12
gtsam==4.2
numpy==1.26.4
```

Comparisons over 10 runs of each:
```
--- Performance Comparison ---
Framework  | Error (m)  | Time (s)   | Improvement
---------------------------------------------
Odometry   | 0.7352 | N/A        | 0.00%     
GTSAM      | 0.4538 | 0.0013 | 38.28%
g2o        | 0.4448 | 0.0016 | 39.50%
```