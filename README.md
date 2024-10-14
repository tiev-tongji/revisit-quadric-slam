# Revisiting Geometric Constraint for Monocular Visual Quadric SLAM

* We first open source the early version of the code．
* For the readme, please temporarily refer to [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2).
* A detailed readme will be released later.

## Abstract
Using Quadrics as the object formulation has the benefits of both generality and closed-form projection derivation between image and world spaces. Although numerous error terms have been proposed to constrain the dual quadric
reconstruction, we found that many of them are imprecise and provide minimal improvements to localization. After scrutinizing the existing error terms, we introduce a simple but precise tangent geometric constraint for object landmarks, which is applied to object reconstruction, frontend pose estimation, and backend bundle adjustment. This constraint is designed to fully leverage precise semantic segmentation, effectively mitigating mismatches between concave object contours and dual quadrics. Experiments on public datasets demonstrate that our approach achieves more accurate object mapping and localization than existing object-based SLAM methods and ORB-SLAM2.
