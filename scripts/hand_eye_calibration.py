import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import argparse

# 读取位姿数据 (t x y z qx qy qz qw)
def read_poses_from_file(filepath):
    poses = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            # 每行数据格式：t x y z qx qy qz qw
            values = list(map(float, line.strip().split()))
            t = np.array(values[1:4])   # 提取平移部分 (x, y, z)
            q = np.array(values[4:8])   # 提取四元数部分 (qx, qy, qz, qw)
            poses.append((t, q))
    return poses



# 将 (t x y z qx qy qz qw) 构建为 4x4 变换矩阵
def build_transform_matrix(t, q):
    R = Rotation.from_quat(q).as_matrix() # 四元数转旋转矩阵
    T = np.eye(4)
    T[0:3, 0:3] = R                       # 旋转矩阵
    T[0:3, 3] = t                         # 平移向量
    return T

# 读取位姿文件并生成4x4变换矩阵
def get_transform_matrices_from_file(filepath):
    poses = read_poses_from_file(filepath)
    transform_matrices = []
    for t, q in poses:
        T = build_transform_matrix(t, q)
        transform_matrices.append(T)
    return transform_matrices

# 手眼标定函数
def hand_eye_calibration(A_matrices, B_matrices):
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for A, B in zip(A_matrices, B_matrices):
        R_gripper2base.append(A[0:3, 0:3])  # 从4x4矩阵提取旋转部分
        t_gripper2base.append(A[0:3, 3])    # 提取平移部分
        R_target2cam.append(B[0:3, 0:3])    # 提取旋转部分
        t_target2cam.append(B[0:3, 3])      # 提取平移部分

    # 使用 Tsai-Lenz 方法进行手眼标定
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    return R_cam2gripper, t_cam2gripper

# 主程序：读取机械臂和相机位姿文件并执行手眼标定
if __name__ == "__main__":
    # 读取机械臂和相机的位姿文件
    arm_poses_file = "slam_cut_pose.csv"   # 机械臂位姿文件路径
    cam_poses_file = "aligned_beiyun_poses.csv"   # 相机位姿文件路径

    A_matrices = get_transform_matrices_from_file(arm_poses_file)
    B_matrices = get_transform_matrices_from_file(cam_poses_file)

    # 进行手眼标定
    R_cam2gripper, t_cam2gripper = hand_eye_calibration(A_matrices, B_matrices)

    # 打印标定结果
    print("Rotation (camera to gripper):\n", R_cam2gripper)
    print("Translation (camera to gripper):\n", t_cam2gripper)
