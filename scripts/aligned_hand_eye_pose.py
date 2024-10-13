import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# 读取姿态文件
def read_poses_from_file(filename):
    poses = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):  # 忽略注释行
                continue
            data = list(map(float, line.split()))
            t, x, y, z, qx, qy, qz, qw = data
            poses.append((t, np.array([x, y, z]), np.array([qx, qy, qz, qw])))
    return poses

# 四元数转旋转矩阵
def quaternion_to_rotation_matrix(quaternion):
    r = R.from_quat(quaternion)
    return r.as_matrix()

# 时间戳补偿
def apply_time_offset(poses, time_offset):
    return [(t + time_offset, pos, quat) for t, pos, quat in poses]

# 找到与 hand pose 时间戳最接近的 eye pose
def find_closest_pose(eye_poses, hand_pose_time):
    closest_pose = min(eye_poses, key=lambda p: abs(p[0] - hand_pose_time))
    return closest_pose

# 将对齐后的姿态保存到文件
def save_aligned_poses(filename, aligned_poses):
    with open(filename, 'w') as f:
        for t, pos, quat in aligned_poses:
            # 将姿态保存为 t x y z qx qy qz qw 格式
            f.write(f"{t} {pos[0]} {pos[1]} {pos[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n")

# 使用 argparse 读取命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description="Hand-eye calibration tool with timestamp compensation")
    parser.add_argument('arm_pose_file', type=str, help='Path to the file containing arm poses')
    parser.add_argument('cam_pose_file', type=str, help='Path to the file containing camera poses')
    parser.add_argument('--time_offset', type=float, default=0.0, help='Time offset for compensating camera pose timestamps (default: 0.0)')
    parser.add_argument('--output_file', type=str, default='aligned_beiyun_poses.txt', help='Output file to save aligned poses')
    return parser.parse_args()

# 主程序
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()

    # 读取机械臂和相机位姿文件
    arm_poses = read_poses_from_file(args.arm_pose_file)
    cam_poses = read_poses_from_file(args.cam_pose_file)

    # 应用时间戳补偿到相机位姿
    cam_poses = apply_time_offset(cam_poses, args.time_offset)

    # 对齐姿态
    aligned_cam_poses = []
    aligned_hand_poses = []

    for hand_pose in arm_poses:
        hand_pose_time, hand_position, hand_quaternion = hand_pose
        closest_cam_pose = find_closest_pose(cam_poses, hand_pose_time)
        
        # 将对齐的姿态添加到结果中
        aligned_cam_poses.append(closest_cam_pose)
        aligned_hand_poses.append(hand_pose)

    # 保存对齐后的姿态到文件
    save_aligned_poses(args.output_file, aligned_cam_poses)
    print(f"Aligned poses saved to {args.output_file}")
