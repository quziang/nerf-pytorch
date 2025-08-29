import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
from scipy.spatial.transform import Rotation as R


def quaternion_to_matrix(quat):
    """
    将四元数转换为旋转矩阵
    
    Args:
        quat: 四元数 [x, y, z, w] 格式
    
    Returns:
        3x3旋转矩阵
    """
    # SPE3R使用[x,y,z,w]格式，scipy使用[x,y,z,w]格式
    rotation = R.from_quat(quat)
    return rotation.as_matrix()


def spe3r_pose_to_nerf(quat, translation):
    """
    将SPE3R的姿态格式转换为NeRF使用的4x4变换矩阵
    
    Args:
        quat: 四元数 [x,y,z,w] - 从航天器坐标系到相机坐标系的旋转
        translation: 平移向量 [x,y,z] - 相机到物体中心的位移
    
    Returns:
        4x4的相机到世界坐标系的变换矩阵 (c2w)
    """
    # 将四元数转换为旋转矩阵
    R_vbs2cam = quaternion_to_matrix(quat)
    
    # SPE3R中的translation是相机到物体的向量，我们需要相机位置
    # 在相机坐标系中，相机在原点，物体在translation位置
    t_cam = np.array(translation)
    
    # 构建世界到相机的变换矩阵(w2c)
    w2c = np.eye(4)
    w2c[:3, :3] = R_vbs2cam
    w2c[:3, 3] = t_cam
    
    # 转换为相机到世界的变换矩阵(c2w)
    c2w = np.linalg.inv(w2c)
    
    # NeRF坐标系转换：调整坐标轴方向
    # SPE3R: z轴向前, NeRF: z轴向前，但需要调整y轴
    transform = np.array([
        [1, 0, 0, 0],   # x轴保持不变
        [0, -1, 0, 0],  # y轴反向 
        [0, 0, -1, 0],  # z轴反向
        [0, 0, 0, 1]
    ])
    
    c2w = c2w @ transform
    
    return c2w.astype(np.float32)


def pose_spherical(theta, phi, radius):
    """
    生成球面坐标系下的相机姿态，用于渲染新视角
    
    Args:
        theta: 偏航角（度）
        phi: 俯仰角（度）  
        radius: 距离
    
    Returns:
        4x4相机到世界变换矩阵
    """
    trans_t = lambda t : torch.Tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()

    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_spe3r_data(basedir, half_res=False, testskip=1, train_split=None, test_split=None, use_masks=True):
    """
    加载SPE3R数据集
    
    Args:
        basedir: 数据集目录路径（如 /path/to/spe3r/soho）
        half_res: 是否将图像分辨率减半
        testskip: 测试集采样间隔
        train_split: 训练集图像索引范围，格式：[(start1, end1), (start2, end2)]
        test_split: 测试集图像索引范围，格式：[(start1, end1), (start2, end2)]
        use_masks: 是否使用掩码去除背景
    
    Returns:
        imgs: 图像数组，形状为(N, H, W, 3)
        poses: 相机姿态数组，形状为(N, 4, 4) 
        render_poses: 用于渲染的相机姿态序列
        [H, W, focal]: 图像尺寸和焦距
        i_split: [i_train, i_val, i_test] 索引分割
    """
    
    # 设置默认的数据分割（按SPE3R论文的设定）
    if train_split is None:
        train_split = [(1, 400), (501, 900)]  # 训练集：1-400, 501-900
    if test_split is None:
        test_split = [(401, 500), (901, 1000)]  # 测试集：401-500, 901-1000
    
    # 加载相机内参
    spe3r_root = os.path.dirname(basedir)  # 上级目录包含camera.json
    camera_path = os.path.join(basedir, '..', 'camera.json')
    
    
    if not os.path.exists(camera_path):
        # 如果没找到，尝试在当前目录
        camera_path = os.path.join(spe3r_root, 'camera.json')
    
    with open(camera_path, 'r') as f:
        camera_params = json.load(f)
    
    # 提取相机参数
    focal = float(camera_params['cameraMatrix'][0][0])  # fx = fy
    H = camera_params['Nu']  # 图像高度：256
    W = camera_params['Nv']  # 图像宽度：256
    
    # 加载姿态标签
    labels_path = os.path.join(basedir, 'labels.json')
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    print(f'Loaded SPE3R data: {len(labels_data)} images, focal={focal}, resolution={H}x{W}')
    if use_masks:
        print(f'Using masks from: {os.path.join(basedir, "masks")}')
    
    # 创建索引映射
    def create_indices(split_ranges):
        indices = []
        for start, end in split_ranges:
            indices.extend(range(start-1, end))  # 转换为0索引
        return indices
    
    train_indices = create_indices(train_split)
    test_indices = create_indices(test_split)
    
    # 验证集使用测试集的一部分
    val_indices = test_indices[::2]  # 测试集的一半作为验证集
    test_indices = test_indices[1::2]  # 另一半作为测试集
    
    # 应用testskip到测试集和验证集
    if testskip > 1:
        val_indices = val_indices[::testskip]
        test_indices = test_indices[::testskip]
    
    all_indices = sorted(train_indices + val_indices + test_indices)
    
    # 加载选定的图像和姿态
    all_imgs = []
    all_poses = []
    all_filenames = []  # 保存原始文件名
    
    images_dir = os.path.join(basedir, 'images')
    masks_dir = os.path.join(basedir, 'masks') if use_masks else None
    
    for idx in all_indices:
        if idx >= len(labels_data):
            continue
            
        label = labels_data[idx]
        img_name = label['filename']
        
        # 加载图像
        img_path = os.path.join(images_dir, f'{img_name}.jpg')
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping")
            continue
            
        img = imageio.imread(img_path)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[...,:3]  # 移除alpha通道如果存在
        
        # 加载并应用掩码
        if use_masks and masks_dir:
            mask_path = os.path.join(masks_dir, f'{img_name}.png')
            if os.path.exists(mask_path):
                mask = imageio.imread(mask_path)
                # 确保掩码是单通道的（取第一个通道）
                if len(mask.shape) == 3:
                    mask = mask[..., 0]  # 取第一个通道
                # 将掩码转换为0-1范围的float，255=前景(1.0), 0=背景(0.0)
                mask = mask.astype(np.float32) / 255.0
                # 先将图像转换到0-1范围
                img = img.astype(np.float32) / 255.0
                # 扩展掩码维度以匹配图像的RGB通道
                mask = mask[..., np.newaxis]
                # 应用掩码：前景保持原样，背景设为白色
                # mask=1(前景)时保持原图，mask=0(背景)时设为白色(1.0)
                img = img * mask + (1.0 - mask) * 1.0  # 背景设为白色
                # 转回uint8格式
                img = (img * 255.0).astype(np.uint8)
            else:
                print(f"Warning: Mask {mask_path} not found for image {img_name}")
        
        all_imgs.append(img)
        all_filenames.append(img_name)  # 保存文件名
        
        # 转换姿态
        quat = label['q_vbs2tango_true']
        trans = label['r_Vo2To_vbs_true']
        
        pose = spe3r_pose_to_nerf(quat, trans)
        all_poses.append(pose)
    
    # 转换为numpy数组
    imgs = np.stack(all_imgs, 0).astype(np.float32) / 255.0
    poses = np.stack(all_poses, 0)
    
    # 创建数据分割索引
    n_train = len(train_indices)
    n_val = len(val_indices) 
    n_test = len(test_indices)
    
    i_train = np.arange(0, n_train)
    i_val = np.arange(n_train, n_train + n_val)
    i_test = np.arange(n_train + n_val, n_train + n_val + n_test)
    
    i_split = [i_train, i_val, i_test]
    
    # 生成渲染路径：围绕物体的球形路径
    render_poses = torch.stack([pose_spherical(angle, -15.0, 5.0) 
                               for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)
    
    # 可选：减半分辨率
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.0
        
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    
    print(f'Loaded SPE3R {basedir}: {imgs.shape}, poses: {poses.shape}')
    print(f'Data split - Train: {len(i_train)}, Val: {len(i_val)}, Test: {len(i_test)}')
    
    # 打印测试集的原始文件名
    print("Test set images:")
    for i, array_idx in enumerate(i_test):
        if array_idx < len(all_filenames):
            print(f"  Test {i}: {all_filenames[array_idx]}")
    
    return imgs, poses, render_poses, [H, W, focal], i_split


# 辅助函数：可视化姿态分布
def visualize_poses(poses, save_path=None):
    """
    可视化相机姿态分布
    
    Args:
        poses: 姿态数组 (N, 4, 4)
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取相机位置
    positions = poses[:, :3, 3]
    
    # 绘制相机位置
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
              c=range(len(positions)), cmap='viridis', s=20)
    
    # 绘制原点（物体中心）
    ax.scatter([0], [0], [0], c='red', s=100, marker='*')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('SPE3R Camera Poses')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # 测试加载函数
    basedir = "/teams/microsate_1687685838/qza/dataset/SPE3R/spe3r/soho"
    
    try:
        imgs, poses, render_poses, hwf, i_split = load_spe3r_data(basedir, half_res=False, use_masks=True)
        
        print("加载成功!")
        print(f"图像形状: {imgs.shape}")
        print(f"姿态形状: {poses.shape}")
        print(f"渲染姿态形状: {render_poses.shape}")
        print(f"图像尺寸和焦距: {hwf}")
        print(f"数据分割: train={len(i_split[0])}, val={len(i_split[1])}, test={len(i_split[2])}")
        
        # 可视化姿态分布
        visualize_poses(poses, 'spe3r_poses.png')
        
    except Exception as e:
        print(f"加载失败: {e}")
        import traceback
        traceback.print_exc()
