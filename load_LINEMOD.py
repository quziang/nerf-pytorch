import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


# 3D变换矩阵函数
# 3D transformation matrix functions

# 沿z轴平移的变换矩阵
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

# 绕x轴旋转的变换矩阵（俯仰角phi）
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

# 绕y轴旋转的变换矩阵（偏航角theta）
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    """
    生成球面坐标系下的相机姿态
    用于围绕物体生成观察视角
    
    Args:
        theta: 偏航角（水平旋转角度，单位：度）
        phi: 俯仰角（垂直旋转角度，单位：度）
        radius: 相机到物体中心的距离
    
    Returns:
        c2w: 相机到世界坐标系的变换矩阵(4x4)
    """
    c2w = trans_t(radius)  # 首先沿z轴平移radius距离
    c2w = rot_phi(phi/180.*np.pi) @ c2w      # 绕x轴旋转phi度（俯仰）
    c2w = rot_theta(theta/180.*np.pi) @ c2w  # 绕y轴旋转theta度（偏航）
    # 应用坐标系变换：从标准坐标系转换到NeRF使用的坐标系
    # NeRF坐标系：x向右，y向上，z向前（朝向物体）
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_LINEMOD_data(basedir, half_res=False, testskip=1):
    """
    加载LINEMOD数据集
    LINEMOD是一个用于6D物体姿态估计的数据集，包含15个纹理化物体的RGB-D图像
    
    Args:
        basedir: 数据集根目录路径
        half_res: 是否将图像分辨率减半（用于加速训练）
        testskip: 测试集采样间隔（1表示使用所有图像，2表示每隔一张使用一张）
    
    Returns:
        imgs: 图像数组，形状为(N, H, W, 4) RGBA格式
        poses: 相机姿态数组，形状为(N, 4, 4)
        render_poses: 用于渲染的相机姿态序列
        [H, W, focal]: 图像高度、宽度和焦距
        K: 相机内参矩阵
        i_split: 训练/验证/测试集的索引分割
        near: 近平面距离
        far: 远平面距离
    """
    # 数据集分割：训练集、验证集、测试集
    splits = ['train', 'val', 'test']
    metas = {}
    
    # 加载每个分割的元数据（JSON格式）
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []   # 存储所有图像
    all_poses = []  # 存储所有相机姿态
    counts = [0]    # 记录每个分割的累计图像数量
    
    # 处理每个数据分割
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        
        # 确定采样间隔
        if s=='train' or testskip==0:
            skip = 1  # 训练集使用所有图像
        else:
            skip = testskip  # 验证集和测试集可以跳过一些图像
            
        # 加载图像和对应的相机姿态
        for idx_test, frame in enumerate(meta['frames'][::skip]):
            fname = frame['file_path']
            if s == 'test':
                print(f"{idx_test}th test frame: {fname}")
            imgs.append(imageio.imread(fname))                    # 读取图像
            poses.append(np.array(frame['transform_matrix']))     # 读取4x4变换矩阵
        
        # 图像归一化到[0,1]范围，保持RGBA 4个通道
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    # 创建数据分割索引
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    # 合并所有数据
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    # 获取图像尺寸和相机内参
    H, W = imgs[0].shape[:2]
    focal = float(meta['frames'][0]['intrinsic_matrix'][0][0])  # 焦距
    K = meta['frames'][0]['intrinsic_matrix']                   # 内参矩阵
    print(f"Focal: {focal}")
    
    # 生成用于渲染的球形相机路径
    # 围绕物体360度旋转，俯仰角-30度，距离4.0单位
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    # 可选：将图像分辨率减半以加速处理
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        # 使用OpenCV调整图像大小
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    # 设置近远平面距离
    # 取训练集和测试集中的最小near和最大far值
    near = np.floor(min(metas['train']['near'], metas['test']['near']))
    far = np.ceil(max(metas['train']['far'], metas['test']['far']))
    
    return imgs, poses, render_poses, [H, W, focal], K, i_split, near, far


