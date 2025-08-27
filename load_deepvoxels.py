import os
import numpy as np
import imageio 


def load_dv_data(scene='cube', basedir='/data/deepvoxels', testskip=8):
    """
    加载DeepVoxels数据集
    DeepVoxels是一个用于神经场景表示研究的合成数据集，
    包含多个3D场景（如立方体、希腊雕像等）的多视角图像
    
    Args:
        scene: 场景名称（如'cube', 'greek', 'vase'等）
        basedir: 数据集根目录路径
        testskip: 测试集和验证集的采样间隔
    
    Returns:
        imgs: 图像数组
        poses: 相机姿态数组
        render_poses: 用于渲染的相机姿态
        [H,W,focal]: 图像尺寸和焦距
        i_split: 数据集分割索引
    """
    

    def parse_intrinsics(filepath, trgt_sidelength, invert_y=False):
        """
        解析相机内参文件
        
        Args:
            filepath: 内参文件路径
            trgt_sidelength: 目标图像边长
            invert_y: 是否反转y轴
        
        Returns:
            full_intrinsic: 完整内参矩阵(4x4)
            grid_barycenter: 网格重心坐标
            scale: 缩放因子
            near_plane: 近平面距离
            world2cam_poses: 是否为世界坐标到相机坐标的姿态
        """
        # Get camera intrinsics
        # 获取相机内参
        with open(filepath, 'r') as file:
            f, cx, cy = list(map(float, file.readline().split()))[:3]  # 焦距和主点坐标
            grid_barycenter = np.array(list(map(float, file.readline().split())))  # 网格重心
            near_plane = float(file.readline())    # 近平面
            scale = float(file.readline())         # 缩放因子
            height, width = map(float, file.readline().split())  # 图像尺寸

            try:
                world2cam_poses = int(file.readline())
            except ValueError:
                world2cam_poses = None

        if world2cam_poses is None:
            world2cam_poses = False

        world2cam_poses = bool(world2cam_poses)

        print(cx,cy,f,height,width)

        # 根据目标分辨率调整内参
        cx = cx / width * trgt_sidelength
        cy = cy / height * trgt_sidelength
        f = trgt_sidelength / height * f

        fx = f
        if invert_y:
            fy = -f  # 反转y轴（某些坐标系约定）
        else:
            fy = f

        # Build the intrinsic matrices
        # 构建内参矩阵
        full_intrinsic = np.array([[fx, 0., cx, 0.],
                                   [0., fy, cy, 0],
                                   [0., 0, 1, 0],
                                   [0, 0, 0, 1]])

        return full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses


    def load_pose(filename):
        """
        从文件加载单个4x4相机姿态矩阵
        
        Args:
            filename: 姿态文件路径
        
        Returns:
            4x4相机姿态矩阵
        """
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4,4]).astype(np.float32)


    H = 512  # 图像高度
    W = 512  # 图像宽度
    deepvoxels_base = '{}/train/{}/'.format(basedir, scene)

    # 解析相机内参
    full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses = parse_intrinsics(os.path.join(deepvoxels_base, 'intrinsics.txt'), H)
    print(full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses)
    focal = full_intrinsic[0,0]  # 提取焦距
    print(H, W, focal)

    
    def dir2poses(posedir):
        """
        从目录加载所有姿态文件并应用坐标变换
        
        Args:
            posedir: 姿态文件目录路径
        
        Returns:
            poses: 变换后的姿态数组，形状为(N, 3, 4)
        """
        # 加载目录中所有.txt格式的姿态文件
        poses = np.stack([load_pose(os.path.join(posedir, f)) for f in sorted(os.listdir(posedir)) if f.endswith('txt')], 0)
        
        # 坐标系变换矩阵：调整DeepVoxels坐标系到NeRF坐标系
        transf = np.array([
            [1,0,0,0],   # x轴保持不变
            [0,-1,0,0],  # y轴反向
            [0,0,-1,0],  # z轴反向
            [0,0,0,1.],
        ])
        poses = poses @ transf
        poses = poses[:,:3,:4].astype(np.float32)  # 取前3x4部分
        return poses
    
    # 加载训练集、测试集和验证集的相机姿态
    posedir = os.path.join(deepvoxels_base, 'pose')
    poses = dir2poses(posedir)  # 训练集姿态
    testposes = dir2poses('{}/test/{}/pose'.format(basedir, scene))      # 测试集姿态
    testposes = testposes[::testskip]  # 按间隔采样测试集
    valposes = dir2poses('{}/validation/{}/pose'.format(basedir, scene)) # 验证集姿态
    valposes = valposes[::testskip]    # 按间隔采样验证集

    # 加载训练集图像
    imgfiles = [f for f in sorted(os.listdir(os.path.join(deepvoxels_base, 'rgb'))) if f.endswith('png')]
    imgs = np.stack([imageio.imread(os.path.join(deepvoxels_base, 'rgb', f))/255. for f in imgfiles], 0).astype(np.float32)
    
    
    # 加载测试集图像
    testimgd = '{}/test/{}/rgb'.format(basedir, scene)
    imgfiles = [f for f in sorted(os.listdir(testimgd)) if f.endswith('png')]
    testimgs = np.stack([imageio.imread(os.path.join(testimgd, f))/255. for f in imgfiles[::testskip]], 0).astype(np.float32)
    
    # 加载验证集图像
    valimgd = '{}/validation/{}/rgb'.format(basedir, scene)
    imgfiles = [f for f in sorted(os.listdir(valimgd)) if f.endswith('png')]
    valimgs = np.stack([imageio.imread(os.path.join(valimgd, f))/255. for f in imgfiles[::testskip]], 0).astype(np.float32)
    
    # 组织数据集分割
    all_imgs = [imgs, valimgs, testimgs]  # [训练集, 验证集, 测试集]
    counts = [0] + [x.shape[0] for x in all_imgs]  # 计算累计数量
    counts = np.cumsum(counts)
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]  # 创建索引分割
    
    # 合并所有数据
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate([poses, valposes, testposes], 0)
    
    # 使用测试集姿态作为渲染路径
    render_poses = testposes
    
    print(poses.shape, imgs.shape)
    
    return imgs, poses, render_poses, [H,W,focal], i_split


