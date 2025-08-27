import numpy as np
import os, imageio


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original
########## LLFF数据加载代码的轻微修改版本
########## 原始版本请参见 https://github.com/Fyusion/LLFF

def _minify(basedir, factors=[], resolutions=[]):
    """
    缩小图像尺寸，创建不同分辨率的图像副本
    
    Args:
        basedir: 数据集基础目录
        factors: 缩放因子列表（例如[2, 4, 8]表示缩小2倍、4倍、8倍）
        resolutions: 具体分辨率列表（例如[[480, 640]]）
    """
    needtoload = False
    # 检查是否需要创建缩放图像
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    # 获取原始图像文件列表
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    # 为每个因子或分辨率创建缩放图像
    for r in factors + resolutions:
        if isinstance(r, int):
            # 按因子缩放
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            # 按具体分辨率缩放
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        # 创建目录并复制图像
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        # 使用ImageMagick的mogrify命令调整图像大小
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        # 删除原格式文件（如果转换为png）
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    """
    加载LLFF数据集的poses、bounds和图像
    
    Args:
        basedir: 数据集基础目录
        factor: 缩放因子
        width: 目标宽度
        height: 目标高度  
        load_imgs: 是否加载图像数据
    
    Returns:
        poses: 相机姿态矩阵
        bds: 深度边界
        imgs: 图像数组（如果load_imgs=True）
    """
    
    # 加载poses和bounds数据
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])  # 重塑为(3,5,N)格式
    bds = poses_arr[:, -2:].transpose([1,0])  # 深度边界
    
    # 获取第一张图像的尺寸
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''  # 后缀名
    
    # 根据参数确定缩放策略
    if factor is not None:
        # 按因子缩放
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        # 按高度缩放
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        # 按宽度缩放
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    # 检查图像目录是否存在
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    # 获取图像文件列表
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    # 更新poses中的图像尺寸和焦距信息
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])  # 高度和宽度
    poses[2, 4, :] = poses[2, 4, :] * 1./factor         # 调整焦距
    
    if not load_imgs:
        return poses, bds
    
    # 图像读取函数
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    # 加载所有图像并归一化到[0,1]
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            
            
    

def normalize(x):
    """向量归一化"""
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    """
    构造视图矩阵（相机坐标系到世界坐标系的变换）
    
    Args:
        z: 相机朝向方向（前向向量）
        up: 上方向向量
        pos: 相机位置
    
    Returns:
        m: 4x4变换矩阵的前3x4部分
    """
    vec2 = normalize(z)         # 前向方向（z轴）
    vec1_avg = up               # 上方向
    vec0 = normalize(np.cross(vec1_avg, vec2))  # 右向方向（x轴）
    vec1 = normalize(np.cross(vec2, vec0))      # 重新计算上方向（y轴）
    m = np.stack([vec0, vec1, vec2, pos], 1)    # 组装变换矩阵
    return m

def ptstocam(pts, c2w):
    """
    将世界坐标系中的点转换到相机坐标系
    
    Args:
        pts: 世界坐标系中的点
        c2w: 相机到世界坐标系的变换矩阵
    
    Returns:
        tt: 相机坐标系中的点
    """
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):
    """
    计算所有相机姿态的平均姿态
    
    Args:
        poses: 相机姿态数组
    
    Returns:
        c2w: 平均的相机到世界坐标系变换矩阵
    """
    hwf = poses[0, :3, -1:]  # 高度、宽度、焦距信息

    center = poses[:, :3, 3].mean(0)      # 平均位置
    vec2 = normalize(poses[:, :3, 2].sum(0))  # 平均朝向方向
    up = poses[:, :3, 1].sum(0)               # 平均上方向
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    """
    生成螺旋形的相机路径用于渲染
    
    Args:
        c2w: 中心相机姿态
        up: 上方向向量
        rads: 螺旋半径（x, y, z方向）
        focal: 焦距
        zdelta: z方向偏移
        zrate: z方向变化率
        rots: 旋转圈数
        N: 生成的姿态数量
    
    Returns:
        render_poses: 渲染路径的相机姿态列表
    """
    render_poses = []
    rads = np.array(list(rads) + [1.])  # 添加齐次坐标
    hwf = c2w[:,4:5]  # 高度、宽度、焦距信息
    
    # 生成螺旋路径上的每个相机姿态
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        # 计算螺旋路径上的相机位置
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        # 计算相机朝向（指向焦点）
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        # 构造相机姿态矩阵
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):
    """
    重新定心相机姿态，将平均姿态作为世界坐标系原点
    
    Args:
        poses: 原始相机姿态数组
    
    Returns:
        poses: 重新定心后的相机姿态数组
    """
    poses_ = poses+0  # 复制数组
    bottom = np.reshape([0,0,0,1.], [1,4])  # 齐次坐标底行
    c2w = poses_avg(poses)  # 计算平均姿态
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)  # 扩展为4x4矩阵
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])  # 为所有姿态添加底行
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    # 应用逆变换，将平均姿态设为原点
    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    """
    将相机姿态球化，生成360度观察的相机路径
    用于处理内向场景（如房间内部）
    
    Args:
        poses: 原始相机姿态
        bds: 深度边界
    
    Returns:
        poses_reset: 重新设置的相机姿态
        new_poses: 球化后的新相机姿态
        bds: 调整后的深度边界
    """
    
    # 辅助函数：将3x4矩阵扩展为4x4矩阵
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]  # 相机光线方向
    rays_o = poses[:,:3,3:4]  # 相机光线起点

    def min_line_dist(rays_o, rays_d):
        """
        计算所有光线的最小距离点（场景中心）
        
        Args:
            rays_o: 光线起点
            rays_d: 光线方向
        
        Returns:
            pt_mindist: 最小距离点
        """
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)  # 计算场景中心
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)  # 平均上方向

    # 构建新的坐标系
    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    # 将所有姿态转换到新坐标系
    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    # 计算场景的球形半径
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    # 标准化场景大小
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    # 生成球形相机路径
    centroid = np.mean(poses_reset[:,:3,3], 0)  # 质心
    zh = centroid[2]  # z高度
    radcircle = np.sqrt(rad**2-zh**2)  # 圆形半径
    new_poses = []
    
    # 在固定高度的圆周上生成相机姿态
    for th in np.linspace(0.,2.*np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])  # 向下看

        vec2 = normalize(camorigin)  # 从中心指向相机的方向
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    # 添加相机内参信息
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    """
    加载LLFF数据集的主函数
    
    Args:
        basedir: 数据集基础目录
        factor: 图像缩放因子（默认8，表示将原图缩小8倍）
        recenter: 是否重新定心相机姿态
        bd_factor: 深度边界缩放因子
        spherify: 是否球化相机姿态（用于360度场景）
        path_zflat: 是否使用平坦的z路径
    
    Returns:
        images: 图像数组
        poses: 相机姿态数组
        bds: 深度边界
        render_poses: 用于渲染的相机路径
        i_test: 测试视图索引
    """
    

    # 加载数据（factor=8表示将原图下采样8倍）
    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    # 修正旋转矩阵的顺序并将变量维度移到轴0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    # 如果提供了bd_factor则重新缩放场景
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc  # 缩放相机位置
    bds *= sc            # 缩放深度边界
    
    if recenter:
        # 重新定心相机姿态
        poses = recenter_poses(poses)
        
    if spherify:
        # 球化相机姿态（用于360度场景）
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        # 计算平均相机姿态
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        ## 生成螺旋路径
        # Get average pose
        # 获取平均姿态
        up = normalize(poses[:, :3, 1].sum(0))  # 平均上方向

        # Find a reasonable "focus depth" for this dataset
        # 为此数据集找到合理的"焦点深度"
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        # 获取螺旋路径的半径
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T  # 相机位置
        rads = np.percentile(np.abs(tt), 90, 0)  # 计算90%分位数作为半径
        c2w_path = c2w
        N_views = 120  # 视图数量
        N_rots = 2     # 旋转圈数
        if path_zflat:
            # 使用平坦的z路径
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        # 生成螺旋路径的姿态
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    # 计算用于测试的视图（距离平均姿态最近的视图）
    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)  # 找到距离最小的视图作为测试视图
    print('HOLDOUT view is', i_test)
    
    # 确保数据类型为float32
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test



