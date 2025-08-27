import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
# 杂项辅助函数
img2mse = lambda x, y : torch.mean((x - y) ** 2)  # 计算两幅图像之间的均方误差
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))  # 将均方误差转换为峰值信噪比(PSNR)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)  # 将浮点数值转换为8位整数(0-255)


# Positional encoding (section 5.1)
# 位置编码(论文第5.1节)
class Embedder:
    """
    位置编码类，用于将低维输入映射到高维空间
    通过正弦和余弦函数的组合来编码位置信息
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        """创建编码函数列表"""
        embed_fns = []
        d = self.kwargs['input_dims']  # 输入维度
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)  # 包含原始输入
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']  # 最大频率的log2值
        N_freqs = self.kwargs['num_freqs']  # 频率数量
        
        if self.kwargs['log_sampling']:
            # 对数采样：频率按对数尺度分布
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            # 线性采样：频率按线性尺度分布
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                # 为每个频率和每个周期函数(sin, cos)创建编码函数
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        """对输入进行编码"""
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    """
    获取位置编码器
    
    Args:
        multires: 多分辨率级别，决定编码的频率范围
        i: 编码器索引，-1表示不使用位置编码
    
    Returns:
        embed: 编码函数
        embedder_obj.out_dim: 编码后的输出维度
    """
    if i == -1:
        return nn.Identity(), 3  # 不使用位置编码，直接返回恒等映射
    
    embed_kwargs = {
                'include_input' : True,      # 包含原始输入
                'input_dims' : 3,            # 输入维度(x, y, z坐标)
                'max_freq_log2' : multires-1, # 最大频率的log2值
                'num_freqs' : multires,       # 频率数量
                'log_sampling' : True,        # 使用对数采样
                'periodic_fns' : [torch.sin, torch.cos],  # 周期函数列表
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)  # 创建编码函数
    return embed, embedder_obj.out_dim


# Model
# NeRF模型
class NeRF(nn.Module):
    """
    NeRF神经辐射场模型
    
    Args:
        D: 网络深度(隐藏层数量)
        W: 网络宽度(每层神经元数量)
        input_ch: 位置编码后的输入通道数
        input_ch_views: 视角方向编码后的输入通道数
        output_ch: 输出通道数(RGB+密度=4)
        skips: 跳跃连接的层索引列表
        use_viewdirs: 是否使用视角方向信息
    """
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # 处理3D点坐标的多层感知机
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        ### 根据官方代码实现 (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # 处理视角方向的网络层
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        ### 根据论文的实现
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            # 使用视角方向时的输出层
            self.feature_linear = nn.Linear(W, W)      # 特征提取层
            self.alpha_linear = nn.Linear(W, 1)        # 密度输出层
            self.rgb_linear = nn.Linear(W//2, 3)       # RGB颜色输出层
        else:
            # 不使用视角方向时的输出层
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """
        NeRF网络前向传播
        
        Args:
            x: 输入张量，包含位置编码和视角方向编码
        
        Returns:
            outputs: 输出张量，包含RGB颜色和密度值
        """
        # 分离位置输入和视角方向输入
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        
        # 通过位置处理网络层
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                # 跳跃连接：将原始位置输入与当前特征拼接
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # 使用视角方向信息
            alpha = self.alpha_linear(h)           # 预测密度值
            feature = self.feature_linear(h)       # 提取特征
            h = torch.cat([feature, input_views], -1)  # 将特征与视角方向拼接
        
            # 通过视角处理网络层
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)               # 预测RGB颜色
            outputs = torch.cat([rgb, alpha], -1)  # 拼接颜色和密度
        else:
            # 不使用视角方向信息，直接输出
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        """
        从Keras模型加载权重
        
        Args:
            weights: Keras模型的权重列表
        """
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        # 加载位置处理层的权重
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        # 加载特征提取层的权重
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        # 加载视角处理层的权重
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        # 加载RGB输出层的权重
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        # 加载密度输出层的权重
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
# 光线辅助函数
def get_rays(H, W, K, c2w):
    """
    生成相机光线（PyTorch版本）
    
    Args:
        H: 图像高度
        W: 图像宽度  
        K: 相机内参矩阵
        c2w: 相机到世界坐标系的变换矩阵
    
    Returns:
        rays_o: 光线起点
        rays_d: 光线方向
    """
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    # 计算相机坐标系下的光线方向
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # 将光线方向从相机坐标系转换到世界坐标系
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 将相机坐标系原点转换到世界坐标系，这是所有光线的起点
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    """
    生成相机光线（NumPy版本）
    
    Args:
        H: 图像高度
        W: 图像宽度
        K: 相机内参矩阵
        c2w: 相机到世界坐标系的变换矩阵
    
    Returns:
        rays_o: 光线起点
        rays_d: 光线方向
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # 计算相机坐标系下的光线方向
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # 将光线方向从相机坐标系转换到世界坐标系
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 将相机坐标系原点转换到世界坐标系，这是所有光线的起点
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    将光线转换到标准化设备坐标(NDC)
    用于处理前向场景，将无限远的光线映射到有限的NDC空间
    
    Args:
        H: 图像高度
        W: 图像宽度
        focal: 焦距
        near: 近平面距离
        rays_o: 光线起点
        rays_d: 光线方向
    
    Returns:
        rays_o: NDC空间中的光线起点
        rays_d: NDC空间中的光线方向
    """
    # Shift ray origins to near plane
    # 将光线起点移动到近平面
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    # 投影变换
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]  # x坐标投影
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]  # y坐标投影
    o2 = 1. + 2. * near / rays_o[...,2]                     # z坐标投影

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])  # x方向投影
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])  # y方向投影
    d2 = -2. * near / rays_o[...,2]                                                        # z方向投影
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
# 分层采样 (论文第5.2节)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    根据概率密度函数进行分层采样
    用于NeRF的精细网络采样，在重要区域采样更多点
    
    Args:
        bins: 深度区间边界
        weights: 每个区间的权重(重要性)
        N_samples: 采样点数量
        det: 是否使用确定性采样
        pytest: 是否为测试模式(使用固定随机种子)
    
    Returns:
        samples: 采样的深度值
    """
    # Get pdf
    # 获取概率密度函数
    weights = weights + 1e-5 # prevent nans # 防止除零错误
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # 归一化权重得到概率密度
    cdf = torch.cumsum(pdf, -1)  # 计算累积分布函数
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins)) # 在开始处添加0

    # Take uniform samples
    # 进行均匀采样
    if det:
        # 确定性采样：使用等间隔采样点
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        # 随机采样：使用随机采样点
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    # 测试模式，使用固定的随机数
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    # 反演累积分布函数
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)  # 找到每个u值在CDF中的位置
    below = torch.max(torch.zeros_like(inds-1), inds-1)  # 下界索引
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)  # 上界索引
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2) # 组合上下界

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # 收集对应的CDF和bins值
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # 线性插值计算最终的采样位置
    denom = (cdf_g[...,1]-cdf_g[...,0])  # 分母：CDF差值
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)  # 防止除零
    t = (u-cdf_g[...,0])/denom  # 插值参数
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])  # 线性插值

    return samples
