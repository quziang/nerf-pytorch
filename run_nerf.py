import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

# 从辅助模块导入工具函数
from run_nerf_helpers import *

# 导入不同数据集的加载函数
# Import different dataset loading functions
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


# 设置设备（GPU或CPU）和随机种子
# Set device (GPU or CPU) and random seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    将函数 'fn' 构造为适用于较小批次的版本
    
    Args:
        fn: 需要批处理的函数
        chunk: 批次大小，如果为None则不进行批处理
    
    Returns:
        ret: 批处理版本的函数
    """
    if chunk is None:
        return fn
    def ret(inputs):
        # 将输入分块处理，每次处理chunk个样本，最后拼接结果
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    准备输入并应用网络函数 'fn'
    
    Args:
        inputs: 3D坐标点 [N_rays, N_samples, 3]
        viewdirs: 视角方向 [N_rays, 3] 
        fn: NeRF网络函数
        embed_fn: 位置编码函数
        embeddirs_fn: 方向编码函数
        netchunk: 网络批处理大小
        
    Returns:
        outputs: 网络输出结果 [N_rays, N_samples, output_dim]
    """
    # 将输入坐标展平为二维 [N_rays*N_samples, 3]
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    # 对3D坐标进行位置编码
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        # 扩展视角方向以匹配每个采样点
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        # 展平视角方向
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # 对视角方向进行编码
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        # 拼接位置编码和方向编码
        embedded = torch.cat([embedded, embedded_dirs], -1)

    # 通过网络前向传播，使用批处理避免内存溢出
    outputs_flat = batchify(fn, netchunk)(embedded)
    # 重新整形为原始维度
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    将光线分成更小的批次进行渲染以避免内存溢出
    
    Args:
        rays_flat: 展平的光线数据 [N_rays, ray_info]
        chunk: 批次大小
        **kwargs: 传递给render_rays的其他参数
        
    Returns:
        all_ret: 包含所有渲染结果的字典
    """
    all_ret = {}
    # 分批处理光线
    for i in range(0, rays_flat.shape[0], chunk):
        # 渲染当前批次的光线
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        # 收集每个批次的结果
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    # 拼接所有批次的结果
    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    渲染光线生成图像
    Args:
      H: int. Height of image in pixels. 图像高度（像素）
      W: int. Width of image in pixels. 图像宽度（像素）
      focal: float. Focal length of pinhole camera. 针孔相机焦距
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
        批处理大小，用于控制最大内存使用量，不影响最终结果
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch. 光线起点和方向 [2, batch_size, 3]
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
        相机到世界坐标的变换矩阵 [3, 4]
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
        是否使用标准化设备坐标系
      near: float or array of shape [batch_size]. Nearest distance for a ray.
        光线的最近距离
      far: float or array of shape [batch_size]. Farthest distance for a ray.
        光线的最远距离
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
        是否在模型中使用空间点的视角方向
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
       静态相机变换矩阵，如果不为None，用于相机位置而c2w用于视角方向
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays. 预测的RGB值
      disp_map: [batch_size]. Disparity map. Inverse of depth. 视差图（深度的倒数）
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray. 沿光线累积的不透明度
      extras: dict with everything returned by render_rays(). 渲染过程的额外信息
    """
    if c2w is not None:
        # special case to render full image
        # 特殊情况：渲染完整图像，需要生成所有像素的光线
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        # 使用提供的光线批次
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        # 将光线方向作为输入提供给网络
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            # 特殊情况：可视化视角方向的效果，使用静态相机位置
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        # 归一化视角方向向量
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3] 保存原始形状
    if ndc:
        # for forward facing scenes
        # 对于前向场景，将光线转换到NDC坐标系
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    # 创建光线批次数据
    rays_o = torch.reshape(rays_o, [-1,3]).float()  # 光线起点
    rays_d = torch.reshape(rays_d, [-1,3]).float()  # 光线方向

    # 设置近远平面距离
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    # 拼接光线信息：起点、方向、近距离、远距离
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        # 如果使用视角方向，则添加到光线数据中
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    # 渲染并重新整形结果
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        # 将结果重新整形为原始图像维度
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # 提取主要的渲染结果
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    """
    沿着给定的相机路径渲染图像序列
    Render images along a given camera path
    
    Args:
        render_poses: 相机位姿序列 [N, 3, 4]
        hwf: 图像高度、宽度和焦距 [H, W, focal]
        K: 相机内参矩阵 [3, 3]
        chunk: 批处理大小
        render_kwargs: 渲染参数字典
        gt_imgs: 真实图像（用于计算PSNR）
        savedir: 保存渲染图像的目录
        render_factor: 下采样因子，用于加速渲染
    
    Returns:
        rgbs: 渲染的RGB图像数组
        disps: 视差图数组
    """

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        # 为了加速，渲染下采样图像
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    # 遍历每个相机位姿进行渲染
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        # 渲染当前位姿的图像
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        # 如果有真实图像，计算PSNR
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        # 保存渲染结果为PNG图像
        if savedir is not None:
            rgb8 = to8b(rgbs[-1])  # 转换为8位图像
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    # 将所有结果堆叠成数组
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    实例化NeRF的多层感知机模型
    
    Args:
        args: 命令行参数对象，包含网络配置参数
        
    Returns:
        render_kwargs_train: 训练时的渲染参数
        render_kwargs_test: 测试时的渲染参数  
        start: 训练开始的迭代数
        grad_vars: 需要梯度更新的参数列表
        optimizer: 优化器
    """
    # 创建位置编码函数
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # 创建视角方向编码函数
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    
    # 如果使用重要性采样，输出通道为5（RGB+密度+额外信息），否则为4（RGB+密度）
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]  # 在第4层添加跳跃连接
    
    # 创建粗糙网络（coarse network）
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        # 创建精细网络（fine network）用于重要性采样
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    # 创建网络查询函数
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    # 创建Adam优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    # 加载检查点
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        # 查找实验目录中的所有检查点文件
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        # 加载最新的检查点
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        # 加载模型权重
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    # 设置训练时的渲染参数
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,     # 网络查询函数
        'perturb' : args.perturb,                  # 是否对采样点添加扰动
        'N_importance' : args.N_importance,        # 重要性采样的点数
        'network_fine' : model_fine,               # 精细网络
        'N_samples' : args.N_samples,              # 粗糙采样点数
        'network_fn' : model,                      # 粗糙网络
        'use_viewdirs' : args.use_viewdirs,        # 是否使用视角方向
        'white_bkgd' : args.white_bkgd,           # 是否使用白色背景
        'raw_noise_std' : args.raw_noise_std,     # 原始输出的噪声标准差
    }

    # NDC only good for LLFF-style forward facing data
    # NDC坐标系仅适用于LLFF风格的前向数据
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    # 设置测试时的渲染参数（关闭随机性）
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False          # 测试时不添加扰动
    render_kwargs_test['raw_noise_std'] = 0.       # 测试时不添加噪声

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    将模型的原始预测转换为具有语义意义的值
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
             模型的原始预测 [光线数, 每条光线采样点数, 4(RGB+密度)]
        z_vals: [num_rays, num_samples along ray]. Integration time.
                积分时间（沿光线的距离值）
        rays_d: [num_rays, 3]. Direction of each ray.
                每条光线的方向向量
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray. 每条光线的估计RGB颜色
        disp_map: [num_rays]. Disparity map. Inverse of depth map. 视差图（深度图的倒数）
        acc_map: [num_rays]. Sum of weights along each ray. 每条光线的权重和（累积透射率）
        weights: [num_rays, num_samples]. Weights assigned to each sampled color. 每个采样颜色的权重
        depth_map: [num_rays]. Estimated distance to object. 到物体的估计距离
    """
    # Lambda函数：将原始密度值转换为透明度值
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    # 计算相邻采样点之间的距离
    dists = z_vals[...,1:] - z_vals[...,:-1]
    # 在最后添加一个很大的距离值，表示到无穷远
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    # 根据光线方向调整距离（考虑光线方向的长度）
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    # 将原始RGB值通过sigmoid函数映射到[0,1]范围
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    
    # 添加噪声用于正则化
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        # 如果是pytest模式，使用固定的随机数
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # 计算透明度值
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    
    # 计算体积渲染权重：weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    # 权重 = 当前点的透明度 * 到当前点为止的累积透射率
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    # 通过权重积分得到最终的RGB值
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    # 计算深度图（权重加权的z值）
    depth_map = torch.sum(weights * z_vals, -1)
    # 计算视差图（深度的倒数）
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    # 计算累积透射率（透明度图）
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        # 如果使用白色背景，将透明部分设为白色
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    体积渲染函数
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
        包含光线所有必要信息的数组：起点、方向、最近距离、最远距离、单位视角方向
      network_fn: function. Model for predicting RGB and density at each point
        in space. 预测空间中每个点RGB和密度的模型函数
      network_query_fn: function used for passing queries to network_fn.
        用于向network_fn传递查询的函数
      N_samples: int. Number of different times to sample along each ray.
        沿每条光线的采样点数
      retraw: bool. If True, include model's raw, unprocessed predictions.
        是否返回模型的原始未处理预测
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
        是否在逆深度空间中线性采样而非深度空间
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time. 是否对采样点添加分层随机扰动
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
        额外的重要性采样点数（仅用于精细网络）
      network_fine: "fine" network with same spec as network_fn.
        与network_fn规格相同的精细网络
      white_bkgd: bool. If True, assume a white background.
        是否假设白色背景
      raw_noise_std: ... 原始输出的噪声标准差
      verbose: bool. If True, print more debugging info.
        是否打印更多调试信息
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
        光线的估计RGB颜色（来自精细模型）
      disp_map: [num_rays]. Disparity map. 1 / depth. 视差图（深度的倒数）
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
        每条光线的累积透明度（来自精细模型）
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
        模型的原始预测
      rgb0: See rgb_map. Output for coarse model. 粗糙模型的RGB输出
      disp0: See disp_map. Output for coarse model. 粗糙模型的视差输出
      acc0: See acc_map. Output for coarse model. 粗糙模型的累积透明度输出
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample. 每个采样点沿光线距离的标准差
    """
    N_rays = ray_batch.shape[0]
    # 提取光线的起点和方向
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    # 如果光线数据包含视角方向信息则提取，否则为None
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    # 提取近远边界
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    # 在[0,1]范围内生成等间距的采样参数
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        # 在深度空间中线性采样
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # 在逆深度空间中线性采样（对于远距离场景更合适）
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    # 扩展z_vals以匹配每条光线
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        # 获取采样点之间的区间，进行分层随机采样
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        # 在这些区间内进行分层采样
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        # Pytest模式下使用固定随机数
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    # 计算3D采样点坐标：光线起点 + 方向 * 距离
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

#     raw = run_network(pts)
    # 通过粗糙网络查询采样点的RGB和密度
    raw = network_query_fn(pts, viewdirs, network_fn)
    # 将原始输出转换为有意义的值
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:
        # 保存粗糙网络的结果
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # 基于粗糙网络的权重进行重要性采样
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        # 合并粗糙采样点和重要性采样点，并排序
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        # 重新计算3D采样点
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        # 选择使用精细网络还是粗糙网络
        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        # 通过选定的网络查询所有采样点
        raw = network_query_fn(pts, viewdirs, run_fn)

        # 使用所有采样点（粗糙+重要性）计算最终结果
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    # 构建返回结果字典
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        # 如果使用了重要性采样，也返回粗糙网络的结果
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    # 检查数值错误
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    """
    配置命令行参数解析器
    Configure command line argument parser
    """
    import configargparse
    parser = configargparse.ArgumentParser()
    # 配置文件路径
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    # 实验名称
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    # 模型和日志保存目录
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    # 数据集目录
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options 训练选项
    # 网络深度（层数）
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    # 网络宽度（每层神经元数）
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    # 精细网络深度
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    # 精细网络宽度
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    # 随机光线批次大小
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    # 学习率
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    # 学习率衰减步数
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    # 光线并行处理数量
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    # 网络并行处理的点数
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    # 是否只从单张图像采样光线
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    # 不重载权重
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    # 特定的权重文件路径
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options 渲染选项
    # 粗糙网络每条光线的采样点数
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    # 精细网络每条光线的额外采样点数
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    # 扰动参数（0表示无抖动，1表示有抖动）
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    # 是否使用5D输入（位置+方向）而非3D
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    # 位置编码设置
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    # 3D位置的位置编码最大频率的log2值
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    # 2D方向的位置编码最大频率的log2值
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    # sigma_a输出的噪声标准差
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # 仅渲染模式（不优化）
    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    # 渲染测试集
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    # 下采样因子
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options 训练选项
    # 中心裁剪的训练步数
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    # 中心裁剪的比例
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options 数据集选项
    # 数据集类型
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    # 测试跳帧数
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags deepvoxels数据集标志
    # 形状类型
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags blender数据集标志
    # 白色背景
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    # 半分辨率
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags LLFF数据集标志
    # 下采样因子
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    # 不使用NDC坐标
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    # 线性视差采样
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    # 球面化
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    # LLFF测试集间隔
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options 日志/保存选项
    # 控制台输出和指标记录频率
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    # tensorboard图像记录频率
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    # 权重检查点保存频率
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    # 测试集保存频率
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    # 视频渲染频率
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train():
    """
    主要的训练函数
    Main training function
    """
    parser = config_parser()
    args = parser.parse_args()

    # Load data 加载数据
    K = None
    if args.dataset_type == 'llff':
        # 加载LLFF数据集（真实场景）
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]  # 提取高度、宽度、焦距
        poses = poses[:,:3,:4]  # 提取位姿矩阵（去掉最后一行）
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            # 自动选择测试图像（每llffhold张取一张）
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test  # 验证集与测试集相同
        # 训练集为除了测试集和验证集之外的所有图像
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            # 不使用NDC坐标时，使用实际的边界距离
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            # 使用NDC坐标时的标准化距离
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        # 加载Blender合成数据集
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        # Blender数据集的固定近远平面
        near = 2.
        far = 6.

        if args.white_bkgd:
            # 处理alpha通道，将透明部分设为白色
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            # 只使用RGB通道
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        # 加载LINEMOD数据集
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':
        # 加载DeepVoxels数据集
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        # 根据相机位置计算近远平面
        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    # 将内参转换为正确的类型
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        # 构建相机内参矩阵
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        # 如果只渲染测试集，使用测试图像的位姿
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    # 创建日志目录并复制配置文件
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    # 保存参数配置
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        # 保存配置文件副本
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    # 创建NeRF模型
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    # 设置边界参数
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    # 将测试数据移至GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    # 如果只是从训练好的模型进行渲染，则提前返回
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                # 渲染测试位姿
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                # 默认渲染更平滑的位姿路径
                images = None

            # 创建渲染保存目录
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            # 沿路径渲染图像
            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            # 生成视频
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    # 如果使用随机光线批处理，准备光线批次张量
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        # 随机光线批处理模式
        print('get rays')
        # 为所有训练图像生成光线
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        # 拼接光线和对应的RGB值
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        # 只保留训练图像的光线
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        # 重新整形为 [所有像素数, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        # 随机打乱光线顺序
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    # 将训练数据移至GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    # 设置训练总迭代数
    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # 摘要写入器（用于TensorBoard等）
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    # 主训练循环
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        # 采样随机光线批次
        if use_batching:
            # Random over all images
            # 从所有图像中随机采样
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]  # 分离光线和目标颜色

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                # 一个epoch结束，重新打乱数据
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            # 从单张图像中随机采样
            img_i = np.random.choice(i_train)  # 随机选择一张训练图像
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                # 生成该图像的所有光线
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    # 中心裁剪训练（早期训练阶段）
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    # 使用整张图像
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                # 随机选择N_rand个像素
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                # 提取对应的光线和目标颜色
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        #####  核心优化循环  #####
        # 渲染当前批次的光线
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        # 清零梯度
        optimizer.zero_grad()
        # 计算图像损失（RGB重建损失）
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]  # 透明度值
        loss = img_loss
        psnr = mse2psnr(img_loss)  # 计算PSNR

        if 'rgb0' in extras:
            # 如果有粗糙网络的输出，也计算其损失
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0  # 总损失 = 精细网络损失 + 粗糙网络损失
            psnr0 = mse2psnr(img_loss0)

        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        ###   更新学习率   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        # 其余部分是日志记录
        if i%args.i_weights==0:
            # 保存模型权重检查点
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            # 开启测试模式，生成视频
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            # 保存RGB视频
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            # 保存深度视频
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # 如果使用视角方向，可以生成静态相机视频（已注释）
            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            # 在测试集上评估并保存结果
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


        if i%args.i_print==0:
            # 打印训练进度信息
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
        # 这里是原来的TensorFlow摘要记录代码（已注释）
        # Original TensorFlow summary logging code (commented out)
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        # 更新全局步数
        global_step += 1


if __name__=='__main__':
    # 设置默认张量类型为CUDA张量（如果有GPU）
    # Set default tensor type to CUDA tensor (if GPU is available)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # 开始训练
    # Start training
    train()
