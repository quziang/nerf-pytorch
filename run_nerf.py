# NeRF (Neural Radiance Fields) PyTorch实现的主要训练脚本
# 此脚本实现了NeRF模型的训练、渲染和测试功能

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

# 导入NeRF辅助函数和网络结构
from run_nerf_helpers import *

# 导入不同数据集的加载器
from load_llff import load_llff_data        # LLFF数据集（真实场景前向数据）
from load_deepvoxels import load_dv_data    # DeepVoxels数据集
from load_blender import load_blender_data  # Blender合成数据集
from load_LINEMOD import load_LINEMOD_data  # LINEMOD数据集
from load_spe3r import load_spe3r_data      # SPE3R数据集

# 设备配置：优先使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)  # 设置随机种子以确保结果可重复
DEBUG = False      # 调试标志


def batchify(fn, chunk):
    """构造函数'fn'的批处理版本，用于处理较小的批次。
    
    Args:
        fn: 需要进行批处理的函数
        chunk: 每个批次的大小，如果为None则不进行批处理
        
    Returns:
        批处理版本的函数
    """
    if chunk is None:
        return fn
    def ret(inputs):
        # 将输入分成小批次，分别处理后拼接结果
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """准备输入并应用网络函数'fn'进行前向传播。
    
    Args:
        inputs: 3D坐标输入 [N_rays, N_samples, 3]
        viewdirs: 视角方向 [N_rays, 3] 或 None
        fn: 网络函数（coarse或fine）
        embed_fn: 位置编码函数
        embeddirs_fn: 方向编码函数
        netchunk: 网络处理的批次大小，用于控制内存使用
        
    Returns:
        网络输出结果
    """
    # 将输入重塑为平坦的形状以便处理
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    # 对3D坐标进行位置编码
    embedded = embed_fn(inputs_flat)

    # 如果提供了视角方向，则进行方向编码并拼接
    if viewdirs is not None:
        # 扩展视角方向以匹配输入的形状
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # 对视角方向进行位置编码
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        # 将位置编码和方向编码拼接
        embedded = torch.cat([embedded, embedded_dirs], -1)

    # 使用批处理方式通过网络进行前向传播
    outputs_flat = batchify(fn, netchunk)(embedded)
    # 恢复原始形状
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """将光线分成较小的批次进行渲染，以避免内存溢出(OOM)。
    
    Args:
        rays_flat: 扁平化的光线数据 [N_rays, ...]
        chunk: 每个批次的光线数量
        **kwargs: 传递给render_rays的其他参数
        
    Returns:
        合并后的渲染结果字典
    """
    all_ret = {}
    # 分批处理光线以避免内存问题
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        # 收集每个批次的结果
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    # 将所有批次的结果拼接起来
    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """渲染光线生成图像
    
    Args:
      H: 图像高度（像素）
      W: 图像宽度（像素）
      K: 相机内参矩阵 [3, 3]
      chunk: 同时处理的最大光线数量，用于控制内存使用，不影响最终结果
      rays: 光线数据，形状为[2, batch_size, 3]，包含光线起点和方向
      c2w: 相机到世界坐标的变换矩阵 [3, 4]
      ndc: 是否使用标准化设备坐标(NDC)表示光线
      near: 光线的最近距离
      far: 光线的最远距离
      use_viewdirs: 是否在模型中使用空间点的视角方向
      c2w_staticcam: 静态相机变换矩阵 [3, 4]，如果不为None，用此矩阵作为相机位置，
                     用c2w参数作为视角方向
                     
    Returns:
      rgb_map: [batch_size, 3] 预测的RGB值
      disp_map: [batch_size] 视差图（深度的倒数）
      acc_map: [batch_size] 沿光线累积的不透明度(alpha)
      extras: 包含render_rays返回的所有其他信息的字典
    """
    if c2w is not None:
        # 特殊情况：渲染完整图像
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # 使用提供的光线批次
        rays_o, rays_d = rays

    if use_viewdirs:
        # 提供光线方向作为输入
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # 特殊情况：可视化viewdirs的效果
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        # 标准化视角方向向量
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # 保存原始形状 [..., 3]
    if ndc:
        # 用于前向场景的NDC坐标变换
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # 创建光线批次数据
    rays_o = torch.reshape(rays_o, [-1,3]).float()  # 光线起点
    rays_d = torch.reshape(rays_d, [-1,3]).float()  # 光线方向

    # 为每条光线设置near和far边界
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    # 拼接光线数据：起点、方向、near、far
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        # 如果使用视角方向，则添加到光线数据中
        rays = torch.cat([rays, viewdirs], -1)

    # 渲染并重塑结果
    all_ret = batchify_rays(rays, chunk, **kwargs)
    # 将结果重塑回原始图像形状
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # 提取主要的渲染结果
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]  # RGB图、视差图、累积透明度图
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}  # 其他辅助信息
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    """沿着指定的相机轨迹渲染图像序列
    
    Args:
        render_poses: 相机位姿序列 [N, 3, 4] 或 [N, 4, 4]
        hwf: 图像高度、宽度和焦距 [H, W, focal]
        K: 相机内参矩阵 [3, 3]
        chunk: 渲染时的批次大小
        render_kwargs: 渲染参数字典
        gt_imgs: 真值图像（用于评估），可选
        savedir: 保存渲染结果的目录，可选
        render_factor: 渲染缩放因子，用于快速预览（设为>0时会降采样）
        
    Returns:
        rgbs: 渲染的RGB图像序列 [N, H, W, 3]
        disps: 渲染的视差图序列 [N, H, W]
    """
    H, W, focal = hwf

    if render_factor!=0:
        # 为了加速渲染进行降采样
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []   # 存储RGB渲染结果
    disps = []  # 存储视差渲染结果

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        # 渲染当前视角
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        # 可选：计算与真值图像的PSNR
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        # 可选：保存渲染结果为图像文件
        if savedir is not None:
            rgb8 = to8b(rgbs[-1])  # 转换为8位图像
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    # 将结果堆叠成数组
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """实例化NeRF的MLP模型。
    
    Args:
        args: 包含模型配置参数的参数对象
        
    Returns:
        render_kwargs_train: 训练渲染参数字典
        render_kwargs_test: 测试渲染参数字典  
        start: 训练开始的迭代步数
        grad_vars: 所有需要梯度的参数列表
        optimizer: 优化器
    """
    # 获取位置编码函数和输入通道数
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    # 初始化视角方向相关参数
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # 如果使用视角方向，获取方向编码函数
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    
    # 设置输出通道数：如果使用hierarchical采样则为5(RGB+密度+不确定性)，否则为4(RGB+密度)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]  # 跳跃连接的层位置
    
    # 创建粗糙网络(coarse network)
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    # 如果使用hierarchical采样，创建精细网络(fine network)
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    # 创建网络查询函数，封装位置编码和网络前向传播
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # 创建Adam优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    # 初始化训练参数
    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################
    # 加载检查点 (Checkpoints)
    ##########################

    # 确定要加载的检查点文件
    if args.ft_path is not None and args.ft_path!='None':
        # 如果指定了特定的检查点路径
        ckpts = [args.ft_path]
    else:
        # 否则查找实验目录下的所有tar文件
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        # 加载最新的检查点
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # 恢复训练步数和优化器状态
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # 加载模型权重
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    # 训练时的渲染参数
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,    # 网络查询函数
        'perturb' : args.perturb,                 # 是否对采样点进行扰动
        'N_importance' : args.N_importance,       # 重要性采样的点数
        'network_fine' : model_fine,              # 精细网络
        'N_samples' : args.N_samples,             # 粗糙采样的点数
        'network_fn' : model,                     # 粗糙网络
        'use_viewdirs' : args.use_viewdirs,       # 是否使用视角方向
        'white_bkgd' : args.white_bkgd,           # 是否使用白色背景
        'raw_noise_std' : args.raw_noise_std,     # 原始输出的噪声标准差
    }

    # NDC坐标只适用于LLFF风格的前向数据
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp  # 是否在视差空间线性采样

    # 测试时的渲染参数（不使用扰动和噪声）
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False      # 测试时不扰动
    render_kwargs_test['raw_noise_std'] = 0.   # 测试时不加噪声

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """将模型的原始预测转换为有意义的输出值。
    
    Args:
        raw: [num_rays, num_samples, 4] 模型的原始预测（RGB + 密度）
        z_vals: [num_rays, num_samples] 沿光线的积分时间（深度值）
        rays_d: [num_rays, 3] 每条光线的方向
        raw_noise_std: 添加到密度输出的噪声标准差（用于正则化）
        white_bkgd: 是否假设白色背景
        pytest: 是否为测试模式（使用固定随机数）
        
    Returns:
        rgb_map: [num_rays, 3] 估计的光线RGB颜色
        disp_map: [num_rays] 视差图（深度图的倒数）
        acc_map: [num_rays] 沿每条光线的权重总和（不透明度）
        weights: [num_rays, num_samples] 分配给每个采样颜色的权重
        depth_map: [num_rays] 估计的到物体的距离
    """
    # 定义从原始密度到alpha值的转换函数
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    # 计算相邻采样点之间的距离
    dists = z_vals[...,1:] - z_vals[...,:-1]
    # 在最后添加一个很大的距离（表示到无穷远）
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    # 将距离乘以光线方向的模长，得到真实的世界坐标距离
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    # 通过sigmoid函数将原始RGB预测转换到[0,1]范围
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    
    # 添加噪声进行正则化（仅在训练时）
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # 如果是测试模式，使用固定的随机数
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # 计算alpha值（不透明度）
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    
    # 计算权重：alpha * 累积透射率
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    # 使用权重对RGB进行加权平均，得到最终的颜色
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    # 计算深度图（加权平均的深度）
    depth_map = torch.sum(weights * z_vals, -1)
    # 计算视差图（深度的倒数）
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    # 计算累积不透明度
    acc_map = torch.sum(weights, -1)

    # 如果使用白色背景，将透明部分设为白色
    if white_bkgd:
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
    """体积渲染函数 - NeRF的核心渲染逻辑。
    
    Args:
      ray_batch: [batch_size, ...] 光线批次数据，包含光线起点、方向、
                 近远边界和视角方向等渲染所需的所有信息
      network_fn: 粗糙网络函数，用于预测空间中每点的RGB和密度
      network_query_fn: 网络查询函数，用于向network_fn传递查询
      N_samples: 沿每条光线采样的点数（粗糙采样）
      retraw: 是否返回模型的原始未处理预测
      lindisp: 是否在视差（深度倒数）空间而非深度空间线性采样
      perturb: 扰动参数（0或1），非零时对采样点进行分层随机扰动
      N_importance: 沿每条光线额外采样的点数（重要性采样）
      network_fine: 精细网络，与network_fn规格相同
      white_bkgd: 是否假设白色背景
      raw_noise_std: 添加到密度输出的噪声标准差
      verbose: 是否打印调试信息
      pytest: 测试模式标志
      
    Returns:
      rgb_map: [num_rays, 3] 估计的光线RGB颜色（来自精细模型）
      disp_map: [num_rays] 视差图（深度倒数）
      acc_map: [num_rays] 沿每条光线的累积不透明度（来自精细模型）
      raw: [num_rays, num_samples, 4] 模型的原始预测
      rgb0: 粗糙模型的RGB输出
      disp0: 粗糙模型的视差输出  
      acc0: 粗糙模型的累积不透明度输出
      z_std: [num_rays] 每条光线采样点距离的标准差
    """
    N_rays = ray_batch.shape[0]  # 光线数量
    # 解析光线数据：起点和方向
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    # 如果有视角方向信息则提取，否则为None
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    # 提取near和far边界
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    # 在[0,1]间创建均匀采样点
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        # 在深度空间线性采样
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # 在视差空间线性采样（适用于某些数据集）
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    # 扩展z_vals以匹配光线数量
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # 对采样点进行分层随机扰动
        # 获取采样点之间的中点
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # 在每个区间内进行分层采样
        t_rand = torch.rand(z_vals.shape)

        # 测试模式时使用固定随机数
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    # 计算3D采样点坐标：光线起点 + 方向 * 距离
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


    # 通过粗糙网络预测RGB和密度
    # raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    # 将原始预测转换为RGB图、视差图等有意义的输出
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    # 如果使用分层采样（hierarchical sampling）
    if N_importance > 0:
        # 保存粗糙网络的输出
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # 基于权重进行重要性采样
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])  # 计算中点
        # 根据权重分布进行PDF采样，获得更重要的采样点
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()  # 不参与梯度计算

        # 合并粗糙采样点和重要性采样点，并排序
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        # 重新计算3D采样点坐标
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        # 选择使用精细网络还是粗糙网络
        run_fn = network_fn if network_fine is None else network_fine
        # 通过网络进行前向传播
        # raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        # 将精细网络的预测转换为最终输出
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    # 准备返回结果
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw  # 如果需要，返回原始预测
    if N_importance > 0:
        # 如果使用了分层采样，也返回粗糙网络的结果
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # 重要性采样点的标准差

    # 检查数值稳定性
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    """创建和配置命令行参数解析器。
    
    Returns:
        parser: 配置好的参数解析器对象
    """
    import configargparse
    parser = configargparse.ArgumentParser()
    
    # 基本配置参数
    parser.add_argument('--config', is_config_file=True, 
                        help='配置文件路径')
    parser.add_argument("--expname", type=str, 
                        help='实验名称')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='存储检查点和日志的目录')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='输入数据目录')

    # 训练选项
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='网络层数')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='每层的通道数')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='精细网络的层数')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='精细网络每层的通道数')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='批次大小（每个梯度步骤的随机光线数量）')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='学习率')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='指数学习率衰减（以1000步为单位）')
    parser.add_argument("--N_iters", type=int, default=200000, 
                        help='训练迭代次数')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='并行处理的光线数量，内存不足时减少此值')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='并行通过网络的点数，内存不足时减少此值')
    parser.add_argument("--no_batching", action='store_true', 
                        help='每次只从1张图像中取随机光线')
    parser.add_argument("--no_reload", action='store_true', 
                        help='不从保存的检查点重新加载权重')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='为粗糙网络重新加载的特定权重文件路径')

    # 渲染选项
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='每条光线的粗糙采样点数')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='每条光线的额外精细采样点数')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='设为0表示无抖动，设为1表示有抖动')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='使用完整的5D输入而不是3D（包含视角方向）')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='设为0使用默认位置编码，设为-1表示不使用')
    parser.add_argument("--multires", type=int, default=10, 
                        help='位置编码的最大频率的log2值（3D位置）')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='位置编码的最大频率的log2值（2D方向）')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='添加到密度输出的噪声标准差，推荐1e0')

    parser.add_argument("--render_only", action='store_true', 
                        help='不进行优化，仅加载权重并渲染指定的相机轨迹')
    parser.add_argument("--render_test", action='store_true', 
                        help='渲染测试集而不是预设的相机轨迹')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='降采样因子以加速渲染，设为4或8可快速预览')

    # 训练选项
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='在中心裁剪区域训练的步数')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='中心裁剪占图像的比例') 

    # 数据集选项
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='数据集类型选项: llff / blender / deepvoxels / LINEMOD / spe3r')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='从测试/验证集加载1/N张图像，对deepvoxels等大数据集有用')
    
    ## SPE3R数据集标志
    parser.add_argument("--no_masks", action='store_true', 
                        help='SPE3R数据集不使用掩码去除背景')

    ## DeepVoxels数据集标志
    parser.add_argument("--shape", type=str, default='greek', 
                        help='形状选项: armchair / cube / greek / vase')

    ## Blender数据集标志
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='在白色背景上渲染合成数据（deepvoxels总是使用）')
    parser.add_argument("--half_res", action='store_true', 
                        help='以400x400而不是800x800加载blender合成数据')

    ## LLFF数据集标志
    parser.add_argument("--factor", type=int, default=8, 
                        help='LLFF图像的降采样因子')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='不使用标准化设备坐标（用于非前向场景）')
    parser.add_argument("--lindisp", action='store_true', 
                        help='在视差空间而非深度空间线性采样')
    parser.add_argument("--spherify", action='store_true', 
                        help='用于球形360度场景')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='每1/N张图像作为LLFF测试集，论文使用8')

    # 日志记录/保存选项
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='控制台打印和指标记录的频率')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='tensorboard图像记录的频率')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='权重检查点保存的频率')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='测试集保存的频率')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='渲染轨迹视频保存的频率')

    return parser


def train():
    """NeRF模型的主要训练函数。
    
    此函数负责：
    1. 解析命令行参数
    2. 加载数据集
    3. 创建NeRF模型
    4. 执行训练循环
    5. 保存检查点和渲染结果
    """
    # 解析配置参数
    parser = config_parser()
    args = parser.parse_args()

    # 加载数据
    K = None  # 相机内参矩阵，将根据数据集类型设置
    print(f"DEBUG: Dataset type is '{args.dataset_type}'")
    if args.dataset_type == 'llff':
        # 加载LLFF数据集（真实场景前向数据）
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]  # 提取高度、宽度、焦距
        poses = poses[:,:3,:4]  # 只保留相机外参的旋转和平移部分
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        # 如果设置了holdout，则自动选择测试图像
        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        # 设置验证集和训练集索引
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        # 根据是否使用NDC设置near和far边界
        if args.no_ndc:
            # 不使用NDC时，使用实际的深度边界
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            # 使用NDC时的标准化边界
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        # 加载Blender合成数据集
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        # Blender数据集的标准near/far设置
        near = 2.
        far = 6.

        # 处理背景：合成数据通常包含alpha通道
        if args.white_bkgd:
            # 使用白色背景：RGB = RGB * alpha + white * (1 - alpha)
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            # 只保留RGB通道，忽略alpha
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    elif args.dataset_type == 'spe3r':
        # 加载SPE3R数据集（航天器图像数据）
        print('DEBUG: Loading SPE3R dataset...')
        use_masks = not args.no_masks  # 默认使用掩码，除非明确指定不使用
        images, poses, render_poses, hwf, i_split = load_spe3r_data(args.datadir, args.half_res, args.testskip, use_masks=use_masks)
        print('Loaded spe3r', images.shape, render_poses.shape, hwf, args.datadir)
        if use_masks:
            print('Using masks to remove background')  # 使用掩码去除背景
        else:
            print('Not using masks - keeping original background')  # 保持原始背景
        i_train, i_val, i_test = i_split

        # SPE3R数据集的near/far设置
        # 由于是航天器场景，相机距离物体较远，需要设置合适的near/far值
        near = 2.0  # 最近距离
        far = 8.0   # 最远距离

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # 将内参转换为正确的类型
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # 如果没有提供相机内参矩阵K，则根据焦距构建标准内参矩阵
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],     # fx, 0, cx
            [0, focal, 0.5*H],     # 0, fy, cy
            [0, 0, 1]              # 0, 0, 1
        ])

    # 如果只渲染测试集，使用测试集的相机位姿
    if args.render_test:
        render_poses = np.array(poses[i_test])

    # 创建日志目录并复制配置文件
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    
    # 保存所有参数到args.txt文件
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    
    # 如果使用了配置文件，也复制一份
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # 创建NeRF模型
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start  # 全局训练步数

    # 更新渲染参数中的边界设置
    bds_dict = {
        'near' : near,  # 最近边界
        'far' : far,    # 最远边界
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # 将测试数据移动到GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # 如果只是渲染模式（不训练），直接从训练好的模型渲染
    if args.render_only:
        print('RENDER ONLY')  # 仅渲染模式
        with torch.no_grad():  # 不计算梯度以节省内存
            if args.render_test:
                # render_test模式：渲染测试集图像
                images = images[i_test]
            else:
                # 默认模式：渲染预设的平滑轨迹
                images = None

            # 创建渲染结果保存目录
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            # 沿着指定轨迹渲染图像
            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            
            # 将渲染结果保存为MP4视频
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return  # 渲染完成后直接返回，不进行训练

    # 如果使用随机光线批处理，准备光线批次张量
    N_rand = args.N_rand  # 每个批次的随机光线数量
    use_batching = not args.no_batching  # 是否使用批处理模式
    if use_batching:
        # 随机光线批处理模式
        print('get rays')
        # 为每个训练图像生成光线
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        # 将光线与对应的RGB值拼接
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        # 重新排列维度以便后续处理
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        # 只保留训练图像的数据
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        # 将所有光线重塑为平坦的形状
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        # 随机打乱光线顺序以避免偏差
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0  # 批次索引初始化

    # 将训练数据移动到GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    # 训练设置
    N_iters = args.N_iters + 1  # 总训练迭代次数
    print('Begin')
    print('TRAIN views are', i_train)  # 打印训练视图索引
    print('TEST views are', i_test)    # 打印测试视图索引
    print('VAL views are', i_val)      # 打印验证视图索引

    # 可选：TensorBoard日志记录器
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    # 初始化用于存储训练指标的列表（用于绘制训练曲线）
    train_iterations = []  # 训练迭代步数
    train_losses = []      # 训练损失值
    train_psnrs = []       # 训练PSNR值
    
    start = start + 1
    # 主要训练循环
    for i in trange(start, N_iters):
        time0 = time.time()  # 记录开始时间

        # 采样随机光线批次
        if use_batching:
            # 批处理模式：从所有图像的光线中随机采样
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]  # 分离光线和目标RGB

            # 更新批次索引
            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                # 一个epoch结束，重新打乱数据
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # 单图像模式：每次从一张图像中随机采样光线
            img_i = np.random.choice(i_train)  # 随机选择一张训练图像
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]  # 获取对应的相机位姿

            if N_rand is not None:
                # 生成该图像的所有光线
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                # 如果在预裁剪阶段，只从图像中心区域采样
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)  # 中心裁剪的半高
                    dW = int(W//2 * args.precrop_frac)  # 中心裁剪的半宽
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    # 正常训练阶段：从整个图像采样
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                # 随机选择N_rand个像素坐标
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                
                # 根据选中的坐标提取对应的光线和目标RGB
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  核心优化循环  #####
        # 渲染当前批次的光线
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        # 清零梯度
        optimizer.zero_grad()
        
        # 计算主要损失：预测RGB与真值RGB之间的均方误差
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]  # 提取透射率信息
        loss = img_loss
        psnr = mse2psnr(img_loss)  # 计算PSNR指标

        # 如果使用了分层采样，还需要计算粗糙网络的损失
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0  # 总损失 = 精细网络损失 + 粗糙网络损失
            psnr0 = mse2psnr(img_loss0)

        # 反向传播和参数更新
        loss.backward()
        optimizer.step()

        # 注意：重要！
        ###   更新学习率（指数衰减）   ###
        decay_rate = 0.1  # 衰减率
        decay_steps = args.lrate_decay * 1000  # 衰减步数
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        # 更新优化器中的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0  # 计算本步训练用时
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           结束            #####

        # 其余部分是日志记录和保存
        
        # 定期保存模型检查点
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,  # 全局步数
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),      # 粗糙网络权重
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),  # 精细网络权重
                'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
            }, path)
            print('Saved checkpoints at', path)

        # 定期生成螺旋轨迹渲染视频
        if i%args.i_video==0 and i > 0:
            # 切换到测试模式（不计算梯度）
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            # 保存RGB视频
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            # 保存深度视频（视差图）
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # 可选：如果使用视角方向，可以生成静态相机视频
            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        # 定期渲染和保存测试集结果
        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        # 定期打印训练进度并记录指标
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            # 存储指标用于后续绘图
            train_iterations.append(i)
            train_losses.append(loss.item())
            train_psnrs.append(psnr.item())
        """
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

        global_step += 1  # 更新全局训练步数

    # 训练完成后生成训练曲线图
    print("\n=== Training completed! Generating training plots ===")
    print("训练完成！正在生成训练曲线图...")
    
    def plot_training_metrics(iterations, losses, psnrs, save_dir):
        """生成并保存训练曲线图"""
        import matplotlib.pyplot as plt
        
        # 创建包含子图的图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制损失曲线
        ax1.plot(iterations, losses, 'b-', linewidth=2, label='训练损失')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('损失')
        ax1.set_title('训练损失 vs 迭代次数')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 绘制PSNR曲线
        ax2.plot(iterations, psnrs, 'r-', linewidth=2, label='训练PSNR')
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('训练PSNR vs 迭代次数')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(save_dir, 'training_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to: {plot_path}")
        
        # 同时保存PDF格式以获得更好的质量
        plot_path_pdf = os.path.join(save_dir, 'training_metrics.pdf')
        plt.savefig(plot_path_pdf, bbox_inches='tight')
        print(f"Training plots (PDF) saved to: {plot_path_pdf}")
        
        plt.close()
        
        # 创建组合图表
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 创建双Y轴图以便同时显示损失和PSNR
        if len(losses) > 0 and len(psnrs) > 0:
            ax2 = ax.twinx()  # 创建第二个y轴
            
            # 绘制损失曲线（左y轴）
            line1 = ax.plot(iterations, losses, 'b-', linewidth=2, label='损失')
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('损失', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            
            # 绘制PSNR曲线（右y轴）
            line2 = ax2.plot(iterations, psnrs, 'r-', linewidth=2, label='PSNR (dB)')
            ax2.set_ylabel('PSNR (dB)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # 添加图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='best')
            
            ax.set_title('训练损失和PSNR vs 迭代次数')
            ax.grid(True, alpha=0.3)
            
            # 保存组合图表
            combined_plot_path = os.path.join(save_dir, 'training_combined.png')
            plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
            print(f"Combined training plot saved to: {combined_plot_path}")
            
            plt.close()
    
    def save_training_data(iterations, losses, psnrs, save_dir):
        """保存训练数据为CSV格式以供后续分析使用"""
        import csv
        
        # 保存为CSV文件
        csv_path = os.path.join(save_dir, 'training_data.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Iteration', 'Loss', 'PSNR'])  # 写入表头
            for i, loss, psnr in zip(iterations, losses, psnrs):
                writer.writerow([i, loss, psnr])
        print(f"Training data saved to: {csv_path}")
        
        # 同时保存为numpy数组格式
        np_path = os.path.join(save_dir, 'training_data.npz')
        np.savez(np_path, 
                iterations=np.array(iterations),
                losses=np.array(losses),
                psnrs=np.array(psnrs))
        print(f"Training data (NumPy) saved to: {np_path}")
    
    # 生成图表并保存数据
    log_dir = os.path.join(basedir, expname)
    
    if len(train_iterations) > 0:
        # 生成训练曲线图
        plot_training_metrics(train_iterations, train_losses, train_psnrs, log_dir)
        # 保存训练数据
        save_training_data(train_iterations, train_losses, train_psnrs, log_dir)
        
        # 打印训练总结统计信息
        print("\n=== Training Summary ===")
        print("=== 训练总结 ===")
        print(f"Total iterations: {len(train_iterations)}")
        print(f"总迭代次数: {len(train_iterations)}")
        print(f"Final Loss: {train_losses[-1]:.6f}")
        print(f"最终损失: {train_losses[-1]:.6f}")
        print(f"Final PSNR: {train_psnrs[-1]:.2f} dB")
        print(f"最终PSNR: {train_psnrs[-1]:.2f} dB")
        print(f"Best PSNR: {max(train_psnrs):.2f} dB (at iteration {train_iterations[train_psnrs.index(max(train_psnrs))]})")
        print(f"最佳PSNR: {max(train_psnrs):.2f} dB (在第 {train_iterations[train_psnrs.index(max(train_psnrs))]} 次迭代)")
        print(f"Lowest Loss: {min(train_losses):.6f} (at iteration {train_iterations[train_losses.index(min(train_losses))]})")
        print(f"最低损失: {min(train_losses):.6f} (在第 {train_iterations[train_losses.index(min(train_losses))]} 次迭代)")
    else:
        print("No training metrics were collected. Make sure i_print > 0 in your config.")
        print("未收集到训练指标。请确保配置中 i_print > 0。")


if __name__=='__main__':
    # 设置默认tensor类型为CUDA FloatTensor（如果有GPU）
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # 开始训练
    train()
