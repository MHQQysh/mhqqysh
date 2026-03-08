from typing import Any

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import PandaStick
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


# extending TableSceneBuilder and only making 2 changes:
# 1.Making table smooth and white, 2. adding support for keyframes of new robots - panda stick
class WhiteTableSceneBuilder(TableSceneBuilder):
    def initialize(self, env_idx: torch.Tensor):
        super().initialize(env_idx)
        b = len(env_idx)
        if self.env.robot_uids == "panda_stick":
            qpos = np.array(
                [
                    0.662,
                    0.212,
                    0.086,
                    -2.685,
                    -0.115,
                    2.898,
                    1.673,
                ]
            )
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    def build(self):
        super().build()
        # cheap way to un-texture table
        for part in self.table._objs:
            for triangle in (
                part.find_component_by_type(sapien.render.RenderBodyComponent)
                .render_shapes[0]
                .parts
            ):
                triangle.material.set_base_color(np.array([255, 255, 255, 255]) / 255)
                triangle.material.set_base_color_texture(None)
                triangle.material.set_normal_texture(None)
                triangle.material.set_emission_texture(None)
                triangle.material.set_transmission_texture(None)
                triangle.material.set_metallic_texture(None)
                triangle.material.set_roughness_texture(None)


@register_env("PushT-v1", max_episode_steps=100)
class PushTEnv(BaseEnv):
    """
    **Task Description:**
    A simulated version of the real-world push-T task from Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

    In this task, the robot needs to:
    1. Precisely push the T-shaped block into the target region, and
    2. Move the end-effector to the end-zone which terminates the episode. [2 Not required for PushT-easy-v1]

    **Randomizations:**
    - 3D T block initial position on table  [-1,1] x [-1,2] + T Goal initial position
    - 3D T block initial z rotation         [0,2pi]

    **Success Conditions:**
    - The T block covers 90% of the 2D goal T's area
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PushT-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_stick"]
    agent: PandaStick

    # # # # # # # # All Unspecified real-life Parameters Here # # # # # # # #
    # Randomizations
    # 3D T center of mass spawnbox dimensions
    tee_spawnbox_xlength = 0.2
    tee_spawnbox_ylength = 0.3

    # translation of the spawnbox from goal tee as upper left of spawnbox
    tee_spawnbox_xoffset = -0.1
    tee_spawnbox_yoffset = -0.1
    #  end randomizations - rotation around z is simply uniform

    # Hand crafted params to match visual of real life setup
    # T Goal initial position on table
    goal_offset = torch.tensor([-0.156, -0.1])
    goal_z_rot = (5 / 3) * np.pi

    # end effector goal - NOTE that chaning this will not change the actual
    # ee starting position of the robot - need to change joint position resting
    # keyframe in table setup to change ee starting location, then copy that location here
    ee_starting_pos2D = torch.tensor([-0.321, 0.284, 1e-3])
    # this will be used in the state observations
    ee_starting_pos3D = torch.tensor([-0.321, 0.284, 0.024])

    # intersection threshold for success in T position
    intersection_thresh = 0.90

    # T block design choices
    T_mass = 0.8
    T_dynamic_friction = 3
    T_static_friction = 3

    def __init__(
        self, *args, robot_uids="panda_stick", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # 框架[1/6]：设备初始化，把运行时张量迁移到当前 device。
        # _load_scene 是一次性迁移这些张量的合适位置。
        self.ee_starting_pos2D = self.ee_starting_pos2D.to(self.device)  # 2D 末端起点放到当前设备
        self.ee_starting_pos3D = self.ee_starting_pos3D.to(self.device)  # 3D 末端起点放到当前设备

        # 框架[2/6]：构建基础场景（桌面与地面）。
        self.table_scene = WhiteTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )  # 创建桌面场景构建器
        self.table_scene.build()  # 实例化地面/桌面 actor

        # 框架[3/6]：创建可动物理 T 与运动学目标 T。
        # create_tee 构建 CAD 风格 T，局部原点放在质心 (0,0,0)。
        TARGET_RED = (
            np.array([194, 19, 22, 255]) / 255
        )  # 与 mani_skill.utils.building.actors.common 的目标红一致

        def create_tee(name="tee", target=False, base_color=TARGET_RED):
            # 用两个 box 拼成 T 形。
            # box2 与 box1 相比：长度为 3/4，且布局上是竖条。
            # 尺寸来自 diffusion policy 使用的 3D CAD: https://cad.onshape.com/documents/f1140134e38f6ed6902648d5/w/a78cf81827600e4ff4058d03/e/f35f57fb7589f72e05c76caf
            box1_half_w = 0.2 / 2  # 横条半宽
            box1_half_h = 0.05 / 2  # 横条半高
            half_thickness = 0.04 / 2 if not target else 1e-4  # 可动 T 有厚度，目标 T 近似薄片

            # 为让旋转绕质心发生，需要把几何中心平移到质心。
            # 竖条是横条尺寸的 3/4，因此质心在 y 方向有偏移。
            # center of mass = (1*com_horiz + (3/4)*com_vert) / (1+(3/4))
            # 代入可得 (0, 0.0375)
            com_y = 0.0375  # 将几何坐标转到质心坐标系的 y 偏移

            builder = self.scene.create_actor_builder()  # tee 的 actor builder
            first_block_pose = sapien.Pose([0.0, 0.0 - com_y, 0.0])  # 横条位姿
            first_block_size = [box1_half_w, box1_half_h, half_thickness]  # 横条半尺寸
            if not target:
                builder._mass = self.T_mass  # 可动 T 质量
                tee_material = sapien.pysapien.physx.PhysxMaterial(
                    static_friction=self.T_dynamic_friction,
                    dynamic_friction=self.T_static_friction,
                    restitution=0,
                )
                builder.add_box_collision(
                    pose=first_block_pose,
                    half_size=first_block_size,
                    material=tee_material,
                )  # 横条碰撞体
                # builder.add_box_collision(pose=first_block_pose, half_size=first_block_size)
            builder.add_box_visual(
                pose=first_block_pose,
                half_size=first_block_size,
                material=sapien.render.RenderMaterial(
                    base_color=base_color,
                ),
            )  # 横条可视体

            # 竖条通过 y 平移 4*box1_half_h-com_y 与横条贴齐。
            # 注意：这里的 CAD T 朝向约定是“倒置”的。
            second_block_pose = sapien.Pose([0.0, 4 * (box1_half_h) - com_y, 0.0])  # 竖条位姿
            second_block_size = [box1_half_h, (3 / 4) * (box1_half_w), half_thickness]  # 竖条半尺寸
            if not target:
                builder.add_box_collision(
                    pose=second_block_pose,
                    half_size=second_block_size,
                    material=tee_material,
                )  # 竖条碰撞体
                # builder.add_box_collision(pose=second_block_pose, half_size=second_block_size)
            builder.add_box_visual(
                pose=second_block_pose,
                half_size=second_block_size,
                material=sapien.render.RenderMaterial(
                    base_color=base_color,
                ),
            )  # 竖条可视体
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])  # 初始位置放在桌面上方
            if not target:
                return builder.build(name=name)  # 动力学 tee actor
            else:
                return builder.build_kinematic(name=name)  # 运动学目标 tee

        self.tee = create_tee(name="Tee", target=False)  # 可交互的可动 T
        self.goal_tee = create_tee(
            name="goal_Tee",
            target=True,
            base_color=np.array([128, 128, 128, 255]) / 255,
        )  # 静态目标 T

        # 框架[4/6]：添加末端终点目标标记（仅可视）。
        builder = self.scene.create_actor_builder()  # ee 目标标记的 builder
        builder.add_cylinder_visual(
            radius=0.02,
            half_length=1e-4,
            material=sapien.render.RenderMaterial(
                base_color=np.array([128, 128, 128, 255]) / 255
            ),
        )  # 薄圆柱标记
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])  # 放在桌面上方
        self.ee_goal_pos = builder.build_kinematic(name="goal_ee")  # 运动学标记 actor

        # 框架[5/6]：准备 2D 伪渲染所需网格数据。
        res = 64  # 重叠图的栅格分辨率
        uv_half_width = 0.15  # uv 平面半宽（米）
        self.uv_half_width = uv_half_width  # 缓存给后续计算复用
        self.res = res  # 缓存分辨率
        oned_grid = torch.arange(res, dtype=torch.float32).view(1, res).repeat(
            res, 1
        ) - (res / 2)  # 以中心为原点的像素网格
        self.uv_grid = (
            torch.cat([oned_grid.unsqueeze(0), (-1 * oned_grid.T).unsqueeze(0)], dim=0)
            + 0.5
        ) / ((res / 2) / uv_half_width)  # 像素坐标转换到 uv 坐标
        self.uv_grid = self.uv_grid.to(self.device)  # 与后续伪渲染计算保持同设备
        self.homo_uv = torch.cat(
            [self.uv_grid, torch.ones_like(self.uv_grid[0]).unsqueeze(0)], dim=0
        )  # 齐次 uv 坐标 [u,v,1]

        # 框架[6/6]：预计算 tee 二值掩码与 world->goal 变换。
        # tee 由两个 box 组成，并且整体经过质心平移。
        self.center_of_mass = (
            0,
            0.0375,
        )  # 该坐标在“倒置 tee 且横条中心”为原点的坐标系下
        box1 = torch.tensor(
            [[-0.1, 0.025], [0.1, 0.025], [-0.1, -0.025], [0.1, -0.025]]
        )  # 横条标准矩形四角
        box2 = torch.tensor(
            [[-0.025, 0.175], [0.025, 0.175], [-0.025, 0.025], [0.025, 0.025]]
        )  # 竖条标准矩形四角
        box1[:, 1] -= self.center_of_mass[1]  # 按质心做 y 方向平移
        box2[:, 1] -= self.center_of_mass[1]  # 按质心做 y 方向平移

        # 把 tee 的坐标转换成图像索引。
        box1 *= (res / 2) / uv_half_width  # 米制坐标 -> 像素尺度
        box1 += res / 2  # 原点平移到图像中心

        box2 *= (res / 2) / uv_half_width  # 米制坐标 -> 像素尺度
        box2 += res / 2  # 原点平移到图像中心

        box1 = box1.long()  # 切片需要整型索引
        box2 = box2.long()  # 切片需要整型索引

        self.tee_render = torch.zeros(res, res)  # tee 的二值栅格图
        # 图像坐标 x/y 与几何轴有交换，用转置写入来纠正。
        self.tee_render.T[box1[0, 0] : box1[1, 0], box1[2, 1] : box1[0, 1]] = 1
        self.tee_render.T[box2[0, 0] : box2[1, 0], box2[2, 1] : box2[0, 1]] = 1
        # 图像 y 方向与 xy 平面相反，这里再翻转回来。
        self.tee_render = self.tee_render.flip(0).to(self.device)  # 最终 tee 模板放到当前设备

        goal_fake_quat = torch.tensor(
            [(torch.tensor([self.goal_z_rot]) / 2).cos(), 0, 0, 0.0]
        ).unsqueeze(0)  # 由目标 z 角构造仅绕 z 的四元数
        zrot = self.quat_to_zrot(goal_fake_quat).squeeze(
            0
        )  # 目标坐标到世界坐标的 3x3 旋转矩阵
        goal_trans = torch.eye(3)  # 齐次变换 T_goal->world
        goal_trans[:2, :2] = zrot[:2, :2]
        goal_trans[0:2, 2] = self.goal_offset
        self.world_to_goal_trans = torch.linalg.inv(goal_trans).to(
            self.device
        )  # 3x3 的 2D 齐次变换矩阵 T_world->goal

    def quat_to_z_euler(self, quats):
        assert len(quats.shape) == 2 and quats.shape[-1] == 4
        # z rotation == can be defined by just qw = cos(alpha/2), so alpha = 2*cos^{-1}(qw)
        # for fixing quaternion double covering
        # for some reason, torch.sign() had bugs???
        signs = torch.ones_like(quats[:, -1])
        signs[quats[:, -1] < 0] = -1.0
        qw = quats[:, 0] * signs
        z_euler = 2 * qw.acos()
        return z_euler

    def quat_to_zrot(self, quats):
        # expecting batch of quaternions (b,4)
        assert len(quats.shape) == 2 and quats.shape[-1] == 4
        # output is batch of rotation matrices (b,3,3)
        alphas = self.quat_to_z_euler(quats)
        # constructing rot matrix with rotation around z
        rot_mats = torch.zeros(quats.shape[0], 3, 3).to(quats.device)
        rot_mats[:, 2, 2] = 1
        rot_mats[:, 0, 0] = alphas.cos()
        rot_mats[:, 1, 1] = alphas.cos()
        rot_mats[:, 0, 1] = -alphas.sin()
        rot_mats[:, 1, 0] = alphas.sin()
        return rot_mats

    def pseudo_render_intersection(self):
        """'pseudo render' algo for calculating the intersection
        made custom 'psuedo renderer' to compute intersection area
        all computation in parallel on cuda, zero explicit loops
        views blocks in 2d in the goal tee frame to see overlap"""
        # we are given T_{a->w} where a == actor frame and w == world frame
        # we are given T_{g->w} where g == goal frame and w == world frame
        # applying T_{a->w} and then T_{w->g}, we get the actor's orientation in the goal tee's frame
        # T_{w->g} is T_{g->w}^{-1}, we already have the goal's orientation, and it doesn't change
        tee_to_world_trans = self.quat_to_zrot(
            self.tee.pose.q
        )  # should be (b,3,3) rot matrices
        tee_to_world_trans[:, 0:2, 2] = self.tee.pose.p[
            :, :2
        ]  # should be (b,3,3) rigid trans matrices

        # these matrices convert egocentric 3d tee to 2d goal tee frame
        tee_to_goal_trans = (
            self.world_to_goal_trans @ tee_to_world_trans
        )  # should be (b,3,3) rigid trans matrices

        # making homogenious coords of uv map to apply transformations to view tee in goal tee frame
        b = tee_to_world_trans.shape[0]
        res = self.uv_grid.shape[1]
        homo_uv = self.homo_uv

        # finally, get uv coordinates of tee in goal tee frame
        tees_in_goal_frame = (tee_to_goal_trans @ homo_uv.view(3, -1)).view(
            b, 3, res, res
        )
        # convert from homogenious coords to normal coords
        tees_in_goal_frame = tees_in_goal_frame[:, 0:2, :, :] / tees_in_goal_frame[
            :, -1, :, :
        ].unsqueeze(
            1
        )  #  now (b,2,res,res)

        # we now have a collection of coordinates xy that are the coordinates of the tees in the goal frame
        # we just extract the indices in the uv map where the egocentic T is, to get the transformed T coords
        # this works because while we transformed the coordinates of the uv map -
        # the indices where the egocentric T is is still the indices of the T in the uv map (indices of uv map never chnaged, just values)
        tee_coords = tees_in_goal_frame[:, :, self.tee_render == 1].view(
            b, 2, -1
        )  #  (b,2,num_points_in_tee)

        # convert tee_coords to indices - this is basically a batch of indices - same shape as tee_coords
        # this is the inverse function of creating the uv map from image indices used in load_scene
        tee_indices = (
            (tee_coords * ((res / 2) / self.uv_half_width) + (res / 2))
            .long()
            .view(b, 2, -1)
        )  #  (b,2,num_points_in_tee)

        # setting all of our work in image format to compare with egocentric image of goal T
        final_renders = torch.zeros(b, res, res).to(self.device)
        # for batch indexing
        num_tee_pixels = tee_indices.shape[-1]
        batch_indices = (
            torch.arange(b).view(-1, 1).repeat(1, num_tee_pixels).to(self.device)
        )

        # # ensure no out of bounds indexing - it's fine to not fully 'render' tee, just need to fully see goal tee which is insured
        # # because we are in the goal tee frame, and 'cad' tee render setup of egocentric view includes full tee
        # # also, the reward isn't miou, it's intersection area / goal area - don't need union -> don't need full T 'render'
        # #ugly solution for now to keep parallelism no loop - set out of bound image t indices to [0,0]
        # # anywhere where x or y is out of bounds, make indices (0,0)
        invalid_xs = (tee_indices[:, 0, :] < 0) | (tee_indices[:, 0, :] >= self.res)
        invalid_ys = (tee_indices[:, 1, :] < 0) | (tee_indices[:, 1, :] >= self.res)
        tee_indices[:, 0, :][invalid_xs] = 0
        tee_indices[:, 1, :][invalid_xs] = 0
        tee_indices[:, 0, :][invalid_ys] = 0
        tee_indices[:, 1, :][invalid_ys] = 0

        final_renders[batch_indices, tee_indices[:, 0, :], tee_indices[:, 1, :]] = 1
        # coord to image fix - need to transpose each image in the batch, then reverse y coords to correctly visualize
        final_renders = final_renders.permute(0, 2, 1).flip(1)

        # finally, we can calculate intersection/goal_area for reward
        intersection = (
            (final_renders.bool() & self.tee_render.bool()).sum(dim=[-1, -2]).float()
        )
        goal_area = self.tee_render.bool().sum().float()

        reward = intersection / goal_area

        # del tee_to_world_trans; del tee_to_goal_trans; del tees_in_goal_frame; del tee_coords; del tee_indices
        # del final_renders; del invalid_xs; del invalid_ys; batch_indices; del intersection; del goal_area
        # torch.cuda.empty_cache()
        return reward

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # setting the goal tee position, which is fixed, offset from center, and slightly rotated
            target_region_xyz = torch.zeros((b, 3))
            target_region_xyz[:, 0] += self.goal_offset[0]
            target_region_xyz[:, 1] += self.goal_offset[1]
            # set a little bit above 0 so the target is sitting on the table
            target_region_xyz[..., 2] = 1e-3
            self.goal_tee.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, 0, self.goal_z_rot),
                )
            )

            # randomization code that randomizes the x, y position of the tee we
            # goal tee is alredy at y = -0.1 relative to robot, so we allow the tee to be only -0.2 y relative to robot arm
            target_region_xyz[..., 0] += (
                torch.rand(b) * (self.tee_spawnbox_xlength) + self.tee_spawnbox_xoffset
            )
            target_region_xyz[..., 1] += (
                torch.rand(b) * (self.tee_spawnbox_ylength) + self.tee_spawnbox_yoffset
            )

            target_region_xyz[..., 2] = (
                0.04 / 2 + 1e-3
            )  # this is the half thickness of the tee plus a little
            # rotation for pose is just random rotation around z axis
            # z axis rotation euler to quaternion = [cos(theta/2),0,0,sin(theta/2)]
            q_euler_angle = torch.rand(b) * (2 * torch.pi)
            q = torch.zeros((b, 4))
            q[:, 0] = (q_euler_angle / 2).cos()
            q[:, -1] = (q_euler_angle / 2).sin()

            obj_pose = Pose.create_from_pq(p=target_region_xyz, q=q)
            self.tee.set_pose(obj_pose)

            # ee starting/ending position marked on table like irl task
            xyz = torch.zeros((b, 3))
            xyz[:] = self.ee_starting_pos2D
            self.ee_goal_pos.set_pose(
                Pose.create_from_pq(
                    p=xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def evaluate(self):
        # success is where the overlap is over intersection thresh and ee dist to start pos is less than it's own thresh
        inter_area = self.pseudo_render_intersection()
        tee_place_success = (inter_area) >= self.intersection_thresh

        success = tee_place_success

        return {"success": success}

    def _get_obs_extra(self, info: dict):
        # ee position is super useful for pandastick robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            # state based gets info on goal position and t full pose - necessary to learn task
            obs.update(
                goal_pos=self.goal_tee.pose.p,
                obj_pose=self.tee.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        # reward for overlap of the tees

        # legacy reward
        # reward = self.pseudo_render_reward()
        # Pose based reward below is preferred over legacy reward
        # legacy reward gets stuck in local maxs of 50-75% intersection
        # and then fails to promote large explorations to perfectly orient the T, for PPO algorithm

        # new pose based reward: cos(z_rot_euler) + function of translation, between target and goal both in [0,1]
        # z euler cosine similarity reward: -- quat_to_z_euler guarenteed to reutrn value from [0,2pi]
        tee_z_eulers = self.quat_to_z_euler(self.tee.pose.q)
        # subtract the goal z rotatation to get relative rotation
        rot_rew = (tee_z_eulers - self.goal_z_rot).cos()
        # cos output [-1,1], we want reward of 0.5
        reward = (((rot_rew + 1) / 2) ** 2) / 2

        # x and y distance as reward
        tee_to_goal_pose = self.tee.pose.p[:, 0:2] - self.goal_tee.pose.p[:, 0:2]
        tee_to_goal_pose_dist = torch.linalg.norm(tee_to_goal_pose, axis=1)
        reward += ((1 - torch.tanh(5 * tee_to_goal_pose_dist)) ** 2) / 2

        # giving the robot a little help by rewarding it for having its end-effector close to the tee center of mass
        tcp_to_push_pose = self.tee.pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reward += ((1 - torch.tanh(5 * tcp_to_push_pose_dist)).sqrt()) / 20

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
