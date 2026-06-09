import os
import sys
sys.path.append(os.getcwd())
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, update_class_from_dict
from isaacgym import gymapi
from datetime import datetime
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml
from isaacgym import gymapi

def _make_video_path(args):
    os.makedirs(args.video_dir, exist_ok=True)
    if args.video_name:
        video_name = args.video_name
    else:
        checkpoint = "latest" if args.checkpoint is None else str(args.checkpoint)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = f"{args.task}_model_{checkpoint}_{timestamp}.mp4"
    if not video_name.endswith(".mp4"):
        video_name += ".mp4"
    return os.path.join(args.video_dir, video_name)

def _set_camera(gym, sim_env, camera_handle, camera_position, camera_target):
    position = gymapi.Vec3(*camera_position.tolist())
    target = gymapi.Vec3(*camera_target.tolist())
    if camera_handle is not None:
        gym.set_camera_location(camera_handle, sim_env, position, target)

def _record_frame(env, camera_handle, video_writer, width, height):
    env.gym.render_all_camera_sensors(env.sim)
    frame = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
    frame = np.reshape(frame, (height, width, 4))[:, :, :3]
    video_writer.append_data(frame)

def _make_state_renderer(width, height):
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax

def _record_state_frame(env, renderer, video_writer):
    fig, ax = renderer
    body_pos = env.rigid_body_states[0, :, :3].detach().cpu().numpy()
    root_pos = env.root_states[0, :3].detach().cpu().numpy()

    ax.clear()
    ax.scatter(body_pos[:, 0], body_pos[:, 1], body_pos[:, 2], c="tab:blue", s=18)
    for pos in body_pos:
        ax.plot([root_pos[0], pos[0]], [root_pos[1], pos[1]], [root_pos[2], pos[2]], c="0.55", linewidth=0.8)
    ax.scatter([root_pos[0]], [root_pos[1]], [root_pos[2]], c="tab:red", s=36)

    ax.set_xlim(root_pos[0] - 1.5, root_pos[0] + 1.5)
    ax.set_ylim(root_pos[1] - 1.5, root_pos[1] + 1.5)
    ax.set_zlim(0.0, 2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=18, azim=-60)
    ax.set_title("H1 policy state trace")
    fig.tight_layout()
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    video_writer.append_data(frame)

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    resume_path = train_cfg.runner.resume_path
    print(resume_path)
    
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.episode_length_s = 100000
    env_cfg.viewer.record_video = False
    env_cfg.viewer.video_width = args.video_width
    env_cfg.viewer.video_height = args.video_height

    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.randomize_load = False
    env_cfg.domain_rand.randomize_gains = False 
    env_cfg.domain_rand.randomize_link_props = False
    env_cfg.domain_rand.randomize_base_mass = False

    env_cfg.commands.resampling_time = 100
    env_cfg.rewards.penalize_curriculum = False
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = 1
    env_cfg.terrain.selected = True
    env_cfg.terrain.selected_terrain_type = "random_uniform"
    env_cfg.terrain.terrain_kwargs = {  # Dict of arguments for selected terrain
        "random_uniform":
            {
                "min_height": -0.00,
                "max_height": 0.00,
                "step": 0.005,
                "downsampled_scale": 0.2
            },
    }

    # prepare     # planeenvironment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    for i in range(env.num_bodies):
        env.gym.set_rigid_body_color(env.envs[0], env.actor_handles[0], i, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.3))
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    cfg_eval = {
        "timesteps": (env_cfg.env.episode_length_s) * 500 + 1,
        'cameraTrack': True, 
        'trackIndex': 0,
        "cameraInit": np.pi*8/10,  
        "cameraVel": 1*np.pi/10,
    }
    camera_rot = np.pi * 8 / 10
    camera_rot_per_sec = 1 * np.pi / 10
    camera_relative_position = np.array([1, 0, 0.8])
    track_index = 0
    camera_handle = None
    video_writer = None
    state_renderer = None

    if args.record_video:
        video_path = _make_video_path(args)
        video_writer = imageio.get_writer(video_path, fps=args.video_fps)
        state_renderer = _make_state_renderer(args.video_width, args.video_height)
        print(f"Recording video to: {video_path}")

    look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
    if env.viewer is not None:
        env.set_camera(look_at + camera_relative_position, look_at, track_index)
    _set_camera(env.gym, env.envs[0], camera_handle, look_at + camera_relative_position, look_at)
    
    _, _ = env.reset()
    obs, critic_obs, _, _, _ = env.step(torch.zeros(
            env.num_envs, env.num_actions, dtype=torch.float, device=env.device))

    timesteps = args.num_steps if args.num_steps is not None else env_cfg.env.episode_length_s * 500 + 1
    try:
        for timestep in tqdm.tqdm(range(timesteps)):
            with torch.inference_mode():
                actions, _ = policy.act_inference(obs, privileged_obs=critic_obs)

                obs, critic_obs, _, _, _ = env.step(actions)
                look_at = np.array(env.root_states[track_index, :3].cpu(), dtype=np.float64)
                camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
                h_scale = 1
                v_scale = 0.8
                camera_relative_position = 2 * \
                    np.array([np.cos(camera_rot) * h_scale,
                             np.sin(camera_rot) * h_scale, 0.5 * v_scale])
                if env.viewer is not None:
                    env.set_camera(look_at + camera_relative_position, look_at, track_index)
                _set_camera(env.gym, env.envs[0], camera_handle, look_at + camera_relative_position, look_at)

                if video_writer is not None and timestep % args.record_interval == 0:
                    _record_state_frame(env, state_renderer, video_writer)

                env.commands[:, 0] = 2.0
                env.commands[:, 1] = 0
                env.commands[:, 2] = 0
                env.commands[:, 3] = 2.0
                env.commands[:, 4] = 0.5
                env.commands[:, 5] = 0.5
                env.commands[:, 6] = 0.2
                env.commands[:, 7] = -0.0
                env.commands[:, 8] = 0.0
                env.commands[:, 9] = 0.0
                env.use_disturb = True
                env.disturb_masks[:] = True
                env.disturb_isnoise[:]= True
                env.disturb_rad_curriculum[:] = 1.0
                env.interrupt_mask[:] = env.disturb_masks[:]
                env.standing_envs_mask[:] = True
                env.commands[env.standing_envs_mask, :3] = 0
    finally:
        if video_writer is not None:
            video_writer.close()
        if state_renderer is not None:
            plt.close(state_renderer[0])

if __name__ == '__main__':
    args = get_args()
    play(args)
