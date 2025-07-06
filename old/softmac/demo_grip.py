import time
from argparse import ArgumentParser

import taichi as ti
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from softmac.engine.taichi_env import TaichiEnv
from softmac.utils import make_gif_from_numpy, render, prepare

np.set_printoptions(precision=4)

class Controller:
    def __init__(
        self, steps=200, substeps=4000, actions_init=None,
        lr=1e-2, warmup=5, decay=1.0, betas=(0.9, 0.999),
    ):
        # actions
        self.steps = steps
        self.substeps = substeps
        if actions_init is None:
            self.actions = torch.zeros(steps, 2, requires_grad=True)
        else:
            if actions_init.shape[0] > steps:
                assert actions_init.shape[0] == substeps
                actions_init = actions_init.reshape(steps, -1, 2).mean(axis=1)
            self.actions = actions_init.clone().detach().requires_grad_(True)

        # optimizer
        self.optimizer = optim.Adam([self.actions, ], betas=betas)

        self.lr = lr
        self.decay = decay
        self.warmup = warmup

        # log
        self.epoch = 0

    def get_actions(self):
        return torch.tensor(self.actions.detach().numpy().repeat(self.substeps // self.steps, axis=0))

    def schedule_lr(self):
        if self.epoch < self.warmup:
            lr = self.lr * (self.epoch + 1) / self.warmup
        else:
            lr = self.lr * self.decay ** (self.epoch - self.warmup)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        self.latest_lr = lr

    def step(self, grad):
        self.schedule_lr()
        if grad.shape[0] > self.steps:
            grad = grad.reshape(self.steps, -1, 2).mean(axis=1)
        actions_grad = grad

        self.actions.backward(actions_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.epoch += 1

def gen_init_state(args, env, log_dir, actions):
    env.reset()
    env.set_copy(False)
    for step in range(args.steps):
        action = actions[step]
        env.step(action)
    images = render(env, n_steps=args.steps, interval=args.steps // 50)
    make_gif_from_numpy(images, log_dir)

    state = env.simulator.get_state(env.simulator.cur)
    print(state.shape)
    # np.save("envs/grip/grip_mpm_init_state.npy", state)
    # np.save("envs/grip/grip_mpm_target_position.npy", state[:, :3])

def get_init_actions(args, env, choice=0):
    if choice == 0:
        actions = torch.zeros(args.steps, 2)
    elif choice == 1:
        actions = torch.ones(args.steps, 2) * torch.tensor([1.0, -1.0]) * 1.2
    elif choice == 2:
        actions = torch.ones(args.steps, 2) * torch.tensor([1.0, -1.0]) * 0.3
    else:
        assert False
    return torch.FloatTensor(actions)

def plot_loss_curve(log_dir, loss_log):
    fig, ax = plt.subplots(figsize=(4, 3))
    fontsize = 14
    plt.plot(loss_log, color="#c11221")
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14])
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(log_dir / "loss_curve.png", dpi=500)
    plt.close()

    losses = np.array(loss_log)
    np.save(log_dir / "losses.npy", losses)

def main(args):
    # Path and Configurations
    log_dir, cfg = prepare(args)
    ckpt_dir = log_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    # Build Environment
    env = TaichiEnv(cfg)
    env.simulator.primitives_contact = [False, True, True]
    # for i in range(10):
    #     # Adamas setExtForce has bug. Result of the first epoch differs from later epochs.
    #     env.step()
    # env.reset()

    # # Prepare Controller
    # actions = get_init_actions(args, env, choice=2)
    # controller = Controller(
    #     steps=args.steps // 10, substeps=args.steps, actions_init=actions,
    #     lr=1e-1, warmup=5, decay=0.99, betas=(0.5, 0.999)
    # )

    # loss_log = []
    # print("Optimizing Trajectory...")
    # for epoch in range(args.epochs):
    #     # preparation
    #     tik = time.time()
    #     ti.ad.clear_all_gradients()
    #     env.reset()
    #     prepare_time = time.time() - tik

    #     # forward
    #     tik = time.time()
    #     actions = controller.get_actions()
    #     for i in range(args.steps):
    #         env.step(actions[i])
    #     forward_time = time.time() - tik

    #     # loss
    #     tik = time.time()
    #     loss, pose_loss, vel_loss, chamfer_loss = 0., 0., 0., 0.
    #     with ti.ad.Tape(loss=env.loss.loss):
    #         for f in range(1500, env.simulator.cur + 1, 20):
    #             loss_info = env.compute_loss(f)
    #             loss = loss_info["loss"]
    #             pose_loss += loss_info["pose_loss"]
    #             vel_loss += loss_info["vel_loss"]
    #             chamfer_loss += loss_info["chamfer_loss"]
    #     loss_time = time.time() - tik

    #     # backward
    #     tik = time.time()
    #     actions_grad = env.backward()
    #     backward_time = time.time() - tik

    #     # optimize
    #     tik = time.time()
    #     controller.step(actions_grad)
    #     optimize_time = time.time() - tik

    #     total_time = prepare_time + forward_time + loss_time + backward_time + optimize_time
    #     print("+============== Epoch {} ==============+ lr: {:.4f}".format(epoch, controller.latest_lr))
    #     print("Time: total {:.2f}, pre {:.2f}, forward {:.2f}, loss {:.2f}, backward {:.2f}, optimize {:.2f}".format(total_time, prepare_time, forward_time, loss_time, backward_time, optimize_time))
    #     print("Loss: {:.4f} pose: {:.4f} vel: {:.4f} chamfer: {:.4f}".format(loss, pose_loss, vel_loss, chamfer_loss))
    #     print("Final pose: {:.4f} vel: {:.4f} chamfer: {:.4f}".format(loss_info["pose_loss"], loss_info["vel_loss"], loss_info["chamfer_loss"]))
    #     rigid_state = env.rigid_simulator.states[-1]
    #     print("Rigid x: {} v: {}".format(rigid_state[0:2].detach().numpy(), rigid_state[2:4].detach().numpy()))

    #     loss_log.append(env.loss.loss.to_numpy())
        
        # if (epoch + 1) % args.render_interval == 0 or epoch == 0:
    images = render(env, n_steps=args.steps, interval=args.steps // 50)
            # make_gif_from_numpy(images, log_dir, f"epoch{epoch}")

    # plot_loss_curve(log_dir, loss_log)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp-name", "-n", type=str, default="grip")
    parser.add_argument("--config", type=str, default="softmac/config/demo_grip_config.py")
    parser.add_argument("--render-interval", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    main(args)
