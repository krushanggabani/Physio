import time
import time
import os

from argparse import ArgumentParser

import taichi as ti
import numpy as np
import torch
import torch.optim as optim


from physio.engine import Env


os.system('cls' if os.name == 'nt' else 'clear')


def main(args):
    """
    Initialize the environment and measure how long each phase takes:
      1. reset()
      2. step()
      3. render()
    Finally, print a breakdown and total.
    """
    # 1) Instantiate environment with user args
    env = Env(args)

    # 2) Time the reset phase
    start = time.perf_counter()
    env.reset()
    reset_time = time.perf_counter() - start

    # 3) Time the forward (step) phase
    start = time.perf_counter()
    while env.gui.running:
        env.step()
        env.render()
    step_time = time.perf_counter() - start

    # 4) Time the render phase
    start = time.perf_counter()
    env.render()
    render_time = time.perf_counter() - start

    # 5) Aggregate total time
    total_time = reset_time + step_time + render_time

    # 6) Print results
    print(f"Experiment Title: {args.Exp_name}")
    print(f"Timing (s): Total={total_time:.3f}, Reset={reset_time:.3f}, Step={step_time:.3f}, Render={render_time:.3f}")



if __name__ == "__main__":

    parser = ArgumentParser(description="Run Physio engine benchmark and report timings.")
    parser.add_argument("--Exp_name", "-n", type=str, default="Robot Tissue Interaction", help="Name of the experiment (display only)")
    parser.add_argument("--Dimension", type=int,default=2)
    parser.add_argument("--Time_step", type=float, default=2e-4)
    
    args = parser.parse_args()
    main(args)


    