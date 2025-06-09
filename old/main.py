import argparse
from simulation import MPMSimulation

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dim", choices=[2,3], type=int, default=2)
    p.add_argument("--shape", choices=["box","sphere","half_circle"], default="box")
    p.add_argument("--links", nargs="+", type=float, default=[0.2,0.2])
    p.add_argument("--renderer", choices=["gui","3d"], default="gui")
    args = p.parse_args()

    sim = MPMSimulation(
        dim=args.dim,
        shape=args.shape,
        renderer=args.renderer,
        link_lengths=args.links
    )
    sim.run()
