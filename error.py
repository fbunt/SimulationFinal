import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import namedtuple

from generate import Solution


def solution_to_bodies(sol):
    Body = namedtuple("Body", ("x", "y", "r", "vx", "vy", "v"))
    bdy1 = Body(
        sol.y[:, 0],
        sol.y[:, 1],
        np.array((sol.y[:, 0], sol.y[:, 1])).T.copy(),
        sol.y[:, 6],
        sol.y[:, 7],
        np.array((sol.y[:, 6], sol.y[:, 7])).T.copy(),
    )
    bdy2 = Body(
        sol.y[:, 2],
        sol.y[:, 3],
        np.array((sol.y[:, 2], sol.y[:, 3])).T.copy(),
        sol.y[:, 8],
        sol.y[:, 9],
        np.array((sol.y[:, 8], sol.y[:, 9])).T.copy(),
    )
    bdy3 = Body(
        sol.y[:, 4],
        sol.y[:, 5],
        np.array((sol.y[:, 4], sol.y[:, 5])).T.copy(),
        sol.y[:, 10],
        sol.y[:, 11],
        np.array((sol.y[:, 10], sol.y[:, 11])).T.copy(),
    )
    return bdy1, bdy2, bdy3


def get_error(x1, x2):
    return np.sum(np.abs(x1 - x2), axis=-1)


def load_data(fname):
    with open(fname, "rb") as fd:
        return pickle.load(fd)


def plot_full(t, error):
    plt.figure(figsize=(10, 10))
    plt.plot(t, error)
    plt.xlabel("Time")
    plt.ylabel("Absolute Error")
    plt.grid()
    plt.title(f"Overall MAE: {error.mean():2.1e}")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-s", "--save_path", default=None, type=str, help="Save figure to path"
    )
    p.add_argument("subset", type=str, help="position, velocity, full")
    p.add_argument("paths", nargs="+", help="Path(s) to solution file(s)")
    return p


def main(subset, paths, save_path):
    vsol, msol = [load_data(path) for path in paths]
    if subset == "position":
        errors = [
            get_error(vb.r, mb.r)
            for vb, mb in zip(
                solution_to_bodies(vsol), solution_to_bodies(msol)
            )
        ]
    elif subset == "velocity":
        errors = [
            get_error(vb.v, mb.v)
            for vb, mb in zip(
                solution_to_bodies(vsol), solution_to_bodies(msol)
            )
        ]
    else:
        plot_full(vsol.t, get_error(vsol.y, msol.y))
        if save_path is not None:
            plt.savefig(save_path, dpi=200)
        plt.show()
        return

    colors = ["#d62728", "#bcbd22", "#17becf"]
    fig = plt.figure(figsize=(10, 10))
    ax1, ax2, ax3 = fig.subplots(3, sharex=True)
    ax1.plot(
        vsol.t,
        errors[0],
        color=colors[0],
        label=f"Mean: {errors[0].mean():2.1e}",
    )
    ax1.set_ylabel("$x_1$ Absolute Error")
    ax1.grid()
    ax1.legend(loc=2)
    ax2.plot(
        vsol.t, errors[1], color=colors[1], label=f"{errors[1].mean():2.1e}"
    )
    ax2.set_ylabel("$x_2$ Absolute Error")
    ax2.grid()
    ax2.legend(loc=2)
    ax3.plot(
        vsol.t, errors[2], color=colors[2], label=f"{errors[2].mean():2.1e}"
    )
    ax3.set_ylabel("$x_3$ Absolute Error")
    ax3.grid()
    ax3.legend(loc=2)

    ax1.set_title(f"{subset.upper()} Overall MAE: {np.mean(errors):2.1e}")
    ax3.set_xlabel("Time")

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(**vars(args))
