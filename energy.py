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
        np.linalg.norm((sol.y[:, 6], sol.y[:, 7]), axis=0),
    )
    bdy2 = Body(
        sol.y[:, 2],
        sol.y[:, 3],
        np.array((sol.y[:, 2], sol.y[:, 3])).T.copy(),
        sol.y[:, 8],
        sol.y[:, 9],
        np.linalg.norm((sol.y[:, 8], sol.y[:, 9]), axis=0),
    )
    bdy3 = Body(
        sol.y[:, 4],
        sol.y[:, 5],
        np.array((sol.y[:, 4], sol.y[:, 5])).T.copy(),
        sol.y[:, 10],
        sol.y[:, 11],
        np.linalg.norm((sol.y[:, 10], sol.y[:, 11]), axis=0),
    )
    return bdy1, bdy2, bdy3


def energy(b1, b2, b3):
    ke = 0.5 * (b1.v ** 2 + b2.v ** 2 + b3.v ** 2)
    pe = (
        (-1 / np.linalg.norm(b1.r - b2.r, axis=1))
        - (1 / np.linalg.norm(b2.r - b3.r, axis=1))
        - (1 / np.linalg.norm(b1.r - b3.r, axis=1))
    )
    tot = ke + pe
    return ke, pe, tot


def plot(t, ke, pe, tot, ax, xlabels):
    ax.plot(t, ke, "tab:green", label="KE")
    ax.plot(t, pe, "tab:blue", label="PE")
    ax.plot(t, tot, "tab:orange", label="Total")
    if xlabels:
        ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.legend(loc=0)
    ax.grid()


def load_data(fname):
    with open(fname, "rb") as fd:
        return pickle.load(fd)


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="Path(s) to solution file(s)")
    return p


def main(paths):
    sols = [load_data(path) for path in paths]
    energies = [energy(*solution_to_bodies(sol)) for sol in sols]
    fig = plt.figure(figsize=(10, 10))
    if len(paths) == 1:
        ax = fig.add_subplot(111)
        ax.set_title(paths[0])
        plot(sols[0].t, *energies[0], ax, True)
    else:
        ax1, ax2 = fig.subplots(2, sharex=True, sharey=True)
        ax1.set_title(paths[0])
        plot(sols[0].t, *energies[0], ax1, False)
        ax2.set_title(paths[1])
        plot(sols[1].t, *energies[1], ax2, True)
    plt.show()


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(**vars(args))
