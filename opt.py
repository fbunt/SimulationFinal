import argparse
import glob
import numpy as np
import os
import pickle
import tqdm
from collections import namedtuple
from scipy.optimize import minimize

from generate import Solution


def load_solutions(data_dir, model=False):
    if model:
        files = sorted(glob.glob(os.path.join(data_dir, "*-model.pkl")))
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "*[0-9].pkl")))
    solutions = []
    for f in files:
        with open(f, "rb") as fd:
            solutions.append(pickle.load(fd))
    return {f: s for f, s in zip(files, solutions)}


def get_output_name(out_dir, f):
    basename = os.path.basename(f)
    base = os.path.splitext(basename)[0]
    name = f"{base}-opt.pkl"
    return os.path.join(out_dir, name)


def save_solution(fname, sol):
    with open(fname, "wb") as fd:
        pickle.dump(sol, fd, protocol=pickle.HIGHEST_PROTOCOL)


def save_solutions(sol_dict):
    for f, s in sol_dict.items():
        save_solution(f, s)


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


def get_energy(y):
    r1 = y[:2]
    r2 = y[2:4]
    r3 = y[4:6]
    v1 = np.linalg.norm(y[6:8])
    v2 = np.linalg.norm(y[8:10])
    v3 = np.linalg.norm(y[10:])
    ke = 0.5 * (v1 ** 2 + v2 ** 2 + v3 ** 2)
    pe = (
        (-1 / np.linalg.norm(r1 - r2))
        - (1 / np.linalg.norm(r2 - r3))
        - (1 / np.linalg.norm(r1 - r3))
    )
    return ke + pe


def loss_func(y, y0, energy0, g1, g2):
    x = y[:6]
    x0 = y0[:6]
    v = y[6:]
    energy = get_energy(y)
    err = np.linalg.norm(energy - energy0) ** 2
    err += g1 * np.linalg.norm(x - x0) ** 2
    # v0 is zero so no need to take norm
    err += g2 * v ** 2
    return np.sum(err)


def optimize_solution(ms, s, g1=1e-10, g2=1e-10):
    energy0 = get_energy(s.y0)
    args = (s.y0, energy0, g1, g2)
    yopt = np.zeros_like(ms.y)
    for i, y in tqdm.tqdm(enumerate(ms.y), ncols=80, total=len(ms.y)):
        x0 = y
        res = minimize(
            loss_func,
            x0,
            args,
            method="Nelder-Mead",
            options={"adaptive": True, "xatol": 1e-7},
        )
        yopt[i] = res.x
    return yopt


def main(val_data_dir, model_data_dir, out_dir):
    val_solutions = load_solutions(val_data_dir, False)
    model_solutions = load_solutions(model_data_dir, True)
    for (vsf, vs), (msf, ms) in tqdm.tqdm(
        zip(val_solutions.items(), model_solutions.items()),
        ncols=80,
        total=len(val_solutions),
    ):
        yopt = optimize_solution(ms, vs)
        sol_opt = Solution(yopt[0], vs.t, yopt)
        fopt = get_output_name(out_dir, msf)
        save_solution(fopt, sol_opt)


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "val_data_dir",
        type=str,
        help="Directory with validation solutions to load",
    )
    p.add_argument(
        "model_data_dir",
        type=str,
        help="Directory with model solutions to load",
    )
    p.add_argument("out_dir", type=str, help="Output directory")
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(**vars(args))
