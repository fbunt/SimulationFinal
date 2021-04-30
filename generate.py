import argparse
import multiprocessing as mp
import numpy as np
import os
import pickle
import queue
import random
import tqdm
from collections import namedtuple
from itertools import repeat
from scipy.integrate import solve_ivp


def forward(t, y):
    """
    y = [x1, y1, x2, y2, x3, y3, dx1dt, dy1dt, dx2dt, dy2dt, dx3dt, dy3dt]
    """
    r1 = y[:2]
    r2 = y[2:4]
    r3 = y[4:6]
    v1 = y[6:8]
    v2 = y[8:10]
    v3 = y[10:]
    r12 = np.linalg.norm(r1 - r2)
    r13 = np.linalg.norm(r1 - r3)
    r23 = np.linalg.norm(r2 - r3)
    dv1dt = -((r1 - r2) / r12 ** 3) - ((r1 - r3) / r13 ** 3)
    dv2dt = -((r2 - r1) / r12 ** 3) - ((r2 - r3) / r23 ** 3)
    dv3dt = -((r3 - r1) / r13 ** 3) - ((r3 - r2) / r23 ** 3)
    return np.array([*v1, *v2, *v3, *dv1dt, *dv2dt, *dv3dt])


def get_random_static_pos():
    x1 = np.array([1.0, 0.0])
    x2 = np.ones(2)
    # Avoid x2[1] being 0 so that all 3 points don't sit on the x-axis. This
    # also prevents the (-0.5, 0.0) bad case where x2 == x3.
    while np.linalg.norm(x2) > 1.0 or x2[1] == 0.0:
        x2[0] = random.uniform(-0.5, 0.0)
        x2[1] = random.uniform(0.0, 1.0)
    x3 = -x1 - x2
    return x1, x2, x3


Solution = namedtuple("Solution", ("y0", "t", "y"))


def save_solution(out_dir, id_, solution):
    fname = f"solution_{id_:06d}.pkl"
    fname = os.path.join(out_dir, fname)
    with open(fname, "wb") as fd:
        pickle.dump(solution, fd, protocol=pickle.HIGHEST_PROTOCOL)


NT = 1000
TF = 4
TOL = 1e-12


def three_body(method="BDF"):
    x1, x2, x3 = get_random_static_pos()
    v0 = np.zeros(2 * 3)
    y0 = np.array([*x1, *x2, *x3, *v0])
    tspan = (0, TF)
    res = solve_ivp(forward, tspan, y0, method=method, atol=TOL, rtol=TOL)
    if res.t.size > NT:
        step = int(np.round(res.t.size / NT))
    else:
        step = 1
    solution = Solution(y0, res.t[::step].copy(), res.y.T[::step].copy())
    return solution


def three_body_parallel(id_q, out_dir, method):
    solution = three_body(method)
    if solution.t[-1] >= TF * 0.99:
        try:
            id_ = id_q.get_nowait()
            save_solution(out_dir, id_, solution)
            return True
        except (ValueError, OSError, queue.Empty):
            return False
    else:
        return False


def generate_solutions(n_max, out_dir, method):
    if not os.path.isdir(out_dir):
        raise IOError(f"Output location does not exist: {out_dir}")
    for i in tqdm.tqdm(range(n_max), ncols=80):
        three_body(out_dir, i, method=method)


def worker(args):
    return three_body_parallel(worker.id_q, *args)


def worker_init(q):
    # Make q accessible to the worker func
    worker.id_q = q


def generate_solutions_parallel(num_solutions, out_dir, method, cores=16):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    id_q = mp.Queue(num_solutions)
    for i in range(num_solutions):
        id_q.put_nowait(i)
    with mp.Pool(cores, worker_init, [id_q]) as pool, tqdm.tqdm(
        ncols=80, total=num_solutions
    ) as pbar:
        count = 0
        for result in pool.imap_unordered(
            worker,
            zip(repeat(out_dir), repeat(method)),
            chunksize=10,
        ):
            if result:
                count += 1
                pbar.update(1)
                if count >= num_solutions:
                    break
        id_q.close()


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-n",
        "--num_solutions",
        default=20_000,
        type=int,
        help="Number of solutions to produce",
    )
    p.add_argument(
        "-m", "--method", type=str, default="BDF", help="Solver method"
    )
    p.add_argument(
        "-c",
        "--cores",
        type=int,
        default=os.cpu_count(),
        help="Number of cores to use",
    )
    p.add_argument("out_dir", type=str, help="Output directory for data")
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    generate_solutions_parallel(**vars(args))
