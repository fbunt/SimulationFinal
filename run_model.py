import argparse
import glob
import numpy as np
import os
import pickle
import torch

from generate import Solution
from model import ThreeBodyMLP
from train import SolutionInputDataset, dataset_to_array


def load_model(path):
    model = ThreeBodyMLP().cuda()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def load_solutions(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    solutions = []
    for f in files:
        with open(f, "rb") as fd:
            solutions.append(pickle.load(fd))
    return {f: s for f, s in zip(files, solutions)}


def save_solutions(sol_dict):
    for f, s in sol_dict.items():
        with open(f, "wb") as fd:
            pickle.dump(s, fd, protocol=pickle.HIGHEST_PROTOCOL)


def get_model_solution(model, ds):
    y0 = ds.y0
    ds = dataset_to_array(ds)
    y = np.zeros((len(ds), 12))
    t = ds[:, 0].copy()
    ds = torch.tensor(ds).float().cuda()
    with torch.no_grad():
        out = model(ds).cpu().numpy()
    x1x, x1y, x2x, x2y, *vs = out.T
    y[:, 0] = x1x
    y[:, 1] = x1y
    y[:, 2] = x2x
    y[:, 3] = x2y
    y[:, 4] = -x1x - x2x
    y[:, 5] = -x1y - x2x
    y[:, 6:] = np.array(vs).T
    return Solution(y0, t, y)


def get_model_out_name(out_dir, f):
    basename = os.path.basename(f)
    base = os.path.splitext(basename)[0]
    name = f"{base}-model.pkl"
    return os.path.join(out_dir, name)


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("model_path", type=str, help="Model location")
    p.add_argument(
        "data_dir", type=str, help="Directory with solutions to load"
    )
    p.add_argument("out_dir", type=str, help="Output directory")
    return p


def main(model_path, data_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.mak(out_dir, exist_ok=True)
    model = load_model(model_path)
    sols = load_solutions(data_dir)
    sols = {f: (s.y0, SolutionInputDataset(s)) for f, s in sols.items()}
    model_sols = {
        get_model_out_name(out_dir, f): get_model_solution(model, *s)
        for f, s in sols.items()
    }
    save_solutions(model_sols)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(**vars(args))
