import argparse
import glob
import numpy as np
import os
import pickle
import torch
import tqdm
from torch.nn.functional import l1_loss
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from generate import Solution
from model import ThreeBodyMLP


class SolutionInputDataset(Dataset):
    def __init__(self, solution):
        self.n = len(solution.y)
        self.t = solution.t
        self.y0 = solution.y0
        # Due to symmetry, only the position of x2 needs to be given. x1's
        # initial position is always (1, 0) and x2 = -x1 - x2.
        self.values = np.zeros(2 + 1)
        self.values[1:] = self.y0[2:4]

    def __getitem__(self, idx):
        self.values[0] = self.t[idx]
        return self.values.copy()

    def __len__(self):
        return len(self.t)


class SolutionLabelDataset(Dataset):
    def __init__(self, solution):
        self.y = solution.y
        mask = np.ones(self.y.shape[1], dtype=bool)
        # Drop body 3 position
        mask[4:6] = False
        self.yp = solution.y[:, mask]

    def __getitem__(self, idx):
        return self.yp[idx]

    def __len__(self):
        return len(self.y)


class ComposedDataset(Dataset):
    """Passes a query index on to a set of datasets and composses the results
    into a list.

    Extremely handy for building more complex data loading.
    """

    def __init__(self, datasets):
        assert len(datasets) > 0, "Must provide datasets"
        assert (
            len(set(len(d) for d in datasets)) == 1
        ), "Dataset sizes must match"
        self.datasets = datasets
        self.size = len(self.datasets[0])

    def __getitem__(self, idx):
        return [d[idx] for d in self.datasets]

    def __len__(self):
        return self.size


def run_model(model, data_iterator, optimizer, logger, epoch, is_train):
    loss_sum = 0.0
    for (input_data, labels) in data_iterator:
        input_data = input_data.cuda()
        labels = labels.cuda()
        if is_train:
            model.zero_grad()
        output = model(input_data)
        loss = l1_loss(output, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item()
    loss_mean = loss_sum / len(data_iterator)
    logger.add_scalar("Loss", loss_mean, epoch)


def train(model, dataloader, optimizer, logger, epoch, total_epochs):
    model.train()
    it = tqdm.tqdm(
        dataloader,
        ncols=80,
        total=len(dataloader),
        desc=f"Train: {epoch + 1}/{total_epochs}",
    )
    run_model(model, it, optimizer, logger, epoch, False)


def test(model, dataloader, optimizer, logger, epoch, total_epochs):
    model.eval()
    it = tqdm.tqdm(
        dataloader,
        ncols=80,
        total=len(dataloader),
        desc=f"-Test: {epoch + 1}/{total_epochs}",
    )
    with torch.no_grad():
        loss = run_model(model, it, optimizer, logger, epoch, False)
    return loss


def train_test_split(x, train_size):
    assert (
        0.0 < train_size < 1.0
    ), "train_size must be less than 1 but greater than 0"
    n = len(x)
    np.random.shuffle(x)
    isplit = int(n * train_size)
    train = x[:isplit]
    test = x[isplit:]
    return train, test


def build_dataset(files):
    files = sorted(files)
    solutions = []
    for f in files:
        with open(f, "rb") as fd:
            solutions.append(pickle.load(fd))
    input_ds = ConcatDataset([SolutionInputDataset(s) for s in solutions])
    label_ds = ConcatDataset([SolutionLabelDataset(s) for s in solutions])
    return ComposedDataset([input_ds, label_ds])


def build_train_test_datasets(files, train_size=0.9):
    train_files, test_files = train_test_split(files, train_size)
    train_ds = build_dataset(train_files)
    test_ds = build_dataset(test_files)
    return train_ds, test_ds


def _validate_dir_path(path):
    if os.path.isdir(path):
        return path
    raise IOError(f"Directory path does not exist: '{path}'")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-b", "--batch_size", type=int, default=50_000, help="Batch size"
    )
    p.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10_000,
        help="Number of training epochs",
    )
    p.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for training",
    )
    p.add_argument(
        "data_dir",
        type=_validate_dir_path,
        help="Directory containing training data",
    )
    return p


def main(data_dir, batch_size, epochs, learning_rate):
    files = glob.glob(os.path.join(data_dir, "*.pkl"))
    train_ds, test_ds = build_train_test_datasets(files, 0.9)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    model = ThreeBodyMLP().cuda()
    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-2
    )
    frac_mstones = np.array([0.3, 0.5, 0.7, 0.8, 0.85, 0.95, 0.98])
    mstones = np.round(frac_mstones * epochs).astype(int)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, mstones, 0.5)
    logger = SummaryWriter("./logs")
    min_loss = np.inf
    for epoch in range(epochs):
        train(model, train_loader, opt, logger, epoch, epochs)
        test_loss = test(model, test_loader, opt, logger, epoch, epochs)
        if min_loss > test_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), "./model.pt")
        sched.step()
    logger.close()


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(**vars(args))
