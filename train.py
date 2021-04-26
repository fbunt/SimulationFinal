import argparse
import glob
import numpy as np
import os
import pickle
import torch
import tqdm
from torch.nn.functional import mse_loss
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


loss_func = mse_loss


def run_model(model, data_iterator, optimizer, gscaler, is_train):
    loss_sum = 0.0
    for (input_data, labels) in data_iterator:
        input_data = input_data.cuda()
        labels = labels.cuda()
        # Initial conditions (t = 0)
        initial = input_data[..., 0] == 0
        non_initial = ~initial
        if is_train:
            model.zero_grad()
        output = model(input_data)
        xout = output[..., :4]
        xlabels = labels[..., :4]
        vout = output[..., 4:]
        vlabels = labels[..., 4:]
        # Position loss
        loss = loss_func(xout[non_initial], xlabels[non_initial])
        # Velocity loss
        loss += loss_func(vout[non_initial], vlabels[non_initial]) * 1e-1
        # Initial condition loss
        if initial.any():
            loss += loss_func(output[initial], labels[initial]) * 1e3
        if is_train:
            # gscaler.scale(loss).backward()
            # gscaler.step(optimizer)
            # gscaler.update()
            loss.backward()
            optimizer.step()
        loss_sum += loss.item()
    loss_mean = loss_sum / len(data_iterator)
    return loss_mean


def train(model, dataloader, optimizer, gscaler, logger, epoch, total_epochs):
    model.train()
    it = tqdm.tqdm(
        dataloader,
        ncols=80,
        total=len(dataloader),
        desc=f"Train: {epoch + 1}/{total_epochs}",
    )
    loss = run_model(model, it, optimizer, gscaler, True)
    logger.add_scalar("Loss", loss, epoch)
    return loss


def test(model, dataloader, optimizer, gscaler, logger, epoch, total_epochs):
    model.eval()
    it = tqdm.tqdm(
        dataloader,
        ncols=80,
        total=len(dataloader),
        desc=f"-Test: {epoch + 1}/{total_epochs}",
    )
    with torch.no_grad():
        loss = run_model(model, it, optimizer, gscaler, False)
        logger.add_scalar("Loss", loss, epoch)
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


def dataset_to_array(ds, dtype=float, progress=True):
    n = len(ds)
    shape = (n, *ds[0].shape)
    ar = np.zeros(shape, dtype=dtype)
    if progress:
        it = tqdm.tqdm(ds, ncols=80, desc="DS to array")
    else:
        it = ds
    for i, x in enumerate(it):
        ar[i] = x
    return ar


def build_dataset(files):
    files = sorted(files)
    solutions = []
    for f in files:
        with open(f, "rb") as fd:
            solutions.append(pickle.load(fd))
    input_ds = ConcatDataset([SolutionInputDataset(s) for s in solutions])
    input_ds = torch.tensor(dataset_to_array(input_ds)).float()
    label_ds = ConcatDataset([SolutionLabelDataset(s) for s in solutions])
    label_ds = torch.tensor(dataset_to_array(label_ds)).float()
    return ComposedDataset([input_ds, label_ds])


def build_train_test_datasets(files, train_size=0.9):
    train_files, test_files = train_test_split(files, train_size)
    train_ds = build_dataset(train_files)
    test_ds = build_dataset(test_files)
    return train_ds, test_ds


FNAME_MODEL = "model.pt"
FNAME_MODEL_TMP = "model.pt.tmp"
FNAME_FULL_SNAPSHOT = "snap_full.pt"
FNAME_FULL_SNAPSHOT_TMP = "snap_full.pt.tmp"
SNAP_KEY_EPOCH = "epoch"
SNAP_KEY_MODEL = "model"
SNAP_KEY_OPTIMIZER = "optimizer"
SNAP_KEY_LR_SCHED = "lr_sched"
SNAP_KEY_CHECK_VAL = "check_val"


class SnapshotHandler:
    def __init__(self, root_dir, model, optimizer, lr_sched):
        self.root_path = os.path.abspath(root_dir)
        self.model = model
        self.opt = optimizer
        self.lr_sched = lr_sched
        self.model_path = os.path.join(self.root_path, FNAME_MODEL)
        self.model_path_tmp = os.path.join(self.root_path, FNAME_MODEL_TMP)
        self.full_snap_path = os.path.join(self.root_path, FNAME_FULL_SNAPSHOT)
        self.full_snap_path_tmp = os.path.join(
            self.root_path, FNAME_FULL_SNAPSHOT_TMP
        )
        self.counter = 0

    def save_model(self):
        # Make sure that there is always a possible recovery mode in case of
        # early termination during file write.
        torch.save(self.model.state_dict(), self.model_path_tmp)
        os.replace(self.model_path_tmp, self.model_path)

    def can_resume(self):
        return os.path.isfile(self.full_snap_path)

    def take_model_snapshot(self):
        print("\nTaking snapshot")
        self.save_model()
        self.counter += 1
        return True

    def take_full_snapshot(self, epoch, check_val):
        snap = {
            SNAP_KEY_EPOCH: epoch,
            SNAP_KEY_MODEL: self.model.state_dict(),
            SNAP_KEY_OPTIMIZER: self.opt.state_dict(),
            SNAP_KEY_LR_SCHED: self.lr_sched.state_dict(),
            SNAP_KEY_CHECK_VAL: check_val,
        }
        # Make sure that there is always a possible recovery mode in case of
        # early termination during file write.
        torch.save(snap, self.full_snap_path_tmp)
        os.replace(self.full_snap_path_tmp, self.full_snap_path)

    def load_best_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        return self.model

    def load_full_snapshot(self):
        print("Loading full snapshot")
        snap = torch.load(self.full_snap_path)
        epoch = snap[SNAP_KEY_EPOCH]
        self.model.load_state_dict(snap[SNAP_KEY_MODEL])
        self.opt.load_state_dict(snap[SNAP_KEY_OPTIMIZER])
        self.lr_sched.load_state_dict(snap[SNAP_KEY_LR_SCHED])
        check_val = snap[SNAP_KEY_CHECK_VAL]
        return epoch, self.model, self.opt, self.lr_sched, check_val


def _validate_dir_path(path):
    if os.path.isdir(path):
        return path
    raise IOError(f"Directory path does not exist: '{path}'")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-b", "--batch_size", type=int, default=5_000, help="Batch size"
    )
    p.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1_000,
        help="Number of training epochs",
    )
    p.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for training",
    )
    p.add_argument(
        "-R",
        "--resume",
        action="store_true",
        help="causes training to resume from the last saved checkpoint",
    )
    p.add_argument(
        "data_dir",
        type=_validate_dir_path,
        help="Directory containing training data",
    )
    return p


def main(data_dir, batch_size, epochs, learning_rate, resume=False):
    files = glob.glob(os.path.join(data_dir, "*.pkl"))
    train_ds, test_ds = build_train_test_datasets(files, 0.9)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 4, shuffle=False, drop_last=False
    )
    model = ThreeBodyMLP().cuda()
    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-3
    )
    frac_mstones = np.array([0.3, 0.5, 0.7, 0.8, 0.85, 0.95, 0.98])
    mstones = np.round(frac_mstones * epochs).astype(int)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, mstones, 0.5)
    train_logger = SummaryWriter("./logs/training")
    test_logger = SummaryWriter("./logs/test")

    snap_handler = SnapshotHandler(".", model, opt, sched)
    resume = resume and snap_handler.can_resume()
    min_loss = np.inf
    last_epoch = 0
    if resume:
        (
            last_epoch,
            model,
            opt,
            sched,
            min_loss,
        ) = snap_handler.load_full_snapshot()

    grad_scaler = torch.cuda.amp.GradScaler()
    for epoch in range(last_epoch, epochs):
        train_logger.add_scalar(
            "learning_rate", next(iter(opt.param_groups))["lr"], epoch
        )
        train(
            model, train_loader, opt, grad_scaler, train_logger, epoch, epochs
        )
        test_loss = test(
            model, test_loader, opt, grad_scaler, test_logger, epoch, epochs
        )
        sched.step()
        if min_loss > test_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), "./model.pt")
            snap_handler.take_model_snapshot()
        snap_handler.take_full_snapshot(epoch + 1, min_loss)
    train_logger.close()
    test_logger.close()


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(**vars(args))
