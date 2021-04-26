import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import namedtuple
from matplotlib.animation import FuncAnimation

from generate import Solution


Body = namedtuple("Body", ("x", "y", "vx", "vy"))


def solution_to_bodies(sol):
    bdy1 = Body(sol.y[:, 0], sol.y[:, 1], sol.y[:, 6], sol.y[:, 7])
    bdy2 = Body(sol.y[:, 2], sol.y[:, 3], sol.y[:, 8], sol.y[:, 9])
    bdy3 = Body(sol.y[:, 4], sol.y[:, 5], sol.y[:, 10], sol.y[:, 11])
    return bdy1, bdy2, bdy3


class SingleSolutionPlotter:
    def __init__(self, solution, fig):
        self.sol = solution
        self.fig = fig
        self.bodies = solution_to_bodies(solution)
        self.bdy1, self.bdy2, self.bdy3 = self.bodies

        self.ax = self.fig.add_subplot(111)
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        plt.axis("equal")

    def init(self):
        self.artists = []
        self.traces = []
        self.tails = []
        self.positions = []
        self.body_colors = []
        # Plot body traces
        for b in self.bodies:
            (trace,) = self.ax.plot(b.x, b.y, color="gray", lw=0.5)
            self.traces.append(trace)
        # Plot trailing tail
        for b in self.bodies:
            (tail,) = self.ax.plot(b.x[:1], b.y[:1])
            color = tail.get_color()
            self.body_colors.append(color)
            self.artists.append(tail)
            self.tails.append(tail)
        # Plot bodies
        for b, c in zip(self.bodies, self.body_colors):
            (pos,) = self.ax.plot(b.x[:1], b.y[:1], "o", color=c)
            self.artists.append(pos)
            self.positions.append(pos)
        return self.artists

    def update(self, i):
        for b, tail in zip(self.bodies, self.tails):
            tail.set_data(
                b.x[max(0, i - 20) : i + 1], b.y[max(0, i - 20) : i + 1]
            )
        for b, pos in zip(self.bodies, self.positions):
            pos.set_data(b.x[i : i + 1], b.y[i : i + 1])
        return self.artists

    def __len__(self):
        return self.sol.t.size


class SolutionAnimation:
    def __init__(self, solutions, figsize=None, interval=10):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=figsize)
        if len(solutions) == 1:
            self.plotter = SingleSolutionPlotter(solutions[0], self.fig)
        else:
            pass

        self.interval = interval
        self.paused = False

    def init(self):
        return self.plotter.init()

    def update(self, i):
        return self.plotter.update(i)

    def on_click(self, event):
        """Toggle play/pause with space bar. Handy for non-jupyter runs."""
        if event.key != " ":
            return
        if self.paused:
            self.ani.event_source.start()
            self.paused = False
        else:
            self.ani.event_source.stop()
            self.paused = True

    def run(self):
        self.fig.canvas.mpl_connect("key_press_event", self.on_click)
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.plotter),
            init_func=self.init,
            interval=self.interval,
            repeat=False,
            blit=True,
        )
        plt.show()


def load_data(fname):
    with open(fname, "rb") as fd:
        return pickle.load(fd)


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to solution file")
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    solution = load_data(args.path)
    ani = SolutionAnimation([solution])
    ani.run()
