import numpy as np
import pylab as pl
import matplotlib
import torch

GRID_DENS = 15
LEVELS = 5
LABELPAD_DIFF = -20
optNames = ["GD", "NGD", "EF"]
optColours = ['r', 'blue', 'green']
label_for = [
    r"Dataset",
    r"GD",
    r"NGD",
    r"EF"
]
DD = 3
theta_lims = [2 - DD, 2 + DD]
thetas = list([np.linspace(theta_lims[0], theta_lims[1], GRID_DENS) for _ in range(2)])


def make_fig_env():
    fig = pl.figure(figsize=(12, 3.2))
    gs = matplotlib.gridspec.GridSpec(1, 4)
    gs.update(left=0.05, right=0.99, wspace=0.1, hspace=0.2, bottom=0.1, top=0.90)
    axes = [fig.add_subplot(gs[i, j]) for i in range(1) for j in range(4)]
    return fig, axes



def plot_dataset(ax, problem, X, y):

    ax.plot(X[:, 1], y, '.', markersize=3, alpha=0.4, color='r', label='input data')

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 20])

    xs = np.linspace(0, 9, 100)
    theta_0 = problem.theta_0.numpy()
    ax.plot(xs, xs * theta_0[1] + theta_0[0], '--', label=r"$y = \theta x + b$", linewidth=1, color="k")

    ax.set_xlim([-1, 10])
    ax.set_ylim([-2, 22])

    ax.legend(prop={'size': 6}, borderpad=0.3)
	

    return

def plot_loss_contour(ax, problem):
    def compute_losses(lossFunc):
        losses = np.zeros((len(thetas[0]), len(thetas[1])))
        for i, t1 in enumerate(thetas[0]):
            for j, t2 in enumerate(thetas[1]):
                theta = torch.from_numpy(np.array([t1, t2]).reshape(-1, 1)).float()
                losses[i, j] = lossFunc(theta)
        return losses

    losses = compute_losses(lambda t: problem._loss(t))
    ax.contour(thetas[0], thetas[1], losses.T, LEVELS, colors=["k"], alpha=0.3)


def plot_vecFields(axes, problem, vecFuncs, with_loss_contour=True):

    def vectorField(vecFunc):
        vector_field = [np.zeros((GRID_DENS, GRID_DENS)), np.zeros((GRID_DENS, GRID_DENS))]
        for i, t1 in enumerate(thetas[0]):
            for j, t2 in enumerate(thetas[1]):
                theta = torch.from_numpy(np.array([t1, t2]).reshape(-1, 1)).float()
                v = vecFunc(theta)
                for d in range(2):
                    vector_field[d][i, j] = v[d]
        return vector_field

    def plot_vecField(ax, vecField, scale=30):
        U = vecField[0].copy().T
        V = vecField[1].copy().T
        cf = ax.quiver(thetas[0], thetas[1], U, V, angles='xy', scale=scale, color='darkgray', width=0.005, headwidth=4, headlength=3)
        return cf

    vecFields = list([vectorField(vecFuncs[i]) for i in range(len(axes))])
    for ax, vecField in zip(axes, vecFields):
        if with_loss_contour:
            plot_loss_contour(ax, problem)
        plot_vecField(ax, vecField)


def plot_gradientDescent(ax, xs, optName, optColour):
    ax.plot(xs[:, 0], xs[:, 1], linestyle='-', color=optColour, linewidth=4, alpha=0.9)
    ax.plot(xs[0, 0], xs[0, 1], 'h', color="k", markersize=8)
    ax.plot(xs[-1, 0], xs[-1, 1], '*', color="k", markersize=12)


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

def plot(X, y, problem, vec_funcs, starts, chains):

    fig, axes = make_fig_env()

    plot_dataset(axes[0], problem, X, y)
    plot_vecFields(axes[1:], problem, vec_funcs, with_loss_contour=True)

    for i, startingpoint in enumerate(starts):
        for j, vecfunc in enumerate(vec_funcs):
            plot_gradientDescent(axes[1 + j], chains[i][j], optNames[j], optColours[j])

    axes[0].set_xlabel("x", labelpad=LABELPAD_DIFF)
    axes[0].set_ylabel("y", labelpad=LABELPAD_DIFF)
    axes[0].set_xticks([0, 10])
    axes[0].set_yticks([0, 20])

    for ax in axes[1:]:
        ax.set_xlim(theta_lims)
        ax.set_ylim(theta_lims)
        ax.set_xticks(theta_lims)
        ax.set_yticks(theta_lims)
        ax.set_xlabel(r"$b$", labelpad=0.5*LABELPAD_DIFF)
        ax.set_ylabel(r"$\theta$", labelpad=LABELPAD_DIFF)

    for i, ax in enumerate(axes):
        ax.set_title(label_for[i])

