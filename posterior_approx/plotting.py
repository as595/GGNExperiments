import numpy as np
import pylab as pl
import matplotlib
import torch
from matplotlib.patches import Ellipse

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

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

    ax.plot(X, y, '.', markersize=3, alpha=0.4, color='r', label='input data')

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 20])

    xs = np.linspace(0, 9, 100)
    theta_0 = problem.theta_0.detach().numpy()
    ax.plot(xs, xs * theta_0[1] + theta_0[0], '--', label=r"$y = \theta x + b$", linewidth=1, color="k")

    ax.set_xlim([-1, 10])
    ax.set_ylim([-2, 22])

    ax.legend(prop={'size': 6}, borderpad=0.3)
	
    return

def plot_sdata(ax, problem, X, y, type='full'):

    if type=='full':
        title = 'Full distribution'
        samples = problem.fdist
    elif type=='kernel':
        title = r'Kernel contribution ($\pm${} $\sigma$)'.format(problem.nsigma)
        samples = problem.kernel
    elif type=='nonkernel':
        title = r'Non-kernel contribution ($\pm${} $\sigma$)'.format(problem.nsigma)
        samples = problem.nkernel
    
    ax.plot(X.detach().numpy(), y.detach().numpy(), '.', markersize=3, alpha=0.4, color='r', label='input data')

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 20])

    xs = np.linspace(0, 9, 100)
    
    theta_0 = problem.theta_0.numpy()
    ax.plot(xs, xs * theta_0[1] + theta_0[0], '--', label=r"$y = \theta x + b$", linewidth=1, color="k")

    for i in range(len(samples)):
        theta = samples[i]
        vector_to_parameters(theta, problem.model.parameters())
        ys = problem.model(torch.from_numpy(xs.reshape(-1,1)).float()).detach().numpy()
        ax.plot(xs, ys, '--', linewidth=1, alpha = 0.2, color="gray")
        
    ax.set_xlim([-1, 10])
    ax.set_ylim([-2, 22])

    ax.legend(prop={'size': 6}, borderpad=0.3)
    ax.set_title(title)
    
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

def plot_samples(ax, problem):

    # sampling:
    precision = problem._hessian(problem.theta_0)
    Chol = torch.linalg.cholesky(precision)
    Sigma = torch.cholesky_inverse(Chol, upper=False)
    Sigma_chol = torch.linalg.cholesky(Sigma, upper=False)
    
    mc_samples = 1000
    covs = (Sigma_chol @ torch.randn(len(problem.theta_0), mc_samples)).t()
    mu = problem.theta_0.squeeze().repeat(mc_samples,1)
    samples = mu + covs

    idx = np.arange(0,mc_samples)
    ix_kernel = np.random.choice(idx, 100, replace=False)
    fdist = torch.zeros_like(samples[ix_kernel,:])
    fdist[:,0], fdist[:,1] = samples[ix_kernel,1], samples[ix_kernel,0] # model uses parameters in reverse order
    problem.fdist = fdist
    
    for sample_set in [samples]:
        ax.scatter(sample_set[:,0], sample_set[:,1], s=3, c='b', alpha=0.3)

    return

def plot_ellipse(ax, problem):

    # sampling:
    precision = problem._hessian(problem.theta_0)
    Chol = torch.linalg.cholesky(precision)
    Sigma = torch.cholesky_inverse(Chol, upper=False)

    # get kernel (major axis) / non-kernel (minor axis) directions 
    vals, vecs = torch.linalg.eigh(Sigma)
    vals, vecs = vals.detach().numpy(), vecs.detach().numpy()
    gradients = [v[1] / v[0] for v in vecs]
    mu = problem.theta_0.squeeze().detach().numpy()
    intercepts = [mu[1] - (gradient*mu[0]) for gradient in gradients]

    n_sigma = 1.
    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = n_sigma * problem.noise_var * np.sqrt(vals) # semi-axes
    
    # plot ellipse on axes:
    e = Ellipse(mu, 2.*w, 2.*h, angle=angle)
    e.set_fill(False)
    ax.add_artist(e)


    # separate kernel / non-kernel samples:
    n_pts = 1000
    _x = np.linspace(theta_lims[0], theta_lims[1], n_pts)
    _y, _d = np.zeros((n_pts,2)), np.zeros((n_pts,2))
    
    for i in range(0,2):
        _y[:,i] = intercepts[i] + _x*gradients[i]
        _d[:,i] = np.sqrt((_x-mu[0])**2 + (_y[:,i]-mu[1])**2)

    ix_1 = np.argwhere(_d[:,0]<w) # non-kernel
    ix_2 = np.argwhere(_d[:,1]<h) # kernel

    x1 = _x[ix_1]; y1 = _y[ix_1,0]
    x2 = _x[ix_2]; y2 = _y[ix_2,1]

    ax.plot(x1, y1, c='g', lw=2., label='non-kernel')
    ax.plot(x2, y2, c='r', lw=2., label='kernel');

    ax.legend(prop={'size': 6}, borderpad=0.3)

    idx = np.arange(0,x1.shape[0])
    ix_nkernel = np.random.choice(idx, 50, replace=False)
    nonkernel = torch.from_numpy(np.hstack((y1[ix_nkernel], x1[ix_nkernel]))).float()

    idx = np.arange(0,x2.shape[0])
    ix_kernel = np.random.choice(idx, 50, replace=False)
    kernel = torch.from_numpy(np.hstack((y2[ix_kernel], x2[ix_kernel]))).float()

    problem.kernel = kernel
    problem.nkernel = nonkernel
    problem.nsigma = n_sigma

    return 


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

def plot(X, y, problem, vec_funcs):

    fig, axes = make_fig_env()

    plot_dataset(axes[0], problem, X, y)
    plot_vecFields(axes[1:], problem, vec_funcs, with_loss_contour=True)

    axes[0].set_xlabel("x", labelpad=0.5*LABELPAD_DIFF)
    axes[0].set_ylabel("y", labelpad=0.5*LABELPAD_DIFF)
    axes[0].set_xticks([0, 10])
    axes[0].set_yticks([0, 20])

    for ax in axes[1:]:
        ax.set_xlim(theta_lims)
        ax.set_ylim(theta_lims)
        ax.set_xticks(theta_lims)
        ax.set_yticks(theta_lims)
        ax.set_xlabel(r"$\theta$", labelpad=0.5*LABELPAD_DIFF)
        ax.set_ylabel(r"$b$", labelpad=LABELPAD_DIFF)

    for i, ax in enumerate(axes):
        ax.set_title(label_for[i])

# ----------------------------------------------------------------------------------------

def funcplot(X, y, problem, vec_funcs):

    fig = pl.figure(figsize=(12, 3.2))
    gs = matplotlib.gridspec.GridSpec(1, 4)
    gs.update(left=0.05, right=0.99, wspace=0.1, hspace=0.2, bottom=0.1, top=0.90)
    axes = [fig.add_subplot(gs[i, j]) for i in range(1) for j in range(4)]

    # plot posterior:
    plot_vecFields([axes[0]], problem, vec_funcs, with_loss_contour=True)
    plot_samples(axes[0], problem)
    plot_ellipse(axes[0], problem)

    # plot the full distribution:
    plot_sdata(axes[1], problem, X, y, type='full')

    # plot the kernel contribution:
    plot_sdata(axes[2], problem, X, y, type='kernel')

    # plot the kernel contribution:
    plot_sdata(axes[3], problem, X, y, type='nonkernel')

    for i in range(1,4):
        axes[i].set_xlabel("x", labelpad=0.5*LABELPAD_DIFF)
        axes[i].set_ylabel("y", labelpad=0.5*LABELPAD_DIFF)
        axes[i].set_xticks([0, 10])
        axes[i].set_yticks([0, 20])

    axes[0].set_xlim(theta_lims)
    axes[0].set_ylim(theta_lims)
    axes[0].set_xticks(theta_lims)
    axes[0].set_yticks(theta_lims)
    axes[0].set_ylabel(r"$b$", labelpad=LABELPAD_DIFF)
    axes[0].set_xlabel(r"$\theta$", labelpad=0.5*LABELPAD_DIFF)

    return
