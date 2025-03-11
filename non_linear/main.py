import numpy as np
import pylab as pl
import scipy as sp
from scipy.optimize import minimize
from tqdm import tqdm
import time

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from problems import *
from plotting import *
from non_linear import nonlinear_regression

# ----------------------------------------------------------------------------------------

def conj_grad(problem):

	theta = torch.zeros([problem.D, 1])

	n_it = 2
	for i in range(n_it):
		H = problem._hessian(theta).detach()
		g = problem._g(theta).detach()
		dif, info = sp.sparse.linalg.cg(H.numpy(),g.numpy()) # no torch alternative...
		assert(info == 0)
		theta-= torch.from_numpy(dif.reshape((problem.D,1)))
	
	return theta, info

# ----------------------------------------------------------------------------------------

def f_opt(theta, problem):

	problem.model.eval()
	_theta = torch.from_numpy(theta).reshape(-1,1).float()
	vector_to_parameters(_theta, problem.model.parameters())
	loss = torch.mean(problem.y - problem.model(problem.X))
	
	return loss.detach().numpy()

def direct_opt(problem):

	theta = 1.9*np.ones(problem.D)
	res = minimize(f_opt, theta, (problem,))
	print(res)
	theta = res.x
	stop
	return theta

# ----------------------------------------------------------------------------------------

def run_chain(vectorFuncs, startingPoints, n_it=50000, step=0.001):

	"""
	Appendix D
	"""
	def gd(vecFunc, startingPoint):
		xs = np.zeros((n_it, 2))
		xs[0, :] = startingPoint.copy().reshape((-1,))

		for t in tqdm(range(1, n_it), leave=False):
			xs_ten = torch.from_numpy(xs[t - 1]).float()
			xs[t] = (xs[t - 1] + step * vecFunc(xs_ten)).reshape((-1,))

		return xs

	results = []
	for i, startingpoint in tqdm(enumerate(startingPoints), total=len(startingPoints), leave=False):
		results.append([])
		for j, vecfunc in tqdm(enumerate(vectorFuncs), total=len(vectorFuncs), leave=False):
			results[i].append(gd(vecfunc, startingpoint))

	return results

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

if __name__ == "__main__":

	action='sample'

	# set seed:
	torch.manual_seed(42)

	# generate some data:
	N = 1000
	X, y = nonlinear_model(N, theta=[2.,2.], noise_var=1.0)

	# define the kind of problem you want to solve:
	model = NonLinearModel()
	problem = nonlinear_regression(X,y,model,prior_var=0.1,noise_var=1.0)

	# get a solution using conj. gradient:
#	theta_0, _ = conj_grad(problem) # replace this... scipy.optimise?
#	theta_0 = direct_opt(problem)
	theta_0 = torch.tensor([2.,2.])
	problem.theta_0 = theta_0
	
	# create different vector functions to map the gradient landscape:
	gammas = [1/20, 1/2, 3] 
	def vec_funcs(gammas, problem):
		return [
			lambda t: - gammas[0] * problem._g(t).detach().numpy(),
			lambda t: - gammas[1] * sp.linalg.solve(problem._hessian(t).detach().numpy() + (1e-8) * np.eye(2), problem._g(t).detach().numpy()),
	#		lambda t: - gammas[2] * sp.linalg.solve(problem._ef(t).detach().numpy() + (1e-8) * np.eye(2), problem._g(t).detach().numpy()),
		]

	# (x,y) swapped w.r.t original code because of param order:
	starts = [
			  np.array([4.5, 2.]).reshape((-1, 1)),
			  np.array([0, 1]).reshape((-1, 1)),
			  np.array([3, 4.5]).reshape((-1, 1)),
			  np.array([3, -0.5]).reshape((-1, 1)),
		]

	if action=='chains':
		# use the differently conditioned gradients for gradient descent:
		chains = run_chain(vec_funcs(gammas, problem), starts)

		# visualise the results:
		plot(X.detach(),y.detach(),problem,vec_funcs(gammas, problem), starts, chains)
		pl.savefig("vecfield.png")
		pl.show()

	elif action=='sample':
		funcplot(X,y,problem,vec_funcs(gammas, problem))
		pl.savefig("kernel.png")
		pl.show()
