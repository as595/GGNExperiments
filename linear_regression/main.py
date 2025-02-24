import numpy as np
import pylab as pl
import scipy as sp
from tqdm import tqdm

from problems import *
from plotting import *
from linear import linear_regression

# ----------------------------------------------------------------------------------------

def conj_grad(problem):

	theta = torch.zeros([problem.D, 1])

	n_it = 2
	for i in range(n_it):
		H = problem._hessian(theta)
		g = problem._g(theta)
		dif, info = sp.sparse.linalg.cg(H.numpy(),g.numpy()) # no torch alternative...
		assert(info == 0)
		theta-= torch.from_numpy(dif.reshape((problem.D,1)))
	
	return theta, info


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

	# generate some data:
	N = 1000
	X, y = gradient_field_problem(N)

	# define the kind of problem you want to solve:
	problem = linear_regression(X,y)

	# get a solution using conj. gradient:
	theta_0, _ = conj_grad(problem) 
	problem.theta_0 = theta_0
#	print(problem.g_prior(theta_0))
#	stop

	# create different vector functions to map the gradient landscape:
	gammas = [1 / 3, 1, 3] 
	def vec_funcs(gammas, problem):
		return [
			lambda t: - gammas[0] * problem._g(t).numpy(),
			lambda t: - gammas[1] * sp.linalg.solve(problem._hessian(t).numpy() + (1e-8) * np.eye(2), problem._g(t).numpy()),
			lambda t: - gammas[2] * sp.linalg.solve(problem._ef(t).numpy() + (1e-8) * np.eye(2), problem._g(t).numpy()),
		]

	starts = [
			  np.array([2, 4.5]).reshape((-1, 1)),
			  np.array([1, 0]).reshape((-1, 1)),
			  np.array([4.5, 3]).reshape((-1, 1)),
			  np.array([-0.5, 3]).reshape((-1, 1)),
		]

	# use the differently conditioned gradients for gradient descent:
	chains = run_chain(vec_funcs(gammas, problem), starts)

	# visualise the results:
	plot(X,y,problem,vec_funcs(gammas, problem), starts, chains)
	pl.savefig("vecfield.png")
	pl.show()
