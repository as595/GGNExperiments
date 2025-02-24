import numpy as np

import torch
from torch.distributions import Normal


class linear_regression():

	def __init__(self, X, y, prior_var=0.1, noise=1):


		assert(len(X.size()) == 2)

		self.X = X
		self.y = y.reshape(X.size(0),1)
		self.N = X.size(0)			# no. of data samples
		self.D = X.size(1)			# no. of parameters
		self.prior_var = prior_var 	# prior variance
		self.noise = noise  		# data noise variance


	def _loss(self, theta):
		f = torch.matmul(self.X, theta)
		nll = -1.*Normal(self.y, self.noise).log_prob(f).mean(0)
		return nll


	def g_prior(self, theta):
		"""
		grad log prior
		why is this divided by the number of data points?
		"""
		return theta.reshape(-1,) / (self.prior_var  * self.N)


	def grads(self, theta):

		"""
		numerical solution to grad loss:
		dL/dtheta = dL/df . df/dtheta = (f-y)/sigma^2 . x
		assumes gaussianity
		"""
		_theta = theta.reshape(-1,1)
		f = torch.matmul(self.X, _theta)
		diff = f - self.y 
		J = (self.X*diff) / self.noise
		return J


	def _g(self, theta):
		"""
		dL/dtheta + d(log_prior)/dtheta
		"""
		return self.grads(theta).mean(0) + self.g_prior(theta)

	def _ef(self, theta):
		J = self.grads(theta) 
		return torch.matmul(J.T, J) / self.N


	def hess_data(self, theta):
		"""
		Eq 29 in Kunstner
		J = df/dtheta = x
		where 
		f = theta_1.x +theta_2
		why is this divided by the noise? just to whiten the data?
		"""
		return torch.matmul(self.X.T, self.X) / (self.noise * self.N)


	def hess_prior(self):
		return torch.eye(self.D) / (self.prior_var * self.N)


	def _hessian(self, theta):
		return self.hess_data(theta) + self.hess_prior()
		