import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from torch.func import grad, jacrev, functional_call

class linear_regression():

	def __init__(self, X, y, model, prior_var=0.1, noise_var=1):


		assert(len(X.size()) == 2)

		self.X = X
		self.y = y.reshape(X.size(0),1)
		self.N = X.size(0)					# no. of data samples
		self.D = len(parameters_to_vector(model.parameters()).detach())	# no. of parameters 
		self.prior_var = prior_var 			# prior variance
		self.noise_var = noise_var  		# data noise variance
		self.model = model					# model

	def _loss(self, theta):
		_theta = theta.reshape(-1,1)
		vector_to_parameters(_theta, self.model.parameters())
		f = self.model(self.X)
		nll = -1.*Normal(self.y, self.noise_var).log_prob(f).mean(0)
		return nll

	def J_theta_f(self, theta):

		"""
		using torch.func
		"""

		_theta = theta.reshape(-1,1)
		vector_to_parameters(_theta, self.model.parameters())
		
		self.model.zero_grad()
		params = dict(self.model.named_parameters())

		res = jacrev(functional_call, argnums=1)(self.model, params, (self.X,)) # returns dict
		J = torch.zeros([self.X.size(0), len(params.keys())])

		for i in range(len(params.keys())):
			key = list(params.keys())[i]
			J[:,i] = res[key].squeeze()
		
		return J

	def J_theta_L(self, theta):

		"""
		using torch.func

		"""

		_theta = theta.reshape(-1,1)
		vector_to_parameters(_theta, self.model.parameters())

		self.model.zero_grad()
		params = dict(self.model.named_parameters())
		
		res = jacrev(functional_call, argnums=1)(self.model, params, (self.X, self.y, 'loss')) # returns dict
		J = torch.zeros([self.X.size(0), len(params.keys())])
		for i in range(len(params.keys())):
			key = list(params.keys())[i]
			J[:,i] = res[key].squeeze()

		# MSE loss doesn't include 1/2\sigma^2 factor in NLL
		return J/(2.*self.noise_var)

	def g_prior(self, theta):
		"""
		grad log prior
		why is this divided by the number of data points?
		"""
		return theta.reshape(-1,) / (self.prior_var  * self.N)

	def grads(self, theta):
		return self.J_theta_L(theta)

	def _g(self, theta):
		"""
		dL/dtheta + d(log_prior)/dtheta
		"""
		return self.grads(theta).mean(0) + self.g_prior(theta)

	def _ef(self, theta):
		J = self.grads(theta) 
		return torch.matmul(J.T, J) / self.N

	def hess_data(self, theta):
		J = self.J_theta_f(theta)
		return torch.matmul(J.T, J) / (self.noise_var * self.N)

	def hess_prior(self):
		return torch.eye(self.D) / (self.prior_var * self.N)


	def _hessian(self, theta):
		return self.hess_data(theta) + self.hess_prior()
		