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

		analytic:
		dL_df = 2*(y_tilde[i] - self.y[i])
		df_dw1= self.X[i]
		dL_dw1 = dL_df*df_dw1
		dL_dw2 = dL_df
		print(dL_dw1, dL_dw2)
		
		"""
		_theta = theta.reshape(-1,1)
		vector_to_parameters(_theta, self.model.parameters())

		y_tilde = self.model(self.X)
		loss_fn = nn.MSELoss()

		grads=[]
		for i in range(self.X.size(0)): # clunky - how to speed this up?
			self.model.zero_grad()
			loss = loss_fn(y_tilde[i], self.y[i])
			loss.backward(retain_graph=True)
			grad = []
			for param in self.model.parameters():
				grad.append(param.grad.view(-1))
			grads.append(torch.cat(grad))

		J = torch.cat(grads).reshape(-1,self.D)/(2.*self.noise_var)
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

		_theta = theta.reshape(-1,1)
		vector_to_parameters(_theta, self.model.parameters())

		y_tilde = self.model(self.X)
		
		grads=[]
		for i in range(self.X.size(0)): # clunky 
			self.model.zero_grad()
			y_tilde[i].backward(retain_graph=True)
			grad = []
			for param in self.model.parameters():
				grad.append(param.grad.view(-1))
			grads.append(torch.cat(grad))

		J = torch.cat(grads).reshape(-1,self.D)
		
		return torch.matmul(J.T, J) / (self.noise_var * self.N)

	def hess_prior(self):
		return torch.eye(self.D) / (self.prior_var * self.N)


	def _hessian(self, theta):
		return self.hess_data(theta) + self.hess_prior()
		