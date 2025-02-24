import torch
from torch.distributions import Normal, LogNormal


def gradient_field_problem(N):

	X = torch.hstack([torch.ones((N,1)), LogNormal(0,0.75).sample([N,1])])
	thetaT = torch.tensor([[2.],[2.]])
	y = torch.matmul(X, thetaT) + Normal(0,1).sample([N,1])

	return X, y


def simple_nn(N):

	X = LogNormal(0,0.75).sample(N)

	return X, y