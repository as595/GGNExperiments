import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal


def gradient_field_problem(N):

	X = torch.hstack([torch.ones((N,1)), LogNormal(0,0.75).sample([N,1])])
	thetaT = torch.tensor([[2.],[2.]])
	y = torch.matmul(X, thetaT) + Normal(0,1).sample([N,1])

	return X, y

# ------------------------------------------------------------------------------

def linear_model(N, theta, noise_var):

	X = LogNormal(0,0.75).sample([N,1])
	theta_0 = torch.tensor(theta).unsqueeze(1)
	
	# define model:
	model = LinearModel() # single neuron with weight and bias
	model.output_layer.weight.data.fill_(theta_0[1][0])
	model.output_layer.bias.data.fill_(theta_0[0][0])
	
	# calculate model:
	y = model(X)

	# add noise:
	noise = Normal(0,noise_var).sample((N,)).unsqueeze(1)
	y+= noise
	
	return X, y

# ------------------------------------------------------------------------------

class LinearModel(nn.Module):
	def __init__(self, **kwargs):
		super(LinearModel, self).__init__()

		"""
		f(x) = w_1 x + b_1
		"""
		
		self.output_layer = nn.Linear(1, 1)
		self.loss = nn.MSELoss(reduction='none')
		
	def forward(self, x, y=None, op='forward'):
		if op=='forward':
			return self.output_layer(x)
		elif op=='loss': # needed for fast jacobian calc.
			y_tilde = self.output_layer(x)
			return torch.nn.functional.mse_loss(y_tilde, y, reduction='none')

	def _loss(self, x, y, reduction='none'):
		y_tilde = self.forward(x)
		return torch.nn.functional.mse_loss(y_tilde, y, reduction=reduction)
		
	
# ------------------------------------------------------------------------------

class NonLinearModel(nn.Module):
	def __init__(self, **kwargs):
		super(NonLinearModel, self).__init__()

		"""
		f(x) = w_1 ReLU(w_2 x)
		"""
		
		self.hidden_layer = nn.Linear(1, 1, bias=False)
		self.output_layer = nn.Linear(1, 1, bias=False)
		self.relu = torch.relu
		
	def forward(self, x):
		return self.output_layer(self.act(self.hidden_layer(x)))
