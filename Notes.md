## Hessian & Jacobian calculations

For a Gaussian likelihood, the Hessian of the log-likelihood w.r.t the model is unity, i.e. $\nabla_f^2 \, \log \mathcal{L} = 1$. So in this case, the GGN matrix, $J^T H J$, is just given by:

$$
GGN = \sum_n (J_{\theta} f(x_n, \theta))^T (J_{\theta} f(x_n, \theta)),
$$

which is the Hessian of the data. 

### Analytic

In the code for [Kunstner+ 2019]() the Hessian of the data is calculated using the analytic form of the Jacobian, because the data model is defined as $f(x) = \theta_1 x + \theta_2$.

In this case it's easy to see that the Jacobian, $J_{\theta}(f)$, will have components: $df/d\theta_1 = x$ & $df/d\theta_2 = 1$, so the implementation in their code is:

```python
def hess_data(self, theta):
    """
    Eq 29
    """
    return torch.matmul(self.X.T, self.X) / (self.noise_var * self.N)
```

### torch

In the case where the model is not explicitly defined as a function, but is instead specified by a pytorch model (which could just be `torch.nn.Linear(1, 1)`), the same thing is calculated using:

```python
def hess_data(self, theta):
    """
    Eq 29 in Kunstner
    """

    vector_to_parameters(theta, self.model.parameters())

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
```

The loop here is required because the `.backward()` step that calculates the gradients accumulates them for the parameters in the model. Even if you use `out.backward(torch.ones_like(self.X))` to get a vector output, the gradients on the model parameters will still be accumulated and give you a single value. 

### torch.func

The `functorch` functions have now been absorbed into `torch.func` and it is possible to extract the Jacobian using vector mapping:

```python
	def J_theta_f(self, theta):

		"""
		using torch.func
		"""

		params = dict(self.model.named_parameters())
		
		res = jacrev(functional_call, argnums=1)(self.model, params, (self.X,)) # returns dict
		J = torch.zeros([self.X.size(0), len(params.keys())])

		for i in range(len(params.keys())):
			key = list(params.keys())[i]
			J[:,i] = res[key].squeeze()
		
		return J

	def hess_data(self, theta):
		J = self.J_theta_f(theta)
		return torch.matmul(J.T, J) / (self.noise_var * self.N)
```
