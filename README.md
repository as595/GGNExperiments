# GGNExperiments
Experiments with Generalised Gauss Newton for NGD and LLA

---

The GGN is often used as a first approximation to the Hessian, where:

$$
GNN =  \sum_n (J_{\theta} b_n(\theta))^T \nabla_b a_n(b_n(\theta)) J_{\theta} b_n(\theta) 
$$

where $a_n(b_n(\theta))$ is the negative log likelihood or cost function $-\log \mathcal{L} = -\log p(y_n | b_n(\theta))$, which is itself a function of the model output $b_n(\theta)$. 

In the case where $b_n(\theta)$ is itself a linear function, the GNN is an exact representation of the Hessian, because $\nabla^2_{\theta} b_n(\theta) = 0$. This was demonstrated nicely in [Kunstner+ 2019]() for a linear regression examples, reproduced here.

![alt text](./linear_regression/vecfield.png)
