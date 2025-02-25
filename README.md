# GGNExperiments
Experiments with Generalised Gauss Newton for NGD and LLA

---

The GGN is often used as an approximation to the Hessian, where:

$$
\nabla^2 \mathcal{L} = \underbrace{ \sum_n (J_{\theta} b_n(\theta))^T \nabla_b a_n(b_n(\theta)) J_{\theta} b_n(\theta) }_{GGN} + \sum_n \nabla_b a_n (b_n (\theta) ) \nabla \nabla b_n (\theta)
$$

where $a_n(b_n(\theta))$ is the negative log likelihood or cost function $-\log \mathcal{L} = -\log p(y_n | b_n(\theta))$, which is itself a function of the model output $b_n(\theta)$.
