# LLM Training Fundamentals
## Training Overview





## Optimization Algorithms

### Minibatch Stochastic Gradient Descent

Batch gradient descent is usually not ideal for neural network application that involves large-size training data, such as image recognization and natural language processing. Evaluating the gradient over the whole set of training data is computational prohibitive; moreover, many samples are similar, making the gradient of the whole data sample is simply the multiplier of the gradient of a much smaller, representative sample data set. 

Minibatch stochastic gradient descent uses a random sample of the training data set to estimate the gradient on each step. A typical algorithm is showed as follows.



```{prf:algorithm} Minibatch stochastic gradient descent algorithm
:label: Minibatch_stochastic_gradient_descent_algorithm

**Inputs** Learning rate $\alpha_k$, iniital model parameter $\theta$.

**Output** $\theta_k$

1. Set $k=1$ 
2. Repeat until stopping criteria is met:
    1. Sample a minibatch of training samples of size $m$: $(x^{(i)},y^{(i)}),i=1,2,...,m$.
    2. Compute a gradient estimate over this minibatch samples via

    $$\hat{g}_k = \frac{1}{m}\nabla_{\theta} \sum_{i=1}^{m} L(f(x^{(i)};\theta),y^{(i)}).$$

    3. Apply update $\theta_k = \theta_k - \alpha_k \hat{g}_k$.
    
    4. Set $k=k+1$.
```


````{prf:remark} choice of minibatch size
* The estimation quality of gradient via minibatch gradient descent is strongly affected by the minibatch size. In general, the gradient estimate is unbiased, irrespective of the choice of minibatch size, but its variance will decrease as the minibatch increases.
* For larger minibatch size, we can increase learning rate since the estimated gradient is more certain. There is an empirical ***Linear Scaling Rule***: When the minibatch size is multiplied by $k$, multiply the learning rate by $k$ {cite:p}`goyal2017accurate`.
````

### Adaptive Gradient Method
#### Adaptive Gradient (AdaGrad)

For simple stochastic gradient methods, we need to set the learning rate hyperparameter or even dynamically schedule learning rate, which is usually a difficult task or problem specific. Further, a uniform learning rate is usually not an effective way for high-dimensional gradient descent methods, since one learning rate could be too large for one-dimension but, on the contrary, too small for another dimension. 

The AdaGrad algorithm{cite:p}`duchi2011adaptive` addresses the issue by choosing different learning rates for each dimension. The algorithm adaptively scales the learning rate for each dimension based on the **accumulated gradient magnitude** on that dimension so far.

Let $G_k$ be the **accumulated gradient** up to iteration $k$, given by

$$G_k = G_{k-1} + \hat{g}_k\odot \hat{g}_k, G_0 = 0.$$

The parameter update is given by 

$$\theta_k = \theta_{k-1} - \frac{\alpha_0}{\delta + \sqrt{G_k}} \hat{g}_k$$

where $\alpha_0$ is the initial learning speed, usually set at a small number (say, 1e-9 to 1e-7) and $\delta$ is a small positive constant to avoid division by zero.

As we can see, the learning rate in AdaGrad is monotonically decreasing, which may dramatically slow down the convergence as the learning rate becomes too small. In general, AdaGrad algorithm performs best for **convex optimization** (however, neural network optimization is usually non-convex).

#### RMSProp

As we mentioned before, AdaGrad tend to shrink learning rate too aggressively. This is an advantage when applying AdaGrad to convex function optimization as it enables the algorithm to converge fast and stably. However, non-convex function optimization usually require large, adaptive learning rate to escape bad local minimum and converge stably to better local minimums. 

The first remedy is to prevent the learning rate from shrinking too fast. Let $G_k$ be the accumulated gradient up to iteration $k$, given by

$$G_k = \rho G_{k-1} + (1- \rho) g_k\odot g_k, G_0 = 0.$$

Then we compute update 

$$\theta_k= \theta_{k-1} - \frac{\alpha_0}{\delta + \sqrt{G_k}} \hat{g}_k.$$

which is the core part of the RMSProp algorithm {cite:p}`Hinton2012Neural`.

How this modification can make the $G_k$ smaller than that in the AdaGrad, as can be seen from following expansion. 

````{prf:remark} Expansion of $G_k$
In RMSProp, we have

$$
\begin{align}
G_k &= \rho G_{k-1} + (1- \rho) g_k\odot g_k \\
	&= \rho G_{k-2} + \rho(1- \rho) g_k\odot g_k + \rho(1 - \rho)g_{k-1}\odot g_{k-1} \\
	&\approx g\cdot g (1 - \rho) (1 + \rho + \cdot + \rho^{k-1})\\
	&= (1 - \rho^k) g\cdot g
\end{align}
$$

where we assume $g_k\cdot g_k \approx g\cdot g, \forall k$. Clearly, we have
roughly, 

$$G_k^{RMSProp} \approx  \frac{(1 - \rho^k)}{k} G_k^{AdaGrad}, G_k^{AdaGrad}\approx k g\odot g.$$

````


### Momentum Method

Simple SGD with small learning rate can lead to extremely slow learning for functional surfaces with long, narrow valleys {cite:p}`sutskever2013importance`. One intuition inspired by the physics of a heavy ball falling down is to add momentum to the gradient descent steps. Mathematically, adding momentum is equivalent to adding historical weighted averaged gradient to the current gradient. The total gradient will then be hopefully large enough to enable fast movement on relatively flat regions.  


```{figure} ../img/chapter_training/training_fundamentals/SGDMomentum.jpg
---
scale: 30%
name: chapter_training_fig_optimizer_SGD_momentum
---
SGD without momentum and with momentum. SGD with momentum can accumulate gradient/velocity in horizontal direction and move faster towards the minimum located at the center.
```

Consider the gradient 

$$\hat{g}_k = \frac{1}{m}\nabla_{\theta} \sum_{i=1}^{m} L(f(x^{(i);\theta}),y^{(i)}),$$

we can compute a **speed** (an intermedidate parameter) via 

$$v_k = \mu v_{k-1} - \alpha_k \hat{g}_k,$$

where $\mu \in [0, 1]$ is the momentum coefficiency, and $\alpha_k$ is the learning rate. 

The speed (with momentum considered) is then used to update parameter $\theta$ via 

$$\theta_k = \theta_k + v_k.$$

To see that the update velocity is the weighted average gradient, we now show that the velocity is an exponentially decaying moving average (similar to AR(1) process) of the negative gradients, given by

$$
\begin{align}
v_k &= \alpha v_{k-1} - \hat{g}_k\\
&= \alpha (\alpha v_{k-2} - \hat{g}_{k-1}) - \hat{g}_k \\
&= \alpha^2 v_{k-2} - \alpha \hat{g}_{k-1} - \hat{g}_k \\
&= \cdots \\
&= \sum_{i=0}^\infty  \alpha^{i} \hat{g}_{k-i}.
\end{align}	
$$


### Combined Together: Adam and AdamW

#### Adam
By combining the ideas of momentum and adaptive learning rate, we yield Adam, one of most popular gradient descent algorithm in deep learning community{cite:p}`kingma2014adam`. The name Adam is derived from adaptive moment estimation. As its name suggests, Adam will compute velocity via momentum type of averaging and adjust the learning rate using inverse of accumulated gradients. 

Specifically, the velocity is computed via

$$
\begin{align}
M_k &= \rho_1 M_{k-1} + (1-\rho_1)\hat{g}_k \\
\tilde{M}_k &= \frac{M_k}{1-\rho_1^k}
\end{align}
$$

and the accumulated gradient magnitude is compute via

$$
\begin{align}
G_k &= \rho_2 G_{k-1} + (1-\rho_2)\hat{g}_k \odot \hat{g}_k \\
\tilde{G}_k &= \frac{G_k}{1-\rho_2^k}
\end{align}
$$

Note that we correct the $M_k$ and $G_k$ be dividing the factor $1 - \rho_i^k, i= 1, 2$ to get the average estimation. 

The final algorithm is given by the following.


```{prf:algorithm} Adam stochastic gradient descent algorithm
:label: Adam_stochastic_gradient_descent_algorithm
**Inputs** Learning rate $\alpha$(set to 0.001), iniital model parameter $\theta$, decay parameters $\rho_1$(set to 0.9), $\rho_2$(set to 0.999). $\delta = 1e-8$

**Output** $\theta_k$

1. Set $k=1$. 
2. Set $M_k = 0, G_k = 0$.
3. Repeat until stopping criteria is met：
    1. Sample a minibatch of training samples of size $m$ $(x^{(i)},y^{(i)}),i=1,2,...,m$.\\
    2. compute gradient estimate over minibatch $N$ samples via

		$$\hat{g}_k = \frac{1}{m}\nabla_{\theta} \sum_{i=1}^{m} L(f(x^{(i);\theta}),y^{(i)}).$$ 
		
    3. Accumulate $M_k = \rho_1 M_{k-1} + (1-\rho_1)\hat{g}_k$. Accumulate $G_k = \rho_2 G_{k-1} + (1-\rho_2)\hat{g}_k \odot \hat{g}_k$.
	
    4. Correct biases

		$$\tilde{M}_k = \frac{M_k}{1-\rho_1^k}, \tilde{G}_k = \frac{G_k}{1-\rho_2^k}.$$
	
    5. Apply update $$\theta_k = \theta_{k-1} -\frac{\alpha \cdot \tilde{M}_k}{\delta + \sqrt{\tilde{G}_k }}.$$
	6. Set $k=k+1$.
	
```

#### $L_2$ Weight Decay and AdamW

$L_2$ regularization on model parameters often reduce model overfitting and improves the generalization ability of the model. In the SGD optimization framework, the implementation of $L_2$ regularization term is often realized via **weight decay**, resulting in an additional term in the gradient that penalize large weights. That is,

$${g}_{k} \leftarrow \nabla f_{k}\left({\theta}_{k-1}\right)+\lambda {\theta}_{k-1},$$

where $\lambda$ is the decay parameter, corresponding the strength of the regularization.

AdamW {cite:p}`loshchilov2017decoupled` is the algorithm that correctly implements Adam with $L_2$ regularization, which is also called Adam with decoupled weight decay.

The algorithm is given by the following.


```{prf:algorithm} Adam stochastic gradient descent algorithm with weight decay
:label: Adam_stochastic_gradient_descent_algorithm_with_weight_decay
**Inputs** Learning rate $\alpha$(set to 0.001), iniital model parameter $\theta$, decay parameters $\rho_1$(set to 0.9), $\rho_2$(set to 0.999). $\delta = 1e-8$.  <span style="background-color: #e4ac94">Weight decay parameter $\lambda \in \mathbb{R}$</span>.

**Output** $\theta_k$

1. Set $k=1$. 
2. Set $M_k = 0, G_k = 0$.
3. Repeat until stopping criteria is met：
    1. Sample a minibatch of training samples of size $m$ $(x^{(i)},y^{(i)}),i=1,2,...,m$.\\
    2. compute gradient estimate over minibatch $N$ samples via

		$$\hat{g}_k = \frac{1}{m}\nabla_{\theta} \sum_{i=1}^{m} L(f(x^{(i);\theta}),y^{(i)}).$$ 
		
    3. Accumulate $M_k = \rho_1 M_{k-1} + (1-\rho_1)\hat{g}_k$. Accumulate $G_k = \rho_2 G_{k-1} + (1-\rho_2)\hat{g}_k \odot \hat{g}_k$.
	
    4. Correct biases

		$$\tilde{M}_k = \frac{M_k}{1-\rho_1^k}, \tilde{G}_k = \frac{G_k}{1-\rho_2^k}.$$
	
    5. <span style="background-color: #e4ac94"> Apply update
    
    $$\theta_k = \theta_{k-1} -\frac{\alpha \cdot \tilde{M}_k}{\delta + \sqrt{\tilde{G}_k }} - \lambda \theta_{k}.$$
    
    </span>
	6. Set $k=k+1$.
	
```

## Pretraining


\begin{figure}[H]
	\centering
	\includegraphics[width=0.7\linewidth]{../figures/deepLearning/ApplicationsNLP_LLM/efficient_training/training_data/training_data_distribution_summary}
	\caption{Ratios of various data sources in the pre-training data for existing LLMs. Image from \cite{zhao2023survey}.}
	\label{fig:trainingdatadistributionsummary}
\end{figure}


## Post-training




## SFT Vs RLHF

% from https://arxiv.org/pdf/2303.18223

SFT adopts a teacher-forcing approach, which directly optimizes the likelihood of a demonstration output. Such a token-level training way essentially does behavior cloning as the supervision label and directly learns to imitate the demonstrations from experts without specifying a reward model as in typical RL algorithms. 

RLHF firstly learns the reward model, and then employs it to improve the LLM with RL training (e.g., PPO).


 preference annotation is much easier than writing the demonstration data, and annotators can even judge the quality of
more superior generations than those they create, making it
possible to explore a broader state space beyond what can
be demonstrated by human annotators

Another key point is that RLHF essentially encourages LLMs to learn correct policies by contrasting the self-generated responses (discriminating between good and bad responses). It no longer forces the model to imitate external demonstration data, and thus can mitigate the hallucination issues with SFT as discussed above

RLHF inherits the drawbacks of classic RL algorithms, e.g., sample inefficiency and
training instability. When adapted to LLMs, RLHF further
relies on a strong SFT model as initial model checkpoint for
efficiently achieving good performance

Overall, SFT is particularly useful to increase the model capacity of pre-trained model checkpoints right after pretraining, while RLHF is promising to further improve the
model capacity of SFT models.





## Bibliography

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
