(chapter_training_sec_LLM_alignment)=
# LLM Alignement and Preference Learning

## Motivation and Overview

The objective function in LLM pretraining is predicting the next token in the training corpus. When the trained model is properly and carefully prompted as demonstration (i.e., in-context learning as in {ref}`content:chapter_foundation:GPT_series:GPT_3`), the model can largely accomplish useful tasks by following these demonstrations. However, these model can often generate un-desired outputs, including un-factual content, biased and harmful text, or simply do not follow the instructions in the prompt. 

This is because the pretraining task of *predicting the next token* is inherently different from the objective of training an LLM to be an instruction-following assistant that avoids generating unintended text. Although **instruction tuning data** ({ref}`chapter_training_sec_LLM_finetuning`), which are (prompt, completion) pairs, can expose the LLM to what humans like to see for given prompts, it is often not enough to prevent model from producing unintended texts. Instead, we need a training methodology to **explicitly reward the model when it is well-behaved and penalize the model when it is mis-behaved**. Training the model to learn the human preference using rewards and penalities are the core of LLM alignment and preference learning. The pioneering approach is using reinforcement learning via the PPO algorithm {cite:p}`ouyang2022traininglanguagemodelsfollow`.

As shown in the {numref}`chapter_training_fig_alignment_model_alignment_motivation`, while SFT on instruction tuning dataset can improve model helpfulness and instruction following abilities, reinforcement learning can help the model achieve much larger gains than SFT.  

```{figure} ../img/chapter_training/alignment/model_alignment_motivation.png
---
scale: 40%
name: chapter_training_fig_alignment_model_alignment_motivation
---
Human evaluations of various models outputs show that how often outputs from each model were preferred to those from the 175B GPT-3 SFT model. The aligned models InstructGPT models (PPO-ptx) as well as variant (PPO) significantly outperform the GPT-3 baselines. Image from {cite:p}`ouyang2022traininglanguagemodelsfollow`.
```



## Alignment Using RLHF


### Overall methodology

The Alignment methodology {cite:p}`ouyang2022traininglanguagemodelsfollow` has the following three steps [{numref}`chapter_training_fig_alignment_RLHF_demo`].

**Step 1: SFT on demeonstration data**: Collect demonstration data showing the target output given different prompt input, and SFT the model to mimic the target output. 

**Step 2: Preference/comparison labeling and reward modeling**: Collect preference data, and train a reward model. The preference data set consists of labeler's preferences towards different model outputs. Such preference data will be used to train a reward model to predict if human would prefer the model output given a model input.

**Step 3 Optimize model generation policy with reward model**: The reward model will be used to guide the model's improvement on producing human preferred outputs. The optimization can be done using reinforcement learning, particularly the PPO algorithm {cite:p}`schulman2017proximalpolicyoptimizationalgorithms`. 

Steps 2 and 3 can be iterated continuously; with model policy improved using reward model, we can collect more preference data to train a new RM and then a new policy. The reason is to make sure that the reward model is adapted to the new input distribution determined by the policy.

```{figure} ../img/chapter_training/alignment/RLHF_PPO/RLHF_demo.png
---
scale: 50%
name: chapter_training_fig_alignment_RLHF_demo
---
A diagram illustrating the three steps of our method: (1) supervised fine-tuning (SFT), (2)
reward model (RM) training, and (3) reinforcement learning via proximal policy optimization (PPO)
on this reward model. Image from {cite:p}`ouyang2022traininglanguagemodelsfollow`.
```

````{prf:reward} The importance of accurate reward model
In classical RL, the reward model is usually deterministic and given. And RL algorithm optimizes the policy to maximize the reward. In LLM alignment, the reward model is not clearly defined and needs to be inferred from the preference data. 

The size, quality, and distribution of the preference data affects how good we can train a reward model that approximates the ground-truth reward model. If there is a gap between the trained reward model and the ground-truth reward model, the gap will translate to the sub-optimality of the learned policy. 
````


### Preference Data and Reward Modeling

After SFT process on positive example, the trained model has improved ability on producing human preferred output. **But it often has the overfitting risk and does not generalize well to unseen input data distributions.** Preference data collections aims to provide both positive and negative examples, which are then used to train a reward model to help guide the model to generalize.

In the preference data collection process, human labelers assess different model outputs given the same prompt input and rank the output based on the human preference. The following table show the scoring standard used in the preference ranking process. 
```{table} How labeler evaluates the response quality
| Metadata | Scale |
| :--- | ---: |
| Overall quality | Likert scale; 1-7 |
| Fails to follow the correct instruction / task | Binary |
| Inappropriate for customer assistant | Binary |
| Hallucination | Binary |
| Satisifies constraint provided in the instruction | Binary |
| Contains sexual content | Binary |
| ... | ... |
```

The objective of **reward modeling** is to train a model that take prompt $x$ and one completion $y$ as input and output a scalar score that align with human preference. 
More specificlly, let $r(x, y)$ be the model's scalar output, we have

$$r(x, y_w) > r(x, y_l) ~\text{if} ~ y_w \succ w_l$$

where $y_w$ and $y_l$ are two completions of prompt $x$, and $y_w$ is the preferred completion compared to $y_l$.

Specifically, the loss function for the reward model (parameterized by $\theta$) is given by:

$$
L_\theta=-\frac{1}{\binom{K}{2}} E_{\left(x, y_w, y_l\right) \sim D}\left[\log \left(\sigma\left(r_\theta\left(x, y_w\right)-r_\theta\left(x, y_l\right)\right)\right)\right]
$$

Here $K$ (between 4 and 9) is the number reponses from the model for a given input, which forms $\binom{K}{2}$ pairwise comparison for the labeler to rank. Usually, all completions associated with a model input are put into a single batch. This makes the training more efficient, as only one forward pass is needed, as well as helps model generation, as the model sees both positive and negative examples at the same time.

The interpretation of the loss function is that it encourages the reward model to give higher score to winning completions $y_w$ then losing completions $y_l$.

The reward model can be initialized from the SFT model (e.g., a 6B model) with the final embedding layer removed and a predictor head added on top to the final token's last layer hidden dimensions. The reward model is trained to take in a prompt and response, and output a scalar reward. 

<!-- Tends to overfitting to highly scored completions.

Instead, we train on all $\binom{K}{2}$ comparisons from each prompt as a single batch element. This is much more computationally efficient because it only requires a single forward pass of the RM for each completion (rather than $\binom{K}{2}$ forward passes for $K$ completions) and, because it no longer overfits, it achieves much improved validation accuracy and log loss.


每一个输入token最终都能够生成一个标量值。对于LLM来说，最后一个输入token的处理结果会采样变成next_token，现在变成了score，作为所有输入token的打分结果（其实也可以取所有token生成的score进行平均，通常是直接取最后一个score，训练的效果更好一些）。
预训练好的Reward模型可以参考：https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-7B-Reward
 -->

### Markov Decision Process (MDP) and Reinforcement learning

To understand how we can use a reward model to guide the improvement of the LLM using reinforcement learning, we can use the **MDP** framework. In the MDP fraemwork, we view model text generation as an agent's sequential decision process. In particular, the MDP agent provides a response to the given prompt, need to decide the action (which token to generate) at each step. The alignment of LLM is equivalent to optimizing the agent on decision making policy to complete a final human preferred sequence. 

Mathematically, an MDP is charcterized by tuple $\langle\mathcal{S}, V, R, \pi, \gamma, T\rangle$.
* The initial state $\boldsymbol{s}_0 \in \mathcal{S}$ is a task-specific prompt represented by $\boldsymbol{x}=\left(x_0, \cdots, x_m\right)$. That is, $\boldsymbol{s}_0 = \boldsymbol{x}$.
* An action in the environment $a_t \in \mathcal{A}$ consists of a token from our vocabulary $V$.
* The transition function or policy $\pi: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}$ decides an action $a_t$ and updatethe state $s_{t-1}=\left(x_0, \cdots, x_m, a_0, \cdots, a_{t-1}\right)$.
* At the end of an episode a reward $R: \mathcal{S} \rightarrow \mathbb{R}^1$ is provided by the reward model. The episode ends when the current time step $t$ exceeds the horizon $T$ or an end of sentence (EOS) token is generated. 

Specifically, we will parameterize the **step-wise policy** by $\theta$ such that a stochastic policy for each step is given by 

$$\pi_s(a|s, \theta) = Pr(a_t = a| s_t = s, \theta).$$

<!-- 
Each environment is an NLP task: we are given a supervised dataset $\mathcal{D}=\left\{\left(\boldsymbol{x}^i, \boldsymbol{y}^i\right)\right\}_{i=1}^N$ of $N$ examples, where $\boldsymbol{x} \in \mathcal{X}$ is an language input and $\boldsymbol{y} \in \mathcal{Y}$ is the target string. 



Each episode in the MDP begins by sampling a datapoint $(\boldsymbol{x}, \boldsymbol{y})$ from our dataset and ends when the current time step $t$ exceeds the horizon $T$ or an end of sentence (EOS) token is generated. 


The input $\boldsymbol{x}=\left(x_0, \cdots, x_m\right)$ is a task-specific prompt that is used as our initial state $\boldsymbol{s}_0=\left(x_0, \cdots, x_m\right)$, where $s_0 \in \mathcal{S}$ and $\mathcal{S}$ is the state space with $x_m \in \mathcal{V}$.  The transition function $P: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$ deterministically appends an action $a_t$ to the end of the state $s_{t-1}=\left(x_0, \cdots, x_m, a_0, \cdots, a_{t-1}\right)$. This continues until the end of the horizon $t \leq T$ and we obtain a state $s_T=\left(x_0, \cdots, x_m, a_0, \cdots, a_T\right)$. At the end of an episode a reward $\mathcal{R}: \mathcal{S} \times \mathcal{A} \times \mathcal{Y} \rightarrow \mathbb{R}^1$ that depends on the $\left(s_T, \boldsymbol{y}\right)$ (e.g., an automated metric like PARENT Dhingra et al. (2019)) is emitted. RL4LMs provides an OpenAI gym (Brockman et al., 2016) style -->


### The PPO Algorithm

PPO is a policy gradient algorithm used to find the optimal policy $\pi^*$. We use the following notations:
* $(x, y)$ are prompt input $x$ and completion $y$ drawn from distribution $D_{\pi}$ dependent on policy $\pi$
* $\pi(y | x) $ is the trajectory level policy that connects to stepwise policy $\pi_s (a | s)$ via

$$\pi(y | x) = \prod_{t <= T} \pi_s(y_t| y_{( < t)}, x).$$


We **maximize** the following objective function in the PPO RL training:

$$
\begin{align*}
&\max_{\phi}\operatorname{Objective}_{\text{PPO}}(\phi) \\
=& \max_{\phi} E_{(x, y) \sim D_{\pi_\phi^{\mathrm{RL}}}}\left[r_\theta(x, y)\right] - \operatorname{KL} (\pi_\phi^{\mathrm{RL}} || \pi_\phi^{\mathrm{SFT}}) \\
=& \max_{\phi} E_{(x, y) \sim D_{\pi_\phi^{\mathrm{RL}}}}\left[\underbrace{r_\theta(x, y)}_{\text{reward gain}}-\beta \underbrace{\log \pi_\phi^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x)}_{\text{KL regularization}}\right]
\end{align*}
$$

which contains the **reward gain** and **KL regularization penality**.
Here  $\pi_\phi^{\mathrm{RL}}$ is the RL policy to be optimized, $\pi^{\mathrm{SFT}}$ is the supervised trained model's model as regularizer. The KL penality term aims to ensure that the optimized policy does not severely deviate from the original policy and overfit to the reward gain. 
  
Besides the KL penality, to further prevent language modeling performance regression, we can add an auxillary objective to maximize the likelihood on texts sampled from pretraining datasets. The final objective, named **PPO-ptx**, is given by

$$
\operatorname{Objective}_{\text{PPO-ptx}}(\phi)= \operatorname{Objective}_{\text{PPO}}(\phi) + \gamma E_{x \sim D_{\text {pretrain }}}\left[\log \left(\pi_\phi^{\mathrm{RL}}(x)\right)\right]
$$

where $D_{\text {pretrain }}$ is the pretraining distribution.he pretraining loss coefficient, $\gamma$, control the strength of the KL penalty and pretraining gradients respectively. 

```{figure} ../img/chapter_training/alignment/RLHF_PPO/RLHF_PPO_training_demo.png
---
scale: 35%
name: chapter_training_fig_alignment_RLHF_demo
---
Illustration of PPO based optimization, which uses a Frozen LLM and KL loss to prevent model optimization from going too far and reward model to encourge model to learn to generate highly-rewarded outputs.
```


````{prf:remark} exploitation and exploration aspects from $\gamma$
$\beta$ controls the balance between exploitation and exploration. When $\gamma \rightarrow 0$, the learning will concentrate on the max reward with full exploitation. When $\gamma \rightarrow \infty$, optimal policy will be the same as $\pi_{\mathrm{sft}}$ with full exploration.
````


<!-- ### The PPO algorithm

There are four models needed in the PPO algorithm:
* Actor LLM 
* Frozen Actor LLM
* Value function model
* Frozen Reward model
  

 -->


### SFT vs RLHF

% from https://arxiv.org/pdf/2303.18223

SFT adopts a teacher-forcing approach, which directly optimizes the likelihood of a demonstration output. Such a token-level training way essentially does behavior cloning to imitate the demonstrations behavior. 

On the other hand, RLHF firstly learns the reward model from preference data, and then employs it to improve the LLM with RL training (e.g., PPO).

In terms of generation demonstration data vs preference labeling,
preference labeling is much easier than writing the demonstration data.

Another key difference is that RLHF essentially encourages LLMs to learn correct policies by contrasting the self-generated responses (**discriminating between positive and negative responses**). It no longer forces the model to imitate external, **positive only** demonstration data, and thus can mitigate the hallucination issues with SFT as discussed above.

Like classic RL algorithms, RLHF has the drawbacks like sample inefficiency, training complexity and instability. When adapted to LLMs, RLHF further relies on a strong SFT model as initial model checkpoint for efficiently achieving good performance

Overall, SFT is particularly useful to increase the model capacity of pre-trained model checkpoints right after pretraining, while RLHF is promising to further improve the model capacity of SFT models.


## DPO

**DPO (Direct Preference Optimization)** {cite:p}`rafailov2024directpreferenceoptimizationlanguage` improves the classical RLHF-PPO algorithm from the following two aspects:

* Reward model is no longer need; Instead, preference data is directly used to train an aligned model in one step.
* Reinforcement learning is simplified. Using mathematical equivalence, the goal of maximizing probabilites of human-preferred output is accomplished via a simplier appraoch like SFT

The following illustrates from the DPO paper to visually compare the differences between RLHF-PPO and DPO

```{figure} ../img/chapter_training/alignment/DPO/DPO_PPO_comparison.png
---
scale: 45%
name: chapter_training_fig_alignmenet_DPO_PPO_comparison
---
DPO optimizes for human preferences while avoiding reinforcement learning. Existing methods for fine-tuning language models with human feedback first fit a reward model to a dataset of prompts and
human preferences over pairs of responses, and then use RL to find a policy that maximizes the learned reward.
In contrast, DPO directly optimizes for the policy best satisfying the preferences with a simple classification
objective. Image from {cite:p}`rafailov2024directpreferenceoptimizationlanguage`.
```

The DPO training pipelines consists of the following two steps:
* Sample completions $y_1, y_2 \sim \pi_{\text {ref }}(\cdot \mid x)$ for every prompt $x$, label with human preferences to construct the offline dataset of preferences $\mathcal{D}=\left\{x^{(i)}, y_w^{(i)}, y_l^{(i)}\right\}_{i=1}^N$ 
*  Optimize the language model $\pi_\theta$ to minimize $\mathcal{L}_{\mathrm{DPO}}$ for the given $\pi_{\text {ref }}$ and $\mathcal{D}$ and desired $\beta$. 
   
$$
\mathcal{L}_{\mathrm{DPO}}\left(\pi_\theta ; \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta\left(y_w \mid x\right)}{\pi_{\mathrm{ref}}\left(y_w \mid x\right)}-\beta \log \frac{\pi_\theta\left(y_l \mid x\right)}{\pi_{\mathrm{ref}}\left(y_l \mid x\right)}\right)\right].
$$

Note: Since the preference datasets are sampled using $\pi^{\mathrm{SFT}}$, we initialize $\pi_{\text {ref }}=\pi^{\mathrm{SFT}}$ whenever available. 


### Preliminary: Preference modeling

The **Bradley-Terry model** {cite:p}`bradley1952rank` is a probability model for the outcome of pairwise comparisons between items, teams, or objects. Given a pair of items $i$ and $j$ drawn from some population, it estimates the probability that the pairwise comparison $i>j$ turns out true, as

$$
\operatorname{Pr}(i\succ j)=\frac{s_i}{s_i+s_j}
$$

where $s_i$ is a positive real-valued score assigned to individual $i$. 

$$
P\left(y_1 \succ y_2 \mid x\right)=\frac{\exp \left[r\left(x, y_1\right)\right]}{\exp \left[r\left(x, y_1\right)\right]+\exp \left[r\left(x, y_2\right)\right]}$$

````{prf:remark} Relationship to logistic regression

$$\operatorname{logit} \operatorname{Pr}(i>j)=\log \frac{\operatorname{Pr}(i>j)}{1-\operatorname{Pr}(i>j)}=\log \frac{\operatorname{Pr}(i>j)}{\operatorname{Pr}(j>i)}=\beta_i-\beta_j
$$

````

### Driving the DPO

Here we outline the key steps to derive the DPO objective function. 

First we start with the objective of LLM alignment with a given **fixed reward function** $r$ with a KL constraint, 

$$
\max_\pi \mathbb{E}_{x, y \sim \mathcal{D}_{\pi}}[r(x, y)]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi(y \mid x) \| \pi_{\text {ref }}(y \mid x)\right].
$$

It turns out that we can obtain the theoretical solution of $\pi_r(y|x)$ given by

$$\pi_r(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta}r(x, y)),$$

where $Z(x)$ is partition funciton dependent only on $x$ and $\pi_{\text{ref}}$, which is given by $Z(x)=\sum_y \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)$.


With some algebra, we can also represent the reward funciton with $\pi_r(y|x)$, given by

$$r(x, y) = \beta \log \frac{\pi_r (y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z.$$

````{prf:remark} Implicit reward
Note that here the reward can be interpreted as the log ratio of the likelihood of a response between the current policy model and the reference model. And the policy is the maximizer of the objective function given the reward.
````


Note that we have just shown that that reward function $r(x, y)$ and its corresponding optimal policy $\pi_{\text{ref}}(y|x)$ are inter-convertable, with a funciton $Z(x)$ independent of $y$. 


This means that instead of **numerically optimizing policy $\pi$, we can also choose optimize the reward function.** When the reward function is optimized, the policy is also optimized at the same time (in other words, we can analytically solve the optimal policy).

Given the available preference data, one formulation to optimize the reward function is the Bradley-Terry (BT) objective, that is

$$\mathcal{L}_{BT} = -\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}\left[\log \sigma (r(y_w, x) - r(r_l, x)) \right].$$

By leveraging the relationship between reward $r$ and policy $\pi$, we can arrive at the DPO loss function:

$$
\mathcal{L}_{\mathrm{DPO}}\left(\pi_\theta ; \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta\left(y_w \mid x\right)}{\pi_{\mathrm{ref}}\left(y_w \mid x\right)}-\beta \log \frac{\pi_\theta\left(y_l \mid x\right)}{\pi_{\mathrm{ref}}\left(y_l \mid x\right)}\right)\right].
$$

where the terms $\beta \log Z$ are canceled during subtraction.


````{prf:remark} How DPO loss work

The gradient of DPO loss function is given by:

$$\nabla_\theta \mathcal{L}_{\mathrm{DPO}}\left(\theta, y_w, y_l\right)=-\left(1-\hat{p}_\theta\right) [\underbrace{\nabla_\theta \log \pi_\theta\left(y_w\right)}_{\text {upweight } y_w}-\underbrace{\nabla_\theta \log \pi_\theta\left(y_l\right)}_{\text {downweight } y_l}]
$$

where for the preference completion pair $y_w \succ y_l$, as long as $\hat{p} < 1$, there will gradients to upweight the probability of generating $y_w$ and downweight the probability of generating $y_l$.
````

<!-- ### Additional remark RL vs SFT vs DPO

% https://mp.weixin.qq.com/s/WKuEcsyMFkaKf19o20Ci3g -->

### DPO vs RL

One fundamental differences between DPO and RL from policy optimization perspective is that
* DPO is learning from **offline** generated preference data and there is no exploration of the input output space during training. The model has a hard time to generate well beyond what is included in the preference data.
* RL is **online** learning with exploration. RL does not need offline generated data; the LLM agent itself generates outputs and learn from the reward signal from reward model. The self-generation approach enables RL to have in theory unlimited amount of data to cover much large ranges of input and output distributions.

From reward modeling perspective, 
* DPO is approximating the ground-truth reward model using limited, offline generated preference data. The size and distribution of preference data affect the gap of trained reward model and the ground-truth reward model. The gap in the reward model will translate to the gap of policy. 
* To overcome the limitation of offline generated preference data, we can use the exploitation idea from reinforcement learning to generate large-scale and broad ranged (in terms of distribution) data for preference labeling, which can be used to improve reward modeling and close the gap to groundtruth reward model. 

## DPO variants

### Smoothing preference label

{cite:p}`Mitchell2023noteondpo` explore more robust DPO approach when the preference labels are noisy. It assumes that the labels have been flipped with some small probability $\epsilon \in(0,0.5)$. We can use a conservative target distribution instead, $p\left(y_w \succ y_l\right)=1-\epsilon$, giving BCE loss:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{DPO}}^\epsilon\left(\theta, y_w, y_l\right) & =-(1-\epsilon) \log \hat{p}_\theta\left(y_w \succ y_l\right)-\epsilon \log \left(1-\hat{p}_\theta\left(y_w \succ y_l\right)\right) \\
& =(1-\epsilon) \mathcal{L}_{\mathrm{DPO}}\left(\theta, y_w, y_l\right)+\epsilon \mathcal{L}_{\mathrm{DPO}}\left(\theta, y_l, y_w\right)
\end{aligned}
$$


The gradient of $\mathcal{L}_{\mathrm{DPO}}^\epsilon\left(\theta, y_w, y_l\right)$ is reduced to the simplified form :

$$
\begin{aligned}
\nabla_\theta \mathcal{L}_{\mathrm{DPO}}^\epsilon\left(\theta, y_w, y_l\right) & =-\left((1-\epsilon)\left(1-\hat{p}_\theta\right)-\epsilon \hat{p}_\theta\right)[\underbrace{\nabla_\theta \log \pi_\theta\left(y_w\right)}_{\text {upweight } y_w}-\underbrace{\nabla_\theta \log \pi_\theta\left(y_l\right)}_{\text {downweight } y_l}] \\
& =\quad\left(\hat{p}_\theta-(1-\epsilon)\right)\left[\nabla_\theta \log \pi_\theta\left(y_w\right)-\nabla_\theta \log \pi_\theta\left(y_l\right)\right]
\end{aligned}
$$

<!-- The gradient is zero when $\hat{p}_\theta\left(y_w \succ y_l\right)=(1-\epsilon)$, i.e., our (implicit) reward assigns the desired confidence level in this training example under the Bradley-Terry model [2]. For normal DPO, the gradient is never zero! Using the shorthand $h_{\pi_\theta}^{y_w, y_l}=\log \frac{\pi_\theta\left(y_w\right)}{\pi_{\text {ref }}\left(y_w\right)}-\log \frac{\pi_\theta\left(y_l\right)}{\pi_{\text {ref }}\left(y_l\right)}$, let's compare the conservative DPO (cDPO?) -->


### Simple DPO

The simple DPO{cite:p}`meng2024simposimplepreferenceoptimization` improve the original DPO from two aspects:
* Make the sequence likelihood function used in implicit reward be aligned with the likelihood of actual sequence decoding. 
* Add a margin to encourage larger reward gap between positive sequence and negative sequence, which will help generalzation. 

First, the authors argue that original DPO derives the closed form implicit reward as the log ratio of the likelihood of a response between the current policy model and the reference model plus a constant only depending on $x$

$$r(x, y)=\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + Const(x).$$

In actual decoding process, the likelihood of a response is usually length averaged (see {ref}`chapter_inference_sec_deconding_beam_search`), for example,

$$p_\theta(y \mid x)=\frac{1}{|y|} \log \pi_\theta(y \mid x)=\frac{1}{|y|} \sum_{i=1}^{|y|} \log \pi_\theta\left(y_i \mid x, y_{<i}\right)$$

Length-normalized reward formulation. Naturally, we consider replacing the reward formulation in DPO with $p_\theta$ in Eq. (3), so that it aligns with the likehood metric that guides generation. This results in a length-normalized reward:

$$
r_{\mathrm{SimPO}}(x, y)=\frac{\beta}{|y|} \log \pi_\theta(y \mid x)=\frac{\beta}{|y|} \sum_{i=1}^{|y|} \log \pi_\theta\left(y_i \mid x, y_{<i}\right)
$$

The author argues that normalizing the reward with response lengths is crucial; removing the length normalization term from the reward formulation results in a bias toward generating longer but lower-quality sequences.

Additionally,a target reward margin term, $\gamma>0$, is added to the Bradley-Terry objective to ensure that the reward for the winning response, $r\left(x, y_w\right)$, exceeds the reward for the losing response, $r\left(x, y_l\right)$, by at least $\gamma$. This is margin idea is also commonly used in constrastive learning. 

$$
p\left(y_w \succ y_l \mid x\right)=\sigma\left(r\left(x, y_w\right)-r\left(x, y_l\right)-\gamma\right)
$$


Combined these ideas together, we arrive at the SimPO loss function:

$$
\mathcal{L}_{\operatorname{SimPO}}\left(\pi_\theta\right)= -\mathbb{E}\left[\log \sigma\left(\frac{\beta}{\left|y_w\right|} \log \pi_\theta\left(y_w \mid x\right)-\frac{\beta}{\left|y_l\right|} \log \pi_\theta\left(y_l \mid x\right)-\gamma\right)\right]
\end{align}
$$

Note that, unlike the traditional DPO, SimPO does not require a reference model, making it more lightweight and easier to implement.


<!-- 提问：DPO的变体有哪些，主要解决DPO的什么问题？

回答：

RSO [1]：由于DPO的蒙特卡洛采样很难达到，所以其实DPO几乎是off-policy的采样方式，RSO主要从DPO的采样方式来解决DPO的问题。
Iterative DPO [2]：同样由于DPO的蒙特卡洛采样很难达到，所以通过on-policy的方式采样来替代off-policy的采样。
IPO [3]：由于BT model的目标是最大化正负response的reward gap，但其实其中忽略了真实情况下我们组的pair可能会有噪音，那么无限去扩大reward gap其实是不准确的，也就是overfit了preference的pair数据，那么解决方案是需要限制这个gap的范围。
DPOP [4]：由于LLM model很难区分编辑距离较小的pair，那么当持续去区分这批case的时候，模型效果会崩塌，现象是正例子和负例子的概率都往下掉。那么DPOP用了一个新项来惩罚正例往下掉的pair，使得正例概率继续提升。
[1] Liu T, Zhao Y, Joshi R, et al. Statistical rejection sampling improves preference optimization[J]. arXiv preprint arXiv:2309.06657, 2023.

[2] Yuan W, Pang R Y, Cho K, et al. Self-rewarding language models[J]. arXiv preprint arXiv:2401.10020, 2024.

[3] Azar M G, Rowland M, Piot B, et al. A general theoretical paradigm to understand learning from human preferences[J]. arXiv preprint arXiv:2310.12036, 2023.

[4] Pal A, Karkhanis D, Dooley S, et al. Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive[J]. arXiv preprint arXiv:2402.13228, 2024. -->

## Bibliography

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
```