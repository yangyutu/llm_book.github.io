# LLM alignement and preference learning

## Motivation

The objective function in LLM pretraining is predicting the next token on a webpage from the internet. When the trained model is properly and carefully prompted as demonstration (i.e., in-context learning as in GPT-3), the model can largely accomplish useful tasks by following these demonstrations. However, these model can often generate un-desired outputs, including un-factual content, biased and harmful text, or simply do not follow the instructions in the prompt. 

This is because the pretraining task of *predicting the next token* is inherently different from the objective training an LLM to be an instruction-following assistant that avoids generating unintended text. While continuing SFT instruction tuning data, which are (prompt, completion) pairs, can expose the LLM to what humans like to see for given prompts, it is often not enough to prevent model from producing unintended texts. Instead, we need a training methodology to explictly reward the model when it is well-behaved and penalize the model when it is mis-behaved. Training the model to learn the human preference using rewards and penalities are the core of LLM alignment and preference learning. The pioneering approach is using reinforcement learning via the PPO algorithm {cite:p}`ouyang2022traininglanguagemodelsfollow`.

As shown in the {numref}`chapter_training_fig_alignment_model_alignment_motivation`, SFT on instruction tuning dataset and reinforcement learning via PPO can improve model helpfulness and instruction following abilities.  

```{figure} ../img/chapter_training/alignment/model_alignment_motivation.png
---
scale: 40%
name: chapter_training_fig_alignment_model_alignment_motivation
---
Human evaluations of various models outputs show that how often outputs from each model were preferred to those from the 175B GPT-3 SFT model. The aligned models InstructGPT models (PPO-ptx) as well as variant (PPO) significantly outperform the GPT-3 baselines. Image from {cite:p}`ouyang2022traininglanguagemodelsfollow`.
```



## RLHF via PPO


### Overall methodology

Our methodology follows that of Ziegler et al. (2019) and Stiennon et al. (2020), who applied it in the stylistic continuation and summarization domains. We start with a pretrained language model (Radford et al., 2019; Brown et al., 2020; Fedus et al., 2021; Rae et al., 2021; Thoppilan et al., 2022), a distribution of prompts on which we want our model to produce aligned outputs, and a team of trained human labelers (see Sections 3.4 for details). We then apply the following three steps (Figure 2).

**Step 1: SFT on demeonstration data**: Collect demonstration data, and train a supervised policy. Our labelers provide demonstrations of the desired behavior on the input prompt distribution (see Section 3.2 for details on this distribution). We then fine-tune a pretrained GPT-3 model on this data using supervised learning.

**Step 2: Preference labeling and reward modeling**: Collect comparison data, and train a reward model. We collect a dataset of comparisons between model outputs, where labelers indicate which output they prefer for a given input. We then train a reward model to predict the human-preferred output.

**Step 3 Optimize policy with reward model**: Optimize a policy against the reward model using PPO. We use the output of the RM as a scalar reward. We fine-tune the supervised policy to optimize this reward using the PPO algorithm (Schulman et al., 2017).

Steps 2 and 3 can be iterated continuously; more comparison data is collected on the current best policy, which is used to train a new RM and then a new policy. In practice, most of our comparison data comes from our supervised policies, with some coming from our PPO policies.





{cite:p}`ouyang2022traininglanguagemodelsfollow`

```{figure} ../img/chapter_training/alignment/RLHF_PPO/RLHF_demo.png
---
scale: 50%
name: chapter_training_fig_alignment_RLHF_demo
---
A diagram illustrating the three steps of our method: (1) supervised fine-tuning (SFT), (2)
reward model (RM) training, and (3) reinforcement learning via proximal policy optimization (PPO)
on this reward model. Image from {cite:p}`ouyang2022traininglanguagemodelsfollow`.
```


### SFT


### Preference data collection

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


### Reward modeling

The objective of reward modeling is to train a model that take prompt $x$ and one completion $y$ as input and output a scalar score that align with human preference. 
More specificlly, let $r(x, y)$ be the model's scalar output, we have

$$r(x, y_w) > r(x, y_l) ~\text{if} ~ y_w \succ w_l$$

where $y_w$ and $y_l$ are two completions of prompt $x$, and $y_w$ is the preferred completion compared to $y_l$.


Specifically, the loss function for the reward model (parameterized by $\theta$) is given by:

$$
L_\theta=-\frac{1}{\binom{K}{2}} E_{\left(x, y_w, y_l\right) \sim D}\left[\log \left(\sigma\left(r_\theta\left(x, y_w\right)-r_\theta\left(x, y_l\right)\right)\right)\right]
$$


Starting from the SFT model with the final unembedding layer removed,
we trained a model to take in a prompt and response, and output a scalar reward. In this paper we
only use 6B RMs


Tends to overfitting to highly scored completions.

Instead, we train on all $\binom{K}{2}$ comparisons from each prompt as a single batch element. This is much more computationally efficient because it only requires a single forward pass of the RM for each completion (rather than $\binom{K}{2}$ forward passes for $K$ completions) and, because it no longer overfits, it achieves much improved validation accuracy and log loss.


每一个输入token最终都能够生成一个标量值。对于LLM来说，最后一个输入token的处理结果会采样变成next_token，现在变成了score，作为所有输入token的打分结果（其实也可以取所有token生成的score进行平均，通常是直接取最后一个score，训练的效果更好一些）。
预训练好的Reward模型可以参考：https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-7B-Reward




### Reinforcement learning

Given the prompt and response, it produces a reward determined by the reward model and ends the episode.
In addition, we add a per-token KL penalty from the SFT model at each token to mitigate over-optimization of the reward model.

We **maximize** the following objective function in the PPO RL training:

$$
\operatorname{Objective}_{\text{PPO}}(\phi)=  E_{(x, y) \sim D_{\pi_\phi^{\mathrm{RL}}}}\left[r_\theta(x, y)-\beta \log \left(\pi_\phi^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x)\right)\right]
$$

Here
* $\pi_\phi^{\mathrm{RL}}$ is the learned RL policy, $\pi^{\mathrm{SFT}}$ is the supervised trained model
* $D_{\text {pretrain }}$ is the pretraining distribution. The KL reward coefficient, $\beta$, and the pretraining loss coefficient, $\gamma$, control the strength of the KL penalty and pretraining gradients respectively. 
  

如果仅仅使用reward loss，容易导致模型在偏好数据上过拟合，损失原本的能力。因此增加了和原始模型输出的KL损失（KL越小越好）


To prevent language modeling performance regression, we can add an auxillary objective to maximize the likelihood on texts sampled from pretraining datasets. The final objective, named **PPO-ptx**, is given by

$$
\operatorname{Objective}_{\text{PPO-ptx}}(\phi)= \operatorname{Objective}_{\text{PPO}}(\phi) + \gamma E_{x \sim D_{\text {pretrain }}}\left[\log \left(\pi_\phi^{\mathrm{RL}}(x)\right)\right]
$$

where $D_{\text {pretrain }}$ is the pretraining distribution.

```{figure} ../img/chapter_training/alignment/RLHF_PPO/RLHF_PPO_training_demo.png
---
scale: 50%
name: chapter_training_fig_alignment_RLHF_demo
---

```

### The PPO algorithm

There are four models needed in the PPO algorithm:
* Actor LLM 
* Frozen Actor LLM
* Value function model
* Frozen Reward model
  

### Preliminary: MDP

Each environment is an NLP task: we are given a supervised dataset $\mathcal{D}=\left\{\left(\boldsymbol{x}^i, \boldsymbol{y}^i\right)\right\}_{i=1}^N$ of $N$ examples, where $\boldsymbol{x} \in \mathcal{X}$ is an language input and $\boldsymbol{y} \in \mathcal{Y}$ is the target string. 

Generation can be viewed as a Markov Decision Process (MDP) $\langle\mathcal{S}, \mathcal{A}, \mathcal{R}, P, \gamma, T\rangle$ using a finite vocabulary $\mathcal{V}$. 

Each episode in the MDP begins by sampling a datapoint $(\boldsymbol{x}, \boldsymbol{y})$ from our dataset and ends when the current time step $t$ exceeds the horizon $T$ or an end of sentence (EOS) token is generated. 
* The initial state $\boldsymbol{s}_0 \in \mathcal{S}$ is a task-specific prompt represented by $\boldsymbol{x}=\left(x_0, \cdots, x_m\right)$. That is, $\boldsymbol{s}_0 = \boldsymbol{x}$.
* An action in the environment $a_t \in \mathcal{A}$ consists of a token from our vocabulary $\mathcal{V}$.
* The transition function $P: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$ deterministically appends an action $a_t$ to the end of the state $s_{t-1}=\left(x_0, \cdots, x_m, a_0, \cdots, a_{t-1}\right)$.
* At the end of an episode a reward $\mathcal{R}: \mathcal{S} \rightarrow \mathbb{R}^1$ is provided by the reward model.

The input $\boldsymbol{x}=\left(x_0, \cdots, x_m\right)$ is a task-specific prompt that is used as our initial state $\boldsymbol{s}_0=\left(x_0, \cdots, x_m\right)$, where $s_0 \in \mathcal{S}$ and $\mathcal{S}$ is the state space with $x_m \in \mathcal{V}$.  The transition function $P: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$ deterministically appends an action $a_t$ to the end of the state $s_{t-1}=\left(x_0, \cdots, x_m, a_0, \cdots, a_{t-1}\right)$. This continues until the end of the horizon $t \leq T$ and we obtain a state $s_T=\left(x_0, \cdots, x_m, a_0, \cdots, a_T\right)$. At the end of an episode a reward $\mathcal{R}: \mathcal{S} \times \mathcal{A} \times \mathcal{Y} \rightarrow \mathbb{R}^1$ that depends on the $\left(s_T, \boldsymbol{y}\right)$ (e.g., an automated metric like PARENT Dhingra et al. (2019)) is emitted. RL4LMs provides an OpenAI gym (Brockman et al., 2016) style


## DPO

DPO (Direct Preference Optimization) {cite:p}`rafailov2024directpreferenceoptimizationlanguage` improves the classical RLHF-PPO algorithm from the following two aspects:

* Reward model is no longer need; Instead, preference data is directly used to train an aligned model in one step.
* Reinforcement learning is simplified .
It no longer uses reinforcement learning methods. Through mathematical reasoning, it simplifies the original preference alignment objective step by step, and finally trains an aligned model using simpler steps similar to SFT (Supervised Fine-Tuning).

The following illustrates from the DPO paper to visually compare the differences between RLHF-PPO and DPO



```{figure} ../img/chapter_training/alignment/DPO/DPO_PPO_comparison.png
---
scale: 45%
name: chapter_foundation_fig_bert_bert_elmo
---
DPO optimizes for human preferences while avoiding reinforcement learning. Existing methods for fine-tuning language models with human feedback first fit a reward model to a dataset of prompts and
human preferences over pairs of responses, and then use RL to find a policy that maximizes the learned reward.
In contrast, DPO directly optimizes for the policy best satisfying the preferences with a simple classification
objective. Image from {cite:p}`rafailov2024directpreferenceoptimizationlanguage`.
```


DPO pipelines
* Sample completions $y_1, y_2 \sim \pi_{\text {ref }}(\cdot \mid x)$ for every prompt $x$, label with human preferences to construct the offline dataset of preferences $\left.\mathcal{D}=\left\{x^{(i)}, y_w^{(i)}, y_l\right)^{(i)}\right\}_{i=1}^N$ 
*  optimize the language model $\pi_\theta$ to minimize $\mathcal{L}_{\mathrm{DPO}}$ for the given $\pi_{\text {ref }}$ and $\mathcal{D}$ and desired $\beta$. 
   
Note: Since the preference datasets are sampled using $\pi^{\mathrm{SFT}}$, we initialize $\pi_{\text {ref }}=\pi^{\mathrm{SFT}}$ whenever available. 


### Preliminary: Preference modeling

The Bradley-Terry model {cite:p}`bradley1952rank` is a probability model for the outcome of pairwise comparisons between items, teams, or objects. Given a pair of items $i$ and $j$ drawn from some population, it estimates the probability that the pairwise comparison $i>j$ turns out true, as

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
\max _\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi}[r(x, y)]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi(y \mid x) \| \pi_{\text {ref }}(y \mid x)\right].
$$

It turns out that we can obtain the theoretical solution of $\pi_r(y|x)$ given by

$$\pi_r(y|x) = \frac{1}{Z(x)}_{\text{ref}}(y|x) \exp(\frac{1}{\beta}r(x, y)),$$

where $Z(x)$ is partition funciton dependent only on $x$ and $\pi_{\text{ref}}$.


With some algebra, we can also represent the reward funciton with $\pi_r(y|x)$, given by

$$r(x, y) = \beta \log \frac{\pi_r (y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z.$$

Note that we have just shown that that reward function $r(x, y)$ and its corresponding optimal policy $\pi_{\text{ref}}(y|x)$ are inter-convertable, with a funciton $Z(x)$ independent of $y$. 


This means that instead of numerically optimizing policy $\pi_r$, we can also choose optimize the reward function. Given the available preference data, one formulation to optimize the reward function is the Bradley-Terry (BT) objective, that is

$$\mathcal{L}_{BT} = - -\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}\left[\log \sigma (r(y_w, x) - r(r_l, x)) \right].$$

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

## DPO variants

### Smoothing preference label

{cite:p}`Mitchell2023noteondpo`
What if preference labels are noisy? Say the labels have been flipped with some small probability $\epsilon \in(0,0.5)$. We can use a conservative target distribution instead, $p\left(y_w \succ y_l\right)=1-\epsilon$, giving BCE loss:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{DPO}}^\epsilon\left(\theta, y_w, y_l\right) & =-(1-\epsilon) \log \hat{p}_\theta\left(y_w \succ y_l\right)-\epsilon \log \left(1-\hat{p}_\theta\left(y_w \succ y_l\right)\right) \\
& =(1-\epsilon) \mathcal{L}_{\mathrm{DPO}}\left(\theta, y_w, y_l\right)+\epsilon \mathcal{L}_{\mathrm{DPO}}\left(\theta, y_l, y_w\right)
\end{aligned}
$$


The gradient of $\mathcal{L}_{\mathrm{DPO}}^\epsilon\left(\theta, y_w, y_l\right)$ is simply the weighted sum of gradients $(1-\epsilon) \nabla_\theta \mathcal{L}\left(\theta, y_w, y_l\right)+\epsilon \nabla_\theta \mathcal{L}\left(\theta, y_l, y_w\right)$, which reduces to the simplified form (ignoring constants; see [3] for the gradient of the original DPO loss):

$$
\begin{aligned}
\nabla_\theta \mathcal{L}_{\mathrm{DPO}}^\epsilon\left(\theta, y_w, y_l\right) & =-\left((1-\epsilon)\left(1-\hat{p}_\theta\right)-\epsilon \hat{p}_\theta\right)[\underbrace{\nabla_\theta \log \pi_\theta\left(y_w\right)}_{\text {upweight } y_w}-\underbrace{\nabla_\theta \log \pi_\theta\left(y_l\right)}_{\text {downweight } y_l}] \\
& =\quad\left(\hat{p}_\theta-(1-\epsilon)\right)\left[\nabla_\theta \log \pi_\theta\left(y_w\right)-\nabla_\theta \log \pi_\theta\left(y_l\right)\right]
\end{aligned}
$$


The gradient is zero when $\hat{p}_\theta\left(y_w \succ y_l\right)=(1-\epsilon)$, i.e., our (implicit) reward assigns the desired confidence level in this training example under the Bradley-Terry model [2]. For normal DPO, the gradient is never zero! Using the shorthand $h_{\pi_\theta}^{y_w, y_l}=\log \frac{\pi_\theta\left(y_w\right)}{\pi_{\text {ref }}\left(y_w\right)}-\log \frac{\pi_\theta\left(y_l\right)}{\pi_{\text {ref }}\left(y_l\right)}$, let's compare the conservative DPO (cDPO?)


### Simple DPO

{cite:p}`meng2024simposimplepreferenceoptimization`



:bibliography:`../llm_book.bib`