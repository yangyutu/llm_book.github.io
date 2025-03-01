(chapter_training_sec_LLM_alignment)=
# LLM Alignment and Preference Learning

## Motivation and Overview

The objective function in LLM pretraining is predicting the next token in the training corpus. When the trained model is properly and carefully prompted with demonstrations (i.e., in-context learning as in {ref}`content:chapter_foundation:GPT_series:GPT_3`), the model can largely accomplish useful tasks by following these demonstrations. However, these model can often generate un-desired outputs, including un-factual content, biased and harmful text, or simply do not follow the instructions in the prompt. 

This is because the pretraining task of *predicting the next token* is inherently different from the objective of training an LLM to be an instruction-following assistant that avoids generating unintended text. Although **instruction tuning data** ({ref}`chapter_training_sec_LLM_finetuning`), which are (prompt, completion) pairs, can expose the LLM to what humans like to see for given prompts, it is often not enough to prevent model from producing unintended texts. As shown {numref}`chapter_training_fig_alignment_SFT_drawback_empirical_data`, when SFT an LLM on prefered harmless responses in the HH-RLHF dataset {cite:p}`bai2022constitutional`, the log probability of preferred and unwanted responses both exhibited a simultaneous increase. This indiciates that despite the cross-entropy loss can effectively guide the model toward the intended domain (e.g., dialogue), the absence of a penalty also increases the probablity of generating unwanted responses. 

```{figure} ../img/chapter_training/alignment/SFT_drawback_empirical_data.png
---
scale: 55%
name: chapter_training_fig_alignment_SFT_drawback_empirical_data
---
Log probabilities for chosen and rejected responses during model fine-tuning on HH-RLHF dataset. Despite only chosen responses being used for SFT, rejected responses show a comparable likelihood of generation. Image from {cite:p}`hong2024reference`.
```


Instead, we need a training methodology to **explicitly reward the model when it is well-behaved and penalize the model when it is mis-behaved**. Training the model to learn the human preference using rewards and penalities are the core of LLM alignment and preference learning. The pioneering approach is using reinforcement learning via the PPO algorithm {cite:p}`ouyang2022traininglanguagemodelsfollow`.

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

The Alignment methodology {cite:p}`ouyang2022traininglanguagemodelsfollow,stiennon2020learning` has the following three steps [{numref}`chapter_training_fig_alignment_RLHF_demo`].

**Step 1: SFT on demeonstration data**: Collect demonstration data showing the target output given different prompt input, and SFT the model to mimic the target output. 

**Step 2: Preference/comparison labeling and reward modeling**: Collect preference data, and train a reward model. The preference data set consists of labeler's preferences towards different model outputs. Such preference data will be used to train a reward model to predict if human would prefer the model output given a model input.

**Step 3 Optimize model generation policy with reward model**: The reward model will be used to guide the model's improvement on producing human preferred outputs. The optimization can be done using reinforcement learning, particularly the PPO algorithm {cite:p}`schulman2017proximalpolicyoptimizationalgorithms`. 

```{prf:example} Iterative reward model and policy improvement
**Steps 2 and 3 can be, and sometimes must be, iterated continuously**; with model policy improved using reward model, the reward model might need be updated to further guide the improvement of the data. The reason is that
* The text generated from improved model will have a different distribution than what is used in reward model training.
* The trained reward model $R_0$ is an approximate to the groundtruth reward model $R_{GT}$. After one round of policy optimization, the model is likely overfitting to the $R_0$, and actually performs poorly under the evaluation of $R_{GT}$ (also see {cite:p}`stiennon2020learning`).

To iterate the reward model to adapte to the new input distribution determined by the policy, we can collect more preference data, and combine with the original preference data to train a new RM and then a new policy. 
```

```{figure} ../img/chapter_training/alignment/RLHF_PPO/RLHF_demo.png
---
scale: 50%
name: chapter_training_fig_alignment_RLHF_demo
---
A diagram illustrating the three steps of our method: (1) supervised fine-tuning (SFT), (2)
reward model (RM) training, and (3) reinforcement learning via proximal policy optimization (PPO)
on this reward model. Image from {cite:p}`ouyang2022traininglanguagemodelsfollow`.
```

```{prf:remark} The importance of accurate reward model
In classical RL, the reward model is usually deterministic and given. And RL algorithm optimizes the policy to maximize the reward. In LLM alignment, the reward model is not clearly defined and needs to be inferred from the preference data. 

The size, quality, and distribution of the preference data affects how good we can train a reward model that approximates the ground-truth reward model. If there is a gap between the trained reward model and the ground-truth reward model, the gap will translate to the sub-optimality of the learned policy. 
```


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

(chapter_training_sec_LLM_alignment_PPO_algorithm)=
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


In {numref}`chapter_training_fig_alignment_PPO_training_demo`, we illustrate the basic workflow of a PPO algorithm used to improve the langauge model. A more complete workflow is discussed in {ref}`chapter_training_sec_LLM_alignment_PPO_algorithm_deep_dive`.

```{figure} ../img/chapter_training/alignment/RLHF_PPO/RLHF_PPO_training_demo.png
---
scale: 35%
name: chapter_training_fig_alignment_PPO_training_demo
---
Illustration of PPO based optimization, which uses a Frozen LLM and KL loss to prevent model optimization from going too far and reward model to encourge model to learn to generate highly-rewarded outputs.
```


```{prf:remark} exploitation and exploration aspects from $\gamma$
$\gamma$ controls the balance between exploitation and exploration. When $\gamma \rightarrow 0$, the learning will concentrate on the max reward with full exploitation. When $\gamma \rightarrow \infty$, optimal policy will be the same as $\pi_{\mathrm{sft}}$ with full exploration.
```

(chapter_training_sec_LLM_alignment_PPO_algorithm_deep_dive)=
### PPO Deep Dive

Given the to-be-optimized poliy $\pi(\phi)$ and a reference policy $\pi_{ref}$, the objective function (to be maximized) used to update $\pi$ is given by

$$
\operatorname{Objective}\left(\theta_{\phi}\right)=\min \left( r(y \mid x)A(s, a), \operatorname{Clip}\left(r(y \mid x), 1-\epsilon, 1+\epsilon\right) A(s, a)\right)
$$

Here
* Here $r(y \mid x)$ is the log probability ratio between $\pi_\phi$ and $\pi_{ref}$ on generating $y$ given $x$, which is given by

$$r(y \mid x) = \frac{\pi_\phi(y \mid x)}{\pi_{\theta_{ref}}(y \mid x)}$$ 

* $\epsilon$ and the $\operatorname{Clip}$ are acting to prevent $\pi_\phi$ from going far away from $\pi_{ref}$.

* $A(y, x)$ is a scalar representing the advantage, in terms of reward, of $y$ with respect to other possible $y'$ under policy $\pi_\phi$ (we will discuss it shortly).

The interpretation of the loss function is as follows:
* **Advantage is positive**: The objective function reduces to

$$
\operatorname{Objective}\left(\pi_\phi\right)=\min \left(r(y \mid x),(1+\epsilon)\right) A(s, a)
$$

Because the advantage is positive, the objective will increase if the action $y$ becomes more likely (i.e., if $r(y \mid x) $ increases). Here the min says that if $r(y \mid x) $ already above $1 + \epsilon$, the policy will be not updated. 
* **Advantage is negative**: The objective function reduces to

$$
\operatorname{Objective}\left(\pi_\phi\right)=\max \left(r(y \mid x),(1-\epsilon)\right) A(s, a)
$$

Because the advantage is positive, the objective will improve if the action $y$ becomes less likely (i.e., if $r(y \mid x) $ decrease). Here the min says that if $r(y \mid x) $ already below $1 - \epsilon$, the policy will be not updated. 

A typical implementation of PPO involves **four models**:
* An actor LLM (to be optimized) which specifies the generation policy.
* An frozen actor LLM, which specifies the reference policy.
* A reward model, which rate the final output $y$ given prompt $x$. Reward model is trained before the PPO.
* A value model, which estimates the expected reward at step $t$ by following **current policy**. The input to the value model is $(y_{<t}, x)$.

Given an output $y$ under current policy, the advantage of this output is given $$A(y \mid x) = R(y \mid x) - V^{\pi}(x).$$

A positive advantage indicates $y$ is an output better than average output from current policy and is worth reinforced; a negative advantage indicates $y$ is an output not better than the average, and it needs to be penalized. Note that the value function is a function of current policy, therefore it needs to be updated when the policy is updated.

There are other advanced methods to estimate more fine-grained level advantages on the token-level, as summarized in {cite:p}`zheng2023secrets`. 


```{figure} ../img/chapter_training/alignment/RLHF_PPO/PPO_deep_dive_workflow.png
---
scale: 65%
name: chapter_training_fig_alignment_PPO_deep_dive_workflow
---
Illustration of complete workflow PPO optimization. Image from {cite:p}`zheng2023secrets`
```



<!-- ### The PPO algorithm

There are four models needed in the PPO algorithm:
* Actor LLM 
* Frozen Actor LLM
* Value function model
* Frozen Reward model
  

### PPO implementation



 -->



### Discussion: SFT vs RLHF

% from https://arxiv.org/pdf/2303.18223

SFT adopts a teacher-forcing approach, which directly optimizes the likelihood of a demonstration output. Such a token-level training way essentially does **behavior cloning** to imitate the demonstrations behavior. 

On the other hand, RLHF firstly learns the reward model from preference data, and then employs it to improve the LLM with RL training (e.g., PPO). **The reward model not just encourages positive behavior, but also discourages undesired responses.**

In terms of generating comprehensive demonstration data vs preference labeling, preference labeling is much easier than writing the demonstration data.

RLHF essentially encourages LLMs to learn correct policies by contrasting the self-generated responses (**discriminating between positive and negative responses**). It not just forces the model to imitate external, **positive only** demonstration data but also forces the model to **know when not to imitate** or to know when to reject when there is uncertainty. 
 
These negative penality signal can mitigate the hallucination or over-generalization issues associated with SFT on positive only data. 

```{prf:example} SFT on positive only data can lead to over-generalization
Let's consider how an LLM learns about geographic facts through SFT. During training, it might see examples like:
Training Data:

Q: "What is the capital of France?" \
A: "Paris is the capital of France." \
Q: "What is the capital of Germany?" \
A: "Berlin is the capital of Germany." \

The model learns a simple pattern: "[City] is the capital of [Country]." While this works for legitimate countries, it leads to problematic overgeneralization for fictional/hypothetical places:

Q: "What is the capital of Mars?" \
A: "Olympus City is the capital of Mars." 

RLHF can mitigate the issue by introducing explicit examples of what not to do:

Question: "What is the capital of Mars?" \
✓ Preferred: "Mars is a planet and does not have a capital city. While there are various geographic features on Mars, including Olympus Mons and Valles Marineris, the concept of a capital city applies to political entities on Earth." \
✗ Rejected: "Olympus Mons is the capital of Mars." \
✗ Rejected: "The Mars Colony Capital was established in 2020."

```

Like classic RL algorithms, RLHF has the drawbacks like sample inefficiency, training complexity and instability. When adapted to LLMs, RLHF further relies on a strong SFT model as initial model checkpoint for efficiently achieving good performance

Overall, SFT is particularly useful to increase the model capacity of pre-trained model checkpoints right after pretraining, while RLHF is promising to further improve the model capacity of SFT models.

(chapter_training_sec_LLM_alignment_reward_model_criticality)=
### Discussion: Reward Model Criticality 

For aligning LLM to human preference, **reward modeling plays crucial role of human preferences and set the optimization direction for the model** - if the reward model is built incorrectly or inaccruately, the model is optimized towards the wrong direction. 

Let the groundtruth reward model be $R_{GT}$. In reward modeling, we are training models $R_0$ to approximate $R_{GT}$. The typical reward modeling involves collecting preference label from labeler and build the reward model by learning from preference labels. 
The gap $R_0$ and $R_{GT}$ is affected by the following factors:
* **The preference data quality, quantity, and distribution**; more specifically,
  * label's consistence with human preference
  * more high-quality preference data, the smaller the gap
  * the distribution should be broad and diverse to reflect the richness of the input space 
* **The model's learning capacity** - a weak model cannot capture intricate aspects of human preferences.

Suppose we obtain a reward model $R_0$. What happen as we optimizes the model policy towards the reward model? Optimizing against reward model $R_0$ is supposed to make our policy align with human preferences, i.e., $R_{GT}$. But the $R_0$ is not a perfect representation of our labeler preferences, as it has limited capacity and only sees a limited amount of preference data from a likely narrow distribution of inputs. Studies from {cite:p}`stiennon2020learning,gao2023scaling` show that optimizing towards an imperfect reward model can run into the overfitting risk, leading to a model achieving high score with respect to $R_0$ but actually low score with respect to $R_{GT}$ [{numref}`chapter_training_fig_alignment_reward_model_overfitting`]. To minimize the gap between $R_0$ and $R_{GT}$, they also empirically show that one can enlarge the model size as well as the training data size.

```{figure} ../img/chapter_training/alignment/reward_model/reward_model_overfitting.png
---
scale: 55%
name: chapter_training_fig_alignment_reward_model_overfitting
---
(Left) The overfitting phenomonon of optimizing the model towards an imperfect reward model, leading to a model achieving high score with respect to $R_0$ (dash line) but actually low score with respect to $R_{GT}$ (solid line). Here the KL distance w.r.t. the initial policy is used to measure the degree of over-optimization. (Right) To reduce the gap to $R_{GT}$, one can enlarge model size as well as enlarge preference training data.
```

Studies from {cite:p}`wang2024secrets` further reveals that 
* **The importance of label quality** - incorrect and ambiguous preference pairs in the dataset may hinder the reward
model from accurately capturing human intent. 
* **Poor generation of reward model** - reward models trained on data from a specific distribution often struggle to generalize to examples outside that distribution and are not suitable for iterative RLHF training.

To mitigate these drawbacks, they proposed that
* One can measure the strength of preferences within the data using ensemble reward models.
* Use labeling smoothing to reduce the impact of noisy labels (also discussed in {ref}`chapter_training_sec_LLM_alignment_label_smoothing_DPO`).
* Use adaptive margin, originated from contrastive learning, in training reward model (also discussed in {ref}`chapter_training_sec_LLM_alignment_simple_DPO`).

## RL Variants
(chapter_training_sec_LLM_alignment_GRPO)=
### Group Relative Policy Optimization (GRPO)



DeepSeek team {cite:p}`shao2024deepseekmath` proposed a modified PPO, known as GRPO, to reduce the computation cost and improve the value function estimation for the original PPO algorithm. As shown in {numref}`shao2024deepseekmath`, GRPO does not need an additional value model; baseline score for advantage computation is directly estimated from group scores, significantly reducing training resources.

```{figure} ../img/chapter_training/alignment/RL_variants/GRPO/PPO_vs_GRPO.png
---
scale: 55%
name: chapter_training_fig_alignment_RL_variants_GRPO_vs_PPO
---
Comparison of PPO and GRPO. GRPO does not need an additional value model; baseline score for advantage computation is directly estimated from group scores, significantly reducing training resources. Image from {cite:p}`shao2024deepseekmath`
```

More specifically, the loss function of GRPO is given by

$$\mathcal{J}_{G R P O}(\theta)=\mathbb{E}_{q \sim P(Q),\left\{o_i\right\}_{i=1}^G \sim \pi_{\theta_{o l d}}(O \mid q)} \left[
\frac{1}{G} \sum_{i=1}^G \frac{1}{\left|o_i\right|} \sum_{t=1}^{\left|o_i\right|}\left(\min \left[r_{i,t} \hat{A}_{i, t}, \operatorname{clip}\left(r_{i,t}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i, t}\right]-\beta \mathbb{D}_{K L}\left[\pi_\theta| | \pi_{r e f}\right]\right)\right]
$$ (chapter_training_sec_LLM_alignment_RL_variants_eq:GRPO)

Here
* Question $q$ is sampled from distribution $P(Q)$; $G$ output $\left\{o_i\right\}$ are sampled from the old policy $\pi_{\theta_{o l d}}(O \mid q)$.
* $r_{i,t}$ is the log probability ratio for output $i$ at step $t$, which is given by

$$r_{i,t} = \frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{o l d}}\left(o_{i, t} \mid q, o_{i,<t}\right)}$$

* $\hat{A}_{i, t}$ is the group-estiamted advantage based on reward model $R$ for output $i$ at step $t$. 
  1. If the reward model is an **outcome reward model** that provides rewards only at the end of the each output $o_t$, then

$$\hat{A}_{i, t}= \hat{A}_{i} = \frac{R_{i}-\operatorname{mean}\left(\left\{R_{1}, R_{2}, \cdots, r_{G}\right\}\right)}{\operatorname{std}\left(\left\{R_{1}, R_{2}, \cdots, r_{G}\right\}\right)}$$  

  2. If the reward model is a **process reward model** that provides token-level rewards, then

$$\hat{A}_{i, t}=\frac{R_{i,t}-\operatorname{mean}\left(\left\{R_{1,t}, R_{2,t}, \cdots, r_{G,t}\right\}\right)}{\operatorname{std}\left(\left\{R_{1,t}, R_{2,t}, \cdots, r_{G,t}\right\}\right)}$$  

* The computation of KL divergence is based on an modified [low-variance unbiased estimator](http://joschu.net/blog/kl-approx.html)

$$\mathbb{D}_{K L}\left[\pi_\theta| | \pi_{r e f}\right]_{i,t}=\frac{\pi_{r e f}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}-\log \frac{\pi_{r e f}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}-1$$

When estimating advantages, GRPO directly use sample uses the average reward of multiple sampled outputs, produced in response to the same question, as the baseline.

Removing value function has critical benefits:
* The value function employed in PPO is typically another model of comparable size as
the policy model - training the value function itself brings a substantial computational burden
* In the LLM context, usually only the last token is assigned a reward score by the reward model, such sparse reward also presents challenge to train an accurate value function. 

We summarize the GRPO algorithm in the following.


```{prf:algorithm} Iterative Group Relative Policy Optimization

**Input:**  Initial policy model $\pi_{\text{init}}$; reward models $R$; task prompts $\mathcal{D}$;  Reward model iteration $I$, number of batches $M$, and gradient steps $\mu$.  

**Output:** $\pi_{\theta}$

1. Initialize policy model $\pi_{\theta} \leftarrow \pi_{\text{init}}$
2. **For** iteration = 1, ..., $I$ **do**  
   1. Set reference model $\pi_{\text{ref}} \leftarrow \pi_{\theta}$
   2. **For** step = 1, ..., $M$ **do**  
      1. Sample a batch of tasks $\mathcal{D}_b$ from $\mathcal{D}$  
      2. Update the old policy model $\pi_{\text{old}} \leftarrow \pi_{\theta}$  
      3. Sample $G$ outputs $\{o_i\}_{i=1}^{G} \sim \pi_{\theta}(\cdot | q) $ for each question $ q \in \mathcal{D}_b $  
      4. Compute rewards for each $o_i$
      5. Compute $\hat{A_{i,t}}$ for the $t$-th token of $o_i$ through group relative advantage estimation.  
      6.  **For** GRPO iteration = 1, ..., \( J \) **do**  
          1.  Update the policy model \( \pi_{\theta} \) by maximizing the GRPO objective {eq}``  
   3.  Update reward model $R$ through continuous training using a replay mechanism.  
```

## DPO
### Overview

**DPO (Direct Preference Optimization)** {cite:p}`rafailov2024directpreferenceoptimizationlanguage` improves the classical RLHF-PPO algorithm from the following two aspects:

* **Additional reward model is no longer need**; Instead, the LLM itself can act as a reward model itself; preference data is directly used to train an aligned model in one step.
* **Reinforcement learning no longer need**. Optimizing the policy of an LLM towards a reward model is mathematially equivalent to directly training the LLM as a reward model on the preference data. 
 
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
* **Preference data construction** - sample completions $y_1, y_2 \sim \pi_{\text {ref }}(\cdot \mid x)$ for every prompt $x$, label with human preferences to construct the offline dataset of preferences $\mathcal{D}=\left\{x^{(i)}, y_w^{(i)}, y_l^{(i)}\right\}_{i=1}^N$ 
*  **Optimize the language model as a reward model**, which is equivalent to optimize $\pi_\theta$ to minimize $\mathcal{L}_{\mathrm{DPO}}$ for the given $\pi_{\text {ref }}$ and $\mathcal{D}$ and desired $\beta$. 
   
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

```{prf:remark} Relationship to logistic regression

$$\operatorname{logit} \operatorname{Pr}(i>j)=\log \frac{\operatorname{Pr}(i>j)}{1-\operatorname{Pr}(i>j)}=\log \frac{\operatorname{Pr}(i>j)}{\operatorname{Pr}(j>i)}=\beta_i-\beta_j
$$

```

### Driving the DPO

Here we outline the key steps to derive the DPO objective function, and explain why optimizing LLM as a reward model is equivalent to optimizing its policy. 

First we start with the objective of LLM alignment with a given **fixed reward function** $r$ with a KL constraint, 

$$
\max_\pi \mathbb{E}_{x, y \sim \mathcal{D}_{\pi}}[r(x, y)]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi(y \mid x) \| \pi_{\text {ref }}(y \mid x)\right].
$$

It turns out that we can obtain the theoretical solution of $\pi_r(y|x)$ given by

$$\pi_r(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta}r(x, y)),$$

where $Z(x)$ is partition funciton dependent only on $x$ and $\pi_{\text{ref}}$, which is given by $Z(x)=\sum_y \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)$.


With some algebra, we can also represent the reward funciton with $\pi_r(y|x)$, given by

$$r(x, y) = \beta \log \frac{\pi_r (y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z.$$

```{prf:remark} Implicit reward
Note that here the reward can be interpreted as the log ratio of the likelihood of a response between the current policy model and the reference model. And the policy is the maximizer of the objective function given the reward.
```


Note that we have just shown that that reward function $r(x, y)$ and its corresponding optimal policy $\pi_{\text{ref}}(y|x)$ are inter-convertable, with a funciton $Z(x)$ independent of $y$. 


This means that instead of **numerically optimizing policy $\pi$, we can also choose optimize the reward function.** When the reward function is optimized, the policy is also optimized at the same time (in other words, we can analytically solve the optimal policy).

Given the available preference data, one formulation to optimize the reward function is the Bradley-Terry (BT) objective, that is

$$\mathcal{L}_{BT} = -\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}\left[\log \sigma (r(y_w, x) - r(r_l, x)) \right].$$

By leveraging the relationship between reward $r$ and policy $\pi$, we can arrive at the DPO loss function:

$$
\mathcal{L}_{\mathrm{DPO}}\left(\pi_\theta ; \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta\left(y_w \mid x\right)}{\pi_{\mathrm{ref}}\left(y_w \mid x\right)}-\beta \log \frac{\pi_\theta\left(y_l \mid x\right)}{\pi_{\mathrm{ref}}\left(y_l \mid x\right)}\right)\right].
$$

where the terms $\beta \log Z$ are canceled during subtraction.


```{prf:remark} How DPO loss work

The gradient of DPO loss function is given by:

$$\nabla_\theta \mathcal{L}_{\mathrm{DPO}}\left(\theta, y_w, y_l\right)=-\left(1-\hat{p}_\theta\right) [\underbrace{\nabla_\theta \log \pi_\theta\left(y_w\right)}_{\text {upweight } y_w}-\underbrace{\nabla_\theta \log \pi_\theta\left(y_l\right)}_{\text {downweight } y_l}]
$$

where for the preference completion pair $y_w \succ y_l$, as long as $\hat{p} < 1$, there will gradients to upweight the probability of generating $y_w$ and downweight the probability of generating $y_l$.
```

```{prf:remark} Monitor DPO training process
The DPO algorithm aims to make winning responses have higher probability and losing responses have lower probability. If the training works as expected, beside the overall loss is descreasing, we will see 
*  $\log \frac{\pi_\theta\left(y_w \mid x\right)}{\pi_{\mathrm{ref}}\left(y_w \mid x\right)}$ becomes larger for the same $y_w$.
* $\log \frac{\pi_\theta\left(y_l  \mid x\right)}{\pi_{\mathrm{ref}}\left(y_l \mid x\right)}$ becomes smaller for the same $y_l$.
```


<!-- ### Additional remark RL vs SFT vs DPO

% https://mp.weixin.qq.com/s/WKuEcsyMFkaKf19o20Ci3g -->

### Discussion: DPO vs RL

**DPO and RL-PPO have the same objective - optimizing LLM's generation behavior towards what human prefer.** Since there is no theorectially correct human preference model $R_{GT} $available; instead of, we use colect preference label data $D$ from human labelers to reflect what human prefer. 

In the RL-PPO approach, a proxy reward model $R_0$ is first learned from human-labeled data to approximate $R_{GT}$; then the PPO algorithm is used to optimize the model policy toward $R_0$, with the hope that optimizing towards $R_0$ will more or less optimizing towards $R_{GT}$. As a comparison, instead of learning a reward model, DPO directly optimizes the policy over preference data.

Normally, the finite-sized $D$ cannot cover the whole input-output space, and the proxy reward model $R_0$ often performs poorly in the face of out-ofdistribution data [Also see {ref}`chapter_training_sec_LLM_alignment_reward_model_criticality`]. Studies from {cite:p}`xu2024dpo` show that with an imperfect reward model, the policy $\pi_{\text{DPO}}$ from DPO, which is trained to maximize $y_w$ and minimize $y_l$ probability, **can unexpectedly favor out of distribution responses** (i.e., output $y$ that is different from $y_w$ and $y_l$ ) and lead to unpredictable behaviors.

The reason for the lack of robustness for DPO compared to RL-PPO is that
* DPO is learning from **limited, offline generated** preference data and there is no additional exploration of the input output space during training. The resulting model has a hard time to generate well beyond what is included in the preference data. In the loss function, this is a limited regularization effect on $\pi_{\text{DPO}}(y)$, when $y$ is largely different from $y_w$ or $y_l$ in the training data. 
* RL-PPO is **online** learning with exploration. After obtaining the reward model, RL-PPO does not need offline generated data to train the policy $\pi_{\text{PPO}}$; the LLM agent itself generates outputs and learn from the reward signal from reward model. The self-generation approach enables RL to have in theory **unlimited amount of data to cover much large ranges of input and output distributions**. Even an output $y$ is largely different from $y_w$ or $y_l$ in the training data, such $y$ might be covered in the online exploration process; As a result, at least the KL regularization in the loss function can still properly guide $\pi_{\text{PPO}}(y)$ to not to be far away from $\pi_{\text{ref}}(y)$.  

Nevertheless, from reward modeling perspective, DPO and RL-PPO share the vulnerbility to reward model
* Both DPO and RL-PPO is approximating the ground-truth reward model using limited, offline generated preference data. The size and distribution of preference data affect the gap of trained reward model and the ground-truth reward model. The gap in the reward model will translate to the gap of policy. 
* To overcome the limitation of reward modeling on offline generated data, we can use iterative approach to improve reward modeling (i.e., collecting additional preference data label on DPO or RL-PPO policies for continous reward modeling improvement). 

```{prf:remark} Effective implementing PPO is critical
Although RL-PPO has better robustness to imperfect reward model {cite:p}`xu2024dpo`, an effective implementation of PPO is critical. This involves tricks like advantage normalization, large batch size, and exponential moving average update for the reference model, etc.
```
### Iterative DPO

DPO can also be used iteratively (e.g., 3-5 iterations)to enhance the alignment results. As shown in {numref}`chapter_training_fig_alignmenet_Iterative_DPO`, 
* Starting with a SFT model checkpoint $M_0$, one can go through the DPO data annotation and training process to arrive at $M_1$
* Preference pair data will be collected from $M_1$ and used to train $M_2$.

Iterative DPO has demonstrated its effectiveness in different scenarios {cite:p}`pang2024iterative,dubey2024llama,touvron2023llama2,yuan2024self,chen2024self`.


```{figure} ../img/chapter_training/alignment/DPO/Iterative_DPO.png
---
scale: 45%
name: chapter_training_fig_alignmenet_Iterative_DPO
---
Workflow of iterative DPO Image from {cite:p}`rafailov2024directpreferenceoptimizationlanguage`.
```




## DPO Variants
(chapter_training_sec_LLM_alignment_label_smoothing_DPO)=
### Smoothing preference label

When the preference label is noisy (e.g., annotation error/noise), a more robust DPO approach is desired {cite:p}`Mitchell2023noteondpo`. It assumes that the labels have been flipped with some small probability $\epsilon \in(0,0.5)$. We can use a conservative target distribution instead, $p\left(y_w \succ y_l\right)=1-\epsilon$, giving BCE loss:

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


(chapter_training_sec_LLM_alignment_simple_DPO)=
### Simple DPO

The simple DPO{cite:p}`meng2024simposimplepreferenceoptimization` improve the original DPO from two aspects:
* Make the sequence likelihood function used in implicit reward be aligned with the likelihood of actual sequence decoding. 
* Add a margin to encourage larger reward gap between positive sequence and negative sequence, which will help generalzation. 

First, the authors argue that original DPO derives the closed form implicit reward as the log ratio of the likelihood of a response between the current policy model and the reference model plus a constant only depending on $x$

$$r(x, y)=\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + Const(x).$$

In actual decoding process, the likelihood of a response is usually **length averaged** (see {ref}`chapter_inference_sec_deconding_beam_search`), for example,

$$p_\theta(y \mid x)=\frac{1}{|y|} \log \pi_\theta(y \mid x)=\frac{1}{|y|} \sum_{i=1}^{|y|} \log \pi_\theta\left(y_i \mid x, y_{<i}\right)$$

Naturally, we consider replacing the reward formulation in DPO with $p_\theta$ above, so that it aligns with the likehood metric that guides generation. This results in a length-normalized reward:

$$
r_{\mathrm{SimPO}}(x, y)=\frac{\beta}{|y|} \log \pi_\theta(y \mid x)=\frac{\beta}{|y|} \sum_{i=1}^{|y|} \log \pi_\theta\left(y_i \mid x, y_{<i}\right)
$$

The author argues that normalizing the reward with response lengths is crucial: **removing the length normalization term from the reward formulation results in a bias toward generating longer but lower-quality sequences.**

Additionally,a target **reward margin term**, $\gamma>0$, is added to the Bradley-Terry objective to ensure that the reward for the winning response, $r\left(x, y_w\right)$, exceeds the reward for the losing response, $r\left(x, y_l\right)$, by at least $\gamma$. This is margin idea is also commonly used in constrastive learning. 

$$
p\left(y_w \succ y_l \mid x\right)=\sigma\left(r\left(x, y_w\right)-r\left(x, y_l\right)-\gamma\right)
$$


Combined these ideas together, we arrive at the SimPO loss function:

$$
\mathcal{L}_{\operatorname{SimPO}}\left(\pi_\theta\right)= -\mathbb{E}\left[\log \sigma\left(\frac{\beta}{\left|y_w\right|} \log \pi_\theta\left(y_w \mid x\right)-\frac{\beta}{\left|y_l\right|} \log \pi_\theta\left(y_l \mid x\right)-\gamma\right)\right]
$$

Note that, unlike the traditional DPO, SimPO does not require a reference model, making it more lightweight and easier to implement.

(chapter_training_sec_LLM_alignment_DPO_variant_DPOP_regularized_DPO)=
### DPO-Positive and Regularized DPO

DPO and its variant usually perform well when the preference paired data consists of strong contrastive pairs, i.e., positive example and negative example are sharply different from edit distance perspective. For these examples, DPO can enhance the probability of generating the positive and reduce the probability of generating the negative. 

However, for paired data that is small edit distance (i.e., positive and negative pairs look similiar), DPO algorithm can lead to **failure mode - that is, both the generating probability of positive and negative example decrease** (although negative ones decrease more).

Authors from {cite:p}`pal2024smaug` not only provides an theoretical understanding of above phenomonon, they will propose one approach to mitigate the failure mode, known as **DPO-Positive** or **DPO-P**.

The key idea is to add a penality term when the model reduces the probability of positive examples. The modified loss function is given by

$$
\mathcal{L}_{\mathrm{DPOP}}\left(\pi_\theta\right)=-\mathbb{E}_{\left(x, y_w, y_l\right) \sim D}\left[\log \sigma \left(\beta \left(\log \frac{\pi_\theta\left(y_w \mid x\right)}{\pi_{\mathrm{ref}}\left(y_w \mid x\right)}\right.\right.\right.  -\log \frac{\pi_\theta\left(y_l \mid x\right)}{\pi_{\mathrm{ref}}\left(y_l \mid x\right)} \\
 \left.\left.\left.-\lambda \cdot \max \left(0, \log \frac{\pi_{\mathrm{ref}}\left(y_w \mid x\right)}{\pi_\theta\left(y_w \mid x\right)}\right)\right)\right)\right]
$$

where $\lambda>0$ is a hyperparameter determining the strength of the penalty. From the Bradley-Terry modeling framework, 
* for the negative $y_l$, the loss function is encourging **minimize** term of

$$
\beta \cdot \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
$$

* for the positive $y_w$, the loss function is encouraging **maximizing** the term of

$$\beta\left[\log \frac{\pi_\theta\left(y_w \mid x\right)}{\pi_{\mathrm{ref}}(y_w \mid x)}-\lambda \cdot \max \left(0, \log \frac{\pi_{\mathrm{ref}}(y_w \mid x)}{\pi_\theta(y_w \mid x)}\right)\right].
$$

Clearly, if we want to maximize these terms for $y_w$, we need to ensure that the generating probability $\pi_{\theta}(y_w|x)$ not to reduce too much from $\pi_{\text{ref}}(y_w|x)$.

On a similar line of thinking, {cite:p}`pang2024iterative` proposed using the negative log likelihood (NLL) loss as a regularizer, which gives the following loss function:

$$
\mathcal{L}_{\mathrm{DPO-R}}\left(\pi_\theta\right)=-\mathbb{E}_{\left(x, y_w, y_l\right) \sim D}\left[\log \sigma \left(\beta \left(\log \frac{\pi_\theta\left(y_w \mid x\right)}{\pi_{\mathrm{ref}}\left(y_w \mid x\right)} -\log \frac{\pi_\theta\left(y_l \mid x\right)}{\pi_{\mathrm{ref}}\left(y_l \mid x\right)}\right)\right) + \frac{\lambda}{|y_w|}\log\left(\pi_\theta(y_w|x)\right) \right]
$$

The authors found that in reasoning tasks, having the NLL loss regularization is critical to promote the likelihood of positive (chosen) sequences [{numref}`chapter_training_fig_alignmenet_DPO_variant_DPO_NLL_regularizer`].


```{figure} ../img/chapter_training/alignment/DPO_variants/DPO_NLL_regularizer.png
---
scale: 55%
name: chapter_training_fig_alignmenet_DPO_variant_DPO_NLL_regularizer
---
Effect on NLL regularizer loss on applying DPO in reasoning tasks. (left) Without regularizer loss, the likelihood of chosen sequences is decreasing when the model is initialized from Llama (left) and a positive-example FT checkpoint (right). After FT, the decreasing of the likelihood is more severe.   Image from {cite:p}`pang2024iterative`.
```

### Cringe Loss

{cite:p}`adolphs2022cringe,xu2023some`


<!-- 提问：DPO的变体有哪些，主要解决DPO的什么问题？

回答：

RSO [1]：由于DPO的蒙特卡洛采样很难达到，所以其实DPO几乎是off-policy的采样方式，RSO主要从DPO的采样方式来解决DPO的问题。
Iterative DPO [2]：同样由于DPO的蒙特卡洛采样很难达到，所以通过on-policy的方式采样来替代off-policy的采样。
IPO [3]：由于BT model的目标是最大化正负response的reward gap，但其实其中忽略了真实情况下我们组的pair可能会有噪音，那么无限去扩大reward gap其实是不准确的，也就是overfit了preference的pair数据，那么解决方案是需要限制这个gap的范围。
DPOP [4]：由于LLM model很难区分编辑距离较小的pair，那么当持续去区分这批case的时候，模型效果会崩塌，现象是正例子和负例子的概率都往下掉。那么DPOP用了一个新项来惩罚正例往下掉的pair，使得正例概率继续提升。
[1] Liu T, Zhao Y, Joshi R, et al. Statistical rejection sampling improves preference optimization[J]. arXiv preprint arXiv:2309.06657, 2023.

[2] Yuan W, Pang R Y, Cho K, et al. Self-rewarding language models[J]. arXiv preprint arXiv:2401.10020, 2024.

[3] Azar M G, Rowland M, Piot B, et al. A general theoretical paradigm to understand learning from human preferences[J]. arXiv preprint arXiv:2310.12036, 2023.

[4] Pal A, Karkhanis D, Dooley S, et al. g: Fixing Failure Modes of Preference Optimisation with DPO-Positive[J]. arXiv preprint arXiv:2402.13228, 2024. -->




## Bibliography

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
```