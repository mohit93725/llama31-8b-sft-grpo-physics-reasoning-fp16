# Group Relative Policy Optimization (GRPO) Training for Physics Reasoning (TRL)

This project fine-tunes a **4-bit Llama-3.1-8B** base model with **SFT** followed by **GRPO** on a Physics reasoning dataset using [Hugging Face TRL].

**What is GRPO?**  
GRPO is a *critic-free* variant of policy optimization related to PPO. For each prompt, the model generates a **group** of \(G\) completions. Rewards within that group are **centered (and optionally normalized)** to form group-relative advantages. The objective also supports an optional **KL regularizer** to keep the updated policy close to a reference/base policy.

---

## GRPO Loss

Let:
- \(i \in \{1,\dots,G\}\) index the completion **within a group** for the same prompt.  
- \(t\) index **token positions** in a completion.  
- \(R_i\) be the **scalar reward** for completion \(i\).  
- \(\bar{R} = \frac{1}{G}\sum_{j=1}^{G} R_j\) be the group mean reward.  
- The **group-relative advantage**:
  \[
  A_i =
  \begin{cases}
  \dfrac{R_i - \bar{R}}{\max(\operatorname{std}(\mathbf{R}), \epsilon)} & \text{if scale\_rewards=True}\\[6pt]
  R_i - \bar{R} & \text{otherwise}
  \end{cases}
  \]
- The **likelihood ratio** per token:
  \[
  r_{i,t} = \exp\big(\log \pi_\theta(a_{i,t}\!\mid s_{i,t}) - \log \pi_{\text{old}}(a_{i,t}\!\mid s_{i,t})\big)
  \]
- Optional per-token KL against a reference policy \(\pi_{\text{ref}}\):
  \[
  \widehat{D_{\mathrm{KL}}}\!\left(\pi_\theta \Vert \pi_{\text{ref}}\right)
  \approx \exp(\Delta\!\log p) - \Delta\!\log p - 1 \quad \text{with } \Delta\!\log p = \log p_\theta - \log p_{\text{ref}}
  \]

The **GRPO objective** (PPO-style clipping with optional KL) is:
\[
\mathcal{L}_{\mathrm{GRPO}}(\theta)
= -\,\mathbb{E}_{i,t}\!\left[
\min\!\Big(r_{i,t}\,A_i,\; \mathrm{clip}\!\big(r_{i,t},\,1-\epsilon,\,1+\epsilon\big)\,A_i\Big)
\;-\; \beta\,\widehat{D_{\mathrm{KL}}}\!\left(\pi_\theta \Vert \pi_{\text{ref}}\right)
\right]
\]

- \(\epsilon \in [0,0.2]\) is the PPO clipping range (prevents overly large updates).  
- \(\beta \ge 0\) weights the KL regularizer (0 disables it).  
- \(\mathbb{E}_{i,t}\) means **average over all tokens \(t\) of all completions \(i\)** (masking padding).  
- The “Dr.GRPO” variant divides the loss by a **constant length** (e.g., `max_completion_length`) to reduce length bias.

**Mini example (group of 4):** If rewards are \([0.2,\;0.7,\;-0.1,\;0.4]\), the mean is \(0.3\), so advantages are \([-0.1,\;0.4,\;-0.4,\;0.1]\). With `scale_rewards=True`, also divide by the group std.

---

## Why GRPO (in practice)

- **No critic needed:** advantages come from *group-relative* rewards (centered/normalized), avoiding value-function training.  
- **Stable updates:** PPO-style clipping (\(\epsilon\)) and optional KL (\(\beta\)) keep the new policy close to the old/reference policy.  
- **Great for reasoning:** sampling multiple completions per prompt lets your reward target the best reasoning trajectory among candidates.

