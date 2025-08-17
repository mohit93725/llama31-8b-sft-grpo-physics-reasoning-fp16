# Group Relative Policy Optimization (GRPO) Training for Physics Reasoning (TRL)

This project fine-tunes a **4-bit Llama-3.1-8B** base model with **SFT** followed by **GRPO** on a Physics reasoning dataset using [Hugging Face TRL].

**What is GRPO?**  
GRPO is a *critic-free* variant of policy optimization related to PPO. For each prompt, the model generates a **group** of \(G\) completions. Rewards within that group are **centered** to form group-relative advantages. The objective also supports an optional **KL regularizer** to keep the updated policy close to a base policy.

---

## GRPO Loss

LGRPO​(θ)=−E_i,t​[min(r_i,t​A_i​,clip(r_i,t​,1−ϵ,1+ϵ)A_i​)−β*D_KL

​where, E_i,t: Expectation over completion "i" and token position "t". 
r: Reward value A: Advantage over the base value function 
ϵ: Generally between [0,0.2]. Avoids model to take larger steps 
β: Weight for the KL-divergence 
D_KL​: KL-divergence term.

---

## Why GRPO (in practice)

- **No critic needed:**  
- **Stable updates:** 
- **Great for reasoning:** 

