
# Learning Algorithms
Implementing core ML algorithms from scratch — Decision Trees, Genetic Algorithms, Logistic Regression, Reinforcement Learning, Neural Networks, and more. (no scikit-learn black boxes here).



## Fuzzy gate:
can be place right before our final `Dense` layer, i.e., `... Conv -> Pool -> Conv -> Pool -> Flatten -> FuzzyGate -> Dense(softmax)`. This way it “modulates” the learned features before classification, but does not touch our conv math.

# 1) What it does (high-level).

It computes a **single scalar gate** $\alpha \in (0,1)$ per sample from a few summary stats of the flattened feature vector $f \in \mathbb{R}^F$. Then it scales the features: $f_{\text{out}} = \alpha \cdot f$. Intuition: when features look “salient,” let them pass more strongly; otherwise, damp them a bit. Fully differentiable, few parameters, easy to train.

# 2) Inputs it looks at.

From $f$ it extracts three smooth statistics per sample.

* Mean: $m = \frac{1}{F}\sum_i f_i$.
* Stddev: $s = \sqrt{\frac{1}{F}\sum_i (f_i - m)^2 + \varepsilon}$.
* “Soft” max: $u = \sum_i f_i \cdot \mathrm{softmax}(\tau f)_i$, where $\mathrm{softmax}(\tau f)_i = \frac{e^{\tau f_i}}{\sum_k e^{\tau f_k}}$ and $\tau \approx 5$ gives a smooth max with gradients.

They will be collected into $\mathbf{z} = [m, s, u] \in \mathbb{R}^3$.

# 4) Fuzzy membership functions (Gaussian).

For each of the three stats, define three linguistic terms: Low, Medium, High. Each term has a center $c_{j,t}$ and width $\sigma_{j,t}$ (strictly positive). Membership:

$$
\mu_{j,t}(\,z_j\,) = \exp\!\left(-\tfrac{1}{2}\,\frac{(z_j - c_{j,t})^2}{\sigma_{j,t}^2}\right).
$$

Here $j \in \{m,s,u\}$, $t \in \{\text{Low},\text{Med},\text{High}\}$. That is 9 centers and 9 widths total.

# 5) Rules and firing strengths.

Define a **tiny rule base** using a smooth AND as product (product T-norm). Examples that work well on vision features.

* R1: IF $m$ is High AND $s$ is High THEN gate is High.
* R2: IF $m$ is Low AND $s$ is Low THEN gate is Low.
* R3: IF $u$ is High AND $s$ is Med THEN gate is Med–High.

For rule $r$, the firing strength is $w_r = \prod_{(j,t)\in \text{antecedents}(r)} \mu_{j,t}(z_j)$. we can start with just three rules above to keep it tiny.

# 6) Sugeno-style aggregation to a scalar gate.

Give each rule a learnable consequent $a_r$ which we squish to $(0,1)$ as $\alpha_r = \sigma(a_r)$. Final gate:

$$
\alpha = \frac{\sum_r w_r\,\alpha_r}{\sum_r w_r + \varepsilon}.
$$

I think this is smooth, stable, and easy to backpropagate.

# 7) Apply the gate.

Output $f_{\text{out}} = \alpha \cdot f$. Shape-wise this is just a broadcast multiply per sample. No shape surprises for our `Dense` layer.

# 8) Training and gradients (practical plan).

Two safe phases so we don’t bite off too much at once.

* **Phase A (easy, very stable).** Treat the stats $\mathbf{z}=[m,s,u]$ as “stop-grad” with respect to $f$ initially. That is, gradients flow to $f$ as $\partial L/\partial f = \alpha \cdot \partial L/\partial f_{\text{out}}$. Separately, the scalar path yields $\partial L/\partial \alpha = \sum_i (\partial L/\partial f_{\text{out},i}) \cdot f_i$, which backprops into rule consequents $a_r$ and membership parameters $c,\sigma$. This already learns useful gating without tricky second-order couplings.
* **Phase B (full).** Later, enable gradients from $\alpha$ back into $f$ through $m,s,u$. The derivatives are simple for $m$ and $s$ and well-behaved for the softmax-max $u$. This gives the model more expressive power once Phase A is stable.

# 9) Initialization that “does no harm.”

* Initialize $a_r$ so that $\alpha_r \approx 0.5$ for all rules. Start neutral.
* Set centers $c_{j,t}$ using quick percentiles from a small calibration batch. For example, Low at 25th percentile, Med at 50th, High at 75th for each stat. Set widths $\sigma_{j,t}$ to about one interquartile range so overlaps are smooth. Clamp $\sigma \ge 1e\!-\!3$.
* Use $\varepsilon = 1e\!-\!8$, $\tau \in [3,8]$. These are robust.

# 10) Integration checklist.

* a new layer `FuzzyGate` inheriting `Layer` base. It needs `forward`, `backward`, `params`, `apply_grads`, `has_params`, and `describe`. Implemented optimizer already supports per-layer momentum and weight decay, so it will just work.
* Insert it in `demo.py` as `net = Model([conv1, pool1, conv2, pool2, Flatten(), FuzzyGate(...), dense])`. No other code changes are required.
* Start with Phase A gradients for simplicity. If training is stable, we switch to Phase B.


* **Interpretability.** we can print memberships and rule firings per sample during training to see why the gate opened or closed.
* **Specialization.** we can tailor rules to a specific domain later (e.g., “if energy is low and sparsity high, damp features”).
* **Tiny overhead.** A handful of parameters and very small compute cost relative to convs.

**micro-steps** next:

The draft is for the minimal `FuzzyGate` skeleton that passes inputs through unchanged but logs the three stats, so we can verify shapes and perf impact.
- Then we turn on Phase A gating with fixed neutral parameters.
- Then we make the parameters learnable.
- Finally, we enable full gradients through the stats.
