# Shift triage under language-variety shift: propositions and procedure

This note states the formal results behind the label-efficient shift-triage procedure and proves them. Empirical
references point to the findings of the study; no number below is an experimental result unless it is derived
from a stated formula.

## Setting and notation

- A classifier produces a score vector $s(x) \in \mathbb{R}^K$ (logits) for an input $x$ with label
  $Y \in \{1, \dots, K\}$. Source and target distributions are $P$ and $Q$ over $(X, Y)$.
- A *post-hoc map* is any measurable $g : \mathbb{R}^K \to \Delta^{K-1}$ applied to the score; temperature
  scaling, bias-corrected TS (BCTS), vector, matrix and isotonic scaling, batch calibration with a fixed batch
  and label-shift correction are all post-hoc maps of $s$.
- Log-loss: $\ell(p, y) = -\log p_y$. Target risk of $g$: $R_Q(g) = \mathbb{E}_Q[\ell(g(s(X)), Y)]$.
- $\pi_Q$ is the target class prior, $H_Q(Y)$ its entropy, $I_Q(Y; S)$ the mutual information between the
  label and the score $S = s(X)$ under $Q$.
- $n$ labelled target calibration items $(X_i, Y_i)_{i \le n}$ are exchangeable with a target test item
  $(X_{n+1}, Y_{n+1})$.

## P1. No post-hoc map can repair a score that carries no information about the label

**Proposition 1.** For every post-hoc map $g$,
$$R_Q(g) \;\ge\; H_Q(Y \mid S) \;=\; H_Q(Y) - I_Q(Y; S),$$
with equality if and only if $g(S) = Q(Y \mid S)$ almost surely. In particular, if $Y$ and $S$ are independent
under $Q$, then no post-hoc map has lower target log-loss than the constant predictor $\pi_Q$, and a constant
predictor that uses the source prior $\pi_P$ has excess risk $\mathrm{KL}(\pi_Q \,\|\, \pi_P)$.

*Proof.* Conditionally on $S = s$, the expected loss of the distribution $q = g(s)$ is the cross-entropy
$\mathbb{E}[-\log q_Y \mid S = s] = H(Q(\cdot \mid s)) + \mathrm{KL}(Q(\cdot \mid s) \,\|\, q) \ge
H(Q(\cdot \mid s))$ (Gibbs' inequality), with equality iff $q = Q(\cdot \mid s)$. Taking the expectation over
$S$ gives $R_Q(g) \ge H_Q(Y \mid S)$. Under independence, $Q(\cdot \mid s) = \pi_Q$ for almost every $s$, so the
bound equals $H_Q(Y)$ and is attained by the constant $\pi_Q$; the constant $\pi_P$ has risk
$H_Q(Y) + \mathrm{KL}(\pi_Q \,\|\, \pi_P)$. $\square$

**Corollary 1 (abstention cannot help).** Let $h(S)$ be any classifier that uses the score and $A$ any
acceptance event measurable with respect to $S$ (for example "confidence above a threshold") with $Q(A) > 0$.
If $Y$ and $S$ are independent under $Q$, the selective error satisfies
$$Q(Y \neq h(S) \mid A) \;=\; \mathbb{E}_Q\!\left[1 - \pi_Q(h(S)) \mid A\right] \;\ge\; 1 - \max_k \pi_Q(k),$$
so every coverage level has selective risk at least the majority-class error, and the risk–coverage curve is
flat at the level of a predictor that ignores the input when $h$ always predicts the same class.

*Proof.* Independence gives $Q(Y = h(S) \mid S) = \pi_Q(h(S))$; condition on $A \in \sigma(S)$ and bound
$\pi_Q(h(S)) \le \max_k \pi_Q(k)$. $\square$

**Remarks.**
1. Proposition 1 formalises the "discrimination collapse" reading of the encoders on OffMix-3L:
   with all target labels, TS stays at the level of the constant predictor, the target temperature is unbounded
   (the minimiser of the target likelihood over $T$ moves to $T \to \infty$, i.e. towards a constant), and
   confidence-based abstention leaves the selective risk at the overall error.
2. A ranking AUROC of 0.5 is necessary but not sufficient for independence; the bound is stated in terms of
   $I_Q(Y; S)$, which the procedure below tests through the exact permutation null of no association.
3. The result applies to the score that is recalibrated. It does not apply to a new head fitted on the
   representation or to another system (an LLM), which is why the procedure routes instead of
   recalibrating when the score is uninformative.

## P2. MAP bias-corrected temperature scaling as empirical-Bayes shrinkage, and when it loses to the MLE

The few-label recalibrators fit $\theta = (\log T, b)$ with $z \mapsto z / T + b$ (BCTS; TS fixes $b = 0$) by
$$\hat\theta_{\mathrm{MAP}} = \arg\min_\theta\; n \hat L_n(\theta) + \tfrac12 (\theta - \theta_s)^\top
\Sigma^{-1} (\theta - \theta_s),$$
where $\hat L_n$ is the mean target log-loss on the $n$ labelled items, $\theta_s$ the source estimate and
$\Sigma$ the prior covariance (diagonal: $\sigma_T^2$ for $\log T$, $\sigma_b^2$ for each free bias), set to
the mean squared oracle deviation $\theta_{\mathrm{target}} - \theta_s$ on the *other* task (a moment estimator
under $\theta_{\mathrm{target}} - \theta_s \sim \mathcal N(0, \Sigma)$ across conditions; empirical Bayes in
spirit, but estimated on a different task, so it is a cross-task plug-in rather than an estimate from the
population the target belongs to). $\theta_s$ is treated as fixed; its own sampling error on the source
calibration split is ignored.

**Proposition 2.** Let $\hat\theta_n$ be the target MLE, $F$ the per-item Fisher information at the target
optimum $\theta^*$, and suppose $\hat L_n$ is locally quadratic, $\hat L_n(\theta) \approx \hat L_n(\hat\theta_n) +
\tfrac12 (\theta - \hat\theta_n)^\top F (\theta - \hat\theta_n)$. Then
$$\hat\theta_{\mathrm{MAP}} \approx W_n \hat\theta_n + (I - W_n)\,\theta_s, \qquad W_n = (nF + \Sigma^{-1})^{-1} nF.$$
For scalar TS with prior variance $\sigma^2$ and $\tau^2 = F^{-1}$ (so $\operatorname{Var}\hat\theta_n \approx
\tau^2 / n$), the weight on the data is $w_n = n\sigma^2 / (n\sigma^2 + \tau^2)$, the data and the prior weigh
equally at $n_{1/2} = \tau^2 / \sigma^2$ labels, and with deviation $\delta = \theta^* - \theta_s$,
$$\operatorname{MSE}(\hat\theta_{\mathrm{MAP}}) = (1 - w_n)^2 \delta^2 + w_n^2 \tau^2 / n
\;<\; \operatorname{MSE}(\hat\theta_n) = \tau^2 / n \iff \delta^2 < 2\sigma^2 + \tau^2 / n.$$

*Proof.* Setting the gradient of the quadratic objective to zero gives $nF(\theta - \hat\theta_n) +
\Sigma^{-1}(\theta - \theta_s) = 0$, i.e. the stated convex combination. For the scalar case,
$\hat\theta_{\mathrm{MAP}} - \theta^* = w_n(\hat\theta_n - \theta^*) - (1 - w_n)\delta$ with
$\mathbb E[\hat\theta_n - \theta^*] \approx 0$, which gives the MSE. The inequality
$(1 - w_n)^2 \delta^2 < (1 - w_n^2)\tau^2 / n$ is $\delta^2 < \frac{\tau^2}{n}\frac{1 + w_n}{1 - w_n}$, and
$\frac{1 + w_n}{1 - w_n} = 1 + \frac{2 w_n}{1 - w_n} = 1 + \frac{2 n \sigma^2}{\tau^2}$. $\square$

**Remarks.**
1. The condition $\delta^2 < 2\sigma^2 + \tau^2/n$ is consistent with the empirical pattern: shrinkage wins on
   every dialect and on Chittagong, where the deviation is small relative to the prior scale, and loses to the
   MLE on OffMix-3L, where the deviation is unbounded (Proposition 1). As $n \to \infty$ the condition becomes
   $\delta^2 < 2\sigma^2$: for a shift larger than the prior scale the MAP estimate has the larger MSE at every
   $n$, but the excess $(1 - w_n)^2 \delta^2 = \delta^2 \tau^4 / (n\sigma^2 + \tau^2)^2$ decays as $n^{-2}$ while
   the MLE's MSE decays as $n^{-1}$, so the ratio of the two MSEs tends to 1.
2. BCTS can move the decision threshold (the argmax) through $b$; TS cannot. The bias prior is what makes 10–50
   labels sufficient when the ranking survives.
3. The quadratic approximation is local; at the optimiser bound the curvature vanishes (unbounded optimum) and
   the approximation fails, which is the collapse case of P1.

## P3. A finite-sample test of ranking survival

Let $\hat A = \frac{1}{n_1 n_0} \sum_{i: Y_i = 1} \sum_{j: Y_j = 0} \left(\mathbb 1[S_i > S_j] +
\tfrac12 \mathbb 1[S_i = S_j]\right)$ be the empirical AUROC of a scalar score (P(positive) for two classes) on
the $n$ labelled target items with $n_1$ positives and $n_0$ negatives.

The score function must not depend on the $n$ labelled items (in the procedure it is the source-TS probability,
fitted on the source calibration split; any strictly increasing transform of the positive-class logit margin, such
as BCTS in the two-class case, leaves $\hat A$ unchanged).

**Proposition 3.** (a) *Exact test of no association.* Conditionally on the labels and on the multiset of
scores, if $Y$ and $S$ are independent and the items are exchangeable, every assignment of the observed scores to
the items is equally likely; the permutation (Mann–Whitney) p-value
$p = \frac{1 + \#\{\pi : \hat A^{\pi} \ge \hat A\}}{1 + B}$ (Monte Carlo with $B$ random permutations, or the exact
distribution) satisfies $\Pr(p \le \alpha) \le \alpha$ under the null for every $n$.
(b) *Lower confidence bound.* For any target distribution, with $m = \min(n_1, n_0)$ and conditionally on
$(n_1, n_0)$,
$$\Pr\!\left(A \ge \hat A - \sqrt{\log(1/\delta) / (2m)}\right) \ge 1 - \delta,$$
where $A$ is the population AUROC.

*Proof.* (a) Under exchangeability and independence the joint law of $(S_1, \dots, S_n)$ given the labels is
invariant under permutations, so $\hat A$ and $\hat A^{\pi}$ are exchangeable and the rank of $\hat A$ among them
is (super-)uniform; the add-one estimator is valid (Phipson & Smyth, 2010). (b) $\hat A$ is a two-sample
U-statistic with kernel in $[0, 1]$ and mean $A$; Hoeffding's (1963) bound for U-statistics,
$\Pr(\hat A - A \ge t) \le \exp(-2 m t^2)$, applied to the lower tail gives the bound. $\square$

**Remarks.**
1. The bound is distribution-free but conservative: with $n = 50$ balanced labels ($m = 25$) and $\delta = 0.05$
   the half-width is $\sqrt{\log 20 / 50} \approx 0.245$; with $m = 12$ it is $\approx 0.353$. A retained ranking
   near 0.9 (the LLMs on OffMix-3L) clears 0.5 at $m = 25$; a collapsed one near 0.5 cannot. Part (a)
   is the more powerful decision rule for "collapse vs survival"; part (b) quantifies how much ranking survives.
2. With several candidate systems the tests are combined with Holm's procedure (valid under arbitrary dependence
   between the p-values, which share the labelled items), which controls the family-wise probability of declaring
   a collapsed system "surviving" at level $\delta$ (the Learn-then-Test view of Angelopoulos et al., 2021: each
   system is a configuration, its null is "the configuration is unsafe"). "Collapsed" means exact independence of
   score and label; a weak but non-zero association (AUROC 0.55, say) is not protected against and may survive.
3. Part (a) holds for any statistic, including the macro one-vs-rest AUROC used for three classes; part (b) is
   stated for two classes (for $K$ classes it applies to each one-vs-rest AUROC with $m_k = \min(n_k, n - n_k)$ and
   a union bound over $k$). The implementation uses part (a) only.
4. The labelled items are a random subset of the finite target calibration pool. If the pool is an i.i.d. sample of
   the target, so is the subset, and the bound holds for the population AUROC. No statement is made about the AUROC of
   the finite pool: Hoeffding's (1963) Theorem 4 covers sums, not two-sample U-statistics.

## P4. Coverage and risk control after choosing a system with the same labels

Let $C_j^{\beta}$ be the split-conformal set of candidate $j \in \{1, \dots, M\}$ calibrated at miscoverage
$\beta$ on the $n$ labelled items, and $\hat\jmath$ any data-dependent choice that may use those same items.

The nonconformity score of each candidate must be a fixed function that does not depend on the calibration items
(in the procedure: the source-TS probabilities; a MAP-BCTS map refitted on the same items would break the
exchangeability argument).

**Proposition 4.** (a) *Bonferroni.* With $\beta = \alpha / M$ and $n \ge M/\alpha - 1$ (so that the conformal
quantile is finite), $\Pr\!\left(Y_{n+1} \in C_{\hat\jmath}^{\alpha/M}(X_{n+1})\right) \ge 1 - \alpha$.
(b) *Risk control.* If $\lambda_j$ is the conformal-risk-control threshold of candidate $j$ for a loss
$L_j \in [0, 1]$ at level $\alpha / M$ (Angelopoulos et al., 2024), then the selected candidate satisfies
$\mathbb E[L_{\hat\jmath}(\lambda_{\hat\jmath})] \le \alpha$. For the false-negative rate the loss is
$\mathbb 1[\text{positive item not flagged}]$ and the threshold is calibrated on the positive items only, so the
guarantee is $\Pr(\text{miss} \mid Y_{n+1} = 1) \le \alpha$ under exchangeability of the positive items; a
non-trivial threshold needs $n_1 \ge M/\alpha - 1$ positives (otherwise every item is flagged).
(c) *Sample splitting.* If the choice uses only $n_s$ items and the set (or threshold) of the chosen candidate is
calibrated on the remaining $n_c = n - n_s$ items, the usual guarantees hold at level $\alpha$ with $n_c$ items.

*Proof.* (a) Each $C_j^{\alpha/M}$ has marginal coverage at least $1 - \alpha/M$ by exchangeability. Whatever
$\hat\jmath$ is, $\{Y_{n+1} \notin C_{\hat\jmath}\} \subseteq \bigcup_j \{Y_{n+1} \notin C_j\}$, and the union bound
gives miscoverage at most $\alpha$. The finiteness condition is $\lceil (n+1)(1 - \alpha/M) \rceil \le n$.
(b) Losses are non-negative, so $L_{\hat\jmath} \le \sum_j L_j$ pointwise and
$\mathbb E[L_{\hat\jmath}] \le \sum_j \mathbb E[L_j] \le M \cdot \alpha / M$. (c) The chosen candidate is
independent of the calibration items, so the single-candidate guarantee applies conditionally on the choice.
$\square$

**Remarks.**
1. Reusing the same $n$ items for choice and calibration at level $\alpha$ only guarantees
   $1 - M\alpha$; (a) costs larger sets, (c) costs labels. For $M = 2$, $\alpha = 0.1$, (a) needs $n \ge 19$ for
   the sets and $n_1 \ge 19$ positives for the FNR threshold. With $n = 25$ and a balanced target ($n_1 \approx 12$)
   the FNR threshold is therefore vacuous (flag everything), which is the behaviour reported in the evaluation.
4. *Abstention and conditional validity.* When the same items also decide whether to deploy (survival test of P3
   on all $n$ items, Bonferroni variant), (a) and (b) bound the joint probability $\Pr(\text{miss} \cap
   \text{deploy}) \le \alpha$, so conditionally on deploying the bound is $\alpha / \Pr(\text{deploy})$. In the
   split variant the deploy decision uses only the selection part and the guarantee holds conditionally on
   deploying. The Bonferroni variant abstains rarely at $n \ge 50$, so the gap is small in practice,
   but the stated guarantee must be the marginal one.
2. Selection-adjusted conformal methods (Liang, Zhu & Barber, 2024) can recover efficiency; Bonferroni and
   splitting are the baselines the procedure reports.
3. All guarantees are marginal over the target calibration and test items and assume exchangeability within the
   target domain; they say nothing about other target domains.

## P5. A common location shift of a calibrated score needs a bias, not a temperature

Binary case first. Let $Z = s_1(X) - s_0(X)$ be the log-odds of the classifier after source temperature scaling, with
$\sigma(z) = 1/(1 + e^{-z})$, and write $\operatorname{logit} \pi = \log(\pi / (1 - \pi))$ for a prior $\pi = P(Y = 1)$.

- (A1) *Source calibration:* $P_P(Y = 1 \mid Z = z) = \sigma(z)$ for $P$-almost every $z$.
- (A2) *Common location shift:* there is one $\delta \in \mathbb R$ such that, for $y \in \{0, 1\}$, the law of $Z$
  given $Y = y$ under $Q$ is the law of $Z + \delta$ given $Y = y$ under $P$. The class prior may move from $\pi_P$
  to $\pi_Q$.

Let $\lambda = \operatorname{logit} \pi_Q - \operatorname{logit} \pi_P$ and $\beta = \lambda - \delta$.

**Proposition 5.** Under (A1) and (A2):
(a) $P_Q(Y = 1 \mid Z = z) = \sigma(z + \beta)$. The target-optimal post-hoc map of $Z$ is the bias map
$z \mapsto \sigma(z + \beta)$ at temperature 1.
(b) The ranking is unchanged: $\mathrm{AUROC}_Q(Z) = \mathrm{AUROC}_P(Z)$.
(c) Every temperature map $g_T(z) = \sigma(z / T)$, $T \in (0, \infty)$, predicts $\mathbb 1[z > 0]$, the decisions of
the source classifier, while the Bayes rule predicts $\mathbb 1[z > -\beta]$. If $\beta \neq 0$, $Z$ has no atoms
and $\mathbb E_Q |Z| < \infty$, then $\inf_{T > 0} R_Q(g_T) - R_Q(\sigma(\cdot + \beta)) > 0$, where $R_Q$ is the
target log-loss; the infimum includes the limit $T \to \infty$ (the constant $1/2$).
(d) *Several classes.* If $\operatorname{softmax}(s)$ is calibrated on $P$ and, for every class $y$, the law of the
logit vector $s$ given $Y = y$ under $Q$ is that of $s + d$ given $Y = y$ under $P$ for one $d \in \mathbb R^K$, then
$P_Q(Y = k \mid s) = \operatorname{softmax}_k(s + b)$ with $b_k = \log(\pi_{Q,k} / \pi_{P,k}) - d_k$: a bias vector
at temperature 1. Temperature scaling never changes the predicted class.

*Proof.* (a) Write $f^P_y, f^Q_y$ for the class-conditional densities of $Z$ (with respect to a common dominating
measure; the argument is the same with Radon–Nikodym derivatives). By Bayes' rule and (A1),
$\log f^P_1(u) / f^P_0(u) = \operatorname{logit} P_P(Y = 1 \mid u) - \operatorname{logit} \pi_P = u -
\operatorname{logit} \pi_P$. By (A2), $f^Q_y(z) = f^P_y(z - \delta)$. Hence
$\operatorname{logit} P_Q(Y = 1 \mid z) = \log f^Q_1(z) / f^Q_0(z) + \operatorname{logit} \pi_Q =
(z - \delta) - \operatorname{logit} \pi_P + \operatorname{logit} \pi_Q = z + \beta$.
(b) $\mathrm{AUROC}_Q = \Pr(Z_1 > Z_0)$ for independent $Z_y \sim f^Q_y$; translating both by $\delta$ leaves the
event unchanged, and AUROC does not depend on the class prior.
(c) $\sigma$ is increasing, so $g_T$ predicts class 1 exactly when $z > 0$, for every $T$. For any post-hoc map $g$,
$R_Q(g) - R_Q(\sigma(\cdot + \beta)) = \mathbb E_Q\, \mathrm{KL}(\mathrm{Bern}(\sigma(Z + \beta)) \,\|\,
\mathrm{Bern}(g(Z)))$, because $\sigma(Z + \beta)$ is the true posterior by (a). The integrand vanishes only where
$z + \beta = z / T$, i.e. at no $z$ if $T = 1$ and at the single point $z^* = -\beta / (1 - 1/T)$ otherwise; without
atoms this is a null set, so the excess $\Delta(T)$ is positive for every $T$. For $p \in (0, 1)$ and
$q = \sigma(z / T)$, $\mathrm{KL}(\mathrm{Bern}(p) \,\|\, \mathrm{Bern}(q)) \le \max(-\log q, -\log(1 - q)) =
\log(1 + e^{|z| / T}) \le \log 2 + |z| / T$, so by dominated convergence $\Delta$ is continuous on $(0, \infty)$ and
$\Delta(T) \to \mathbb E_Q\,\mathrm{KL}(\mathrm{Bern}(\sigma(Z + \beta)) \,\|\, \mathrm{Bern}(1/2)) > 0$ as
$T \to \infty$. As $T \to 0$, $\sigma(z / T)$ tends to 0 or 1 for every $z \neq 0$ while $\sigma(z + \beta) \in (0, 1)$,
so the integrand tends to $\infty$ on $\{Z \neq 0\}$ and $\Delta(T) \to \infty$ by Fatou's lemma. A continuous positive
function on $(0, \infty)$ with a positive limit at $\infty$ and an infinite limit at 0 has a positive infimum.
(d) As in (a), calibration gives $\log f^P_k(u) = u_k - \log \pi_{P,k} + h(u)$ with $h$ common to all classes, and
(A2) gives $f^Q_k(s) = f^P_k(s - d)$; the class-independent term cancels in the softmax. A temperature divides every
logit by the same $T > 0$ and keeps their order. $\square$

**Remarks.**
1. *Why this matters for the regime map.* (b) and (c) are the signature of the threshold-shift regime R2: the ranking
   is retained, the predicted label marginal moves, and no temperature changes a decision. A temperature fitted on
   target labels under (A2) is not 1 although the Bayes correction has temperature 1: it rises to hedge the misplaced
   threshold (`src/location_shift_simulation.py`: oracle log T about 1.1 at $|\beta| = 2$ in the Gaussian model).
   A finite oracle temperature ratio therefore does not show a temperature shift; it can be a bias shift read
   through the wrong knob.
2. *Prior change versus score shift.* $\beta = \lambda - \delta$ separates a change of the class prior ($\lambda$,
   classical label shift; Saerens et al., 2002) from a shift of the score itself ($\delta$). EM label-shift correction
   assumes $\delta = 0$ and attributes every change of the predicted marginal to the prior; under $\delta \neq 0$ it
   estimates the wrong prior, consistent with its failure on targets whose ranking survives.
   `src/operating_point_analysis.py` reports this decomposition empirically (oracle bias, its prior part
   $\lambda$ and the residual).
3. *When (A2) fails.* If the classes move by different amounts, the target log-odds are no longer $z + \beta$.
   In the equal-variance Gaussian model they stay linear in $z$ with another slope, so a temperature and a bias
   (BCTS) are both needed and the AUROC changes; the simulation checks this boundary. (A2) can be probed on a
   labelled target sample by the slope that BCTS estimates ($\log \hat T \approx 0$ under (A2)).
4. The result is an elementary consequence of Bayes' rule. Its use here is to name the assumption under which a
   bias term is the right repair and temperature scaling provably is not.

## Shift-triage procedure

Input: candidate systems (source-calibrated encoder runs and LLM systems), an unlabeled target stream, a label
budget $n$, levels $\alpha$ (coverage or FNR) and $\delta$ (decision error).

1. **Label-free screening.** Fit the character n-gram domain classifier (source cal vs unlabeled target cal) and
   compute the encoder's predicted-marginal shift. If neither indicates a shift, deploy the
   encoder with source TS; otherwise request $n$ target labels. Label-free indicators only trigger label
   collection; they never replace the tests below.
2. **Split.** Divide the labelled items into a selection part ($n_s$) and a calibration part ($n_c$), or keep all
   $n$ items and use the Bonferroni variant of P4.
3. **Ranking survival (P3).** For each candidate, test the null of no association between its score and the label
   on the selection part (exact permutation test); combine with Holm at level $\delta$. Candidates whose null is
   rejected "survive".
4. **Abstain under collapse (P1).** If no candidate survives, abstain on the shifted stream (defer to humans); no
   recalibration, conformal set or abstention threshold of these scores can do better than ignoring the input.
5. **Choose.** Among survivors, choose the candidate with the lowest log-loss after MAP BCTS fitted and evaluated
   on the selection part (in-sample penalised log-loss; the guarantees below do not depend on the choice rule).
6. **Recalibrate and certify.** *Split variant:* MAP BCTS of the chosen candidate fitted on the selection part
   gives the point predictions and the score for the conformal set and FNR threshold, calibrated on the
   calibration part at level $\alpha$ (P4 c). *Bonferroni variant:* MAP BCTS refitted on all $n$ items gives the
   point predictions (P2); the conformal set and FNR threshold use the fixed source-TS score, calibrated on all
   $n$ items at level $\alpha / M$ (P4 a, b).

Guarantees: marginal coverage $\ge 1 - \alpha$ or FNR $\le \alpha$ for the deployed system (conditionally on
deploying for the split variant, jointly with the deploy event for the Bonferroni variant; P4 remark 4);
family-wise probability $\le \delta$ of deploying a collapsed candidate; with no surviving candidate, abstention is
optimal among post-hoc uses of the candidates' scores (P1).

Evaluation (on the saved outputs, no GPU): every condition, encoder run and LLM system, $n \in \{25, 50, 100\}$
with the resamples of `src/shift_recalibration.py`; realised coverage, FNR and FPR, the rate of wrong survival
decisions against the full-target AUROC, macro-F1 of the deployed system and the abstention rate; compared with
always-encoder, always-LLM, the reuse-the-same-labels router and the oracle. Step 1 is not part of
the evaluation: every evaluated condition is shifted and detected, so the evaluation starts at step 2.

## Verification record (2026-11-07, Proposition 5)

- Proof re-read once; the first draft bounded the integrand by a term that was not needed and gave the wrong set
  for the $T \to 0$ limit (every $z \neq 0$ contributes, not only the items between the two thresholds); corrected.
- Monte Carlo (`src/location_shift_simulation.py`, `location_shift_simulation.csv`; 236 checks, all hold): under a
  common shift the bias-only fit reaches the Bayes log-loss within 0.005 nats and BCTS keeps $|\log \hat T| \le 0.05$
  for a Gaussian and a skewed non-Gaussian score model; temperature scaling never changes a decision and its excess
  log-loss is positive whenever $\beta \neq 0$; AUROC is unchanged within 0.01; under a differential shift the bias
  alone is not enough and BCTS reaches the Bayes log-loss.
- Not read by someone other than the authors.

## Verification record (2026-11-01)

Second reading of P1–P4 against the implementation (`src/shift_triage.py`, `src/shift_recalibration.py`):
- P1, Corollary 1: the proofs are correct as stated.
- P2: the algebra (stationarity condition, weight $w_n$, MSE and the condition $\delta^2 < 2\sigma^2 + \tau^2/n$)
  was re-derived and checked by Monte Carlo in the Gaussian model (400,000 draws per setting; the sign of
  MSE(MAP) − MSE(MLE) agreed with the condition in every setting, including one next to the boundary). Wording
  corrected: the prior variance is a cross-task plug-in, and the large-$n$ remark now states the decay rates.
- P3: correct; added that the score must not be fitted on the labelled items (the implementation uses source-TS
  probabilities), that the multi-class version uses part (a) only, and the note on the random subset of the pool; the
  half-width for $m = 12$ is 0.353 (was printed as 0.354).
- P4: correct for fixed nonconformity scores. Two gaps in the procedure text were closed so that it matches the
  implementation: the Bonferroni variant certifies the source-TS score (not the refitted MAP-BCTS map) and the
  split variant fits MAP BCTS on the selection part only. Added the positives-only form of the FNR guarantee with
  its minimum of $M/\alpha - 1$ positives (checked by simulation: miss rates 0.049–0.051 at the nominal 0.05 and
  0.096 at 0.1) and the conditional-on-deployment caveat for the Bonferroni variant.
- Procedure step 5 said "cross-fitted"; the implementation (and now the text) uses the in-sample penalised
  log-loss on the selection part. This does not affect any guarantee.
- Still not done: a reading by someone other than the authors.

## References

- Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L. (2021). Learn then Test: Calibrating
  predictive algorithms to achieve risk control. arXiv:2110.01052.
- Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024). Conformal risk control. ICLR.
- Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. JASA 58, 13–30.
- Liang, R., Zhu, W., & Barber, R. F. (2024). Conformal prediction after data-dependent model selection.
  arXiv:2408.07066.
- Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero. SAGMB 9(1).
- Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori
  probabilities: A simple procedure. Neural Computation 14(1), 21–41.
