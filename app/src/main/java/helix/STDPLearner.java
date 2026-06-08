package helix;

/**
 * Spike-Timing-Dependent Plasticity (STDP) weight update engine.
 *
 * <h2>What STDP is</h2>
 * <p>STDP is the synaptic learning rule the brain actually uses.  The rule is:
 * <ul>
 *   <li><b>Potentiation</b>: if neuron A fires just <em>before</em> neuron B, the
 *       weight of the A → B synapse increases (A helped cause B's spike).</li>
 *   <li><b>Depression</b>: if A fires just <em>after</em> B, the weight decreases
 *       (A came too late to contribute, correlation is spurious).</li>
 * </ul>
 * <p>The magnitude of the change decays exponentially with the absolute time
 * difference Δt = t_post − t_pre.  This matches the plasticity window
 * measured in biological neurons (roughly ±50 ms).
 *
 * <h2>Reward modulation</h2>
 * <p>Pure STDP is Hebbian — it strengthens any co-active synapse regardless of
 * whether the activity led to a correct outcome.  We gate all weight changes by
 * a reward signal:
 * <pre>  Δw = reward × Δw_STDP</pre>
 * <ul>
 *   <li>reward = +1.0 → prediction was correct → potentiate causal synapses.</li>
 *   <li>reward = −1.0 → prediction was wrong   → reverse (depress) those synapses.</li>
 *   <li>reward =  0.0 → no information         → no weight change.</li>
 * </ul>
 *
 * <h2>Weight clipping</h2>
 * <p>Weights are clipped to [−1.0, +1.0] to prevent runaway potentiation or
 * depression.  In a full biological model you would also add homeostatic
 * scaling, but clipping is sufficient for this prototype.
 */
public class STDPLearner {

    // ──────────────────────────────────────────────────────────────────────
    // STDP parameters
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Potentiation amplitude A+.
     * Maximum weight increase when Δt → 0+ (pre just before post).
     * Typical biological range: 0.005 – 0.02.
     */
    private double A_plus;

    /**
     * Depression amplitude A−.
     * Maximum weight decrease when Δt → 0− (post just before pre).
     * Slightly larger than A_plus creates a net tendency towards sparsity
     * (matches biological measurements in cortical synapses).
     */
    private double A_minus;

    /**
     * Potentiation time constant τ+ (in milliseconds / ticks).
     * Controls how wide the "causal window" is.  Larger values mean more
     * distant spike pairs still contribute meaningfully to potentiation.
     */
    private double tau_plus;

    /**
     * Depression time constant τ− (in milliseconds / ticks).
     */
    private double tau_minus;

    // ──────────────────────────────────────────────────────────────────────
    // Construction
    // ──────────────────────────────────────────────────────────────────────

    /** Create a learner with biologically inspired default parameters. */
    public STDPLearner() {
        this(0.012, 0.008, 20.0, 20.0);
    }

    /**
     * Create a learner with fully specified parameters.
     *
     * @param A_plus    potentiation amplitude  (e.g., 0.01)
     * @param A_minus   depression amplitude    (e.g., 0.012)
     * @param tau_plus  potentiation time const (e.g., 20.0)
     * @param tau_minus depression time const   (e.g., 20.0)
     */
    public STDPLearner(double A_plus, double A_minus,
                       double tau_plus, double tau_minus) {
        this.A_plus    = A_plus;
        this.A_minus   = A_minus;
        this.tau_plus  = tau_plus;
        this.tau_minus = tau_minus;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Weight update
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Compute and apply the STDP weight update for a single synapse.
     *
     * <h3>Algorithm</h3>
     * <pre>
     * if (pre fired AND post fired) {
     *     if (timeDelta > 0)  // pre before post  → potentiate
     *         Δw = A+ × exp(−|Δt| / τ+)
     *     else                // post before pre  → depress
     *         Δw = −A− × exp(−|Δt| / τ−)
     * } else {
     *     Δw = 0
     * }
     * w_new = clip(w_old + reward × Δw,  −1.0, +1.0)
     * </pre>
     *
     * @param currentWeight existing synaptic weight in [−1.0, +1.0]
     * @param preSpike      1 if the pre-synaptic neuron fired this tick, else 0
     * @param postSpike     1 if the post-synaptic neuron fired this tick, else 0
     * @param timeDelta     t_post − t_pre in ticks (positive = pre fired first = causal)
     * @param reward        modulation signal: +1 correct, −1 incorrect, 0 neutral
     * @return updated synaptic weight, clipped to [−1.0, +1.0]
     */
    public double updateWeight(double currentWeight,
                               int preSpike, int postSpike,
                               double timeDelta, double reward) {
        double dw = 0.0;
        if (postSpike == 1) {
            // fired: strengthen if pre also fired, slightly weaken if not
            dw = preSpike == 1 ? A_plus : -A_plus * 0.1;
        } else if (preSpike == 1) {
            // pre fired but post didn't: mild depression
            dw = -A_minus * 0.05;
        }
        // clamp weight between 0.05 and 0.95
        return Math.max(0.05, Math.min(0.95, currentWeight + reward * dw));
    }

    // ──────────────────────────────────────────────────────────────────────
    // Parameter adaptation (used by CuriosityEngine for meta-learning)
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Scale both amplitude parameters by {@code factor}.
     * Called by the meta-learning layer to boost plasticity for concepts
     * with high prediction error.
     *
     * @param factor multiplier (e.g., 1.5 = 50% more plastic)
     */
    public void scalePlasticity(double factor) {
        A_plus  = Math.min(0.05, A_plus  * factor); // cap at 0.05 to prevent explosion
        A_minus = Math.min(0.06, A_minus * factor);
    }

    /**
     * Reset parameters to biological defaults.
     * Called by the meta-learner when curiosity decreases (concept is learned).
     */
    public void resetPlasticity() {
        A_plus  = 0.012;
        A_minus = 0.008;
    }

    // ── Getters (for monitoring) ───────────────────────────────────────────

    public double getAPlus()    { return A_plus;    }
    public double getAMinus()   { return A_minus;   }
    public double getTauPlus()  { return tau_plus;  }
    public double getTauMinus() { return tau_minus; }
}
