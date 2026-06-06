package helix;

/**
 * Leaky Integrate-and-Fire (LIF) neuron.
 *
 * <p>Analogous to a bucket with a hole:
 * <ul>
 *   <li>Input pours "water" (charge) in.</li>
 *   <li>The leak drains it at rate τ each tick.</li>
 *   <li>When voltage ≥ θ, the neuron fires a spike (returns 1) and resets.</li>
 *   <li>After firing, the neuron is silent for {@code refracPeriod} ticks
 *       (refractory period), mimicking the biological hyperpolarisation phase.</li>
 * </ul>
 *
 * <p>State update equation per tick:
 * <pre>  v(t+1) = v(t) × (1 − τ)  +  input × scale</pre>
 *
 * @see SNNLayer for running many LIF neurons in parallel.
 */
public class LIFNeuron {

    /** Current membrane voltage (accumulates charge, leaks, resets on spike). */
    private double voltage = 0.0;

    /**
     * Leak rate in (0, 1].
     * <ul>
     *   <li>0.1 → fast leak, neuron forgets quickly, fires only on strong signals.</li>
     *   <li>0.9 → slow leak, neuron integrates over a long window, fires more easily.</li>
     * </ul>
     */
    private final double tau;

    /**
     * Firing threshold θ.
     * When {@code voltage >= threshold} the neuron fires.
     * A value of 1.0 is a sensible default; lower values make neurons more excitable.
     */
    private final double threshold;

    /**
     * Ticks remaining in the refractory period.
     * During this time the neuron cannot fire and its voltage is clamped to 0.
     */
    private int refracTimer = 0;

    /**
     * Total quiet ticks after each spike.
     * Corresponds to the absolute refractory period of a real neuron.
     */
    private final int refracPeriod;

    /**
     * Input scaling factor.
     * Prevents runaway voltage when raw input signals are large.
     */
    private static final double INPUT_SCALE = 0.05;

    // ──────────────────────────────────────────────────────────────────────
    // Construction
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Create a LIF neuron with explicit parameters.
     *
     * @param tau          leak rate; try 0.1–0.5 for most applications
     * @param threshold    spike threshold; 1.0 is the standard starting point
     * @param refracPeriod silent ticks after a spike; 3 is biologically inspired
     */
    public LIFNeuron(double tau, double threshold, int refracPeriod) {
        if (tau <= 0 || tau > 1)
            throw new IllegalArgumentException("tau must be in (0, 1], got: " + tau);
        if (refracPeriod < 0)
            throw new IllegalArgumentException("refracPeriod must be >= 0");

        this.tau          = tau;
        this.threshold    = threshold;
        this.refracPeriod = refracPeriod;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Core update
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Advance the neuron by one discrete time step.
     *
     * <p>Algorithm:
     * <ol>
     *   <li>If in refractory period: decrement timer, clamp voltage to 0, return 0.</li>
     *   <li>Otherwise: leak current voltage, add scaled input.</li>
     *   <li>If voltage crosses threshold: reset, start refractory period, return 1 (spike).</li>
     *   <li>Else: return 0 (no spike).</li>
     * </ol>
     *
     * @param input raw stimulus value (e.g., sensor reading, upstream spike sum)
     * @return 1 if the neuron fired a spike this tick, 0 otherwise
     */
    public int tick(double input) {
        // ── Refractory check ───────────────────────────────────────────
        if (refracTimer > 0) {
            refracTimer--;
            voltage = 0.0;
            return 0; // cannot fire while recovering
        }

        // ── Leak-integrate ─────────────────────────────────────────────
        // Exponential leak: residual voltage decays by factor (1 - tau) each tick.
        // New charge from input is scaled to prevent overflow.
        voltage = voltage * (1.0 - tau) + input * INPUT_SCALE;

        // ── Threshold check ────────────────────────────────────────────
        if (voltage >= threshold) {
            voltage     = 0.0;           // hard reset (not soft reset)
            refracTimer = refracPeriod;  // enter refractory silence
            return 1;                    // 🔥 SPIKE
        }

        return 0; // sub-threshold, no spike
    }

    // ──────────────────────────────────────────────────────────────────────
    // Accessors (useful for monitoring and STDP weight updates)
    // ──────────────────────────────────────────────────────────────────────

    /** @return current membrane voltage (before this tick's update) */
    public double getVoltage()      { return voltage; }

    /** @return leak rate τ */
    public double getTau()          { return tau; }

    /** @return firing threshold θ */
    public double getThreshold()    { return threshold; }

    /** @return remaining refractory ticks (0 means available to fire) */
    public int getRefracTimer()     { return refracTimer; }
}
