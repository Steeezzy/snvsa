package helix;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Intrinsic motivation engine — tracks prediction error per concept and
 * boosts STDP plasticity for concepts the model doesn't yet understand.
 *
 * <h2>Why curiosity matters</h2>
 * <p>Without intrinsic motivation, a learning system allocates plasticity
 * uniformly across all inputs.  Concepts it already knows well "waste" weight
 * updates.  The curiosity engine solves this by computing a <em>learning boost
 * multiplier</em> per concept that is proportional to:
 * <ul>
 *   <li><b>Prediction error</b> — how often the model gets this concept wrong.
 *       High error → high curiosity → more STDP updates.</li>
 *   <li><b>Novelty</b> — how rarely this concept has been seen.
 *       Low visit count → extra boost to ensure sparse concepts aren't ignored.</li>
 * </ul>
 *
 * <h2>Error tracking (exponential moving average)</h2>
 * <p>Rather than storing all historical rewards, we maintain an exponential
 * moving average of the error per concept:
 * <pre>  error(t+1) = DECAY × error(t) + (1−DECAY) × error_this_tick</pre>
 * {@code DECAY = 0.95} means recent observations count roughly 20× more than
 * observations 60 ticks ago.  This lets the engine adapt quickly when the model
 * starts improving on a concept.
 *
 * <h2>Boost formula</h2>
 * <pre>  boost = 1.0 + (error × 0.5) + (novelty × 0.3)</pre>
 * where {@code novelty = 1 / (1 + log(1 + visits))}.
 * A never-seen concept gets boost ≈ 1.8.
 * A perfectly-known concept gets boost ≈ 1.0 (neutral).
 * The boost is passed to {@link STDPLearner#scalePlasticity}.
 *
 * <h2>Meta-learning connection</h2>
 * <p>The curiosity engine is the "Phase 5" meta-learner: it doesn't adjust
 * individual synaptic weights (that's Phase 4's job), it adjusts the
 * <em>learning rate itself</em> for each concept.  This is the second level of
 * the learning hierarchy:
 * <pre>
 *   Phase 4 → updates weights W
 *   Phase 5 → updates the STDP amplitudes A+, A− per concept
 * </pre>
 */
public class CuriosityEngine {

    // ──────────────────────────────────────────────────────────────────────
    // State
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Exponential moving average of prediction error per concept.
     * Error is defined as {@code 1 − |reward|} (reward=±1 → error=0, reward=0 → error=1).
     */
    private final Map<String, Double>  errorHistory = new HashMap<>();

    /** Total times each concept has been encountered. */
    private final Map<String, Integer> visitCount   = new HashMap<>();

    /** EMA decay factor: 0.95 gives a ~20-tick effective window. */
    private static final double DECAY          = 0.95;

    /** Weight of the error term in the boost formula. */
    private static final double ERROR_WEIGHT   = 0.50;

    /** Weight of the novelty term in the boost formula. */
    private static final double NOVELTY_WEIGHT = 0.30;

    /** Maximum boost cap — prevents unbounded STDP amplitudes. */
    private static final double MAX_BOOST      = 3.0;

    // ──────────────────────────────────────────────────────────────────────
    // Core update
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Record an observation for a concept after a prediction + reward.
     *
     * <p>Call this every tick after calling {@link VSAMemory#query} and
     * {@link SelfLearningLoop#evaluateReward}.
     *
     * @param concept the predicted concept label (e.g., "high", "low")
     * @param reward  the reward signal for this tick (in [−1, +1])
     */
    public void observe(String concept, double reward) {
        // Prediction error = 1 − |reward|  (perfect reward → zero error)
        double error = 1.0 - Math.abs(reward);

        // Exponential moving average update
        errorHistory.merge(concept, error,
            (oldError, newError) -> oldError * DECAY + newError * (1.0 - DECAY));

        // Increment visit counter
        visitCount.merge(concept, 1, Integer::sum);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Boost computation
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Compute the learning-rate multiplier for a concept.
     *
     * <pre>
     * novelty = 1 / (1 + log(1 + visits))   // decays from 1.0 → 0 as visits grow
     * boost   = 1.0 + error × ERROR_WEIGHT + novelty × NOVELTY_WEIGHT
     * </pre>
     *
     * <p>Pass this multiplier to {@link STDPLearner#scalePlasticity(double)}
     * before applying weight updates for the concept.
     *
     * @param concept concept label
     * @return boost multiplier ≥ 1.0 (capped at {@value #MAX_BOOST})
     */
    public double getLearningBoost(String concept) {
        double error   = errorHistory.getOrDefault(concept, 1.0);       // unseen → max error
        int    visits  = visitCount.getOrDefault(concept, 0);

        // Novelty bonus: large when visits ≈ 0, approaches 0 as visits → ∞
        double novelty = 1.0 / (1.0 + Math.log1p(visits));

        double boost = 1.0 + (error * ERROR_WEIGHT) + (novelty * NOVELTY_WEIGHT);
        return Math.min(MAX_BOOST, boost);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Meta-learning application
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Apply the curiosity boost to the STDP learner for a specific concept.
     *
     * <p>Call this before processing each tick's STDP updates.  The STDP
     * amplitudes are temporarily scaled up for high-curiosity concepts.
     * After updates are applied, you may want to call
     * {@link STDPLearner#resetPlasticity()} to return to baseline.
     *
     * @param concept the concept label being learned this tick
     * @param stdp    the STDP learner to tune
     * @return the boost factor that was applied
     */
    public double applyBoost(String concept, STDPLearner stdp) {
        double boost = getLearningBoost(concept);
        stdp.scalePlasticity(boost);
        return boost;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Introspection
    // ──────────────────────────────────────────────────────────────────────

    /**
     * @return an unmodifiable view of the error EMA per concept.
     *         Concepts with high values are the ones the model is most curious about.
     */
    public Map<String, Double> getErrorMap() {
        return Collections.unmodifiableMap(errorHistory);
    }

    /**
     * @return an unmodifiable view of how many times each concept was seen.
     */
    public Map<String, Integer> getVisitMap() {
        return Collections.unmodifiableMap(visitCount);
    }

    /**
     * Find the concept the engine is currently most curious about.
     *
     * @return concept name with the highest boost, or "none" if empty
     */
    public String mostCurious() {
        return errorHistory.keySet().stream()
            .max((a, b) -> Double.compare(getLearningBoost(a), getLearningBoost(b)))
            .orElse("none");
    }

    /**
     * Find the concept the model knows best (lowest error).
     *
     * @return concept name with the lowest error EMA, or "none" if empty
     */
    public String bestKnown() {
        return errorHistory.entrySet().stream()
            .min(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse("none");
    }

    /** Clear all history (useful for resetting the meta-learner between experiments). */
    public void reset() {
        errorHistory.clear();
        visitCount.clear();
    }
}
