package helix;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The self-learning loop — HELIX-1's continuous online learning engine.
 *
 * <h2>What it does</h2>
 * <p>This {@link Runnable} runs on a background virtual thread and cycles through
 * the full perception-prediction-learning pipeline every millisecond:
 * <ol>
 *   <li><b>Generate input</b> — sine wave stub (replace with real data stream).</li>
 *   <li><b>Fire the SNN layer</b> — get a 256-element spike train.</li>
 *   <li><b>Encode into VSA</b> — convert spikes to a 10k-dim query hypervector.</li>
 *   <li><b>Predict</b> — nearest-neighbour lookup in {@link VSAMemory}.</li>
 *   <li><b>Evaluate reward</b> — compare prediction to ground truth.</li>
 *   <li><b>Apply STDP</b> — update synaptic weights toward better predictions.</li>
 *   <li><b>Notify observers</b> — publish current state for the curiosity engine.</li>
 * </ol>
 *
 * <h2>Plugging in real data</h2>
 * <p>Override or replace {@link #generateInput(double)} with your actual sensor
 * stream.  Override {@link #evaluateReward(String, double)} with domain-specific
 * ground truth.  Everything else is generic.
 *
 * <h2>Thread safety</h2>
 * <p>The loop runs on its own virtual thread.  External code reads state via
 * {@link #getLatestState()} which returns an immutable snapshot.
 */
public class SelfLearningLoop implements Runnable {

    // ──────────────────────────────────────────────────────────────────────
    // Dependencies
    // ──────────────────────────────────────────────────────────────────────

    private final SNNLayer     snn;
    private final VSAMemory    memory;
    private final STDPLearner  stdp;

    // ──────────────────────────────────────────────────────────────────────
    // State
    // ──────────────────────────────────────────────────────────────────────

    /** Synaptic weights: one per neuron in the SNN layer. */
    private final double[] synapticWeights;

    /** Cumulative reward over the last 1000 ticks (sliding window). */
    private final double[] rewardWindow = new double[1_000];
    private int windowIdx = 0;

    /** Thread-safe snapshot for external observers. */
    private final AtomicReference<LoopState> latestState = new AtomicReference<>(LoopState.INITIAL);

    /** Set to false to request graceful shutdown. */
    private final AtomicBoolean running = new AtomicBoolean(true);

    /** Total ticks elapsed. */
    private long tick = 0L;

    // ──────────────────────────────────────────────────────────────────────
    // Construction
    // ──────────────────────────────────────────────────────────────────────

    /**
     * @param snn    the SNN layer to drive
     * @param memory the VSA memory to query and update
     * @param stdp   the STDP learner for weight updates
     */
    public SelfLearningLoop(SNNLayer snn, VSAMemory memory, STDPLearner stdp) {
        this.snn    = snn;
        this.memory = memory;
        this.stdp   = stdp;

        this.synapticWeights = new double[snn.getSize()];
        Arrays.fill(synapticWeights, 0.5); // start at mid-range
    }

    // ──────────────────────────────────────────────────────────────────────
    // Main loop
    // ──────────────────────────────────────────────────────────────────────

    @Override
    public void run() {
        int[] lastSpikes = new int[snn.getSize()];
        double time      = 0.0;

        while (running.get()) {
            try {
                // ── 1. Generate input ──────────────────────────────────────────
                double input = generateInput(time);

                // ── 2. Fire SNN layer ──────────────────────────────────────────
                int[] spikes = snn.tick(input);

                // ── 3. Encode spike train → VSA hypervector ───────────────────
                int[] queryHV = memory.encodeSpikeTraIn(spikes);

                // ── 4. Predict (nearest-neighbour VSA query) ──────────────────
                String prediction = memory.query(queryHV);
                String label      = extractLabel(prediction);

                // ── 5. Evaluate reward ────────────────────────────────────────
                double reward = evaluateReward(label, input);

                // ── 6. STDP weight update for every neuron ────────────────────
                for (int i = 0; i < spikes.length; i++) {
                    // timeDelta: positive if current neuron fired, negative if it didn't.
                    // This is a simplified per-neuron proxy for Δt_post − Δt_pre.
                    double timeDelta = spikes[i] == 1
                        ? (lastSpikes[i] == 1 ? 1.0 : 2.0)   // both fired or only now
                        : (lastSpikes[i] == 1 ? -2.0 : 0.0); // only last or neither

                    synapticWeights[i] = stdp.updateWeight(
                        synapticWeights[i],
                        lastSpikes[i],
                        spikes[i],
                        timeDelta,
                        reward
                    );
                }

                // ── 7. Track reward & publish state ──────────────────────────
                rewardWindow[windowIdx % rewardWindow.length] = reward;
                windowIdx++;

                latestState.set(new LoopState(
                    tick, input, prediction, reward,
                    averageReward(), spikes.clone(), synapticWeights.clone()
                ));

                lastSpikes = spikes;
                time       += 1.0;  // 1 ms per tick
                tick++;

                Thread.sleep(1); // 1 ms real-time pacing (remove for max speed)

            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                System.err.println("[HELIX] Loop error at tick " + tick + ": " + e.getMessage());
                // Continue running — don't let a single bad tick kill the learner
            }
        }

        System.out.println("[HELIX] Self-learning loop stopped after " + tick + " ticks.");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Pluggable input & reward  (override in subclasses or replace with lambdas)
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Generate a scalar input for this tick.
     *
     * <p>Default: 5 Hz sine wave with amplitude 20.0, tuned to reliably drive spikes
     * in the default SNNLayer (threshold=1.0, tau≈0.1–0.5, input_scale=0.05).
     * The peak voltage increment per tick = 20.0 × 0.05 = 1.0, which equals θ,
     * ensuring neurons at the bottom of their tau range fire on positive half-cycles.
     * Replace this method with your real sensor data.
     *
     * @param time current tick count (used as time in ms)
     * @return scalar stimulus value
     */
    protected double generateInput(double time) {
        return Math.sin(2.0 * Math.PI * 5.0 * time / 1000.0) * 20.0;
    }

    /**
     * Compute a reward signal for the current prediction.
     *
     * <h3>Default reward logic</h3>
     * <ul>
     *   <li>Input > 1.0  and prediction = "high"    → +1.0 (correct)</li>
     *   <li>Input < −1.0 and prediction = "low"     → +1.0 (correct)</li>
     *   <li>Input > 0.3  and prediction = "rising"  → +0.5 (partial credit)</li>
     *   <li>Everything else                          → −0.5 (incorrect)</li>
     * </ul>
     *
     * <p>Replace with a domain-specific evaluator (e.g., compare to labelled
     * ground truth, compare to next-tick actual value, etc.).
     *
     * @param prediction best-matching concept label from VSA query
     * @param actualInput the raw input value this tick
     * @return reward in [−1.0, +1.0]
     */
    protected double evaluateReward(String prediction, double actualInput) {
        if (actualInput > 1.0  && "high".equals(prediction))    return  1.0;
        if (actualInput < -1.0 && "low".equals(prediction))     return  1.0;
        if (actualInput > 0.3  && "rising".equals(prediction))  return  0.5;
        if (actualInput < -0.3 && "falling".equals(prediction)) return  0.5;
        if (Math.abs(actualInput) < 0.2 && "stable".equals(prediction)) return 0.3;
        return -0.5; // wrong concept predicted
    }

    // ──────────────────────────────────────────────────────────────────────
    // Control
    // ──────────────────────────────────────────────────────────────────────

    /** Request graceful shutdown.  The loop will exit after the current tick. */
    public void stop() {
        running.set(false);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Observation API
    // ──────────────────────────────────────────────────────────────────────

    /** @return most-recent loop state snapshot (thread-safe) */
    public LoopState getLatestState() {
        return latestState.get();
    }

    /** @return a copy of current synaptic weights */
    public double[] getWeights() {
        return synapticWeights.clone();
    }

    /** @return total ticks elapsed */
    public long getTick() {
        return tick;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────────────

    private double averageReward() {
        int    filled = (int) Math.min(tick, rewardWindow.length);
        double sum    = 0.0;
        for (int i = 0; i < filled; i++) sum += rewardWindow[i];
        return filled > 0 ? sum / filled : 0.0;
    }

    private static String extractLabel(String queryResult) {
        int idx = queryResult.indexOf(' ');
        return idx >= 0 ? queryResult.substring(0, idx) : queryResult;
    }

    // ──────────────────────────────────────────────────────────────────────
    // State snapshot record
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Immutable snapshot of the loop at one point in time.
     * Published after every tick so external observers (e.g., {@link CuriosityEngine})
     * can read state without locks.
     */
    public record LoopState(
        long     tick,
        double   input,
        String   prediction,
        double   reward,
        double   avgReward,
        int[]    spikes,
        double[] weights
    ) {
        /** Sentinel "not yet started" state. */
        static final LoopState INITIAL = new LoopState(
            -1L, 0.0, "none", 0.0, 0.0, new int[0], new double[0]);
    }
}
