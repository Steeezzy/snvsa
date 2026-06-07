package helix;

import java.util.concurrent.StructuredTaskScope;
import java.util.Random;

/**
 * A layer of {@link LIFNeuron}s that all process the same scalar input in parallel.
 *
 * <h2>Why 256 neurons?</h2>
 * <p>A single neuron produces a binary stream that carries almost no information.
 * 256 neurons, each with a slightly different τ, collectively produce a rich
 * population code — a high-dimensional binary vector where each dimension
 * represents a different temporal scale of integration.
 *
 * <h2>Why virtual threads / StructuredTaskScope?</h2>
 * <p>Java 23's {@link StructuredTaskScope} uses virtual threads (Project Loom).
 * Virtual threads are ultra-lightweight coroutines managed by the JVM, not the OS.
 * Launching 256 of them costs almost nothing; they are parked on a small shared
 * carrier pool and only consume CPU when actually computing.  On Apple Silicon's
 * unified memory architecture this maps efficiently to the efficiency cores.
 *
 * <h2>Biological diversity</h2>
 * <p>Neurons in the real brain are not identical. We vary τ linearly across the
 * layer so that neurons span a spectrum from "fast-forgetting" (low τ) to
 * "long-memory" (high τ).  This creates a population that encodes both transient
 * and sustained features of the input signal simultaneously.
 */
public class SNNLayer {

    /** Number of neurons in the layer.  Adjust freely; 256 is a good default. */
    private final int size;

    /** The neuron bank. Index i is a neuron with τ = BASE_TAU + i * TAU_STEP. */
    private final LIFNeuron[] neurons;

    // ── Neuron parameter constants ─────────────────────────────────────────
    private static final double BASE_TAU      = 0.10; // fastest leaker
    private static final double TAU_STEP      = 0.003;// increment per neuron
    private static final double MAX_TAU_DELTA = 0.40; // cap for wrap-around
    private static final double THRESHOLD     = 1.0;
    private static final int    REFRAC_PERIOD = 3;    // ticks of silence after spike

    // ──────────────────────────────────────────────────────────────────────
    // Construction
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Construct a layer of {@code size} LIF neurons with linearly varying τ values.
     *
     * @param size number of neurons (must be > 0)
     */
    public SNNLayer(int size) {
        if (size <= 0) throw new IllegalArgumentException("Layer size must be > 0");
        this.size    = size;
        this.neurons = new LIFNeuron[size];
        Random rng = new Random();
        for (int i = 0; i < size; i++) {
            double tau       = 0.03 + (rng.nextDouble() * 0.08); // 0.03 to 0.11
            double threshold = 0.10 + (rng.nextDouble() * 0.15); // 0.10 to 0.25
            int refrac       = 1 + rng.nextInt(4);               // 1 to 4 ticks
            neurons[i] = new LIFNeuron(tau, threshold, refrac);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Core update
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Advance every neuron by one time step, all in parallel using virtual threads.
     *
     * <p>Uses Java 23's {@code StructuredTaskScope.ShutdownOnFailure}: if any
     * neuron's tick throws, the scope shuts down all forks and re-throws.
     *
     * @param input the shared scalar stimulus for this tick
     * @return int[] of length {@code size}; each element is 0 (no spike) or 1 (spike)
     * @throws Exception if any forked task fails or the calling thread is interrupted
     */
    @SuppressWarnings("preview") // StructuredTaskScope is a preview API in Java 23
    public int[] tick(double input) throws Exception {
        int[] spikes = new int[size];

        try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
            for (int i = 0; i < size; i++) {
                final int idx = i;
                scope.fork(() -> {
                    spikes[idx] = neurons[idx].tick(input);
                    return null;
                });
            }
            scope.join().throwIfFailed();
        }

        return spikes;
    }

    @SuppressWarnings("preview")
    public int[] tick(double input, double[] weights) throws Exception {
        int[] spikes = new int[size];
        try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
            for (int i = 0; i < size; i++) {
                final int idx = i;
                scope.fork(() -> {
                    // weight modulates how strongly each neuron receives input
                    spikes[idx] = neurons[idx].tick(input * weights[idx]);
                    return null;
                });
            }
            scope.join().throwIfFailed();
        }
        return spikes;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Accessors
    // ──────────────────────────────────────────────────────────────────────

    /** @return number of neurons in the layer */
    public int getSize() { return size; }

    /**
     * Expose a neuron by index for inspection (e.g., voltage monitoring).
     *
     * @param idx neuron index in [0, size)
     * @return the LIF neuron at that index
     */
    public LIFNeuron getNeuron(int idx) {
        if (idx < 0 || idx >= size)
            throw new IndexOutOfBoundsException("Neuron index " + idx + " out of range [0, " + size + ")");
        return neurons[idx];
    }

    /**
     * Count the total number of spikes in the current output array.
     * Convenience helper for quick sanity checks.
     *
     * @param spikes output of {@link #tick}
     * @return number of 1s in the array
     */
    public static int spikeCount(int[] spikes) {
        int count = 0;
        for (int s : spikes) count += s;
        return count;
    }
}
