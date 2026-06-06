package helix;

import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Random;

/**
 * SIMD-accelerated hypervector operations for Vector Symbolic Architecture (VSA).
 *
 * <h2>What is VSA?</h2>
 * <p>Every concept (e.g., "high temperature", "rising", "danger") is represented as
 * a random binary vector of {@value #DIM} elements, each element being +1 or -1.
 * In 10,000 dimensions, two independently chosen random vectors have a cosine
 * similarity ≈ 0 — they are nearly orthogonal.  This makes accidental "collision"
 * between concepts astronomically unlikely.
 *
 * <h2>Operations</h2>
 * <ul>
 *   <li><b>Bind</b> (XOR-like): combines two concepts into a new one that is
 *       dissimilar to both parents.  Used to encode "A in context of B".</li>
 *   <li><b>Bundle</b> (sum + sign): merges a set of concepts into their centroid.
 *       Used to encode "A or B or C".  The result is closest to the most
 *       frequent member.</li>
 *   <li><b>Similarity</b> (cosine): measures how closely a query matches a stored
 *       concept.  Returns a value in [−1, +1]; 1.0 = identical, 0.0 = unrelated.</li>
 * </ul>
 *
 * <h2>Why the Java Vector API?</h2>
 * <p>The {@link jdk.incubator.vector} API allows the JVM to emit SIMD instructions
 * directly.  {@code IntVector.SPECIES_256} processes 8 × 32-bit integers per CPU
 * instruction on Apple Silicon (ARM NEON with 256-bit registers).  For 10,000-dim
 * vectors this reduces the loop iteration count from 10,000 to 1,250, giving
 * roughly 8× throughput on XOR and add operations.
 */
public final class HypervectorOps {

    // ──────────────────────────────────────────────────────────────────────
    // Constants
    // ──────────────────────────────────────────────────────────────────────

    /** Dimensionality of every hypervector.  10k dims → near-zero collision probability. */
    public static final int DIM = 10_000;

    /**
     * SIMD vector species: 256-bit register width, 32-bit integer lanes.
     * On Apple Silicon ARM NEON: 8 lanes × 32 bits = 256 bits per SIMD op.
     */
    private static final VectorSpecies<Integer> SPEC = IntVector.SPECIES_256;

    // ──────────────────────────────────────────────────────────────────────
    // Construction (utility class — no instances)
    // ──────────────────────────────────────────────────────────────────────

    private HypervectorOps() {}

    // ──────────────────────────────────────────────────────────────────────
    // Factory
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Generate a random base hypervector for a new concept.
     *
     * <p>Each element is independently drawn from {+1, −1} with equal probability.
     * The resulting vector is the permanent "identity" for that concept.
     *
     * @return a fresh random hypervector of length {@value #DIM}
     */
    public static int[] createHV() {
        int[] hv  = new int[DIM];
        Random rng = new Random();
        for (int i = 0; i < DIM; i++) {
            hv[i] = rng.nextBoolean() ? 1 : -1;
        }
        return hv;
    }

    /**
     * Create a deterministic hypervector seeded by {@code seed}.
     * Useful for reproducible tests.
     *
     * @param seed RNG seed
     * @return seeded random hypervector
     */
    public static int[] createHV(long seed) {
        int[] hv  = new int[DIM];
        Random rng = new Random(seed);
        for (int i = 0; i < DIM; i++) {
            hv[i] = rng.nextBoolean() ? 1 : -1;
        }
        return hv;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Binding  (encodes "A and B together")
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Bind two hypervectors using element-wise multiplication (equivalent to XOR
     * over {+1, −1} since (+1)(+1) = +1 and (+1)(−1) = −1).
     *
     * <p>Properties:
     * <ul>
     *   <li>Result is dissimilar to both inputs (new concept).</li>
     *   <li>Operation is commutative and invertible: bind(bind(a,b), b) ≈ a.</li>
     *   <li>Used to encode key-value pairs or temporal sequences.</li>
     * </ul>
     *
     * <p>Implemented with SIMD integer XOR via the Java Vector API.
     *
     * @param a first hypervector
     * @param b second hypervector
     * @return bound hypervector (length {@value #DIM})
     */
    public static int[] bind(int[] a, int[] b) {
        int[] result    = new int[DIM];
        int   loopBound = SPEC.loopBound(DIM);

        // SIMD fast path — processes SPEC.length() = 8 elements per iteration
        for (int i = 0; i < loopBound; i += SPEC.length()) {
            IntVector.fromArray(SPEC, a, i)
                     .lanewise(VectorOperators.XOR, IntVector.fromArray(SPEC, b, i))
                     .intoArray(result, i);
        }
        // Scalar tail for elements that don't fit a full SIMD register
        for (int i = loopBound; i < DIM; i++) {
            result[i] = a[i] * b[i]; // mul over {+1,-1} is equivalent to XOR
        }
        return result;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Bundling  (encodes "A or B or C present")
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Bundle two hypervectors by element-wise addition (superposition).
     *
     * <p>The result is a "noisy" blend; call {@link #binarize} on the final
     * accumulated sum to snap it back to {+1, −1} for further operations.
     *
     * @param a first hypervector
     * @param b second hypervector
     * @return element-wise sum (integer values, not yet binarized)
     */
    public static int[] bundle(int[] a, int[] b) {
        int[] result    = new int[DIM];
        int   loopBound = SPEC.loopBound(DIM);

        for (int i = 0; i < loopBound; i += SPEC.length()) {
            IntVector.fromArray(SPEC, a, i)
                     .add(IntVector.fromArray(SPEC, b, i))
                     .intoArray(result, i);
        }
        for (int i = loopBound; i < DIM; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    /**
     * Binarize a bundled (summed) hypervector back to {+1, −1}.
     * Elements > 0 → +1; elements ≤ 0 → −1.
     *
     * @param hv accumulated bundle (may contain values outside {+1, −1})
     * @return binarized copy
     */
    public static int[] binarize(int[] hv) {
        int[] result = new int[DIM];
        for (int i = 0; i < DIM; i++) {
            result[i] = hv[i] > 0 ? 1 : -1;
        }
        return result;
    }

    // ──────────────────────────────────────────────────────────────────────
    // Similarity  (cosine distance, range [−1, +1])
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Compute the cosine similarity between two hypervectors.
     *
     * <p>For {+1, −1} vectors the magnitudes are always √DIM, so this reduces
     * to the normalised dot product: {@code dot(a,b) / DIM}.
     *
     * <ul>
     *   <li>1.0  → identical vectors (same concept).</li>
     *   <li>0.0  → orthogonal vectors (unrelated concepts).</li>
     *   <li>−1.0 → opposite vectors (antonyms, if intentionally encoded).</li>
     * </ul>
     *
     * @param a first hypervector
     * @param b second hypervector
     * @return cosine similarity in [−1.0, +1.0]
     */
    public static double similarity(int[] a, int[] b) {
        long dot  = 0L;
        long magA = 0L;
        long magB = 0L;

        for (int i = 0; i < DIM; i++) {
            dot  += (long) a[i] * b[i];
            magA += (long) a[i] * a[i];
            magB += (long) b[i] * b[i];
        }

        if (magA == 0L || magB == 0L) return 0.0; // guard against zero vectors
        return dot / (Math.sqrt(magA) * Math.sqrt(magB));
    }

    // ──────────────────────────────────────────────────────────────────────
    // Utilities
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Create a zero-filled hypervector (useful as an accumulator start).
     *
     * @return int[DIM] all zeros
     */
    public static int[] zeros() {
        return new int[DIM];
    }

    /**
     * Permute a hypervector by rotating it left by {@code shift} positions.
     * Permutation is used to encode temporal order (sequence position).
     *
     * @param hv    input hypervector
     * @param shift number of positions to rotate left
     * @return permuted copy
     */
    public static int[] permute(int[] hv, int shift) {
        int[] result = new int[DIM];
        int   s      = ((shift % DIM) + DIM) % DIM; // handle negative shifts
        System.arraycopy(hv, s,   result, 0,     DIM - s);
        System.arraycopy(hv, 0,   result, DIM - s, s);
        return result;
    }
}
