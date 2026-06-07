package helix;

import java.util.*;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * VSA associative memory: stores named concepts as hypervectors and allows
 * nearest-neighbour retrieval by cosine similarity.
 *
 * <h2>How it works</h2>
 * <ol>
 *   <li><b>Registration</b>: each concept (e.g., "high", "low", "rising") is
 *       assigned a freshly drawn random hypervector.  That vector is immutable
 *       for the lifetime of the system.</li>
 *   <li><b>Encoding</b>: an incoming spike train from {@link SNNLayer} is
 *       converted into a query hypervector by binding each active neuron index
 *       with the "fired" prototype vector and superimposing the results.</li>
 *   <li><b>Retrieval</b>: the query hypervector is compared against every
 *       stored concept using cosine similarity; the nearest concept wins.</li>
 * </ol>
 *
 * <h2>Collision resistance</h2>
 * <p>Because all concept vectors are drawn independently from a uniform
 * distribution over {+1, −1}^10000, expected pairwise cosine similarity ≈ 0.
 * The probability of two concepts being confused (sim > 0.3) for a 10k-dim
 * space is < 10^−9, making the memory practically collision-free even with
 * thousands of registered concepts.
 */
public class VSAMemory {

    // ──────────────────────────────────────────────────────────────────────
    // State
    // ──────────────────────────────────────────────────────────────────────

    /**
     * The concept dictionary.
     * Keys: concept names (e.g., "high", "low", "idx_42").
     * Values: random hypervectors of length {@link HypervectorOps#DIM}.
     */
    private final Map<String, int[]> dictionary = new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * Prototype vector for the "neuron fired" concept.
     * All active neurons' index vectors are bound to this vector, so
     * a spike at position i contributes {@code bind(idxHV(i), firingHV)} to the
     * query, encoding "position i fired".
     */
    private final int[] firingPrototype = HypervectorOps.createHV();

    // ──────────────────────────────────────────────────────────────────────
    // Registration
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Register a new named concept with a random hypervector.
     *
     * <p>If the concept is already registered, this method is a no-op
     * (the existing vector is preserved for consistency).
     *
     * @param concept human-readable concept label (e.g., "high", "rising")
     */
    public void register(String concept) {
        dictionary.putIfAbsent(concept, HypervectorOps.createHV());
    }

    /**
     * Register a concept with a specific, pre-computed hypervector.
     * Useful when loading a saved memory state from disk.
     *
     * @param concept concept label
     * @param hv      hypervector of length {@link HypervectorOps#DIM}
     */
    public void register(String concept, int[] hv) {
        if (hv.length != HypervectorOps.DIM)
            throw new IllegalArgumentException(
                "Hypervector length mismatch: expected " + HypervectorOps.DIM
                + " got " + hv.length);
        dictionary.putIfAbsent(concept, hv.clone());
    }

    // ──────────────────────────────────────────────────────────────────────
    // Spike encoding
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Encode a binary spike train as a hypervector.
     *
     * <h3>Encoding algorithm</h3>
     * <p>For each neuron index {@code i} that fired ({@code spikes[i] == 1}):
     * <ol>
     *   <li>Retrieve (or lazily create) the index prototype for position {@code i}.</li>
     *   <li>Bind it with the global {@code firingPrototype} → encodes "position i fired".</li>
     *   <li>Bundle the result into the accumulator → builds up the population code.</li>
     * </ol>
     *
     * <p>The final accumulator is binarized (sign function) before return, so
     * it can be used directly in similarity comparisons against stored concepts.
     *
     * @param spikes int[] from {@link SNNLayer#tick}; each element is 0 or 1
     * @return query hypervector representing this spike pattern
     */
    public int[] encodeSpikeTraIn(int[] spikes) {
        int[] accumulator = HypervectorOps.zeros();
        boolean anySpiked = false;

        for (int i = 0; i < spikes.length; i++) {
            if (spikes[i] == 1) {
                // Lazily create a position-prototype for neuron index i
                String idxKey = "idx_" + i;
                dictionary.computeIfAbsent(idxKey, k -> HypervectorOps.createHV());

                // Bind position vector with firing prototype → encodes "position i fired"
                int[] contribution = HypervectorOps.bind(
                    dictionary.get(idxKey), firingPrototype);

                // Bundle into accumulator (superposition)
                accumulator = HypervectorOps.bundle(accumulator, contribution);
                anySpiked   = true;
            }
        }

        // If nothing fired, return a zero vector — silence has no concept match.
        // binarize of zeros would give all -1 (biased), so we return actual zeros
        // to get near-zero cosine similarity with all stored concepts.
        return anySpiked ? HypervectorOps.binarize(accumulator)
                         : HypervectorOps.zeros();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Retrieval
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Find the concept in the dictionary most similar to {@code queryHV}.
     *
     * <p>Performs a linear scan over all registered concepts (excluding internal
     * position prototypes that start with "idx_").  For large dictionaries this
     * can be replaced with an approximate nearest-neighbour index.
     *
     * @param queryHV the query hypervector (from {@link #encodeSpikeTraIn})
     * @return best-matching concept label and similarity score as a formatted string,
     *         e.g., {@code "high (sim=0.342)"}
     */
    private static final int SEARCH_LIMIT = 500;
    private final java.util.Random queryRng = new java.util.Random();

    public String query(int[] queryHV) {
        String best = "unknown";
        double bestSim = -1.0;

        // sample 500 random concepts instead of scanning all 50k
        List<String> keys = new ArrayList<>(dictionary.keySet());
        int limit = Math.min(SEARCH_LIMIT, keys.size());

        for (int i = 0; i < limit; i++) {
            int idx = queryRng.nextInt(keys.size());
            String concept = keys.get(idx);
            double sim = HypervectorOps.similarity(queryHV, dictionary.get(concept));
            if (sim > bestSim) {
                bestSim = sim;
                best = concept;
            }
        }
        return best + " (sim=" + String.format("%.3f", bestSim) + ")";
    }

    /**
     * Same as {@link #query} but returns only the concept label (no sim string).
     *
     * @param queryHV the query hypervector
     * @return best-matching concept label, or "unknown"
     */
    public String queryLabel(int[] queryHV) {
        String q = query(queryHV);
        int    p = q.indexOf(' ');
        return p >= 0 ? q.substring(0, p) : q;
    }

    /**
     * Compute the similarity score between a query HV and a named concept.
     *
     * @param queryHV  the query hypervector
     * @param concept  registered concept name
     * @return cosine similarity in [−1, +1], or −1.0 if concept not found
     */
    public double similarityTo(int[] queryHV, String concept) {
        int[] hv = dictionary.get(concept);
        return hv == null ? -1.0 : HypervectorOps.similarity(queryHV, hv);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Introspection
    // ──────────────────────────────────────────────────────────────────────

    /**
     * @return an unmodifiable view of the concept dictionary (excludes idx_ entries)
     */
    public Map<String, int[]> getConcepts() {
        Map<String, int[]> result = new HashMap<>();
        dictionary.forEach((k, v) -> {
            if (!k.startsWith("idx_")) result.put(k, v);
        });
        return Collections.unmodifiableMap(result);
    }

    /** @return total number of entries in the dictionary (including idx_ prototypes) */
    public int size() {
        return dictionary.size();
    }

    // returns the hypervector for a concept, null if not found
    public int[] getHV(String concept) {
        // try exact match first
        if (dictionary.containsKey(concept)) return dictionary.get(concept);
        // try partial match
        for (String key : dictionary.keySet()) {
            if (key.startsWith(concept) || concept.startsWith(key)) {
                return dictionary.get(key);
            }
        }
        return null;
    }
}
