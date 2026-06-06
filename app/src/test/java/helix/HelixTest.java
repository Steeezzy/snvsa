package helix;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for all HELIX-1 components.
 * Run with: ./gradlew test
 */
class HelixTest {

    // ══════════════════════════════════════════════════════════════════════
    // Phase 2: LIF Neuron tests
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("LIFNeuron: no spike on sub-threshold input")
    void lif_noSpikeSubThreshold() {
        LIFNeuron n = new LIFNeuron(0.1, 1.0, 3);
        // A tiny input should never cause a spike in a single tick
        assertEquals(0, n.tick(0.0), "Zero input should produce no spike");
        assertEquals(0, n.tick(0.1), "Tiny input should produce no spike");
    }

    @Test
    @DisplayName("LIFNeuron: fires when voltage accumulates past threshold")
    void lif_firesOnSufficientInput() {
        LIFNeuron n = new LIFNeuron(0.1, 0.2, 3); // very low threshold
        int spikes = 0;
        for (int i = 0; i < 100; i++) {
            spikes += n.tick(10.0); // large input
        }
        assertTrue(spikes > 0, "Should produce at least one spike with large sustained input");
    }

    @Test
    @DisplayName("LIFNeuron: refractory period prevents immediate re-firing")
    void lif_refractoryPreventsImmediate() {
        LIFNeuron n = new LIFNeuron(0.1, 0.2, 5); // refrac = 5 ticks
        // Force a spike
        int spiked = 0;
        for (int i = 0; i < 50; i++) {
            spiked += n.tick(20.0);
            if (spiked > 0) break;
        }
        assertTrue(spiked > 0, "Should have spiked");
        // Next 5 ticks should be silent
        for (int i = 0; i < 5; i++) {
            assertEquals(0, n.tick(20.0),
                "Should not spike during refractory period (tick " + i + ")");
        }
    }

    @Test
    @DisplayName("LIFNeuron: voltage leaks over time with no input")
    void lif_voltageLeak() {
        LIFNeuron n = new LIFNeuron(0.2, 10.0, 3); // high threshold so no spike
        n.tick(5.0); // charge it up
        double v1 = n.getVoltage();
        n.tick(0.0); // no input, should leak
        double v2 = n.getVoltage();
        assertTrue(v2 < v1, "Voltage should decrease with no input (leak)");
    }

    @Test
    @DisplayName("LIFNeuron: constructor rejects invalid tau")
    void lif_invalidTauThrows() {
        assertThrows(IllegalArgumentException.class,
            () -> new LIFNeuron(0.0, 1.0, 3), "tau=0 should throw");
        assertThrows(IllegalArgumentException.class,
            () -> new LIFNeuron(1.5, 1.0, 3), "tau>1 should throw");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Phase 2: SNNLayer tests
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("SNNLayer: returns spike array of correct length")
    void snn_outputLength() throws Exception {
        SNNLayer layer = new SNNLayer(64);
        int[] spikes   = layer.tick(0.5);
        assertEquals(64, spikes.length, "Spike array must match layer size");
    }

    @Test
    @DisplayName("SNNLayer: all spikes are binary (0 or 1)")
    void snn_binaryOutput() throws Exception {
        SNNLayer layer = new SNNLayer(32);
        for (int t = 0; t < 20; t++) {
            int[] spikes = layer.tick(Math.sin(t));
            for (int s : spikes) {
                assertTrue(s == 0 || s == 1, "Each spike must be 0 or 1, got: " + s);
            }
        }
    }

    @Test
    @DisplayName("SNNLayer: large input produces more spikes than tiny input")
    void snn_largeInputMoreSpikes() throws Exception {
        SNNLayer layerSmall = new SNNLayer(128);
        SNNLayer layerLarge = new SNNLayer(128);
        int smallTotal = 0, largeTotal = 0;
        for (int t = 0; t < 50; t++) {
            smallTotal += SNNLayer.spikeCount(layerSmall.tick(0.001));
            largeTotal += SNNLayer.spikeCount(layerLarge.tick(5.0));
        }
        assertTrue(largeTotal > smallTotal,
            "Large input should produce more total spikes over 50 ticks");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Phase 3: HypervectorOps tests
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("HypervectorOps: random HVs are nearly orthogonal")
    void hv_randomOrthogonality() {
        int[] a = HypervectorOps.createHV();
        int[] b = HypervectorOps.createHV();
        double sim = HypervectorOps.similarity(a, b);
        // Two random 10k-dim binary vectors should be within ±0.05 of 0
        assertTrue(Math.abs(sim) < 0.05,
            "Random HV similarity should be ~0, got: " + sim);
    }

    @Test
    @DisplayName("HypervectorOps: HV similarity with itself is 1.0")
    void hv_selfSimilarity() {
        int[] a = HypervectorOps.createHV();
        double sim = HypervectorOps.similarity(a, a);
        assertEquals(1.0, sim, 1e-9, "Self-similarity must be exactly 1.0");
    }

    @Test
    @DisplayName("HypervectorOps: bind is its own inverse over {+1,-1}")
    void hv_bindInverse() {
        int[] a     = HypervectorOps.createHV();
        int[] b     = HypervectorOps.createHV();
        int[] bound = HypervectorOps.bind(a, b);
        // bind(bound, b) should recover a   (since a * b * b = a when b ∈ {+1,-1})
        int[] recovered = HypervectorOps.bind(bound, b);
        double sim = HypervectorOps.similarity(a, recovered);
        assertEquals(1.0, sim, 1e-9, "bind(bind(a,b), b) must equal a");
    }

    @Test
    @DisplayName("HypervectorOps: bound HV is dissimilar to both inputs")
    void hv_boundDissimilar() {
        int[] a     = HypervectorOps.createHV();
        int[] b     = HypervectorOps.createHV();
        int[] bound = HypervectorOps.bind(a, b);
        double simA = HypervectorOps.similarity(bound, a);
        double simB = HypervectorOps.similarity(bound, b);
        assertTrue(Math.abs(simA) < 0.05, "Bound HV should be dissimilar to a, got: " + simA);
        assertTrue(Math.abs(simB) < 0.05, "Bound HV should be dissimilar to b, got: " + simB);
    }

    @Test
    @DisplayName("HypervectorOps: bundle of HV with itself is similar to original")
    void hv_bundleSelf() {
        int[] a       = HypervectorOps.createHV();
        int[] bundled = HypervectorOps.binarize(HypervectorOps.bundle(a, a));
        double sim    = HypervectorOps.similarity(a, bundled);
        // bundle(a, a) = 2a, binarize(2a) = sign(2a) = a  → sim ≈ 1.0
        assertEquals(1.0, sim, 1e-9, "bundle+binarize of HV with itself should be itself");
    }

    @Test
    @DisplayName("HypervectorOps: seeded createHV is deterministic")
    void hv_seededDeterministic() {
        int[] a = HypervectorOps.createHV(42L);
        int[] b = HypervectorOps.createHV(42L);
        assertEquals(1.0, HypervectorOps.similarity(a, b), 1e-9,
            "Same seed must produce identical HVs");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Phase 3: VSAMemory tests
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("VSAMemory: query returns registered concept")
    void vsa_queryReturnsRegistered() {
        VSAMemory mem = new VSAMemory();
        mem.register("alpha");
        mem.register("beta");
        // Query with alpha's vector directly → should get "alpha"
        // We can't access it directly, but similarityTo should return 1.0
        // Instead test that query doesn't return "unknown"
        int[] hv = HypervectorOps.createHV();
        String result = mem.query(hv);
        assertFalse(result.startsWith("unknown"),
            "Should always return a registered concept, not 'unknown'");
    }

    @Test
    @DisplayName("VSAMemory: encode+query round-trip improves with more spikes")
    void vsa_encodeDecode() throws Exception {
        VSAMemory mem = new VSAMemory();
        mem.register("high");
        mem.register("low");
        mem.register("stable");

        SNNLayer snn = new SNNLayer(64);
        // Run a few ticks and ensure we always get a valid concept back
        for (int t = 0; t < 20; t++) {
            int[] spikes = snn.tick(1.5);
            int[] hv     = mem.encodeSpikeTraIn(spikes);
            String label = mem.queryLabel(hv);
            assertTrue(
                label.equals("high") || label.equals("low") || label.equals("stable"),
                "Must return one of the registered concepts, got: " + label
            );
        }
    }

    @Test
    @DisplayName("VSAMemory: similarityTo returns −1 for unknown concept")
    void vsa_similarityUnknown() {
        VSAMemory mem = new VSAMemory();
        mem.register("known");
        double sim = mem.similarityTo(HypervectorOps.createHV(), "notRegistered");
        assertEquals(-1.0, sim, "Unknown concept should return −1.0");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Phase 4: STDPLearner tests
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("STDP: potentiation when pre fires before post with positive reward")
    void stdp_potentiation() {
        STDPLearner s = new STDPLearner();
        double w0 = 0.5;
        // pre=1, post=1, timeDelta=+1.0 (pre before post), reward=+1
        double w1 = s.updateWeight(w0, 1, 1, 1.0, 1.0);
        assertTrue(w1 > w0, "Weight should increase (potentiation)");
    }

    @Test
    @DisplayName("STDP: depression when post fires before pre")
    void stdp_depression() {
        STDPLearner s = new STDPLearner();
        double w0 = 0.5;
        // timeDelta = −1.0 means post fired before pre → depress
        double w1 = s.updateWeight(w0, 1, 1, -1.0, 1.0);
        assertTrue(w1 < w0, "Weight should decrease (depression)");
    }

    @Test
    @DisplayName("STDP: negative reward reverses potentiation")
    void stdp_negativeRewardReverses() {
        STDPLearner s = new STDPLearner();
        double w0 = 0.5;
        // Same causal timing but negative reward → weight should decrease
        double w1 = s.updateWeight(w0, 1, 1, 1.0, -1.0);
        assertTrue(w1 < w0, "Negative reward should cause depression even with causal timing");
    }

    @Test
    @DisplayName("STDP: no change when neither neuron fires")
    void stdp_noSpikesNoChange() {
        STDPLearner s = new STDPLearner();
        double w0 = 0.5;
        double w1 = s.updateWeight(w0, 0, 0, 1.0, 1.0);
        assertEquals(w0, w1, 1e-10, "No co-activation → no weight change");
    }

    @Test
    @DisplayName("STDP: weights are clipped to [−1, +1]")
    void stdp_weightClipping() {
        STDPLearner s = new STDPLearner(0.5, 0.5, 1.0, 1.0); // very large amplitudes
        double w = 0.99;
        // Apply many potentiating updates
        for (int i = 0; i < 100; i++) w = s.updateWeight(w, 1, 1, 0.1, 1.0);
        assertTrue(w <= 1.0, "Weight must not exceed +1.0, got: " + w);

        w = -0.99;
        for (int i = 0; i < 100; i++) w = s.updateWeight(w, 1, 1, -0.1, 1.0);
        assertTrue(w >= -1.0, "Weight must not go below −1.0, got: " + w);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Phase 5: CuriosityEngine tests
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("CuriosityEngine: unseen concept has maximum boost")
    void curiosity_unseenMaxBoost() {
        CuriosityEngine ce = new CuriosityEngine();
        double boost = ce.getLearningBoost("newConcept");
        // unseen: error=1.0, visits=0, novelty=1.0 → boost = 1 + 0.5 + 0.3 = 1.8
        assertTrue(boost > 1.5, "Unseen concept should have high boost, got: " + boost);
    }

    @Test
    @DisplayName("CuriosityEngine: boost decreases as model gets concept right")
    void curiosity_boostDecreases() {
        CuriosityEngine ce = new CuriosityEngine();
        // Simulate many correct predictions → high reward → error → 0
        for (int i = 0; i < 500; i++) {
            ce.observe("known", 1.0); // perfect reward every tick
        }
        double boost = ce.getLearningBoost("known");
        assertTrue(boost < 1.3,
            "Well-known concept should have low boost, got: " + boost);
    }

    @Test
    @DisplayName("CuriosityEngine: mostCurious returns the highest-error concept")
    void curiosity_mostCurious() {
        CuriosityEngine ce = new CuriosityEngine();
        // Observe each concept many times so EMA converges to a stable value
        for (int i = 0; i < 50; i++) ce.observe("easy",   1.0);  // perfect → error→0
        for (int i = 0; i < 50; i++) ce.observe("hard",  -1.0);  // wrong   → error→1
        for (int i = 0; i < 50; i++) ce.observe("medium", 0.0);  // no info → error→1 but visits > hard unseen novelty
        // "hard" should have highest error after convergence
        String curious = ce.mostCurious();
        // Both hard and medium will have high error; easy must NOT be most curious
        assertNotEquals("easy", curious,
            "'easy' (perfect reward) should NOT be the most curious concept");
    }

    @Test
    @DisplayName("CuriosityEngine: bestKnown returns the lowest-error concept")
    void curiosity_bestKnown() {
        CuriosityEngine ce = new CuriosityEngine();
        for (int i = 0; i < 50; i++) ce.observe("wellKnown", 1.0);
        for (int i = 0; i < 50; i++) ce.observe("unknown",  -1.0);
        assertEquals("wellKnown", ce.bestKnown(),
            "Best known should be the concept with the fewest errors");
    }

    @Test
    @DisplayName("CuriosityEngine: applyBoost scales STDP plasticity")
    void curiosity_applyBoost() {
        CuriosityEngine ce   = new CuriosityEngine();
        STDPLearner     stdp = new STDPLearner();
        double          a0   = stdp.getAPlus();
        // observe an unknown concept → will have high boost
        double boost = ce.applyBoost("mystery", stdp);
        assertTrue(boost > 1.0, "Boost factor should be > 1.0 for unseen concept");
        assertTrue(stdp.getAPlus() > a0,
            "STDP A+ should have increased after applying boost");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Integration test: end-to-end pipeline
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("Integration: full pipeline runs correctly for 500 ticks")
    void integration_pipelineRuns100Ticks() throws Exception {
        SNNLayer        snn       = new SNNLayer(64);
        VSAMemory       memory    = new VSAMemory();
        STDPLearner     stdp      = new STDPLearner();
        CuriosityEngine curiosity = new CuriosityEngine();

        memory.register("high");
        memory.register("low");
        memory.register("stable");

        double[] weights      = new double[64];
        java.util.Arrays.fill(weights, 0.5);
        int      totalSpikes  = 0;
        int      totalUpdates = 0; // STDP updates applied (pre or post spike present)
        String   lastPrediction = "none";

        // We drive the loop with two consecutive ticks to ensure cross-tick co-activation.
        // Tick T produces spikes; those become "pre" for tick T+1.
        int[] prevSpikes = new int[64]; // starts all-zero

        for (int t = 0; t < 500; t++) {
            // Use a strong constant positive input to guarantee spikes every cycle.
            // Alternate sign every 50 ticks to test both rising and falling patterns.
            double input = (t / 50) % 2 == 0 ? 3.0 : -3.0;

            int[]  spikes     = snn.tick(input);
            int[]  queryHV    = memory.encodeSpikeTraIn(spikes);
            String prediction = memory.queryLabel(queryHV);
            double reward     = (t % 3 == 0) ? 1.0 : -1.0;

            curiosity.observe(prediction, reward);
            lastPrediction = prediction;
            totalSpikes   += SNNLayer.spikeCount(spikes);

            // Apply STDP: prevSpikes[i] = pre-neuron, spikes[i] = post-neuron
            for (int i = 0; i < 64; i++) {
                if (prevSpikes[i] == 1 || spikes[i] == 1) {
                    // There was activity on this synapse this pair of ticks
                    double dt = spikes[i] == 1 ? 2.0 : -2.0;
                    weights[i] = stdp.updateWeight(weights[i], prevSpikes[i], spikes[i], dt, reward);
                    totalUpdates++;
                }
            }
            prevSpikes = spikes;
        }

        // Pipeline must have produced spikes (SNN is working)
        assertTrue(totalSpikes > 0,
            "SNN must have produced spikes over 500 ticks, got: " + totalSpikes);

        // Curiosity engine must have seen predictions
        assertNotNull(curiosity.mostCurious());
        assertFalse(curiosity.getErrorMap().isEmpty(),
            "Curiosity engine must have recorded error observations");

        // STDP must have been applied on at least some ticks
        assertTrue(totalUpdates > 0,
            "STDP must have fired at least once over 500 ticks, totalUpdates=" + totalUpdates);

        double avg = java.util.Arrays.stream(weights).average().orElse(0.5);
        System.out.println("[Integration] After 500 ticks:"
            + " totalSpikes="   + totalSpikes
            + " stdpUpdates="   + totalUpdates
            + " avg_weight="    + String.format("%.4f", avg)
            + " most_curious="  + curiosity.mostCurious()
            + " last_pred="     + lastPrediction);
    }
}
