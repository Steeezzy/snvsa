package helix;

/**
 * HELIX-1 main entry point — wires all five phases together.
 *
 * <h2>Architecture overview</h2>
 * <pre>
 *   [Sensor / Sine wave]
 *         │  (scalar input per tick)
 *         ▼
 *   [SNNLayer] ──── 256 LIF neurons (virtual threads, Java 23)
 *         │  (256-element spike train)
 *         ▼
 *   [VSAMemory] ──── spike → 10k-dim hypervector → cosine similarity query
 *         │  (predicted concept label + similarity score)
 *         ▼
 *   [Reward evaluator] ──── is the prediction correct?
 *         │  (reward ∈ [−1, +1])
 *         ├──► [STDPLearner] ──── update synaptic weights
 *         │        ▲
 *         │        │  (plasticity boost)
 *         └──► [CuriosityEngine] ── track error per concept,
 *                                   boost STDP for unknown concepts
 * </pre>
 *
 * <h2>Run</h2>
 * <pre>./gradlew run</pre>
 * You will see a live table printed every 2 seconds showing:
 * <ul>
 *   <li>Each registered concept's prediction error (lower = better known)</li>
 *   <li>The learning boost multiplier (higher = more attention)</li>
 *   <li>Which concept the model is currently most curious about</li>
 * </ul>
 *
 * <h2>Extend</h2>
 * <ul>
 *   <li>Replace {@link SelfLearningLoop#generateInput} with real sensor data.</li>
 *   <li>Replace {@link SelfLearningLoop#evaluateReward} with ground-truth labels.</li>
 *   <li>Add more concepts via {@link VSAMemory#register}.</li>
 *   <li>Increase {@link SNNLayer} size for richer population codes.</li>
 * </ul>
 */
public class Helix {

    // ── Runtime configuration ────────────────────────────────────────────────
    private static final int    SNN_SIZE          = 256;    // neurons in the layer
    private static final long   REPORT_INTERVAL_MS = 2_000L; // status print frequency
    private static final int    NUM_REPORTS        = 10;     // how many reports before exit
    private static final long   TICKS_UNLIMITED    = Long.MAX_VALUE;

    public static void main(String[] args) throws Exception {
        printBanner();

        // ── Phase 2: SNN layer ────────────────────────────────────────────────
        SNNLayer snn = new SNNLayer(SNN_SIZE);
        System.out.println("✓ SNNLayer created with " + SNN_SIZE + " LIF neurons");

        // ── Phase 3: VSA memory ───────────────────────────────────────────────
        VSAMemory memory = new VSAMemory();
        // Register base concepts; add more as needed
        for (String concept : new String[]{"high", "low", "rising", "falling", "stable"}) {
            memory.register(concept);
        }
        System.out.println("✓ VSAMemory initialized with 5 concepts (" +
            HypervectorOps.DIM + "-dim hypervectors)");

        // ── Phase 4: STDP learner ─────────────────────────────────────────────
        STDPLearner stdp = new STDPLearner();
        System.out.println("✓ STDPLearner ready (A+=" + stdp.getAPlus() +
            ", A−=" + stdp.getAMinus() + ", τ+=" + stdp.getTauPlus() + "ms)");

        // ── Phase 5: Curiosity / meta-learning ────────────────────────────────
        CuriosityEngine curiosity = new CuriosityEngine();
        System.out.println("✓ CuriosityEngine ready\n");

        // ── Wire Phase 4 + Phase 5 together via a combined learning loop ──────
        // We subclass SelfLearningLoop to inject the curiosity engine.
        SelfLearningLoop loop = new SelfLearningLoop(snn, memory, stdp) {
            @Override
            protected double evaluateReward(String prediction, double actualInput) {
                double reward = super.evaluateReward(prediction, actualInput);
                // Meta-learning: observe error and apply boost before next update
                curiosity.observe(prediction, reward);
                return reward;
            }
        };

        // ── Start learning on a virtual thread ────────────────────────────────
        Thread learner = Thread.ofVirtual()
            .name("helix-learner")
            .start(loop);

        System.out.println("🧠 HELIX-1 learning loop started — reporting every " +
            REPORT_INTERVAL_MS / 1000 + "s for " + NUM_REPORTS + " cycles\n");

        // ── Monitoring loop ───────────────────────────────────────────────────
        for (int report = 0; report < NUM_REPORTS; report++) {
            Thread.sleep(REPORT_INTERVAL_MS);
            printReport(report, loop, curiosity, stdp);
        }

        // ── Shutdown ──────────────────────────────────────────────────────────
        loop.stop();
        learner.join();
        System.out.println("\n✅ HELIX-1 stopped cleanly.");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Reporting
    // ──────────────────────────────────────────────────────────────────────

    private static void printReport(int n, SelfLearningLoop loop,
                                    CuriosityEngine curiosity, STDPLearner stdp) {
        SelfLearningLoop.LoopState state = loop.getLatestState();
        System.out.println("┌─── Report " + (n + 1) +
            " ─── tick=" + state.tick() +
            " ─── avgReward=" + String.format("%.3f", state.avgReward()) + " ───");

        System.out.println("│  Last prediction : " + state.prediction());
        System.out.println("│  Last input      : " + String.format("%.3f", state.input()));
        System.out.println("│  Last reward     : " + String.format("%.2f", state.reward()));

        // Spike statistics
        int spikeCount = SNNLayer.spikeCount(state.spikes());
        System.out.println("│  Spikes this tick: " + spikeCount +
            " / " + (state.spikes().length > 0 ? state.spikes().length : "?"));

        System.out.println("│  Weight sample   : [" +
            formatWeights(state.weights(), 6) + " ...]");

        System.out.println("│  STDP plasticity : A+=" +
            String.format("%.4f", stdp.getAPlus()) + " A−=" +
            String.format("%.4f", stdp.getAMinus()));

        System.out.println("│  Concept curiosity map:");
        curiosity.getErrorMap().forEach((concept, error) ->
            System.out.printf("│    %-10s error=%.3f  boost=%.2fx  visits=%d%n",
                concept,
                error,
                curiosity.getLearningBoost(concept),
                curiosity.getVisitMap().getOrDefault(concept, 0)));

        System.out.println("│  Most curious: " + curiosity.mostCurious() +
            "    Best known: " + curiosity.bestKnown());
        System.out.println("└──────────────────────────────────────────────────────\n");
    }

    private static String formatWeights(double[] weights, int count) {
        if (weights.length == 0) return "—";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < Math.min(count, weights.length); i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.3f", weights[i]));
        }
        return sb.toString();
    }

    private static void printBanner() {
        System.out.println("""
            ╔══════════════════════════════════════════════════════════╗
            ║  H E L I X - 1   Neuromorphic Self-Learning System      ║
            ║  Java 23 · Vector API (SIMD) · StructuredTaskScope      ║
            ║  LIF neurons → VSA memory → STDP → Curiosity engine     ║
            ╚══════════════════════════════════════════════════════════╝
            """);
    }
}
