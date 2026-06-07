package helix;

import java.util.Random;

public class TextEncoder {
    private final VSAMemory memory;
    private final double[] projection;

    public TextEncoder(VSAMemory memory) {
        this.memory = memory;
        this.projection = new double[HypervectorOps.DIM];
        Random rng = new Random(1337);
        double norm = 0;
        for (int i = 0; i < HypervectorOps.DIM; i++) {
            projection[i] = rng.nextGaussian();
            norm += projection[i] * projection[i];
        }
        // normalize
        norm = Math.sqrt(norm);
        for (int i = 0; i < HypervectorOps.DIM; i++) {
            projection[i] /= norm;
        }
    }

    // converts a word to a scalar in [0.1, 1.0] for SNN input
    public double encode(String word) {
        int[] hv = memory.getHV(word.toLowerCase().trim());
        if (hv == null) return 0.5; // unknown word → neutral

        double dot = 0;
        for (int i = 0; i < HypervectorOps.DIM; i++) {
            dot += hv[i] * projection[i];
        }
        // map to [0.1, 1.0]
        return 0.1 + 0.9 * ((Math.tanh(dot / 50.0) + 1.0) / 2.0);
    }
}
