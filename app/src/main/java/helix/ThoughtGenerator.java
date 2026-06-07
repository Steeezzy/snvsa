package helix;

import java.util.*;

public class ThoughtGenerator {
    private final VSAMemory memory;
    private String lastThought = "...";
    private final Random rng = new Random();

    public ThoughtGenerator(VSAMemory memory) {
        this.memory = memory;
    }

    // given a reasoning HV, generate a thought string
    // samples 3 random concept directions and finds the closest
    public String generate(int[] reasoningHV) {
        if (reasoningHV == null) return "...";

        // find top concept
        String primary = memory.query(reasoningHV)
            .split(" \\(sim=")[0];

        // find a related concept by slightly perturbing the HV
        int[] shifted = shift(reasoningHV, 0.15);
        String related = memory.query(shifted)
            .split(" \\(sim=")[0];

        // find a contrasting concept by negating a portion
        int[] negated = negate(reasoningHV, 0.3);
        String contrast = memory.query(negated)
            .split(" \\(sim=")[0];

        // form a thought sentence
        String[] templates = {
            primary + " relates to " + related,
            related + " near " + primary,
            primary + " not " + contrast,
            "if " + related + " then " + primary,
            primary + " like " + related,
        };

        lastThought = templates[rng.nextInt(templates.length)];
        return lastThought;
    }

    // slightly perturb an HV by flipping some bits
    private int[] shift(int[] hv, double rate) {
        int[] result = hv.clone();
        for (int i = 0; i < result.length; i++) {
            if (rng.nextDouble() < rate) result[i] *= -1;
        }
        return result;
    }

    // negate a portion of an HV
    private int[] negate(int[] hv, double rate) {
        int[] result = hv.clone();
        for (int i = 0; i < (int)(result.length * rate); i++) {
            result[i] *= -1;
        }
        return result;
    }

    public String getLastThought() { return lastThought; }
}
