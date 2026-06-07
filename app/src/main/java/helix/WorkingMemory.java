package helix;

import java.util.*;

public class WorkingMemory {
    private static final int CAPACITY = 50;
    private final Deque<int[]> buffer = new ArrayDeque<>();
    private int[] bundledState = null;

    // add a new HV to working memory
    public void store(int[] hv) {
        buffer.addFirst(hv.clone());
        if (buffer.size() > CAPACITY) buffer.removeLast();
        rebundle();
    }

    // bundle all stored HVs into one summary vector
    private void rebundle() {
        int[] result = new int[HypervectorOps.DIM];
        for (int[] hv : buffer) {
            result = HypervectorOps.bundle(result, hv);
        }
        bundledState = result;
    }

    // how similar is a new HV to everything we've seen?
    public double recognize(int[] hv) {
        if (bundledState == null) return 0.0;
        return Math.max(0, HypervectorOps.similarity(hv, bundledState));
    }

    // VSA analogy: given A and B, find what relates them
    // bind(A, B) = a new HV encoding "A in context of B"
    public int[] formAnalogy(int[] percept, int[] word) {
        if (percept == null || word == null) return percept;
        return HypervectorOps.bind(percept, word);
    }

    public boolean isEmpty() { return buffer.isEmpty(); }
    public int size() { return buffer.size(); }
}
