package helix;

import java.util.*;

public class EpisodicMemory {
    private static final int CAPACITY = 200;

    private record Episode(int[] hv, double reward, long tick) {}

    private final Deque<Episode> episodes = new ArrayDeque<>();
    private final Random rng = new Random();

    public void store(int[] hv, double reward, long tick) {
        episodes.addFirst(new Episode(hv.clone(), reward, tick));
        if (episodes.size() > CAPACITY) episodes.removeLast();
    }

    // replay a random high-reward episode, returns its HV
    // returns null if no positive episodes exist
    public int[] replay() {
        List<Episode> positive = episodes.stream()
            .filter(e -> e.reward() > 0)
            .toList();
        if (positive.isEmpty()) return null;
        return positive.get(rng.nextInt(positive.size())).hv();
    }

    // how well does current HV match our best memories?
    public double matchBest(int[] hv) {
        return episodes.stream()
            .filter(e -> e.reward() > 0.3)
            .mapToDouble(e -> HypervectorOps.similarity(hv, e.hv()))
            .max()
            .orElse(0.0);
    }

    public int size() { return episodes.size(); }
}
