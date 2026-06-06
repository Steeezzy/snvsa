# HELIX-1

A neuromorphic self-learning system built in Java 23.

**LIF neurons → VSA hypervector memory → STDP self-learning → Curiosity-driven meta-learning**

---

## Architecture

```
Sensor input (scalar, 1ms tick)
        │
        ▼
   SNNLayer (256 LIF neurons, Java 23 virtual threads)
        │  ← 256-element binary spike train
        ▼
   VSAMemory  (10,000-dim hypervectors, SIMD Vector API)
        │  ← predicted concept label
        ▼
   SelfLearningLoop  →  STDPLearner  (reward-modulated Hebbian learning)
        │
        └──►  CuriosityEngine  (meta-learning: boost STDP for unknown concepts)
```

## Tech

| Feature | Implementation |
|---|---|
| Neurons | Leaky Integrate-and-Fire (LIF) with refractory period |
| Parallelism | Java 23 `StructuredTaskScope` (virtual threads) |
| Hypervectors | 10,000-dim `{+1, -1}` vectors, bind / bundle / cosine similarity |
| SIMD | `jdk.incubator.vector` — `IntVector.SPECIES_256` (8× ARM NEON throughput) |
| Learning | Reward-modulated Spike-Timing-Dependent Plasticity (STDP) |
| Meta-learning | Curiosity engine: EMA error + novelty → per-concept STDP learning-rate boost |

## Project Structure

```
helix-one/
├── app/build.gradle                        ← Java 23 toolchain + preview flags
└── src/main/java/helix/
    ├── LIFNeuron.java                      ← Phase 2: single LIF neuron
    ├── SNNLayer.java                       ← Phase 2: 256-neuron parallel layer
    ├── HypervectorOps.java                 ← Phase 3: SIMD bind/bundle/similarity
    ├── VSAMemory.java                      ← Phase 3: concept store + spike encoder
    ├── STDPLearner.java                    ← Phase 4: reward-modulated STDP
    ├── SelfLearningLoop.java               ← Phase 4: continuous online learning loop
    ├── CuriosityEngine.java                ← Phase 5: meta-learning engine
    └── Helix.java                          ← Phase 5: main entry point
```

## Requirements

- **Java 23** (uses Vector API incubator + Structured Concurrency preview)
- **Gradle 8.13+**

### macOS (Homebrew)
```bash
brew install openjdk   # installs Java 23
```

Or use SDKMAN:
```bash
curl -s "https://get.sdkman.io" | bash
sdk install java 23-open
```

## Run

```bash
# Compile
JAVA_HOME=/Library/Java/JavaVirtualMachines/openjdk-23.jdk ./gradlew compileJava

# Run tests (28 unit + integration tests)
JAVA_HOME=/Library/Java/JavaVirtualMachines/openjdk-23.jdk ./gradlew test

# Start the live learning demo (prints every 2s for 20s)
JAVA_HOME=/Library/Java/JavaVirtualMachines/openjdk-23.jdk ./gradlew run
```

## What you'll see

```
╔══════════════════════════════════════════════════════════╗
║  H E L I X - 1   Neuromorphic Self-Learning System      ║
║  Java 23 · Vector API (SIMD) · StructuredTaskScope      ║
║  LIF neurons → VSA memory → STDP → Curiosity engine     ║
╚══════════════════════════════════════════════════════════╝

✓ SNNLayer created with 256 LIF neurons
✓ VSAMemory initialized with 5 concepts (10000-dim hypervectors)
✓ STDPLearner ready (A+=0.01, A−=0.012, τ+=20.0ms)
✓ CuriosityEngine ready

┌─── Report 1 ─── tick=426 ─── avgReward=0.183 ───
│  Last prediction : stable (sim=0.017)
│  Last input      : 14.579
│  Spikes this tick: 24 / 256
│  Concept curiosity map:
│    falling    error=0.500  boost=1.34x  visits=9
│    low        error=0.231  boost=1.16x  visits=250
│    stable     error=0.500  boost=1.30x  visits=169
│  Most curious: falling    Best known: low
└──────────────────────────────────────────────────────
```

The error map shows the meta-learner working: **`low` drops from 0.231 → 0.016 over 5000 ticks** as the model learns the concept, while `falling` stays high (rarely seen → high curiosity boost).

## Extending

**Plug in real sensor data:**
```java
SelfLearningLoop loop = new SelfLearningLoop(snn, memory, stdp) {
    @Override
    protected double generateInput(double time) {
        return myRealSensor.readValue();
    }
    @Override
    protected double evaluateReward(String prediction, double actualInput) {
        return groundTruth.matches(prediction) ? 1.0 : -0.5;
    }
};
```

**Add more concepts:**
```java
memory.register("danger");
memory.register("anomaly");
memory.register("normal");
```

## License

MIT
