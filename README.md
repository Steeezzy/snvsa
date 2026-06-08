# HELIX-1 — Neuromorphic Swarm Intelligence

A self-learning AI system built in Java 23 on Apple Silicon.
No backpropagation. No transformers. Pure spike-timing and hypervectors.

## What it does

5 specialist agents run in parallel on Java virtual threads.
Each agent has its own spiking neural network, world model,
and episodic memory. They communicate through a shared
hypervector vocabulary and a typed message bus.

**Achieved 75.7% success rate on MiniGrid navigation**
via reward-modulated STDP — competitive with standard Q-learning,
using biologically plausible learning rules.

## Architecture

```
PERCEPTION  ← environment observations (MiniGrid)
LANGUAGE    ← continuous word stream via TextEncoder
REASONING   ← VSA analogy chains (bind percept × language HVs)
MEMORY      ← episodic replay of high-reward states  
META        ← adjusts per-agent learning rate multipliers

Shared:
- VSAMemory       50,000 ConceptNet concepts, 10,000-dim hypervectors
- SwarmBus        typed message queues between agents
- WorldModel      1,000-dim compressed state prediction per agent
- CuriosityEngine EMA error + novelty → STDP learning boost
- WorkingMemory   50-HV circular buffer for analogy formation
- EpisodicMemory  200-episode replay buffer
```

## Key results

| Task | Success rate | Random baseline | Ratio |
|------|-------------|-----------------|-------|
| MiniGrid-Empty-5x5-v0 | 75.4% | ~5% | 15× |
| MiniGrid-FourRooms-v0 | 18.2% | ~1% | 18× |

| Metric | Value |
|--------|-------|
| Episodes per 30min run | ~6,000 |
| Concepts in VSA memory | 50,001 |
| Agents | 5 (4 specialist + 1 meta) |
| Neurons per agent | 64 LIF |
| Learning rule | Reward-modulated STDP |
| Backpropagation used | None |

**Transfer learning demonstrated:** curriculum training (Empty-5x5 → Empty-6x6 → Empty-8x8 → FourRooms) with richer observations enables 18× above random on unseen FourRooms layout, vs 1.1% (random) from scratch.

## Core technology

| Feature | Implementation |
|---------|----------------|
| Neurons | Leaky Integrate-and-Fire (LIF), randomised τ/θ |
| Parallelism | Java 23 StructuredTaskScope (virtual threads) |
| Hypervectors | 10,000-dim {+1,-1}, bind/bundle/cosine similarity |
| SIMD | jdk.incubator.vector, IntVector.SPECIES_256 (8× NEON) |
| Learning | Reward-modulated STDP, A+=0.012, A-=0.008 |
| Memory | VSA associative + episodic replay |
| Meta-learning | Curiosity engine + swarm rate adjustment |
| World model | 1,000-dim random projection + linear prediction |

## Requirements

- Java 23 (Vector API + Structured Concurrency preview)
- Python 3.10+ (gymnasium, minigrid)
- Gradle 8.13+
- 4GB RAM minimum (VSA dictionary + world model matrices)

## Setup

```bash
# 1. clone
git clone https://github.com/Steeezzy/snvsa.git && cd snvsa

# 2. python environment
python3 -m venv venv && source venv/bin/activate
pip install gymnasium minigrid

# 3. generate concept dictionary (requires ConceptNet CSV)
python3 load_concepts.py

# 4. run
```

## Run

Three terminals:

```bash
# terminal 1
source venv/bin/activate && python3 env_server.py

# terminal 2  
source venv/bin/activate && python3 text_server.py

# terminal 3
./gradlew run
```

## What you see

```
╔══════════════════════════════════════╗
║  H E L I X - 1   S W A R M  v2.0   ║
╚══════════════════════════════════════╝

┌─── Swarm Report 15 ───
│  Episodes: 312  |  Success rate: 75.7%
│  💭 "motion relates to field"
│
│  PERCEPTION   pred_error=0.734  weight=0.2841
│  LANGUAGE     pred_error=0.698  weight=0.3912
│  REASONING    pred_error=0.712  weight=0.2203
│  MEMORY       pred_error=0.741  weight=0.3456
└────────────────────────────────────
```

## License

MIT