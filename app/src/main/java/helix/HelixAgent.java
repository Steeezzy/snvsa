package helix;

import java.util.concurrent.atomic.AtomicBoolean;

public class HelixAgent implements Runnable {
    public final AgentRole role;
    private final SNNLayer snn;
    private final VSAMemory memory;
    private final STDPLearner stdp;
    private final WorldModel worldModel;
    private final CuriosityEngine curiosity;
    private final SwarmBus bus;
    private final AtomicBoolean running = new AtomicBoolean(true);

    private double[] synapticWeights;
    private int[] lastHV = null;
    private double learningRateMultiplier = 1.0;

    private WorkingMemory workingMemory = new WorkingMemory();
    private final EpisodicMemory episodicMemory = new EpisodicMemory();
    private int[] lastPerceptHV = null;  // from PERCEPTION agent via bus
    private int[] lastLanguageHV = null; // from LANGUAGE agent via bus
    private long tick = 0;
    private int successWindowRemaining = 0;

    private volatile String currentWord = "the";
    private TextEncoder textEncoder = null;
    private ThoughtGenerator thoughtGenerator = null;
    private String lastThought = "...";

    // shared input — perception agent writes, others read
    private volatile double currentInput = 0.0;
    private volatile double currentEnvReward = 0.0;

    public HelixAgent(AgentRole role, VSAMemory memory, SwarmBus bus) {
        this.role = role;
        this.memory = memory;
        this.bus = bus;
        this.snn = new SNNLayer(64); // 64 neurons per agent (was 256)
        this.stdp = new STDPLearner();
        this.worldModel = new WorldModel();
        this.curiosity = new CuriosityEngine();
        this.synapticWeights = new double[64];
        java.util.Arrays.fill(synapticWeights, 0.5);
    }

    public void setInput(double input, double envReward) {
        this.currentInput = input;
        this.currentEnvReward = envReward;
    }

    public void setTextInput(String word) {
        this.currentWord = word;
    }

    public void initTextEncoder(VSAMemory memory) {
        this.textEncoder = new TextEncoder(memory);
    }

    public void initThoughtGenerator(VSAMemory memory) {
        this.thoughtGenerator = new ThoughtGenerator(memory);
    }

    public String getLastThought() { return lastThought; }

    @Override
    public void run() {
        int[] lastSpikes = new int[64];
        tick = 0;

        while (running.get()) {
            try {
                // 1. check inbox for messages from other agents
                processMessages();

                // 2. fire neurons
                double effectiveInput = (role == AgentRole.LANGUAGE && textEncoder != null)
                    ? textEncoder.encode(currentWord)
                    : currentInput;

                int[] spikes = snn.tick(effectiveInput, synapticWeights);

                // DEBUG - remove after fix
                long spikeCount = 0;
                for (int s : spikes) spikeCount += s;
                if (spikeCount > 0) System.out.println(
                    "[" + role + "] tick fired " + spikeCount + " spikes, input=" + currentInput
                );

                // 3. encode to HV
                int[] hv = memory.encodeSpikeTraIn(spikes);

                // 4. world model prediction (skip for LANGUAGE - incompatible HV space)
                double predError = (role == AgentRole.LANGUAGE)
                    ? worldModel.getLastPredictionError()
                    : worldModel.tick(hv, lastHV, currentEnvReward);
                lastHV = hv;

                // 5. query memory for concept
                String prediction = memory.query(hv);

                // 6. compute reward
                // success window: when goal reached, positive reward lasts 50 ticks
                if (currentEnvReward >= 1.0) successWindowRemaining = 50;
                if (successWindowRemaining > 0) successWindowRemaining--;

                double successBonus = successWindowRemaining > 0 ? 0.8 : 0.0;

                double reward = successWindowRemaining > 0 ? 1.0
                    : currentEnvReward > 0 ? 1.0
                    : successBonus;

                double baseReward = reward;

                // REASONING: bind percept + language HVs, check working memory
                if (role == AgentRole.REASONING) {
                    if (lastPerceptHV != null && lastLanguageHV != null) {
                        int[] analogy = workingMemory.formAnalogy(lastPerceptHV, lastLanguageHV);
                        double recognition = workingMemory.recognize(analogy);
                        workingMemory.store(analogy);
                        // recognized a pattern = positive reward
                        if (recognition > 0.3) reward += 0.4;
                        hv = analogy; // reason over the analogy, not raw spikes

                        // generate thought every 100 ticks
                        if (tick % 100 == 0 && thoughtGenerator != null) {
                            lastThought = thoughtGenerator.generate(hv);
                        }
                    }

                    // refresh working memory every 500 ticks to prevent saturation
                    if (tick % 500 == 0) {
                        workingMemory = new WorkingMemory();
                    }
                }

                // MEMORY: store episodes, replay good ones to reinforce
                if (role == AgentRole.MEMORY) {
                    episodicMemory.store(hv, baseReward, tick);
                    int[] replayed = episodicMemory.replay();
                    if (replayed != null) {
                        double matchScore = episodicMemory.matchBest(hv);
                        if (matchScore > 0.2) reward += 0.3; // familiar good state
                        hv = HypervectorOps.bundle(hv, replayed); // blend with memory
                    }
                }

                // 7. curiosity boost
                curiosity.observe(prediction, reward);
                double boost = curiosity.getLearningBoost(prediction)
                               * learningRateMultiplier;

                // 8. STDP update
                for (int i = 0; i < spikes.length; i++) {
                    double timeDelta = spikes[i] == 1 ? 1.0 : -1.0;
                    synapticWeights[i] = stdp.updateWeight(
                        synapticWeights[i], lastSpikes[i], spikes[i],
                        timeDelta, reward * boost
                    );
                }
                lastSpikes = spikes;

                // 9. broadcast HV and prediction error to swarm
                bus.broadcast(new SwarmMessage(
                    role, null, "HV", hv, predError
                ));

                // 10. send reward to meta agent
                bus.send(new SwarmMessage(
                    role, AgentRole.META, "REWARD", null, reward
                ));

                tick++;
                Thread.sleep(0, 100000);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                System.err.println("[" + role + "] CRASH: " + e.getMessage());
                e.printStackTrace();
                // don't break — keep the agent alive
            }
        }
    }

    private void processMessages() {
        SwarmMessage msg;
        while ((msg = bus.poll(role)) != null) {
            switch (msg.type) {
                case "RATE" -> learningRateMultiplier = msg.value;
                case "HV" -> {
                    if (msg.hypervector != null) {
                        if (msg.from == AgentRole.PERCEPTION) {
                            lastPerceptHV = msg.hypervector;
                        } else if (msg.from == AgentRole.LANGUAGE) {
                            lastLanguageHV = msg.hypervector;
                        }
                    }
                }
            }
        }
    }

    public void stop() { running.set(false); }

    public double getAvgWeight() {
        double sum = 0;
        for (double w : synapticWeights) sum += w;
        return sum / synapticWeights.length;
    }

    public double getPredictionError() { return worldModel.getLastPredictionError(); }
    public CuriosityEngine getCuriosity() { return curiosity; }
    public String getCurrentWord() { return currentWord; }
}
