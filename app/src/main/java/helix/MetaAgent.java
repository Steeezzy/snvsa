package helix;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

public class MetaAgent implements Runnable {
    private final SwarmBus bus;
    private final List<HelixAgent> agents;
    private final AtomicBoolean running = new AtomicBoolean(true);

    // tracks reward per agent
    private final Map<AgentRole, Double> avgRewards = new EnumMap<>(AgentRole.class);

    public MetaAgent(SwarmBus bus, List<HelixAgent> agents) {
        this.bus = bus;
        this.agents = agents;
        for (AgentRole r : AgentRole.values()) avgRewards.put(r, 0.0);
    }

    @Override
    public void run() {
        while (running.get()) {
            try {
                // read reward messages
                SwarmMessage msg;
                while ((msg = bus.poll(AgentRole.META)) != null) {
                    if ("REWARD".equals(msg.type)) {
                        double old = avgRewards.get(msg.from);
                        avgRewards.put(msg.from, old * 0.95 + msg.value * 0.05);
                    }
                }

                // every 500ms adjust learning rates
                for (HelixAgent agent : agents) {
                    double reward = avgRewards.get(agent.role);
                    // struggling agent gets boosted rate
                    double newRate = reward < 0 ? 2.0
                                  : reward > 0.5 ? 0.8
                                  : 1.0;
                    bus.send(new SwarmMessage(
                        AgentRole.META, agent.role, "RATE", null, newRate
                    ));
                }

                Thread.sleep(500);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    public void stop() { running.set(false); }
    public Map<AgentRole, Double> getRewards() { return avgRewards; }
}
