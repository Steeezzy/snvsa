package helix;

import java.util.concurrent.*;
import java.util.*;

public class SwarmBus {
    // each agent has its own inbox queue
    private final Map<AgentRole, LinkedBlockingQueue<SwarmMessage>> inboxes
        = new EnumMap<>(AgentRole.class);

    public SwarmBus() {
        for (AgentRole role : AgentRole.values()) {
            inboxes.put(role, new LinkedBlockingQueue<>(100));
        }
    }

    // send a message to one agent
    public void send(SwarmMessage msg) {
        if (msg.to != null) {
            inboxes.get(msg.to).offer(msg);
        }
    }

    // broadcast to all agents except sender
    public void broadcast(SwarmMessage msg) {
        for (AgentRole role : AgentRole.values()) {
            if (role != msg.from) {
                inboxes.get(role).offer(
                    new SwarmMessage(msg.from, role,
                        msg.type, msg.hypervector, msg.value)
                );
            }
        }
    }

    // agent polls its own inbox, returns null if empty
    public SwarmMessage poll(AgentRole role) {
        return inboxes.get(role).poll();
    }
}
