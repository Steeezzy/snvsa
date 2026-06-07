package helix;

public class SwarmMessage {
    public final AgentRole from;
    public final AgentRole to;      // null = broadcast to all
    public final String type;       // "HV", "REWARD", "ERROR", "RATE"
    public final int[] hypervector; // for HV messages
    public final double value;      // for scalar messages

    public SwarmMessage(AgentRole from, AgentRole to,
                        String type, int[] hv, double value) {
        this.from = from;
        this.to = to;
        this.type = type;
        this.hypervector = hv;
        this.value = value;
    }
}
