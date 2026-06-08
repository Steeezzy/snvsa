package helix;

import java.util.*;
import java.nio.file.*;

public class Helix {
    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════╗");
        System.out.println("║  H E L I X - 1   S W A R M  v2.0   ║");
        System.out.println("╚══════════════════════════════════════╝");

        // shared memory and bus
        VSAMemory memory = new VSAMemory();
        SwarmBus bus = new SwarmBus();

        // load concepts
        String json = new String(Files.readAllBytes(Path.of("data/concepts.json")));
        json = json.replace("[","").replace("]","").replace("\"","");
        String[] concepts = json.split(",");
        int loaded = 0;
        for (String c : concepts) {
            String trimmed = c.trim();
            if (!trimmed.isEmpty()) { memory.register(trimmed); loaded++; }
        }
        System.out.println("loaded " + loaded + " concepts into shared VSA memory");

        // create 4 specialist agents (meta is separate)
        List<HelixAgent> agents = new ArrayList<>();
        for (AgentRole role : new AgentRole[]{
            AgentRole.PERCEPTION, AgentRole.LANGUAGE,
            AgentRole.REASONING,  AgentRole.MEMORY
        }) {
            HelixAgent agent = new HelixAgent(role, memory, bus);
            if (role == AgentRole.LANGUAGE) {
                agent.initTextEncoder(memory);
            }
            if (role == AgentRole.REASONING) {
                agent.initThoughtGenerator(memory);
            }
            agents.add(agent);
        }

        // meta agent watches all
        MetaAgent meta = new MetaAgent(bus, agents);

        // connect to environment
        EnvironmentBridge env = new EnvironmentBridge("localhost", 9999);
        env.connect();
        System.out.println("connected to MiniGrid environment");

        // start all agents on virtual threads
        List<Thread> threads = new ArrayList<>();
        for (HelixAgent agent : agents) {
            threads.add(Thread.ofVirtual()
                .name("agent-" + agent.role)
                .start(agent));
        }
        Thread metaThread = Thread.ofVirtual().name("meta").start(meta);

        System.out.println("swarm online — 4 specialist agents + 1 meta agent");
        System.out.println("reporting every 3s for 30 cycles\n");

        // connect language agent to text server
        HelixAgent languageAgent = agents.stream()
            .filter(a -> a.role == AgentRole.LANGUAGE)
            .findFirst().get();

        HelixAgent reasoningAgent = agents.stream()
            .filter(a -> a.role == AgentRole.REASONING)
            .findFirst().get();

        Thread.ofVirtual().name("text-reader").start(() -> {
            try {
                var sock = new java.net.Socket("localhost", 9998);
                var reader = new java.io.BufferedReader(
                    new java.io.InputStreamReader(sock.getInputStream()));
                String word;
                while ((word = reader.readLine()) != null) {
                    languageAgent.setTextInput(word);
                }
            } catch (Exception e) {
                System.err.println("text server disconnected: " + e.getMessage());
            }
        });

        // main loop — perception agent drives env interaction
        HelixAgent perception = agents.get(0);

        // seed initial observation so agents don't start at zero
        {
            var initial = env.step(0);
            for (HelixAgent agent : agents) {
                agent.setInput(initial[0], initial[1]);
            }
        }

        // background thread: drives env continuously
        var latestObs    = new java.util.concurrent.atomic.AtomicReference<>(new double[]{0.2, 0.0});
        var envRunning   = new java.util.concurrent.atomic.AtomicBoolean(true);

        Thread envDriver = Thread.ofVirtual().name("env-driver").start(() -> {
            while (envRunning.get()) {
                try {
                    // mix learned weight with exploration noise
                    double noise = Math.random();
                    int action = noise < 0.3
                        ? (int)(Math.random() * 7)          // 30% random exploration
                        : (int)(Math.abs(perception.getAvgWeight() * 6)) % 7; // 70% learned
                    
                    double[] data = env.step(action);
                    double obs        = data[0];
                    double envReward  = data[1];

                    // parse success_rate from env (add third element)
                    // update env_server.py to also include it in array
                    double successRate = env.getSuccessRate();

                    // boost reward signal when success rate is improving
                    double adjustedReward = envReward + (successRate > 0.1 ? 0.2 : 0.0);

                    latestObs.set(data);
                    for (HelixAgent agent : agents) {
                        agent.setInput(obs, adjustedReward);
                    }
                    Thread.sleep(0, 500_000); // 0.5ms between steps
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                } catch (Exception e) {
                    System.err.println("env error: " + e.getMessage());
                    break;
                }
            }
        });

        // report loop — just prints, doesn't drive env
        for (int report = 1; report <= 600; report++) {
            Thread.sleep(3000);

            // print swarm report
            System.out.println("┌─── Swarm Report " + report + " ───");
            System.out.printf("│  Episodes: %d  |  MiniGrid success rate: %.1f%%%n",
                env.getEpisodes(), env.getSuccessRate() * 100);
            System.out.printf("│  💭 Thought: \"%s\"%n",
                reasoningAgent.getLastThought());
            System.out.println("│");
            for (HelixAgent agent : agents) {
                System.out.printf(
                    "│  %-12s pred_error=%.3f  weight=%.4f  word=%-12s%n",
                    agent.role,
                    agent.getPredictionError(),
                    agent.getAvgWeight(),
                    agent.role == AgentRole.LANGUAGE
                        ? languageAgent.getCurrentWord()
                        : "-"
                );
            }
            System.out.println("│  Meta rewards:");
            meta.getRewards().forEach((role, r) -> {
                if (role != AgentRole.META)
                    System.out.printf("│    %-12s avg=%.3f%n", role, r);
            });
            System.out.println("└────────────────────────────────────");
        }

        // shutdown
        envRunning.set(false);
        envDriver.interrupt();
        agents.forEach(HelixAgent::stop);
        meta.stop();
        env.close();
        threads.forEach(t -> t.interrupt());
        metaThread.interrupt();
        System.out.println("swarm stopped cleanly");
    }
}
