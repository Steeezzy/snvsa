import gymnasium as gym
import minigrid
import socket
import json

env = gym.make("MiniGrid-Empty-5x5-v0")
obs, _ = env.reset()

episode_steps = 0
total_episodes = 0
successes = 0

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("localhost", 9999))
server.listen(1)
print("env server ready on port 9999")

conn, _ = server.accept()
print("HELIX-1 connected")
f = conn.makefile('r')
fw = conn.makefile('w')

while True:
    try:
        data = f.readline().strip()
        if not data:
            break
        action = int(data)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_steps += 1

        if terminated or truncated:
            success = terminated
            total_episodes += 1
            if success:
                successes += 1
            print(f"episode {total_episodes} | steps={episode_steps} | success={success} | rate={successes/total_episodes:.2f}")
            episode_steps = 0
            obs, _ = env.reset()
            reward = 1.0 if success else 0.0

        image = obs["image"]
        scalar = float(image[:,:,0].sum()) / 500.0
        scalar = min(1.0, max(0.1, scalar))

        response = json.dumps({
            "reward": float(reward),
            "obs": scalar,
            "success_rate": round(successes / max(1, total_episodes), 3),
            "episodes": total_episodes
        }) + "\n"
        fw.write(response)
        fw.flush()

    except Exception as e:
        print("error:", e)
        break

conn.close()
server.close()