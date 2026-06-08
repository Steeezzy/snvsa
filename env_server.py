import gymnasium as gym
import minigrid
import socket
import json

def get_env_for_episode(episode_num):
    if episode_num < 500:
        return gym.make("MiniGrid-Empty-5x5-v0")
    elif episode_num < 1500:
        return gym.make("MiniGrid-Empty-6x6-v0")
    elif episode_num < 3000:
        return gym.make("MiniGrid-Empty-8x8-v0")
    else:
        return gym.make("MiniGrid-FourRooms-v0")

env = get_env_for_episode(0)
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

            # curriculum: upgrade environment
            new_env = get_env_for_episode(total_episodes)
            if type(new_env).__name__ != type(env).__name__:
                print(f">>> CURRICULUM: upgrading to {new_env.spec.id}")
            env = new_env
            obs, _ = env.reset()
            reward = 1.0 if success else 0.0

        image = obs["image"]

        # 1. what is directly in front of the agent (center column, front row)
        forward_cell = float(image[3][6][0]) / 10.0  # 0=empty, 1=wall, 2=goal

        # 2. agent direction encoded as a scalar
        direction = float(obs.get("direction", 0)) / 4.0  # 0-3 normalized

        # 3. density of empty cells in visible area (navigability)
        navigability = float((image[:,:,0] == 1).sum()) / (7 * 7)

        # combine into one signal with different weights
        scalar = (forward_cell * 0.5) + (direction * 0.3) + (navigability * 0.2)
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