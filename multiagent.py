from pettingzoo.classic import tictactoe_v3

print("=== 5 GAMES OF MULTI AGENT TIC TAC TOE ===\n")

for game in range(5):
    print(f"--- Game {game+1} ---")
    
    env = tictactoe_v3.env(render_mode="ansi")
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            if reward == 1:
                print(f"🏆 {agent} WINS!")
            elif reward == -1:
                print(f"😵 {agent} LOSES!")
            else:
                print("🤝 Its a draw!")
            env.step(None)
        else:
            action = env.action_space(agent).sample(observation["action_mask"])
            env.step(action)

    env.close()
    print()

print("All 5 games done!")