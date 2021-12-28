from kaggle_environments import make, evaluate

# Create the game environment
# Set debug=True to see the errors if your agent refuses to run
env = make("connectx", debug=True)

# List of available default agents
print(list(env.agents))

# Two random agents play one game round
env.run(["random", "random"])

# Show the game
env.render(mode="ipython")