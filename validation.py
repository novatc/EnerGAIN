from gymnasium import register, make
from stable_baselines3 import SAC

# Load the model
try:
    model = SAC.load("agents/sac")
except Exception as e:
    print("Error loading model: ", e)
    exit()

# Register the validation environment
register(
    id='energy-validation-v0',
    entry_point='environment:EnergyEnv',
    kwargs={'data_path': "data/clean/test_set.csv"}
)

# Create the validation environment
try:
    eval_env = make('energy-validation-v0')
except Exception as e:
    print("Error creating environment: ", e)
    exit()

# Reset the environment
obs, _ = eval_env.reset()

# Initialize the score
score = 0

# Run one episode
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = eval_env.step(action)
    score += reward

print("Total reward:", score)

# Call the render function
eval_env.render()
