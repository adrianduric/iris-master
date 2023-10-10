import gym

env = gym.make('CarRacing-v0')
observation = env.reset()
for t in range(1500):
    #print(t)
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    #print(done)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()