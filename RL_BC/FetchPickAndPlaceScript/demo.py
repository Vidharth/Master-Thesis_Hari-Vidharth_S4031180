state = env.reset()

goal = state['desired_goal']
objectPos = state['observation'][3:6]
object_rel_pos = state['observation'][6:9]

object_oriented_goal = object_rel_pos.copy()
object_oriented_goal[2] += 0.03

timeStep = 0

while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
    # env.render()
    action = [0, 0, 0, 0]
    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03

    for i in range(len(object_oriented_goal)):
        action[i] = object_oriented_goal[i]*6

    action[len(action)-1] = 0.05

    obsDataNew, reward, dones, info = env.step(action)
    timeStep += 1

    dist_block_goal = np.linalg.norm(obsDataNew["achieved_goal"]-obsDataNew["desired_goal"])
    dist_eef_block = np.linalg.norm(obsDataNew["observation"][0:3]-obsDataNew["achieved_goal"])
    dist_eef_goal = np.linalg.norm(obsDataNew["observation"][0:3]-obsDataNew["desired_goal"])

    if info["is_success"] == 0.0:
        done = False
        reward = -(3.0 * dist_block_goal) -(2.0 * dist_eef_block) -(1.0 * dist_eef_goal)

    if info["is_success"] == 1.0:
        done = True
        reward = 0.01 * (1/dist_eef_goal)

    o_clip = np.clip(obsDataNew["observation"], -200, 200)
    g_clip = np.clip(obsDataNew["desired_goal"], -200, 200)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -5, 5)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -5, 5)
    observation = np.concatenate([o_norm, g_norm])

    agent.demo_remember(observation, action, reward, observation, done)
    agent.agent_remember(observation, action, reward, observation, done)

    objectPos = obsDataNew['observation'][3:6]
    object_rel_pos = obsDataNew['observation'][6:9]

while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
    # env.render()
    action = [0, 0, 0, 0]
    for i in range(len(object_rel_pos)):
        action[i] = object_rel_pos[i]*6

    action[len(action)-1] = -0.005

    obsDataNew, reward, dones, info = env.step(action)
    timeStep += 1

    dist_block_goal = np.linalg.norm(obsDataNew["achieved_goal"]-obsDataNew["desired_goal"])
    dist_eef_block = np.linalg.norm(obsDataNew["observation"][0:3]-obsDataNew["achieved_goal"])
    dist_eef_goal = np.linalg.norm(obsDataNew["observation"][0:3]-obsDataNew["desired_goal"])

    if info["is_success"] == 0.0:
        done = False
        reward = -(3.0 * dist_block_goal) -(2.0 * dist_eef_block) -(1.0 * dist_eef_goal)

    if info["is_success"] == 1.0:
        done = True
        reward = 0.01 * (1/dist_eef_goal)

    o_clip = np.clip(obsDataNew["observation"], -200, 200)
    g_clip = np.clip(obsDataNew["desired_goal"], -200, 200)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -5, 5)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -5, 5)
    observation = np.concatenate([o_norm, g_norm])

    agent.demo_remember(observation, action, reward, observation, done)
    agent.agent_remember(observation, action, reward, observation, done)

    objectPos = obsDataNew['observation'][3:6]
    object_rel_pos = obsDataNew['observation'][6:9]


while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
    # env.render()
    action = [0, 0, 0, 0]
    for i in range(len(goal - objectPos)):
        action[i] = (goal - objectPos)[i]*6

    action[len(action)-1] = -0.005

    obsDataNew, reward, dones, info = env.step(action)
    timeStep += 1

    dist_block_goal = np.linalg.norm(obsDataNew["achieved_goal"]-obsDataNew["desired_goal"])
    dist_eef_block = np.linalg.norm(obsDataNew["observation"][0:3]-obsDataNew["achieved_goal"])
    dist_eef_goal = np.linalg.norm(obsDataNew["observation"][0:3]-obsDataNew["desired_goal"])

    if info["is_success"] == 0.0:
        done = False
        reward = -(3.0 * dist_block_goal) -(2.0 * dist_eef_block) -(1.0 * dist_eef_goal)

    if info["is_success"] == 1.0:
        done = True
        reward = 0.01 * (1/dist_eef_goal)

    o_clip = np.clip(obsDataNew["observation"], -200, 200)
    g_clip = np.clip(obsDataNew["desired_goal"], -200, 200)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -5, 5)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -5, 5)
    observation = np.concatenate([o_norm, g_norm])

    agent.demo_remember(observation, action, reward, observation, done)
    agent.agent_remember(observation, action, reward, observation, done)

    objectPos = obsDataNew['observation'][3:6]
    object_rel_pos = obsDataNew['observation'][6:9]

while True: #limit the number of timesteps in the episode to a fixed duration
    # env.render()
    action = [0, 0, 0, 0]
    action[len(action)-1] = -0.005 # keep the gripper closed

    obsDataNew, reward, dones, info = env.step(action)
    timeStep += 1

    dist_block_goal = np.linalg.norm(obsDataNew["achieved_goal"]-obsDataNew["desired_goal"])
    dist_eef_block = np.linalg.norm(obsDataNew["observation"][0:3]-obsDataNew["achieved_goal"])
    dist_eef_goal = np.linalg.norm(obsDataNew["observation"][0:3]-obsDataNew["desired_goal"])

    if info["is_success"] == 0.0:
        done = False
        reward = -(3.0 * dist_block_goal) -(2.0 * dist_eef_block) -(1.0 * dist_eef_goal)

    if info["is_success"] == 1.0:
        done = True
        reward = 0.01 * (1/dist_eef_goal)

    o_clip = np.clip(obsDataNew["observation"], -200, 200)
    g_clip = np.clip(obsDataNew["desired_goal"], -200, 200)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -5, 5)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -5, 5)
    observation = np.concatenate([o_norm, g_norm])

    agent.demo_remember(observation, action, reward, observation, done)
    agent.agent_remember(observation, action, reward, observation, done)

    objectPos = obsDataNew['observation'][3:6]
    object_rel_pos = obsDataNew['observation'][6:9]

    if dones:
        break
