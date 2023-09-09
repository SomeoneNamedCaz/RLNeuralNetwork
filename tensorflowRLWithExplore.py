"""
I'm going to see if the explore stuff is better before getting my one neural network working
"""

if __name__ == '__main__':
    import tensorflow as tf
    import gymnasium as gym
    import numpy as np

    env = gym.make("LunarLander-v2")#, render_mode="human")
    observation, info = env.reset(seed=42)
    observations = []
    rewards = []
    actions = []
    observationsPlusActions = []
    for _ in range(1000):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        observations.append(observation)
        rewards.append(reward)
        actions.append(action)
        # print(observation, action)
        observationsPlusActions.append(np.append(observation, np.array(action)))
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    print(observationsPlusActions[0].shape)
    envEstimator = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(observationsPlusActions[0].shape)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(observations[0].shape[0], activation='sigmoid')
    ])
    rewardEstimator = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=observations[0].shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    policy = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    #
    # mnist = tf.keras.datasets.mnist
    #
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    envLoss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #
    envEstimator.compile(optimizer='adam',
                         loss="mse",
                         metrics=['accuracy'])
    rewardEstimator.compile(optimizer='adam',
                            loss="mse",
                            metrics=['accuracy'])

    policy.compile(optimizer='adam',
                            loss="mse",
                            metrics=['accuracy'])
    print(np.array(observationsPlusActions).shape)
    print(np.array(observations[1:]).shape)
    print(envEstimator.output_shape)
    envEstimator.fit(np.array(observationsPlusActions[:-1]), np.array(observations[1:]), epochs=5)
    rewardEstimator.fit(np.array(observations), np.array(rewards), epochs=5)
    policyWeights = policy.get_weights()
    # policyWeights.
    # policy.set_weights()



