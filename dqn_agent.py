from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import random

StateInfo = namedtuple('StateInfo', ['observation', 'reward', 'is_done'])
BufferEntry = namedtuple(
    'BufferEntry', ['current_state', 'action', 'reward', 'is_done', 'next_state']
)
BatchEntry = namedtuple('BatchEntry', [
    'current_states', 'actions', 'rewards', 'dones', 'next_states'
])


class ReplayBuffer:
    """A generic replay buffer class of max_length entries."""
    def __init__(self, max_length):
        """
        :param max_length: int: Maximum entries permitted in the buffer.
        """
        self.max_length = max_length
        self.buffer = deque(maxlen=max_length)

    def append(self, buffer_entry):
        """
        Appends an entry to the buffer.

        If the buffer already has max_length entries, oldest entry is evicted.
        :param buffer_entry: Entry to add
        :return: void
        """
        self.buffer.append(buffer_entry)

    def sample(self, sample_size):
        """
        Returns a sample of size sample_size without replacement.

        Can throw an exception if sample_size is less than number of buffer entries.
        :param sample_size: int: Number of samples to return.
        :return: array: Returns an array of 'sample_size' samples.
        """
        sample_indices = np.random.choice(
            len(self.buffer), sample_size, replace=False
        )
        return [self.buffer[i] for i in sample_indices]

    def length(self):
        """Returns the number of entries in the buffer."""
        return len(self.buffer)


class EnvWrapper:
    """A wrapper over the env class for convenience."""
    def __init__(self, env, brain_name, num_actions, train_mode):
        self.env = env
        self.brain_name = brain_name
        self.num_actions = num_actions
        self.train_mode = train_mode
        self.env_info = None
        self.reset()

    def getStateInfo(self):
        """
        Returns the current state information.
        Note that vector_observations is an array of shape (1, 37).
        :return: StateInfo: object corresponding to current state.
        """
        return StateInfo(
            self.env_info.vector_observations,
            self.env_info.rewards[0],
            1 if self.env_info.local_done[0] else 0
        )

    def reset(self):
        """
        Resets the environment.
        :return: StateInfo: object representing the reset state.
        """
        self.env_info = self.env.reset(self.train_mode)[self.brain_name]
        return self.getStateInfo()

    def step(self, action):
        """
        Updates the state of the environment after taking 'action'
        Arguments:
            action: Action to take.
        Returns:
            The state info object representing the updated state.
        """
        self.env_info = self.env.step(action)[self.brain_name]
        return self.getStateInfo()

    def sample_action(self):
        """
        Returns a randomly chosen action
        :return: (int), the randomly chosen action.
        """
        return np.random.choice(self.num_actions)


def getBatch(replay_buffer, sample_size):
    """
    Returns a sample of 'sample_size' from 'replay_buffer'.
    The replay buffer is assumed to contain tuples of the form
    (current_state, action, reward, done, next_state).
    :param replay_buffer: ReplayBuffer: Buffer to sample from.
    :param sample_size: int: Number of elements to sample.
    :return: BatchEntry: object representing the batch.
    """
    samples = replay_buffer.sample(sample_size)
    # Samples produced above are zipped tuples. Unzip them.
    current_states, actions, rewards, is_dones, next_states = zip(*samples)
    return BatchEntry(
        np.array(current_states), np.array(actions), np.array(rewards),
        np.array(is_dones), np.array(next_states)
    )


class Agent:
    def __init__(
        self, env_wrapper, replay_buffer,
        model, target_model, device, optimizer,
        gamma, batch_size, min_replay_size,
        target_update_steps=5, double_dqn=False,
        tao=1e-3, seed=0, soft_update=False
    ):
        self.env_wrapper = env_wrapper
        self.replay_buffer = replay_buffer
        self.model = model
        self.target_model = target_model
        self.device = device
        self.optimizer = optimizer
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.target_update_steps = target_update_steps
        self.double_dqn = double_dqn
        self.tao = tao
        self.seed = random.seed(seed)
        self.soft_update = soft_update
        self.env_wrapper.reset()
        self.steps = 0

    def updateTargetModel(self):
        """
        Copies the weights from self.model to self.target.

        Used for experimentation. In the final version, used softUpdate instead.
        The code is generic and allows either updateTargetModel or softUpdate
        to be used based on soft_update flag.
        :return: void
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def softUpdate(self):
        """
        Does a soft update of weights of self.target_model.

        Instead of copying the weights from self.model, does a soft update.
        :return: void
        """
        for tp, lp in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tao * lp.data + (1- self.tao) * tp.data)

    def _getLoss(self, batch):
        """
        Returns the MSE loss.

        This compares the next_state value predicted using the current state
        compared with the same value predicted using the reward and next state.
        When self.double_dqn is true, double_dqn functionality is implemented.

        :param batch: BatchEntry
        :return: Returns the MSE loss.
        """
        current_states, actions, rewards, dones, next_states = batch
        states_v = torch.tensor(current_states).to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.ByteTensor(dones).to(self.device)
        next_states_v = torch.tensor(next_states).to(self.device)

        # Compute the action values for the current state.
        model_output = self.model(states_v.float())
        state_action_values = model_output.gather(
            2, actions_v.unsqueeze(-1).unsqueeze(-1)
        ).squeeze(-1)

        if self.double_dqn:
            next_state_output = self.model(next_states_v.float())
            next_state_output_argmax = next_state_output.max(-1)[1].unsqueeze(-1)
            next_state_values = self.target_model(next_states_v.float()).gather(
                2, next_state_output_argmax
            ).squeeze(-1).squeeze(-1)
        else:
            # Compute the maximum values for the next state.
            next_state_output = self.target_model(next_states_v.float())
            next_state_values = \
                next_state_output.max(-1)[0].squeeze(-1).squeeze(-1)

        # For states which are done, there are no next states.
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = \
            next_state_values.float() * self.gamma + rewards_v.float()
        return nn.MSELoss()(
            state_action_values.squeeze(-1), expected_state_action_values
        )

    def optimize(self):
        """
        Runs a single optimization step.

        The optimization step is only run if there are atleast self.min_replay_size
        entries in the replay buffer.

        :return: void.
        """
        if self.replay_buffer.length() < self.min_replay_size:
            return

        batch = getBatch(self.replay_buffer, self.batch_size)
        loss = self._getLoss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.soft_update:
            self.softUpdate()
        elif self.steps % self.target_update_steps == 0:
                self.updateTargetModel()

    def getAction(self, epsilon=0.0):
        """
        Returns the next action to take.
        :param epsilon: Pick a random action with probability epsilon, else use model's prediction.
        :return: The next action to take.
        """
        if np.random.random() < epsilon:
            return self.env_wrapper.sample_action()

        current_state, _, _ = self.env_wrapper.getStateInfo()
        state_v = torch.tensor(current_state).float().to(self.device)
        q_vals_v = self.model(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        return int(act_v.item())

    def step(self, epsilon):
        """
        Take a single step.
        Specifically
            1) Choose the next action.
            2) Update the environment to take the chosen action.
            3) Add the state and reward tuple to the replay buffer.
            4) Reset environment if is_done is true.

        :param epsilon: float: exploration-exploitation trade off parameter.
        :return: (reward, is_done) tuple.
        """
        current_state, _, _ = self.env_wrapper.getStateInfo()
        action = self.getAction(epsilon)
        next_state, reward, is_done = self.env_wrapper.step(action)
        buffer_entry = BufferEntry(
            current_state, action, reward, is_done, next_state
        )
        self.replay_buffer.append(buffer_entry)
        if is_done:
            self.env_wrapper.reset()
        return reward, is_done

    def getReplayBuffer(self):
        """Returns the replay buffer"""
        return self.replay_buffer


def dqn_agent(agent, model_file_name, n_episodes=3000,
              n_max_steps=1000, eps_start=1.0, eps_decay=0.995,
              eps_end=0.01, target_reward=13.0,
              target_reward_window=100):
    """
    Runs the training for the agent.

    :param agent: Agent: Agent to train.
    :param model_file_name: string: file name to save the trained model to.
    :param n_episodes: int: number of episodes to train for.
    :param n_max_steps: int: maximum number of steps in an episode.
    :param eps_start: float: The starting value of epsilon.
    :param eps_decay: float: The decay applied to epsilon after each step.
    :param eps_end: float: The minimum value of epsilon.
    :param target_reward: float: Target reward to achieve. Training stops early if the
        average reward over 'target_reward_window' exceeds 'target_reward'.
    :param target_reward_window: float: The window over which to compute the average
        reward over.
    :return: (int, float): If after training, the average reward exceeds
        'target_reward', (n_episode, average_reward) is returned. n_episode is the
        episode when the average reward was exceeded. If the average reward fails to
        exceed target_reward, (None, None) is returned.
    """
    epsilon = eps_start
    episode_rewards = []
    for n_ep in range(1, n_episodes + 1):
        episode_reward = 0.0
        episode_done = False
        for n_s in range(n_max_steps):
            epsilon = max(epsilon * eps_decay, eps_end)
            reward, is_done = agent.step(epsilon)
            agent.optimize()
            episode_reward += reward
            if is_done:
                episode_done = True
                break

        if episode_done:
            episode_rewards.append(episode_reward)
        if len(episode_rewards) >= target_reward_window:
            average_reward = sum(episode_rewards[-target_reward_window:]) / target_reward_window
            if n_ep % 100 == 0:
                print('Episode {0}, average reward {1:.3f}'.format(n_ep, average_reward))
            if average_reward > target_reward:
                # The model achieved the target score
                print('\nSolved in {:d} episodes!\t Avg Score: {:.2f}'.format(n_ep, average_reward))
                torch.save(agent.model.state_dict(), model_file_name)
                return n_ep, average_reward, episode_rewards
    return None, None, None




