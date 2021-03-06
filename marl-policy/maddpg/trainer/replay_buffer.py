import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)

    def sequence_sample_index(self, finish_index, step_size=16):
        obs_seq, act_seq, rew_seq, obs_tp_seq, dones_seq = [], [], [], [], []
        begin_index = [x - step_size for x in finish_index]
        zero_data = self._storage[0]
        zero_data = [[x * 0 for x in zero_data]]

        for start, end in zip(begin_index, finish_index):
            data = self._storage[max(start, 0):end]

            for i in range(len(data) - 2, -1, -1):
                done = data[i][4]
                if done:
                    data = data[i + 1:]
                    break

            while (len(data) < step_size):
                data = zero_data + data

            obs = np.stack([x[0] for x in data], axis=0)
            act = np.stack([x[1] for x in data], axis=0)
            rew = np.stack([x[2] for x in data], axis=0)
            obs_tp = np.stack([x[3] for x in data], axis=0)
            dones = np.stack([x[4] for x in data], axis=0)

            obs_seq.append(obs)
            act_seq.append(act)
            rew_seq.append(rew)
            obs_tp_seq.append(obs_tp)
            dones_seq.append(dones)

        obs_seq = np.stack(obs_seq, axis=1)
        act_seq = np.stack(act_seq, axis=1)
        rew_seq = np.stack(rew_seq, axis=1)
        obs_tp_seq = np.stack(obs_tp_seq, axis=1)
        dones_seq = np.stack(dones_seq, axis=1)

        return obs_seq, act_seq, rew_seq, obs_tp_seq, dones_seq


class SeqReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _encode_seq_sample(self, idx_list):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in range(len(idx_list)):
            obses_t.append([])
            actions.append([])
            rewards.append([])
            obses_tp1.append([])
            dones.append([])
            for j in idx_list[i]:
                data = self._storage[j]
                obs_t, action, reward, obs_tp1, done = data
                obses_t[i].append(np.array(obs_t, copy=False))
                actions[i].append(np.array(action, copy=False))
                rewards[i].append(reward)
                obses_tp1[i].append(np.array(obs_tp1, copy=False))
                dones[i].append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_seq_index(self, batch_size, seq_length):
        idx_list = []
        for i in range(batch_size):
            idx_list.append([])
            start_idx = random.randint(0, len(self._storage) - 1)
            for j in range(seq_length):
                idx_list[i].append(start_idx+j)
        return idx_list

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def seq_sample_index(self, idxes):
        return self._encode_seq_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def seq_sample(self, batch_size):
        idxes = self.make_seq_index(batch_size, seq_length = 3)
        return self._encode_seq_sample(idxes)

    def collect(self):
        return self.sample(-1)
