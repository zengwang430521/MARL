import sys
sys.path.append('..')
sys.path.append('../../marl-env')

import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg.trainer.maddpg import MADDPGAgentTrainerRecurrent

import tensorflow.contrib.layers as layers


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")

    # Core training parameters
    parser.add_argument("--step-size", type=int, default=16, help="recurrent step num")
    parser.add_argument("--burn-in-step", type=int, default=8, help="burn in step")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='recurrent', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./checkpoint/recurrent/recurrent", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def q_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def p_model(input, state, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        # mlp to process the obs
        # input is in shape B * F

        input = layers.fully_connected(input, num_outputs=num_units, activation_fn=tf.nn.relu)
        input = layers.fully_connected(input, num_outputs=num_units, activation_fn=tf.nn.relu)
        # LSTM
        lstm = tf.contrib.rnn.BasicLSTMCell(num_units)
        lstm_out, state = tf.nn.dynamic_rnn(lstm, input, state, initial_state=state)

        # mpl to get the p value
        out = layers.fully_connected(lstm_out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)

        # mlp to predict next obs
        next_obs = tf.concat([lstm_out, out], dim=1)
        next_obs = layers.fully_connected(next_obs, num_outputs=num_units, activation_fn=tf.nn.relu)
        next_obs = layers.fully_connected(next_obs, num_outputs=num_units, activation_fn=tf.nn.relu)
        next_obs = layers.fully_connected(next_obs, num_outputs=tf.shape(input, 2), activation_fn=tf.nn.relu)

        return out, state, next_obs


def p_policy_model(input, state, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        # mlp to process the obs
        # input is in shape B * F

        input = layers.fully_connected(input, num_outputs=num_units, activation_fn=tf.nn.relu)
        input = layers.fully_connected(input, num_outputs=num_units, activation_fn=tf.nn.relu)
        '''
        # LSTM
        lstm = tf.contrib.rnn.BasicLSTMCell(num_units)
        lstm_out, state = tf.nn.dynamic_rnn(lstm, input, sequence_length=1, initial_state=state)
        '''
        # GRU
        input = tf.expand_dims(input, 1)
        gru = tf.contrib.rnn.GRUCell(num_units)
        out, state = tf.nn.dynamic_rnn(gru, input, initial_state=state)
        out = tf.squeeze(out, 1)
        # mpl to get the p value
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out, state


def p_predict_model(act, state, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        # mlp to predict next obs
        next_obs = tf.concat([act, state], axis=1)
        next_obs = layers.fully_connected(next_obs, num_outputs=num_units, activation_fn=tf.nn.relu)
        next_obs = layers.fully_connected(next_obs, num_outputs=num_units, activation_fn=tf.nn.relu)
        next_obs = layers.fully_connected(next_obs, num_outputs=num_outputs, activation_fn=tf.nn.relu)
        return next_obs


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, state_shape_n, arglist):
    trainers = []
    trainer = MADDPGAgentTrainerRecurrent

    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, p_policy_model, p_predict_model, q_model, obs_shape_n, env.action_space, state_shape_n, i,
            arglist, local_q_func=(arglist.adv_policy=='ddpg')))

    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, p_policy_model, p_predict_model, q_model, obs_shape_n, env.action_space, state_shape_n, i,
            arglist, local_q_func=(arglist.adv_policy=='ddpg')))

    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        state_shape_n = [(64, ) for i in range(env.n)]
        trainers = get_trainers(env, num_adversaries, obs_shape_n, state_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()

        obs_n = env.reset()
        state_n = [agent.p_init_state(1) for agent in trainers]

        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            ## get action
            temp = [agent.take_action(obs, state) for agent, obs, state in zip(trainers, obs_n, state_n)]
            action_n = [x[0] for x in temp]
            new_state_n = [x[1] for x in temp]

            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)

            # collect experience
            ## need to be modified
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n
            state_n = new_state_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                state_n = [agent.p_init_state(1) for agent in trainers]
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1


            #################################################
            #       need to be modified                     #
            #################################################
            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step, arglist.step_size, arglist.burn_in_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break



if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
