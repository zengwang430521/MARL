import sys
sys.path.append('..')
sys.path.append('../../marl-env')

import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg_full import MADDPGAgentTrainerFull

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
    parser.add_argument("--step-size", type=int, default=8, help="recurrent step num")
    parser.add_argument("--burn-in-step", type=int, default=4, help="burn in step")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='recurrent', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./checkpoint/recurrent/recurrent", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=50, help="save model once every time this many episodes are completed")
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


def p_policy_model(obs, state, obs_pred, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        # mlp to process the obs
        # input is in shape B * F
        input = tf.concat([obs, obs_pred], axis=1)
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
        gru_out, state = tf.nn.dynamic_rnn(gru, input, initial_state=state)
        gru_out = tf.squeeze(gru_out, 1)
        out = gru_out
        # mpl to get the p value
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out, gru_out, state


def p_predict_model(act, state, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        # mlp to predict next obs
        next_obs = tf.concat([act, state], axis=1)
        next_obs = layers.fully_connected(next_obs, num_outputs=num_units, activation_fn=tf.nn.relu)
        next_obs = layers.fully_connected(next_obs, num_outputs=num_units, activation_fn=tf.nn.relu)
        next_obs = layers.fully_connected(next_obs, num_outputs=num_outputs, activation_fn=None)
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
    trainer = MADDPGAgentTrainerFull

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
        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        episode_begin_num = 0

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)
            fname = './learning_curves/' + arglist.exp_name + '_rewards.pkl'
            final_ep_rewards = pickle.load(open(fname, 'rb'))
            fname = './learning_curves/' + arglist.exp_name + '_agrewards.pkl'
            final_ep_ag_rewards = pickle.load(open(fname, 'rb'))
            episode_begin_num = arglist.save_rate * len(final_ep_rewards)


        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()

        obs_n = env.reset()
        state_n = [agent.p_init_state(1) for agent in trainers]
        pred_n = [agent.init_pred(1) for agent in trainers]

        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            ## get action
            temp = [agent.take_action(obs, state, pred) for agent, obs, state, pred in zip(trainers, obs_n, state_n, pred_n)]
            action_n = [x[0] for x in temp]
            new_state_n = [x[1] for x in temp]
            gru_out_n = [x[2] for x in temp]
            new_pred_n = [agent.predict(act[None], gru_out) for agent, act, gru_out in zip(trainers, action_n, gru_out_n)]

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
            # pred_n = [x.eval() for x in new_pred_n]
            pred_n = new_pred_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                state_n = [agent.p_init_state(1) for agent in trainers]
                pred_n = [agent.init_pred(1) for agent in trainers]
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.05)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step, arglist.step_size, arglist.burn_in_step)

            # save model, display training output
            episode_num = len(episode_rewards) + episode_begin_num
            if terminal and (episode_num % arglist.save_rate == 0):
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, episode_num, np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, episode_num, np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(episode_num))

                U.save_state(arglist.save_dir, saver=saver)

            if episode_num > arglist.num_episodes:
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
