import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]

def p_train_new_obs(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        new_obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p, new_obs = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), int(obs_ph_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        ob_loss = tf.reduce_mean(tf.square(new_obs-new_obs_ph_n[p_index]))
        theta = 0.001
        #loss = pg_loss + p_reg * 1e-3
        loss = pg_loss + p_reg * 1e-3 + theta * ob_loss

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + new_obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train_new_obs(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class MADDPGAgentTrainerNewObs(AgentTrainer):
    def __init__(self, name, p_model, q_model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train_new_obs(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=q_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train_new_obs(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=p_model,
            q_func=q_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + obs_next_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]



def p_train_recurrent(make_obs_ph_n, make_state_ph_n, make_obs_next_n, make_obs_pred_n,
               act_space_n, p_index, p_policy, p_predict, q_func,
               optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions


        # set up placeholders
        obs_ph_n = make_obs_ph_n              # all obs, in shape Agent_num * batch_size * time_step * obs_shape
        obs_next_n = make_obs_next_n
        state_ph_n = make_state_ph_n
        obs_pred_n = make_obs_pred_n

        # used for action
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        # p_input is local obs of an agent
        obs_input = obs_ph_n[p_index]
        state_input = state_ph_n[p_index]
        act_input = act_ph_n[p_index]
        obs_next = obs_next_n[p_index]
        obs_pred_input = obs_pred_n[p_index]

        # get output and state
        p, gru_out, state = p_policy(obs_input, state_input, obs_pred_input, int(act_pdtype_n[p_index].param_shape()[0]),
                                     scope="p_policy", num_units=num_units)
        act_pd = act_pdtype_n[p_index].pdfromflat(p)                    # wrap parameters in distribution
        act_sample = act_pd.sample()                                    # sample an action

        # predict the next obs
        obs_pred = p_predict(act_input, gru_out, int(obs_input.shape[1]), scope="p_predict", num_units=num_units)

        # variables for optimization
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_policy")) + U.scope_vars(U.absolute_scope_name("p_predict"))

        pred_loss = tf.reduce_mean(tf.square(obs_next - obs_pred))      # predict loss
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))           # reg item
        # use critic net to get the loss about policy
        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()                          # only modify the action of this agent
        q_input = tf.concat(obs_ph_n + act_input_n, 1)                  # get the input for Q net (all obs + all action)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]    # get q values
        pg_loss = -tf.reduce_mean(q)        # calculate loss to maximize Q values

        loss = pg_loss + p_reg * 1e-3 + pred_loss * 1e-3
        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)  # update p Net parameters

        # Create callable functions
        # update P NET
        train = U.function(inputs=obs_ph_n + state_ph_n + act_ph_n + obs_next_n + obs_pred_n,
                           outputs=loss, updates=[optimize_expr])
        # return action and state
        step = U.function(inputs=[obs_ph_n[p_index]] + [state_ph_n[p_index]] + [obs_pred_n[p_index]],
                          outputs=[act_sample] + [state] + [gru_out])
        p_values = U.function(inputs=[obs_ph_n[p_index]] + [state_ph_n[p_index]] + [obs_pred_n[p_index]], outputs=p)


        # target network
        target_p, target_gru_out, target_state = \
            p_policy(obs_input, state_input, obs_pred_input,
                     int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_policy", num_units=num_units)
        target_obs_pred = p_predict(act_input, target_gru_out, int(obs_input.shape[1]),
                                    scope="target_p_predict", num_units=num_units)

        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_policy")) + \
                             U.scope_vars(U.absolute_scope_name("target_p_predict"))
        # update the parameters θ'i = τθi + (1 − τ)θ'i
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)
        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()

        target_step = U.function(inputs=[obs_ph_n[p_index]] + [state_ph_n[p_index]] + [obs_pred_n[p_index]],
                                 outputs=[target_act_sample] + [target_state] + [target_gru_out])

        # return predicted obs
        gru_temp = tf.placeholder(tf.float32, [None] + [num_units], name='gru_out')
        pred_temp = p_predict(act_input, gru_temp, int(obs_input.shape[1]), scope="p_predict", num_units=num_units)
        predict = U.function(inputs=[act_ph_n[p_index]]+[gru_temp], outputs=pred_temp)
        target_pred_temp = p_predict(act_input, gru_temp, int(obs_input.shape[1]), scope="target_p_predict", num_units=num_units)
        target_predict = U.function(inputs=[act_ph_n[p_index]]+[gru_temp], outputs=target_pred_temp)

        return step, predict, train, update_target_p, {'p_values': p_values, 'target_step': target_step, 'target_predict': target_predict}



class MADDPGAgentTrainerRecurrent(AgentTrainer):
    def __init__(self, name, p_policy, p_predict, q_model, obs_shape_n, act_space_n, state_shape_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.obs_shape = obs_shape_n[agent_index]
        self.state_shape = state_shape_n[agent_index]
        self.p_predict = p_predict
        obs_ph_n = []
        obs_next_n = []
        obs_pred_n = []
        state_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())
            obs_next_n.append(U.BatchInput(obs_shape_n[i], name="next_obs"+str(i)).get())
            obs_pred_n.append(U.BatchInput(obs_shape_n[i], name="pred_obs"+str(i)).get())
            state_ph_n.append(U.BatchInput(state_shape_n[i], name="state"+str(i)).get())

        # Create all the functions necessary to train the critic net
        # q_train is used for optimize Q net according to the loss in this batch
        # q_update is used to update the parameter of target net θ'i = τθi + (1 − τ)θ'i

        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=q_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )



        # step return the action and new_state given the obs and state
        # p_train is used to optimize p Net
        # p_update is used to update target p net as θ'i = τθi + (1 − τ)θ'i
        self.step, self.predict, self.p_train, self.p_update, self.p_debug = p_train_recurrent(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            make_state_ph_n=state_ph_n,
            act_space_n=act_space_n,
            make_obs_next_n=obs_next_n,
            make_obs_pred_n=obs_pred_n,
            p_index=agent_index,
            p_policy=p_policy,
            p_predict=p_predict,
            q_func=q_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            reuse=tf.AUTO_REUSE
        )

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    '''
    def predict(self, act_input, gru_out):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            obs_pred = self.p_predict(act_input[None], gru_out, int(self.obs_shape[0]), scope="p_predict", num_units=self.args.num_units)
            return obs_pred

    def target_predict(self, act_input, gru_out):
        with tf.variable_scope(self.name, reuse=None):
            obs_pred = self.p_predict(act_input, gru_out, int(self.obs_shape[0]), scope="target_p_predict", num_units=self.args.num_units)
            return obs_pred
    '''

    # return the zero state of GRU
    def p_init_state(self, batch_size):
        return np.zeros([batch_size, self.state_shape[0]])

    def init_pred(self, batch_size):
        return np.zeros([batch_size, self.obs_shape[0]])

    # given the obs and current state, return the action and new state
    def take_action(self, obs, state, pred):
        act, new_state, gru_out = self.step(obs[None], state, pred)
        act = act[0]
        return act, new_state, gru_out

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t, step_size=16, burn_in_step=8):
        if len(self.replay_buffer) < self.max_replay_buffer_len:    # replay buffer is not large enough
            return
        if not t % 100 == 0:                                        # only update every 100 steps
            return
        # sample experience
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_seq_n = []
        obs_next_seq_n = []
        act_seq_n = []
        finish_index = self.replay_sample_index
        for i in range(self.n):
            obs_seq, act_seq, rew_seq, obs_next_seq, done_seq = agents[i].replay_buffer.sequence_sample_index(finish_index, step_size)
            obs_seq_n.append(obs_seq)
            obs_next_seq_n.append(obs_next_seq)
            act_seq_n.append(act_seq)

        obs_seq, act_seq, rew_seq, obs_next_seq, done_seq = self.replay_buffer.sequence_sample_index(finish_index, step_size)

        state_n = [agents[i].p_init_state(self.args.batch_size) for i in range(self.n)]
        pred_n = [agents[i].init_pred(self.args.batch_size) for i in range(self.n)]
        target_state_n = [agents[i].p_init_state(self.args.batch_size) for i in range(self.n)]
        target_pred_n = [agents[i].init_pred(self.args.batch_size) for i in range(self.n)]

        act_n = [x[0] for x in act_seq_n]
        temp = [agents[i].p_debug['target_step'](obs_seq_n[i][0], target_state_n[i], target_pred_n[i]) for i in range(self.n)]
        target_state_n = [x[1] for x in temp]
        target_gru_out_n = [x[2] for x in temp]
        target_pred_n = [agents[i].p_debug['target_predict'](act_n[i], target_gru_out_n[i]) for i in range(self.n)]

        # burn in stage, don't update the net
        for step in range(burn_in_step):
            act_n = [x[step] for x in act_seq_n]
            act_next_n = [x[step + 1] for x in act_seq_n]

            # target agent step
            temp = [agents[i].p_debug['target_step'](obs_next_seq_n[i][step], target_state_n[i], target_pred_n[i])
                    for i in range(self.n)]
            target_state_n = [x[1] for x in temp]
            target_gru_out_n = [x[2] for x in temp]
            target_pred_n = [agents[i].p_debug['target_predict'](act_next_n[i], target_gru_out_n[i]) for i in range(self.n) ]

            # agents step
            temp = [agents[i].step(obs_seq_n[i][step], state_n[i], pred_n[i]) for i in range(self.n)]
            state_n = [x[1] for x in temp]
            gru_out_n = [x[2] for x in temp]
            pred_n = [agents[i].predict(act_n[i], gru_out_n[i]) for i in range(self.n) ]

        q_loss = 0
        p_loss = 0
        # update the agents
        for step in range(burn_in_step, step_size):
            obs_n = [x[step] for x in obs_seq_n]
            act_n = [x[step] for x in act_seq_n]
            if step < (step_size-1):
                act_next_n = [x[step + 1] for x in act_seq_n]
            obs_next_n = [x[step] for x in obs_next_seq_n]

            # target agents step, get the action in the next step
            temp = [agents[i].p_debug['target_step'](obs_next_seq_n[i][step], target_state_n[i], target_pred_n[i])
                    for i in range(self.n)]
            target_act_n = [x[0] for x in temp]
            target_state_n = [x[1] for x in temp]
            target_gru_out_n = [x[2] for x in temp]
            if step < (step_size - 1):
                target_pred_n = [agents[i].p_debug['target_predict'](act_next_n[i], target_gru_out_n[i]) for i in range(self.n)]
            # infer y from target action
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_n))
            target_q = rew_seq[step] + self.args.gamma * (1.0 - done_seq[step]) * target_q_next
            q_loss += self.q_train(*(obs_n + act_n + [target_q]))

            p_loss += self.p_train(*(obs_n + state_n + act_n + obs_next_n + pred_n))
            # agents step
            state_n = [x[1] for x in temp]
            gru_out_n = [x[2] for x in temp]
            pred_n = [agents[i].predict(act_n[i], gru_out_n[i]) for i in range(self.n)]

        # update the target net
        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew_seq[step]), np.mean(target_q_next), np.std(target_q)]