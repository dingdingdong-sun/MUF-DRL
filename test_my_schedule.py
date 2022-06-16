import math

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import tensorflow as tf
import gym
import os
import sys
import time
from src.spinup.spinup.utils.logx import EpochLogger
from src.spinup.spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from src.spinup.spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from src.spinup.spinup.utils.logx import restore_tf_graph
import os.path as osp
from multi_HPCSimPickJobs import *

def load_policy(trained_model):
    sess = tf.Session()
    model = restore_tf_graph(sess, trained_model)
    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])

    x_ph = model['x']
    a_ph = model['a']
    last_state_a = model['a_last_state']
    last_state_v = model['v_last_state']
    user_f = model['user']
    mask_ph = model['mask']
    adv_ph = model['adv']
    ret_ph = model['ret']
    logp_old_ph = model['logp_old_ph']

    pi = model['pi']
    v = model['v']
    prio = model['priotity']
    next_state_a = model['a_next_state']
    next_state_v = model['v_next_state']
    # logits = model['logits']
    logp = model['logp']
    logp_pi = model['logp_pi']
    pi_loss = model['pi_loss']
    v_loss = model['v_loss']
    approx_ent = model['approx_ent']
    approx_kl = model['approx_kl']
    clipfrac = model['clipfrac']
    clipped = model['clipped']

    # Optimizers
    # graph = tf.get_default_graph()
    # op = sess.graph.get_operations()
    # [print(m.values()) for m in op]
    # train_pi = graph.get_tensor_by_name('pi/conv2d/kernel/Adam:0')
    # train_v = graph.get_tensor_by_name('v/conv2d/kernel/Adam:0')
    train_pi = tf.get_collection("train_pi")[0]
    train_v = tf.get_collection("train_v")[0]
    # train_pi_optimizer = MpiAdamOptimizer(learning_rate=pi_lr, name='AdamLoad')
    # train_pi = train_pi_optimizer.minimize(pi_loss)
    # train_v_optimizer = MpiAdamOptimizer(learning_rate=vf_lr, name='AdamLoad')
    # train_v = train_v_optimizer.minimize(v_loss)
    # sess.run(tf.variables_initializer(train_pi_optimizer.variables()))
    # sess.run(tf.variables_initializer(train_v_optimizer.variables()))
    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, last_state_a, last_state_v, user_f, a_ph, mask_ph, adv_ph, ret_ph, logp_old_ph]
    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi, prio, next_state_a, next_state_v]


def maxminnorm(array):
    # print(array)
    s = np.sum(array)
    if s == 0:
        array_out = np.ones_like(array)
        array_out = array_out/len(array)
    else:
        array_out = array/s
    return array_out

def sort_dict_by_key(d):
    k = sorted(d.keys())
    out = []
    for i in k:
        out.append(list(d[i]))
    return out

def change_dict_by_key(d, l):
    k = sorted(d.keys())
    x = 0
    for i in k:
        d[i] = l[x]
        x += 1

def test_fcfs_fair(workload_file, model_path, ac_kwargs=dict(), seed=0,
        attn=False,trained_model = None,
        shuffle=False,
        backfil=False, skip=False, score_type=0, batch_job_slice=0):

    np.random.seed(seed)

    with open("first_filter_index.txt", "r") as f:
        tmp = f.readline().strip().split(",")
        start_index = [int(x) for x in tmp]

    env = HPCEnv(shuffle=shuffle, backfil=backfil, skip=skip, job_score_type=score_type,
                 batch_job_slice=batch_job_slice, build_sjf=False)
    env.seed(seed)
    env.my_init(workload_file=workload_file, sched_file=model_path)
    print(env.loads.max_nodes)

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['attn'] = attn

    mean = []
    std = []
    job_wait_ratio_list = []
    user_waits = []
    job_waits = []
    sess = tf.Session()
    model = restore_tf_graph(sess, trained_model)
    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])

    x_ph = model['x']
    a_ph = model['a']
    last_state_a = model['a_last_state']
    last_state_v = model['v_last_state']
    user_f = model['user']
    mask_ph = model['mask']
    adv_ph = model['adv']
    ret_ph = model['ret']
    logp_old_ph = model['logp_old_ph']

    pi = model['pi']
    v = model['v']
    prio = model['priotity']
    next_state_a = model['a_next_state']
    next_state_v = model['v_next_state']
    # logits = model['logits']
    logp = model['logp']
    logp_pi = model['logp_pi']
    pi_loss = model['pi_loss']
    v_loss = model['v_loss']
    approx_ent = model['approx_ent']
    approx_kl = model['approx_kl']
    clipfrac = model['clipfrac']
    clipped = model['clipped']


    train_pi = tf.get_collection("train_pi")[0]
    train_v = tf.get_collection("train_v")[0]

    all_phs = [x_ph, last_state_a, last_state_v, user_f, a_ph, mask_ph, adv_ph, ret_ph, logp_old_ph]
    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi, prio, next_state_a, next_state_v]

    ii = 0
    [o, co, user_index_list, done], d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset_test(ii * JOB_SEQUENCE_SIZE + 256), False, 0, 0, 0, 0, 0

    env.start = start_index[ii]
    num_total = 0
    max_nodes = env.loads.max_nodes
    l_states_a = {}
    l_states_v = {}
    while not ii == 200:
        # print("start index is {}".format(env.start))
        users_job = {}
        users_job_index_map = {}
        users_mask = {}
        l_states_a_new = {}
        l_states_v_new = {}
        all_user_info = o[MAX_QUEUE_SIZE * JOB_FEATURES:]
        cluster_info = np.sum(all_user_info.reshape(-1, max_nodes), axis=0)
        for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):

            #     users_job[o[i + 4]].append(o[i:i + JOB_FEATURES])
            #     if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
            #             users_mask[o[i+4]].append(0)
            #         elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
            #             users_mask[o[i+4]].append(0)
            #         else:
            #             users_mask[o[i+4]].append(1)
            # else:
            if o[i + 4] not in users_job.keys():
                users_job_index_map[o[i + 4]] = []
                users_job[o[i + 4]] = []
                users_mask[o[i + 4]] = []
            users_job[o[i + 4]].append(np.concatenate((o[i:i + JOB_FEATURES], cluster_info[:]), axis=0))
            users_job_index_map[o[i + 4]].append(int(i / JOB_FEATURES))
            if all(o[i:i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                users_mask[o[i + 4]].append(0)
            elif all(o[i:i + JOB_FEATURES] == [1] * JOB_FEATURES):
                users_mask[o[i + 4]].append(0)
            else:
                users_mask[o[i + 4]].append(1)

        # 用户编号1.0的任务是用于完成队列补充的，实际并不存在这个用户，因此删除
        if 1.0 in users_mask.keys():
            users_mask.pop(1.0)
            users_job.pop(1.0)
            users_job_index_map.pop(1.0)
        user_info = []
        user_index_tmp = []
        for u in sorted(users_job.keys()):
            i = int(u * len(env.loads.users_id.keys()))
            user_index_tmp.append(i)
            user_f_tmp = all_user_info[i * max_nodes:(i + 1) * max_nodes]
            user_info.append(user_f_tmp)
            while len(users_job[u]) < MAX_QUEUE_SIZE:
                fea_tmp = np.array([0] + [1] * (JOB_FEATURES - 2) + [0], dtype=np.float32)
                users_job[u].append(np.concatenate((fea_tmp, cluster_info[:]), axis=0))
                users_mask[u].append(0)
            if u not in sorted(l_states_a.keys()):
                l_states_a_new[u] = np.zeros([1, 16], dtype=np.float32)
                l_states_v_new[u] = np.zeros([1, 16], dtype=np.float32)
            else:
                l_states_a_new[u] = l_states_a[u].reshape(1, 16)
                l_states_v_new[u] = l_states_v[u].reshape(1, 16)
        # print(sort_dict_by_key(l_states_a_new))
        # print("next")
        job_index_map = sort_dict_by_key(users_job_index_map)
        a, v_t, logp_t, prio_t, l_state_a, l_state_v = sess.run(get_action_ops,
                                                                feed_dict={
                                                                    x_ph: np.array(sort_dict_by_key(users_job)).reshape(
                                                                        -1, MAX_QUEUE_SIZE, JOB_FEATURES + max_nodes),
                                                                    mask_ph: np.array(
                                                                        sort_dict_by_key(users_mask)).reshape(-1,
                                                                                                              MAX_QUEUE_SIZE),
                                                                    last_state_a: np.array(
                                                                        sort_dict_by_key(l_states_a_new)).reshape(-1,
                                                                                                                  16),
                                                                    last_state_v: np.array(
                                                                        sort_dict_by_key(l_states_v_new)).reshape(-1,
                                                                                                                  16),
                                                                    user_f: np.array(user_info,
                                                                                     dtype=np.float32).reshape(-1,
                                                                                                               max_nodes)})
        # print(ou1)
        # print(ou2)

        change_dict_by_key(l_states_a_new, l_state_a)
        change_dict_by_key(l_states_v_new, l_state_v)
        for key in l_states_a_new.keys():
            l_states_a[key] = l_states_a_new[key]
            l_states_v[key] = l_states_v_new[key]
        # a = np.squeeze(np.array(a))
        # v_t = np.squeeze(np.array(v_t))
        # logp_t = np.squeeze(np.array(logp_t))
        # prio_t = np.squeeze(np.array(prio_t))

        a_real = []
        for l in range(len(a)):
            a_real.append(job_index_map[l][a[l]])
        # print(a_real)
        # print(prio_t)
        if a.shape[0] > 1:
            prio_t = maxminnorm(prio_t)
            a_joint = np.random.choice(a_real, 1, True, prio_t)
        else:
            prio_t = prio_t
            a_joint = a_real

        num_total += 1
        o, r, d, r2, sjf_t, f1_t = env.step_for_fair(a_joint[0])




            #
            # print("mean for traj {} is: {}".format(t, m))
            # print("std for traj {} is: {}".format(t, st))


        if d:
            # t += 1
            l_states_a = {}
            l_states_v = {}
            user_waits.append(r)
            wps = np.array(r)
            m = np.mean(wps)
            print("平均{}".format(m))
            st = np.std(wps)
            job_wait_ratio = 0
            for i in range(env.start, env.last_job_in_batch):
                wait = env.loads[i].scheduled_time - env.loads[i].submit_time
                job_wait_ratio += wait / float(env.loads[i].run_time + wait)
                job_waits.append(wait / float(env.loads[i].run_time + wait))
                # print("wait:{}".format(job_wait_ratio))
                # print(env.loads[i].run_time)
            average_job_wait_ratio = job_wait_ratio / JOB_SEQUENCE_SIZE
            job_wait_ratio_list.append(average_job_wait_ratio)

            mean.append(m)
            std.append(st)
            [o, co, user_index_list, done], d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset_test(ii * JOB_SEQUENCE_SIZE + 256), False, 0, 0, 0, 0, 0
            ii += 1
            print(ii)
            env.start = start_index[ii]
            r = [0.0 for _ in range(len(user_index_list))]
    np.save(r".\user_wait\S-MUF-DRl", np.concatenate(user_waits, axis=0))
    # np.save(r".\job_wait\MUF-DRl_backfill", np.array(job_waits, dtype=np.float32))
    mean = np.mean(mean)
    job_wait_ratio_list = np.array(job_wait_ratio_list)
    job_wait_ratio_all_average = np.mean(job_wait_ratio_list)
    print("用户等待时间比均值:{}".format(mean))
    print("作业等待时间比均值:{}".format(job_wait_ratio_all_average))
    # std = np.array(std)
    #
    # mid = np.median(std)
    # mean = np.mean(mid)
    # activate_index = []
    # for l in range(len(std)):
    #     if std[l] >= mid and std[l] <= 2*mean:
    #         activate_index.append(l+1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str,
                        default='./data/lublin_256.swf')  # RICC-2010-2 lublin_256.swf SDSC-SP2-1998-4.2-cln.swf
    parser.add_argument('--model', type=str, default='./data/lublin_256.schd')
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--trajs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--pre_trained', type=int, default=0)
    parser.add_argument('--trained_model', type=str, default='./data/logs/ppo_temp/ppo_temp_s0')
    parser.add_argument('--attn', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=1)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=100000)
    args = parser.parse_args()

    from src.spinup.spinup.utils.run_utils import setup_logger_kwargs

    # build absolute path for using in hpc_env.
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, './data/logs/')
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed, data_dir=log_data_dir)
    model_file = os.path.join(current_dir, args.trained_model)

    test_fcfs_fair(workload_file, args.model, trained_model=os.path.join(model_file, "simple_save"), seed=args.seed,  attn=args.attn, shuffle=args.shuffle,
                  backfil=args.backfil,
                  skip=args.skip, score_type=args.score_type, batch_job_slice=args.batch_job_slice)