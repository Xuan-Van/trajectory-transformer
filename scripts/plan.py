import json
import pdb
from os.path import join

import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.search import (
    beam_plan,
    make_prefix,
    extract_actions,
    update_context,
)

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'  # 数据集名称
    config: str = 'config.offline'  # 配置文件名称

#######################
######## 设置 ########
#######################

args = Parser().parse_args('plan')  # 解析命令行参数

#######################
####### 模型 ########
#######################

dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath,
        'data_config.pkl')  # 从配置文件加载数据集

gpt, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.gpt_loadpath,
        epoch=args.gpt_epoch, device=args.device)  # 加载GPT模型及其训练轮数

#######################
####### 数据集 #######
#######################

env = datasets.load_environment(args.dataset)  # 加载环境
renderer = utils.make_renderer(args)  # 创建渲染器
timer = utils.timer.Timer()  # 创建计时器

discretizer = dataset.discretizer  # 离散化器
discount = dataset.discount  # 折扣因子
observation_dim = dataset.observation_dim  # 观测维度
action_dim = dataset.action_dim  # 动作维度

value_fn = lambda x: discretizer.value_fn(x, args.percentile)  # 价值函数
preprocess_fn = datasets.get_preprocess_fn(env.name)  # 预处理函数

#######################
###### 主循环 ######
#######################

observation = env.reset()  # 重置环境并获取初始观测
total_reward = 0  # 总奖励初始化为0

## 用于渲染的观测序列
rollout = [observation.copy()]

## 用于条件化Transformer的先前（标记化）转换
context = []

T = env.max_episode_steps  # 最大步数
for t in range(T):

    observation = preprocess_fn(observation)  # 预处理当前观测

    if t % args.plan_freq == 0:
        ## 将先前的转换和当前观测连接起来作为模型的输入
        prefix = make_prefix(discretizer, context, observation, args.prefix_context)

        ## 从模型中采样序列，以`prefix`为起点
        sequence = beam_plan(
            gpt, value_fn, prefix,
            args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
            discount, args.max_context_transitions, verbose=args.verbose,
            k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
        )

    else:
        sequence = sequence[1:]  # 如果不是规划频率的倍数，则去掉序列的第一个元素

    ## [ horizon x transition_dim ] 将采样的标记转换为连续轨迹
    sequence_recon = discretizer.reconstruct(sequence)

    ## [ action_dim ] 从采样的轨迹中提取第一个动作
    action = extract_actions(sequence_recon, observation_dim, action_dim, t=0)

    ## 在环境中执行动作
    next_observation, reward, terminal, _ = env.step(action)

    ## 更新总奖励
    total_reward += reward
    score = env.get_normalized_score(total_reward)

    ## 更新观测序列和上下文转换
    rollout.append(next_observation.copy())
    context = update_context(context, discretizer, observation, action, reward, args.max_context_transitions)

    print(
        f'[ plan ] t: {t} / {T} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f} | '
        f'time: {timer():.2f} | {args.dataset} | {args.exp_name} | {args.suffix}\n'
    )

    ## 可视化
    if t % args.vis_freq == 0 or terminal or t == T:

        ## 保存当前计划
        renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), sequence_recon, env.state_vector())

        ## 保存到目前为止的回放
        renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

    if terminal: break  # 如果终止，则退出循环

    observation = next_observation  # 更新观测

## 将结果保存为json文件
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal, 'gpt_epoch': gpt_epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)