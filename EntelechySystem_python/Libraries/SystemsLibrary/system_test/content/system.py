"""
测试用的系统
"""
# import pygame
import numpy as np
import pygame

from engine.tools.Tools import Tools
from engine.externals import Path, pd
import matplotlib.pyplot as plt

import logging
# from AgentsWorldSystem_python.Libraries.WorldLibrary.中国象棋.world_environment import *
# from AgentsWorldSystem_python.Libraries.WorldLibrary.烧水倒水.world_environment import *

from engine.libraries.models.model import Model
from engine.libraries.world_environment.world_environment import Scenario, World


def system(para: dict, gb: dict):
    ## #NOW 导入智能体大脑模型
    m = Model(gb)

    ## #NOW 导入临时的简单的世界环境

    env = Scenario(
        render_mode='human',
        local_ratio=0.5,
    )
    # observations, infos = my_env.reset()
    env.reset()

    ## #NOW 导入临时的简单的先验知识
    ### 初始化状态下，直接先预置一些概念
    folderpath_conceptions_data = Path(gb['folderpath_data'] / 'conceptions')
    df_基础概念_现代汉语字符库 = pd.read_pickle(folderpath_conceptions_data / '基础概念_现代汉语字符库.pkl')
    ### 导入先验知识
    # 预先导入基础类别为【数字集】、【英文标点符号和字符集】、【中文字符集】，以及【汉字集】里的常用字
    df_基础概念_现代汉语字符库_常用字 = df_基础概念_现代汉语字符库[df_基础概念_现代汉语字符库['是否常用字'] == '是']

    ## 预置先验知识到智能体大脑模型
    ids = range(0, len(df_基础概念_现代汉语字符库_常用字))
    m.op_units_Conception.content[ids] = np.vectorize(lambda char: char.encode('utf-8'))(df_基础概念_现代汉语字符库_常用字['内容_010'])
    m.op_units_Conception.state_on[ids] = True

    ## #NOW 开始交互式学习
    gb['is_interactive_learning'] = True

    # times_to_interactive = 10
    # while times_to_interactive > 0:
    #     times_to_interactive -= 1
    #     pass  # while
    #
    # pass  # function

    while env.agents:

        # 手动采样
        actions = {}
        for agent in env.agents:
            action_说话 = "你好，世界！"
            action_说话_encode = Tools.encode_unicode_to_array(action_说话)
            action = {
                # '说话': actions[agent]['说话'],
                '说话': Tools.encode_unicode_to_array(Tools.process_string_to_fix_length(action_说话)[0]),
                # '移动运动': np.clip(1, my_env.action_space(agent)['移动运动'].low, my_env.action_space(agent)['移动运动'].high),
                '移动运动': np.int64(0),
                '抓取运动': np.int64(0),
                '表情': np.int64(0),
                '睡眠': np.int64(0),
                '饮食': np.int64(0),
            }
            actions[agent] = action
            pass  # for

        # #DEBUG
        print("actions: ")
        for i in actions:
            print(f"agent: {i}")
            print(f"说话: {Tools.decode_unicode_array_to_string(actions[i]['说话'])}")
            print(f"移动运动: {actions[i]['移动运动']}")
            print(f"抓取运动: {actions[i]['抓取运动']}")
            print(f"表情: {actions[i]['表情']}")
            print(f"睡眠: {actions[i]['睡眠']}")
            print(f"饮食: {actions[i]['饮食']}")

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # #NOW 把 observations 里的 Numpy 数组内容转换成人类可阅读的内容
        # #DEBUG
        print("observations: ")
        for agent_name, agent_observations in observations.items():
            print(f"{agent_name}:")
            # 逆向 observations 里的 Numpy 数组内容为一个个具体的观察值
            observation_看到的内容 = agent_observations[:env.world.dim_视觉]
            #  将其重新变成一个图像形状的 Numpy 数组，然后用 Matplotlib 可视化显示该图像
            observation_看到的内容 = np.clip(observation_看到的内容.reshape(env.world.shape_视觉), 0, 255).astype(np.uint8)
            print(f"看到的内容: {observation_看到的内容}")
            plt.imshow(observation_看到的内容)
            plt.title('Observation 看到的内容')
            # plt.show()

            agent_observations = agent_observations[env.world.dim_视觉:]
            observation_听到的内容 = agent_observations[:env.world.dim_听觉]
            print(f"听到的内容: {Tools.decode_unicode_array_to_string(observation_听到的内容)}")
            agent_observations = agent_observations[env.world.dim_听觉:]
            observation_摸到的内容 = agent_observations[:env.world.dim_触觉]
            print(f"摸到的内容: {observation_摸到的内容}")  # BUG 获取错误信息，还是之前听到的内容，而不是所需的
            agent_observations = agent_observations[env.world.dim_触觉:]
            observation_闻到的内容 = agent_observations[:env.world.dim_嗅觉]
            print(f"闻到的内容: {observation_闻到的内容}")  # BUG 获取到错误信息，还是之前听到的内容，而不是所需的
            agent_observations = agent_observations[env.world.dim_嗅觉:]
            observation_感知的温度 = agent_observations[0]
            print(f"感知的温度: {observation_感知的温度}")  # BUG 获取到错误信息，还是之前听到的内容，而不是所需的
            agent_observations = agent_observations[1:]
            observation_说话状态 = agent_observations[0]
            print(f"说话状态: {observation_说话状态}")  # BUG 获取到错误信息，还是之前听到的内容，而不是所需的
            agent_observations = agent_observations[1:]
            observation_抓取状态 = agent_observations[0]
            print(f"抓取状态: {observation_抓取状态}")  # BUG 获取到错误信息，还是之前听到的内容，而不是所需的
            agent_observations = agent_observations[1:]
            observation_困倦状态 = agent_observations[0]
            print(f"困倦状态: {observation_困倦状态}")  # BUG 获取到错误信息，还是之前听到的内容，而不是所需的
            agent_observations = agent_observations[1:]
            observation_饥饿状态 = agent_observations[0]
            print(f"饥饿状态: {observation_饥饿状态}")  # BUG 获取到错误信息，还是之前听到的内容，而不是所需的
            agent_observations = agent_observations[1:]
            observation_呈现的表情 = agent_observations[0]
            print(f"呈现的表情: {observation_呈现的表情}")  # BUG 获取到错误信息，还是之前听到的内容，而不是所需的
            agent_observations = agent_observations[1:]
            pass  # for

        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()

        pass  # while 结束交互式学习


if __name__ == '__main__':
    pass  # if
