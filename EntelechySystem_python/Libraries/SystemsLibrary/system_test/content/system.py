"""
测试用的系统
"""
from engine.externals import Path, pd
import logging
# from ComplexIntelligenceSystem_python.Core.define_units import define_neurons_units, define_op_units
from ComplexIntelligenceSystem_python.Core.define_units import NeuronsUnits, NeuronsUnits_ForHumanRead, OperationUnits
from ComplexIntelligenceSystem_python.Core.tools import Tools
from ComplexIntelligenceSystem_python.Core.settings import Settings
# from AgentsWorldSystem_python.Libraries.WorldLibrary.中国象棋.world_environment import *
# from AgentsWorldSystem_python.Libraries.WorldLibrary.烧水倒水.world_environment import *

from ComplexIntelligenceSystem_python.Libraries.ModelsLibrary.model_test.model import Model
from ElementalConceptionSystem_python.Libararies.ConceptionLibrary.conception_test.conception import Conception


def system(para: dict, gb: dict):
    ## #NOW 导入智能体模型
    model = Model(gb)

    ## #NOW 导入临时的简单的世界环境

    ## #NOW 导入临时的简单的先验知识
    ### 初始化状态下，直接先预置一些概念
    folderpath_conceptions_data = Path(gb['folderpath_data'] / 'conceptions')
    df_基础概念_现代汉语字符库 = pd.read_pickle(folderpath_conceptions_data / '基础概念_现代汉语字符库.pkl')


    ## #NOW 开始交互式学习
    gb['is_interactive_learning'] = True
    times_to_interactive = 10
    while times_to_interactive > 0:
        times_to_interactive -= 1
        pass  # while

    pass  # function


if __name__ == '__main__':
    pass  # if
