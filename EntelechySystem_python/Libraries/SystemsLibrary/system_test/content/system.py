"""
测试用的系统
"""

import logging
# from ComplexIntelligenceSystem_python.Core.define_units import define_neurons_units, define_op_units
from ComplexIntelligenceSystem_python.Core.define_units import NeuronsUnits, NeuronsUnits_ForHumanRead, OperationUnits
from ComplexIntelligenceSystem_python.Core.tools import Tools
from ComplexIntelligenceSystem_python.Core.settings import Settings
# from AgentsWorldSystem_python.Libraries.WorldLibrary.中国象棋_test.world_environment import *
import logging



def system(para: dict, gb: dict):
    ## #NOW 导入智能体模型
    model = Model(gb)

    ## #NOW 导入临时的简单的世界环境

    ## #NOW 导入临时的简单的先验知识

    ## #NOW 开始交互式学习
    gb['is_interactive_learning'] = True
    times_to_interactive = 10
    while times_to_interactive > 0:
        times_to_interactive -= 1
        pass  # while

    pass  # function


if __name__ == '__main__':
    pass  # if
