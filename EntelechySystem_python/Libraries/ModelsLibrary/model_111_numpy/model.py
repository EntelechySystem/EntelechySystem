"""
模型
"""
from scipy.sparse import coo_matrix

from engine.externals import np, logging
# from model_define import NeuralNetUnit, NeuralNetUnit_ForHumanRead, OperationUnits
# from engine.libraries.models.model_define import ModelDefine
from engine.functions.ComplexIntelligenceSystem.Core.tools import Tools
from .model_define import ModelDefine
from .model_settings import ModelSettings


class Model:
    ne_units = None
    op_units_Control = None
    op_units_Container = None
    op_units_Goal = None
    op_units_Task = None
    op_units_Conception = None
    model_function = None

    def __init__(self, gb: dict):
        if gb['is_init_model']:
            self.init_model(gb)
        else:
            self.model_content(gb)
            pass  # if
        # model_function = ModelFunctions()
        pass  # function

    def model_content(self, gb: dict):

        pass  # function

    pass  # class

    def init_model(self, gb):

        ## 初始化单元众

        ### 定义神经元
        self.N_ne_units = gb['神经元预留位总数量']
        self.N_op_on = int(gb['运作单元预留位总数量'] / 2)
        self.N_op_units_Control = int(self.N_op_on / 64)
        self.N_op_units_Container = int(self.N_op_on / 8)
        self.N_op_units_Goal = int(self.N_op_on / 8)
        self.N_op_units_Task = int(self.N_op_on / 8)
        self.N_op_units_Conception = int(self.N_op_on / 2)

        ## 初始化单元众

        ### 定义神经元
        self.ne_units = ModelDefine.NeuralNetUnit(self.N_ne_units, gb['单个神经元连接预留位总数量'])

        # ### 定义用于人类阅读的神经元数据
        # self.ne_units_human = ModelDefine.NeuralNetUnit_ForHumanRead(self.N_ne_units, gb['单个神经元连接预留位总数量'])

        # 打印初始化的神经元
        logging.info("初始化的神经元")
        Tools.print_units_values(self.ne_units)
        # Tools.print_units_values(self.ne_units_human)

        gb['起始gid'] = 0

        ### 初始化控制运作单元
        self.op_units_Control = ModelDefine.OperationUnits(
            self.N_op_units_Control,
            gb['单个运作单元连接预留位总数量'],
            ModelSettings.dict_written_type_of_Units['control'],
            gb['起始gid']
        )
        logging.info("初始化的控制运作单元")
        Tools.print_units_values(self.op_units_Control)

        ### 初始化容器运作单元
        gb['起始gid'] += self.N_op_units_Control
        self.op_units_Container = ModelDefine.OperationUnits(
            self.N_op_units_Container,
            gb['单个运作单元连接预留位总数量'],
            ModelSettings.dict_written_type_of_Units['container'],
            gb['起始gid']
        )

        ### 初始化目标运作单元
        gb['起始gid'] += self.N_op_units_Container
        self.op_units_Goal = ModelDefine.OperationUnits(
            self.N_op_units_Goal,
            gb['单个运作单元连接预留位总数量'],
            ModelSettings.dict_written_type_of_Units['goal'],
            gb['起始gid']
        )
        logging.info("初始化的容器运作单元")
        Tools.print_units_values(self.op_units_Goal)

        ### 初始化任务运作单元
        gb['起始gid'] += self.N_op_units_Goal
        self.op_units_Task = ModelDefine.OperationUnits(
            self.N_op_units_Task,
            gb['单个运作单元连接预留位总数量'],
            ModelSettings.dict_written_type_of_Units['task'],
            gb['起始gid']
        )
        logging.info("初始化的任务运作单元")
        Tools.print_units_values(self.op_units_Task)

        ### 初始化概念运作单元
        gb['起始gid'] += self.N_op_units_Task
        self.op_units_Conception = ModelDefine.OperationUnits(
            self.N_op_units_Conception,
            gb['单个运作单元连接预留位总数量'],
            ModelSettings.dict_written_type_of_Units['conception'],
            gb['起始gid']
        )
        logging.info("初始化的概念运作单元")
        Tools.print_units_values(self.op_units_Conception)

        # 初始化模型单元结构

        ## 初始化控制单元结构（基于 Numpy 版本）

        #### 总控制中心

        # 选取 64 个控制单元做为总控制中心（一级控制中心）。这些控制单元之间相互连接，形成一个全连接网络。
        N_units_controlCenter = 8  # 一级控制中心之控制单元数量
        ids_point = 0  # 用于记录当前要开始选取的 ID 偏移值
        ids_from = np.arange(N_units_controlCenter)
        ids_to = ids_from.copy()
        ids_level1Center = ids_from.copy()
        # indices = np.vstack((ids_from, ids_to))
        # values = np.ones(N_units_controlCenter)
        self.op_units_Control.links_id[np.ix_(ids_from, ids_to)] = 1
        ids_point += N_units_controlCenter

        # 分级控制中心
        # 二级控制中心

        # 再选取 64 个控制单元做为2级控制中心。这些控制单元之间相互连接，形成一个全连接网络。二级控制中心
        N_controlUnits_level2Center = 4
        N_level2Center = 4
        for i in range(N_level2Center):

            # 同一个控制中心内部的控制单元之间相互连接，形成一个全连接网络。
            ids_from = np.arange(N_units_controlCenter, N_units_controlCenter + N_controlUnits_level2Center)
            ids_level2Center = ids_from.copy()
            ids_to = np.arange(N_units_controlCenter, N_units_controlCenter + N_controlUnits_level2Center)

            # indices = np.vstack((ids_from, ids_to))
            # values = np.ones(N_controlUnits_level2Center)
            self.op_units_Control.links_id[np.ix_(ids_from, ids_to)] = 1
            # 自己与自己不连接
            for i in range(N_controlUnits_level2Center):
                self.op_units_Control.links_id[ids_from[i], ids_to[i]] = 0

            # 同一级的控制中心之间暂时不连接，但是与上级控制中心连接
            ids_from = ids_level2Center[0]
            ids_to = ids_level1Center[i]
            self.op_units_Control.links_id[ids_from, ids_to] = 1
            ids_point += N_controlUnits_level2Center

            # 三级控制中心
            # 再选取 64 个控制单元做为3级控制中心。这些控制单元之间相互连接，形成一个全连接网络。
            N_controlUnits_level3Center = 4
            N_level3Center = 4
            for i in range(N_level3Center):

                # 同一个控制中心内部的控制单元之间相互连接，形成一个全连接网络。
                ids_from = np.arange(N_units_controlCenter, N_units_controlCenter + N_controlUnits_level3Center)
                ids_level3Center = ids_from.copy()
                ids_to = np.arange(N_units_controlCenter, N_units_controlCenter + N_controlUnits_level3Center)

                # indices = np.vstack((ids_from, ids_to))
                # values = np.ones(N_controlUnits_level3Center)
                self.op_units_Control.links_id[np.ix_(ids_from, ids_to)] = 1
                # 自己与自己不连接
                for i in range(N_controlUnits_level3Center):
                    self.op_units_Control.links_id[ids_from[i], ids_to[i]] = 0

                # 同一级的控制中心之间暂时不连接，但是与上级控制中心连接
                ids_from = ids_level3Center[0]
                ids_to = ids_level1Center[i]
                self.op_units_Control.links_id[ids_from, ids_to] = 1
                ids_point += N_controlUnits_level3Center

                # #NOW 每一个三级控制中心之每一个控制单元都连接一个概念单元
                ids_from = ids_level3Center[0]
                ids_to = np.arange(self.N_op_units_Control, self.N_op_units_Control + self.N_op_units_Conception)

        pass  # function

    def model_content(self, gb: dict):

        # 基本的概念运作单元

        # 感知模块
        # 设置基本的感知单元
        # 视觉感知单元

        pass  # function

    pass  # class
