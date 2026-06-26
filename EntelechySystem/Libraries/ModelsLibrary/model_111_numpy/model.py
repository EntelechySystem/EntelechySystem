"""
模型
"""

import logging
import numpy as np
# from model_define import NeuralNetUnit, NeuralNetUnit_ForHumanRead, OperationUnits
# from EntelechySystem.engine.libraries.models.model_define import ModelDefine
try:
    from engine.tools.encode_decode_tools import EncodeDecodeTools
except ModuleNotFoundError:
    from EntelechySystem.engine.tools.encode_decode_tools import EncodeDecodeTools
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

    def init_model(self, gb: dict) -> None:

        ## 初始化单元众

        ### 定义神经元
        self.N_ne_units = int(gb['计算神经元预留位总数量'])
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
        EncodeDecodeTools.print_units_values(self.ne_units)
        # Tools.print_units_values(self.ne_units_human)

        gb['起始gid'] = 0

        ### 初始化运作单元实体
        self.op_entity_units = ModelDefine.OperationUnitEntities(
            gb['运作单元预留位总数量'],
        )

        ### 初始化控制运作单元
        self.op_units_Control = ModelDefine.OperationUnits(
            self.N_op_units_Control,
            gb['单个运作单元连接预留位总数量'],
            ModelSettings.dict_written_type_of_Units['control'],
            gb['起始gid']
        )
        logging.info("初始化的控制运作单元")
        EncodeDecodeTools.print_units_values(self.op_units_Control)
        self.op_entity_units.uid[gb['起始gid']:gb['起始gid'] + self.N_op_units_Control] = self.op_units_Control.uid

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
        EncodeDecodeTools.print_units_values(self.op_units_Goal)

        ### 初始化任务运作单元
        gb['起始gid'] += self.N_op_units_Goal
        self.op_units_Task = ModelDefine.OperationUnits(
            self.N_op_units_Task,
            gb['单个运作单元连接预留位总数量'],
            ModelSettings.dict_written_type_of_Units['task'],
            gb['起始gid']
        )
        logging.info("初始化的任务运作单元")
        EncodeDecodeTools.print_units_values(self.op_units_Task)

        ### 初始化概念运作单元
        gb['起始gid'] += self.N_op_units_Task
        self.op_units_Conception = ModelDefine.OperationUnits(
            self.N_op_units_Conception,
            gb['单个运作单元连接预留位总数量'],
            ModelSettings.dict_written_type_of_Units['conception'],
            gb['起始gid']
        )
        logging.info("初始化的概念运作单元")
        EncodeDecodeTools.print_units_values(self.op_units_Conception)

        # 初始化模型单元结构

        ## 初始化控制单元结构（基于 Numpy 版本）

        #### 总控制中心

        # 选取 64 个控制单元做为总控制中心（一级控制中心）。这些控制单元之间相互连接，形成一个全连接网络。
        N_units_controlCenter = 8  # 一级控制中心之控制单元数量
        ids_point = 0  # 用于记录当前要开始选取的 ID 偏移值
        ids_level1Center = np.arange(ids_point, ids_point + N_units_controlCenter, dtype=np.int64)
        self.op_units_Control.connect_cartesian(ids_level1Center, ids_level1Center, include_self=True)
        ids_point += N_units_controlCenter

        # 分级控制中心
        # 二级控制中心

        # 再选取 64 个控制单元做为2级控制中心。这些控制单元之间相互连接，形成一个全连接网络。二级控制中心
        N_controlUnits_level2Center = 4
        N_level2Center = 4
        for level2_idx in range(N_level2Center):

            # 同一个控制中心内部的控制单元之间相互连接，形成一个全连接网络。
            ids_level2Center = np.arange(ids_point, ids_point + N_controlUnits_level2Center, dtype=np.int64)
            self.op_units_Control.connect_cartesian(ids_level2Center, ids_level2Center, include_self=False)

            # 同一级的控制中心之间暂时不连接，但是与上级控制中心连接
            self.op_units_Control.connect_pairs(
                np.asarray([ids_level2Center[0]], dtype=np.int64),
                np.asarray([ids_level1Center[level2_idx]], dtype=np.int64),
            )
            ids_point += N_controlUnits_level2Center

            # 三级控制中心
            # 再选取 64 个控制单元做为3级控制中心。这些控制单元之间相互连接，形成一个全连接网络。
            N_controlUnits_level3Center = 4
            N_level3Center = 4
            for level3_idx in range(N_level3Center):

                # 同一个控制中心内部的控制单元之间相互连接，形成一个全连接网络。
                ids_level3Center = np.arange(ids_point, ids_point + N_controlUnits_level3Center, dtype=np.int64)
                self.op_units_Control.connect_cartesian(ids_level3Center, ids_level3Center, include_self=False)

                # 同一级的控制中心之间暂时不连接，但是与上级控制中心连接
                self.op_units_Control.connect_pairs(
                    np.asarray([ids_level3Center[0]], dtype=np.int64),
                    np.asarray([ids_level2Center[level3_idx % ids_level2Center.size]], dtype=np.int64),
                )
                ids_point += N_controlUnits_level3Center

                # #NOW 每一个三级控制中心之每一个控制单元都连接一个概念单元
                control_src_uids = self.op_units_Control.gid[ids_level3Center].astype(np.int64, copy=False)
                conception_dst_uids = self.op_units_Conception.gid.astype(np.int64, copy=False)

                if conception_dst_uids.size > 0:
                    src_expand = np.repeat(control_src_uids, conception_dst_uids.size)
                    dst_expand = np.tile(conception_dst_uids, control_src_uids.size)
                    self.op_units_Control.connect_uid_pairs(src_expand, dst_expand)
        pass  # function

    def model_content(self, gb: dict):

        # 基本的概念运作单元

        # 感知模块
        # 设置基本的感知单元
        # 视觉感知单元

        pass  # function

    pass  # class
