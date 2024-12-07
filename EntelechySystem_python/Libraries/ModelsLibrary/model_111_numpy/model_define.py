"""
定义单元众 Units 及其相关操作
"""
from dataclasses import dataclass

import numpy as np
# from numba import njit

import torch
from scipy.sparse import coo_matrix, lil_matrix

from engine.functions.ComplexIntelligenceSystem.Core.tools import Tools
from .model_settings import ModelSettings
from dataclasses import dataclass


@dataclass
class ModelDefine():
    @dataclass
    class NeuralNetUnit():
        """
        定义神经网络器之结构化数组的数据类型。

        PyTorch 版本

        """

        def __init__(self, N_units: int, max_N_links: int):
            self.gid = torch.arange(N_units, dtype=torch.int64)  # 单元之全局 ID（N）
            self.uid = torch.arange(N_units, dtype=torch.int64)  # 单元之 ID（N）
            self.pos_x = torch.zeros(N_units, dtype=torch.float64)  # 单元之物理空间之 X 坐标
            self.pos_y = torch.zeros(N_units, dtype=torch.float64)  # 单元之物理空间之 Y 坐标
            # self.pos_z = torch.zeros(N_units, dtype=torch.float64)# 单元之物理空间之 Z 坐标 #NOTE 如果需要再启用
            self.input_units = torch.empty((N_units), dtype=torch.float32)  # 单元之输入
            self.output_units = torch.empty((N_units), dtype=torch.float32)  # 单元之输出
            # self.contents_obj = torch.empty((N_units), dtype=torch.string)  # 单元之内容
            # self.containers_obj = torch.empty((N_units), dtype=torch.string)  # 单元之容器
            # self.nodes_obj = torch.empty((N_units), dtype=torch.string)  # 单元之节点
            # self.links = torch.empty((N_units, max_N_links), dtype=torch.int32)  # 单元之连接
            # self.units_name = torch.array([Tools.generate_unique_identifier() for i in range(N_units)], np.dtype('S32'))
            # self.units_type = torch.array(np.full(N_units, ModelSettings.dict_written_type_of_Units['neuron']), dtype=torch.uint8)

            pass  # function

        pass  # class

    # @dataclass
    # class NeuralNetUnit_ForHumanRead():
    #     """
    #     定义专门用于人类观察可读的神经网络器之结构化数组的数据类型。
    #
    #     PyTorch 版本
    #     """
    #
    #     def __init__(self, N_units: int, max_num_links: int):
    #         """
    #         初始化可人类观察的神经单元之结构化数组的数据类型
    #
    #         Args:
    #             N_units:
    #             max_num_links:
    #         """
    #         self.gid = torch.arange(N_units, dtype=torch.int64)
    #         self.units_name = torch.array([Tools.generate_unique_identifier() for i in range(N_units)], np.dtype('S32'))
    #         self.units_type = torch.array(np.full(N_units, ModelSettings.dict_written_type_of_Units['neuron']), dtype=torch.uint8)
    #         pass  # function
    #
    #     pass  # class

    @dataclass()
    class OperationUnits():
        """
        定义运作单元（机器件）之结构化数组的数据

        Numpy 版本  #NOTE 如果需要再启用

        """

        def __init__(self, N_units: np.uint64, max_N_links: np.uint64, unit_type: np.uint8, init_gid: np.uint64):
            """
            初始化运作单元（机器件）之结构化数组的数据

            Args:
                N_units: 运作单元容量
                max_N_links: 运作单元之最大连接数
                unit_type: 运作单元之类型
                init_gid: 初始全局 ID 偏移值
            """
            self.gid = np.arange(init_gid, init_gid + N_units)  # 单元之全局 ID
            self.uid = np.arange(N_units)  # 单元之 ID
            self.state_on = np.full(N_units, False)  # 运作单元在物理层面上是否被启用，True 表示启用，False 表示未启用
            self.units_name = np.array([Tools.generate_unique_identifier() for i in range(N_units)])  # 运作单元之唯一名称
            self.units_type = np.full(N_units, unit_type)  # 运作单元之类型
            self.input_units = np.full(N_units, ' ', np.dtype('S128'))  # 运作单元之输入
            self.output_units = np.full(N_units, ' ', np.dtype('S128'))  # 运作单元之输出
            # self.links_soft = -np.ones((N_units, max_N_links), np.dtype(np.int32))  # 运作单元之软连接（N×M）
            # self.links_id = -np.ones((N_units, max_N_links), np.dtype(np.int32))  # 运作单元之 id 硬连接（N×M）
            self.links_soft = coo_matrix((N_units, N_units), dtype=np.int32)  # 运作单元之软连接（N×M）
            # self.links_id = coo_matrix((N_units, N_units), dtype=np.bool)  # 运作单元之 id 硬连接（N×M）
            self.links_id = lil_matrix((N_units, N_units), dtype=np.bool)  # 运作单元之 id 硬连接（N×M）
            self.content = np.full(N_units, ' ', np.dtype('S128'))  # 运作单元之内容
            self.explanation = np.full(N_units, ' ', np.dtype('S4096'))  # 运作单元之解释
            self.notes = np.full(N_units, ' ', np.dtype('S4096'))  # 运作单元之备注

            pass  # function

    # @dataclass()
    # class OperationUnitsForHuman():
    #     """
    #     定义可用于人类观察可读的运作单元（机器件）之结构化数组的数据
    #
    #     Numpy 版本  #NOTE 如果需要再启用
    #
    #     注意，连接的数据类型为 int32，因为连接的值可能为负数。值为 -1 表示未连接。
    #     """
    #
    #     def __init__(self, N_units: np.uint32, max_N_links: np.uint32, unit_type: np.uint8, init_gid: np.uint32):
    #         """
    #         初始化可用于人类观察可读的运作单元（机器件）之结构化数组的数据
    #
    #         Args:
    #             N_units: 运作单元容量
    #             max_N_links: 运作单元之最大连接数
    #             unit_type: 运作单元之类型
    #             init_gid: 初始全局 ID 偏移值
    #         """
    #         self.gid = np.arange(init_gid, init_gid + N_units)  # 单元之全局 ID
    #         self.explanation = np.full(N_units, ' ', np.dtype('S65536'))  # 运作单元之解释
    #         self.notes = np.full(N_units, ' ', np.dtype('S65536'))  # 运作单元之备注
    #         pass  # function
    #
    #     pass  # class

    @dataclass()
    class KeyData():
        """
        #TODO 定义匹配钥匙对数据结构
        """
        pass  # class


pass  # class
