"""
定义单元众 Units 及其相关操作
"""
from dataclasses import dataclass

import numpy as np

import torch

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
            self.links = torch.empty((N_units, max_N_links), dtype=torch.int32)  # 单元之连接
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

        PyTorch 版本  #NOTE 如果需要再启用

        字段如下：
        - gid: 单元之全局 ID
        - uid: 单元之 ID
        - state_on: 运作单元在物理层面上是否被启用，True 表示启用，False 表示未启用
        - units_type: 运作单元之类型
        - input_units: 运作单元之输入
        - output_units: 运作单元之输出
        - links_soft: 运作单元之软连接（N×N）
        - links_id: 运作单元之 id 硬连接（N×N）
        - content: 运作单元之内容
        - units_name: 运作单元之唯一名称
        - explanation: 运作单元之解释
        - notes: 运作单元之备注

        """

        def __init__(self, N_units: torch.uint64, N_char: torch.uint64, N_char_explanation: torch.uint64, N_char_notes: torch.uint64, unit_type: torch.uint8, init_gid: torch.uint64):
            """
            初始化运作单元（机器件）之结构化数组的数据

            Args:
                N_units: 运作单元容量
                N_char: 字符容量
                N_char_explanation: 用于解释的字符容量
                N_char_notes: 用于备注的字符容量
                unit_type: 运作单元之类型
                init_gid: 初始全局 ID 偏移值
            """
            self.gid = torch.arange(init_gid, init_gid + N_units)  # 运作单元之全局 ID
            self.uid = torch.arange(N_units, dtype=torch.int64)  # 运作单元之 ID
            self.state_on = torch.full((N_units,), False)  # 运作单元在物理层面上是否被启用，True 表示启用，False 表示未启用
            self.units_type = torch.from_numpy((np.ones(N_units) * unit_type).astype(np.uint8))  # 运作单元之类型
            self.input_units = torch.empty((N_units, N_char), dtype=torch.uint32)  # 运作单元值输入
            self.output_units = torch.empty((N_units, N_char), dtype=torch.uint32)  # 运作单元值输出
            self.links_soft = torch.tensor((N_units, N_units), dtype=torch.int32).to_sparse()  # 运作单元之 id 软连接（N×N COO 存储格式稀疏矩阵）
            self.links_id = torch.tensor((N_units, N_units), dtype=torch.bool).to_sparse()  # 运作单元之 id 硬连接（N×N COO 存储格式稀疏矩阵）
            self.content = torch.empty((N_units, N_char), dtype=torch.uint32)  # 运作单元之内容
            self.units_name = torch.from_numpy(np.array([Tools.encode_ascii_string_array_to_pytorch_tensor(Tools.generate_unique_identifier()) for i in range(N_units)]))  # 运作单元之名称
            self.explanation = torch.empty((N_units, N_char_explanation), dtype=torch.uint32)  # 运作单元之解释
            self.notes = torch.empty((N_units, N_char_notes), dtype=torch.uint32)  # 运作单元之备注
            pass  # function

        pass  # class

    @dataclass()
    class OperationUnitsForHuman():
        """
        定义可用于人类观察可读的运作单元（机器件）之结构化数组的数据

        PyTorch 版本  #NOTE 如果需要再启用

        字段如下：
        - gid: 单元之全局 ID
        - units_name: 运作单元之唯一名称
        - explanation: 运作单元之解释
        - notes: 运作单元之备注

        """

        def __init__(self, N_units: torch.uint64, N_char_explanation: torch.uint64, N_char_notes: torch.uint64, init_gid: torch.uint64):
            """
            初始化可用于人类观察可读的运作单元（机器件）之结构化数组的数据

            Args:
                N_units: 运作单元数量
                N_char_explanation: 用于解释的字符容量
                N_char_notes: 用于备注的字符容量
                init_gid: 初始全局 ID 偏移值
            """
            self.gid = torch.arange(init_gid, init_gid + N_units)  # 运作单元之全局 ID
            self.units_name = torch.from_numpy(np.array([Tools.encode_ascii_string_array_to_pytorch_tensor(Tools.generate_unique_identifier()) for i in range(N_units)]))  # 运作单元之名称
            self.explanation = torch.empty((N_units, N_char_explanation), dtype=torch.uint32)  # 运作单元之解释
            self.notes = torch.empty((N_units, N_char_notes), dtype=torch.uint32)  # 运作单元之备注
            pass  # function

        pass  # class

    @dataclass()
    class KeyData():
        """
        #TODO 定义匹配钥匙对数据结构
        """
        pass  # class


pass  # class
