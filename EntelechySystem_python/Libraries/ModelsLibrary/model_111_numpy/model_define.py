"""
定义单元众 Units 及其相关操作
"""

import numpy as np
# from numba import njit

from dataclasses import dataclass

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from engine.tools.encode_decode_tools import EncodeDecodeTools
except ModuleNotFoundError:
    from EntelechySystem_python.engine.tools.encode_decode_tools import EncodeDecodeTools

try:
    from .recs_relation_adapter import RecsRelationMatrixProxy, load_recs_classes
except ImportError:
    from recs_relation_adapter import RecsRelationMatrixProxy, load_recs_classes


RECS, _ = load_recs_classes()


@dataclass
class ModelDefine():
    @dataclass
    class OperationUnitEntities():
        """
        定义运作单元实体众之结构化数组的数据类型。
        """

        def __init__(self, N_units: int, max_N_links: int | None = None):
            self.gid = np.arange(N_units, dtype=np.uint64)  # 单元之全局 ID
            self.tid = np.zeros(N_units, dtype=np.uint64)  # 单元之类型 ID
            self.uid = np.zeros(N_units, dtype=np.uint64)  # 单元之 ID
            self.state_on = np.full(N_units, False)  # 运作单元在物理层面上是否被启用，True 表示启用，False 表示未启用

        pass  # class

    @dataclass
    class NeuralNetUnit():
        """
        定义神经网络器之结构化数组的数据类型。

        PyTorch 版本

        """

        def __init__(self, N_units: int, max_N_links: int):
            """初始化神经网络单元结构。

            当环境没有安装 PyTorch 时，自动回退为 NumPy 数组，保证最小可运行性。

            Args:
                N_units: 神经单元容量。
                max_N_links: 单元最大连接数（当前保留参数）。
            """
            if torch is None:
                self.gid = np.arange(N_units, dtype=np.int64)  # 单元之全局 ID（N）
                self.uid = np.arange(N_units, dtype=np.int64)  # 单元之 ID（N）
                self.pos_x = np.zeros(N_units, dtype=np.float64)  # 单元之物理空间之 X 坐标
                self.pos_y = np.zeros(N_units, dtype=np.float64)  # 单元之物理空间之 Y 坐标
                self.input_units = np.empty((N_units), dtype=np.float32)  # 单元之输入
                self.output_units = np.empty((N_units), dtype=np.float32)  # 单元之输出
                return

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
            n_units_int = int(N_units)
            init_gid_int = int(init_gid)

            self.gid = np.arange(init_gid_int, init_gid_int + n_units_int, dtype=np.uint64)  # 单元之全局 ID
            self.uid = np.arange(n_units_int, dtype=np.uint64)  # 单元之 ID
            self.state_on = np.full(n_units_int, False, dtype=np.bool_)  # 运作单元是否启用
            self.units_name = np.array([EncodeDecodeTools.generate_unique_identifier() for _ in range(n_units_int)])  # 运作单元之唯一名称
            self.units_type = np.full(n_units_int, unit_type, dtype=np.uint8)  # 运作单元之类型
            self.input_units = np.full(n_units_int, ' ', np.dtype('S128'))  # 运作单元之输入
            self.output_units = np.full(n_units_int, ' ', np.dtype('S128'))  # 运作单元之输出
            self.content = np.full(n_units_int, ' ', np.dtype('S128'))  # 运作单元之内容
            self.explanation = np.full(n_units_int, ' ', np.dtype('S4096'))  # 运作单元之解释
            self.notes = np.full(n_units_int, ' ', np.dtype('S4096'))  # 运作单元之备注

            # RECS 实体池：用于承载后续关系语义查询与批量属性更新。
            self.entities = RECS(
                capacity=n_units_int,
                attr_dtypes={
                    'gid': np.uint64,
                    'uid_local': np.uint64,
                    'units_type': np.uint8,
                    'state_on': np.bool_,
                },
            )
            self.entities.add(
                n_units_int,
                gid=self.gid,
                uid_local=self.uid,
                units_type=self.units_type,
                state_on=self.state_on,
                o=self.state_on,
            )

            self.links_soft = RecsRelationMatrixProxy(
                src_uids=self.gid,
                dst_uids=self.gid,
                relation_name=f'links_soft_type_{int(unit_type)}',
                initial_capacity=max(8, n_units_int * max(1, int(max_N_links) // 4)),
            )
            self.links_id = RecsRelationMatrixProxy(
                src_uids=self.gid,
                dst_uids=self.gid,
                relation_name=f'links_id_type_{int(unit_type)}',
                initial_capacity=max(8, n_units_int * max(1, int(max_N_links) // 2)),
            )

            pass  # function

        def sync_state_on_to_entities(self) -> None:
            """同步 `state_on` 到 RECS 实体池。

            Returns:
                None
            """
            all_indices = np.arange(self.entities.size, dtype=np.int64)
            state_values = self.state_on.astype(np.bool_, copy=False)
            self.entities.assign(all_indices, {'state_on': state_values, 'o': state_values})

        def connect_cartesian(
            self,
            src_local_ids: np.ndarray | list[int],
            dst_local_ids: np.ndarray | list[int],
            include_self: bool = True,
        ) -> None:
            """连接本地下标集合的笛卡尔积关系。

            Args:
                src_local_ids: 源本地下标集合。
                dst_local_ids: 目标本地下标集合。
                include_self: 是否保留自连接。

            Returns:
                None
            """
            self.links_id.connect_cartesian(src_local_ids, dst_local_ids, include_self=include_self)

        def connect_pairs(
            self,
            src_local_ids: np.ndarray | list[int],
            dst_local_ids: np.ndarray | list[int],
        ) -> None:
            """连接本地下标一一配对关系。

            Args:
                src_local_ids: 源本地下标集合。
                dst_local_ids: 目标本地下标集合。

            Returns:
                None
            """
            self.links_id.connect_pairs(src_local_ids, dst_local_ids)

        def disconnect_pairs(
            self,
            src_local_ids: np.ndarray | list[int],
            dst_local_ids: np.ndarray | list[int],
        ) -> None:
            """断开本地下标一一配对关系。

            Args:
                src_local_ids: 源本地下标集合。
                dst_local_ids: 目标本地下标集合。

            Returns:
                None
            """
            self.links_id.disconnect_pairs(src_local_ids, dst_local_ids)

        def connect_uid_pairs(
            self,
            src_uid_values: np.ndarray | list[int],
            dst_uid_values: np.ndarray | list[int],
        ) -> None:
            """连接 uid 一一配对关系（用于跨实体集合连接）。

            Args:
                src_uid_values: 源实体 uid 集合。
                dst_uid_values: 目标实体 uid 集合。

            Returns:
                None
            """
            self.links_id.connect_uid_pairs(src_uid_values, dst_uid_values)

        def export_links_dense(self, dtype: np.dtype = np.int8) -> np.ndarray:
            """导出 links_id 的局部稠密矩阵快照。

            Args:
                dtype: 导出矩阵 dtype。

            Returns:
                np.ndarray: 稠密矩阵。
            """
            return self.links_id.export_dense(dtype=dtype)

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
