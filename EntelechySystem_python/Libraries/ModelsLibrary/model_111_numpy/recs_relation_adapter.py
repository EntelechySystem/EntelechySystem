"""RECS 关系适配器。

该模块用于把 ES 里原有的二维矩阵关系赋值语法，平滑迁移为 RECS 的边表关系。
适配器重点提供两类能力：

1. 集合到集合（笛卡尔积）连接。
2. 一一配对连接/断开。

同时提供 `__setitem__` 兼容接口，便于旧代码逐步替换。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import numpy as np


def load_recs_classes() -> tuple[type, type]:
    """加载 RECS 与 Relation 类。

    优先从当前 Python 环境导入 `recs`，若不可用则回退到工作区同级
    `relation-entity-component-system` 源码路径。

    Returns:
        tuple[type, type]: `(RECS, Relation)` 类对象。

    Raises:
        ModuleNotFoundError: 当无法导入 `recs` 时抛出。
    """
    try:
        from recs import RECS, Relation

        return RECS, Relation
    except ModuleNotFoundError:
        candidate = Path(__file__).resolve().parents[5] / "relation-entity-component-system"
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

        from recs import RECS, Relation

        return RECS, Relation


_, Relation = load_recs_classes()


@dataclass
class RecsRelationMatrixProxy:
    """用 RECS Relation 模拟矩阵关系操作的代理。

    Args:
        src_uids: 源实体 uid 空间（用于本地下标到 uid 的映射）。
        dst_uids: 目标实体 uid 空间（用于本地下标到 uid 的映射）。
        relation_name: 关系名称。
        initial_capacity: 关系边表初始容量。
    """

    src_uids: np.ndarray
    dst_uids: np.ndarray
    relation_name: str
    initial_capacity: int = 1024

    def __post_init__(self) -> None:
        """初始化内部 Relation 和边索引映射。"""
        self.src_uids = np.asarray(self.src_uids, dtype=np.int64)
        self.dst_uids = np.asarray(self.dst_uids, dtype=np.int64)
        self._relation = Relation(
            name=self.relation_name,
            capacity=max(8, int(self.initial_capacity)),
            attr_dtypes={"value": np.int8},
        )
        self._edge_index: dict[tuple[int, int], int] = {}

    @property
    def relation(self):
        """返回底层 RECS Relation 对象。"""
        return self._relation

    @property
    def edge_count(self) -> int:
        """返回关系边数量。"""
        return int(self._relation.size)

    def __setitem__(self, key: tuple[Any, Any], value: Any) -> None:
        """兼容矩阵式赋值写法。

        Args:
            key: 二元下标 `(rows, cols)`。
            value: 赋值标量。`0` 表示断开，其他值表示连接。

        Raises:
            TypeError: 当 key 不是二维下标或 value 不是标量时抛出。
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("关系赋值必须使用二维下标 key=(rows, cols)。")

        scalar = np.asarray(value)
        if scalar.size != 1:
            raise TypeError("当前适配器仅支持标量赋值。")

        rows = self._normalize_selector(key[0], self.src_uids.shape[0])
        cols = self._normalize_selector(key[1], self.dst_uids.shape[0])
        src_local, dst_local = self._expand_pairs(rows, cols)

        src_uid = self.src_uids[src_local]
        dst_uid = self.dst_uids[dst_local]

        if int(scalar.item()) == 0:
            self.disconnect_uid_pairs(src_uid, dst_uid)
        else:
            self.connect_uid_pairs(src_uid, dst_uid)

    def connect_cartesian(
        self,
        src_local_ids: np.ndarray | list[int],
        dst_local_ids: np.ndarray | list[int],
        include_self: bool = True,
    ) -> None:
        """连接两个本地下标集合的笛卡尔积关系。

        Args:
            src_local_ids: 源实体本地下标集合。
            dst_local_ids: 目标实体本地下标集合。
            include_self: 当源和目标在同一集合中时，是否保留自连接。
        """
        src_local = self._normalize_vector_indices(src_local_ids, self.src_uids.shape[0])
        dst_local = self._normalize_vector_indices(dst_local_ids, self.dst_uids.shape[0])
        if src_local.size == 0 or dst_local.size == 0:
            return

        src_expand = np.repeat(src_local, dst_local.size)
        dst_expand = np.tile(dst_local, src_local.size)

        if not include_self:
            mask = src_expand != dst_expand
            src_expand = src_expand[mask]
            dst_expand = dst_expand[mask]

        self.connect_uid_pairs(self.src_uids[src_expand], self.dst_uids[dst_expand])

    def connect_pairs(
        self,
        src_local_ids: np.ndarray | list[int],
        dst_local_ids: np.ndarray | list[int],
    ) -> None:
        """连接本地下标的一一配对关系。

        Args:
            src_local_ids: 源实体本地下标集合。
            dst_local_ids: 目标实体本地下标集合。

        Raises:
            ValueError: 当两端长度不一致时抛出。
        """
        src_local = self._normalize_vector_indices(src_local_ids, self.src_uids.shape[0])
        dst_local = self._normalize_vector_indices(dst_local_ids, self.dst_uids.shape[0])
        if src_local.shape[0] != dst_local.shape[0]:
            raise ValueError("配对连接要求源和目标长度一致。")

        self.connect_uid_pairs(self.src_uids[src_local], self.dst_uids[dst_local])

    def disconnect_pairs(
        self,
        src_local_ids: np.ndarray | list[int],
        dst_local_ids: np.ndarray | list[int],
    ) -> None:
        """断开本地下标的一一配对关系。

        Args:
            src_local_ids: 源实体本地下标集合。
            dst_local_ids: 目标实体本地下标集合。

        Raises:
            ValueError: 当两端长度不一致时抛出。
        """
        src_local = self._normalize_vector_indices(src_local_ids, self.src_uids.shape[0])
        dst_local = self._normalize_vector_indices(dst_local_ids, self.dst_uids.shape[0])
        if src_local.shape[0] != dst_local.shape[0]:
            raise ValueError("配对断开要求源和目标长度一致。")

        self.disconnect_uid_pairs(self.src_uids[src_local], self.dst_uids[dst_local])

    def connect_uid_pairs(
        self,
        src_uid_values: np.ndarray | list[int],
        dst_uid_values: np.ndarray | list[int],
    ) -> None:
        """按 uid 一一配对添加关系边。

        Args:
            src_uid_values: 源实体 uid 集合。
            dst_uid_values: 目标实体 uid 集合。

        Raises:
            ValueError: 当两端长度不一致时抛出。
        """
        src_uid = np.asarray(src_uid_values, dtype=np.int64).reshape(-1)
        dst_uid = np.asarray(dst_uid_values, dtype=np.int64).reshape(-1)
        if src_uid.shape[0] != dst_uid.shape[0]:
            raise ValueError("uid 配对连接要求源和目标长度一致。")
        if src_uid.size == 0:
            return

        new_src: list[int] = []
        new_dst: list[int] = []
        new_keys: list[tuple[int, int]] = []

        for src_value, dst_value in zip(src_uid, dst_uid):
            key = (int(src_value), int(dst_value))
            if key in self._edge_index:
                continue
            new_src.append(key[0])
            new_dst.append(key[1])
            new_keys.append(key)

        if not new_src:
            return

        added_indices = self._relation.add(
            np.asarray(new_src, dtype=np.int64),
            np.asarray(new_dst, dtype=np.int64),
            value=np.ones(len(new_src), dtype=np.int8),
        )
        for key, rel_index in zip(new_keys, added_indices):
            self._edge_index[key] = int(rel_index)

    def disconnect_uid_pairs(
        self,
        src_uid_values: np.ndarray | list[int],
        dst_uid_values: np.ndarray | list[int],
    ) -> None:
        """按 uid 一一配对删除关系边。

        Args:
            src_uid_values: 源实体 uid 集合。
            dst_uid_values: 目标实体 uid 集合。

        Raises:
            ValueError: 当两端长度不一致时抛出。
        """
        src_uid = np.asarray(src_uid_values, dtype=np.int64).reshape(-1)
        dst_uid = np.asarray(dst_uid_values, dtype=np.int64).reshape(-1)
        if src_uid.shape[0] != dst_uid.shape[0]:
            raise ValueError("uid 配对断开要求源和目标长度一致。")
        if src_uid.size == 0:
            return

        remove_keys = {(int(src_value), int(dst_value)) for src_value, dst_value in zip(src_uid, dst_uid)}
        if not remove_keys:
            return

        rel_size = int(self._relation.size)
        if rel_size == 0:
            return

        current_src = self._relation.d["src_uid"][:rel_size]
        current_dst = self._relation.d["dst_uid"][:rel_size]

        remove_mask = np.fromiter(
            ((int(src_value), int(dst_value)) in remove_keys for src_value, dst_value in zip(current_src, current_dst)),
            dtype=bool,
            count=rel_size,
        )
        if np.any(remove_mask):
            keep_mask = ~remove_mask
            keep_count = int(np.count_nonzero(keep_mask))
            for field_name, field_values in self._relation.d.items():
                field_values[:keep_count] = field_values[:rel_size][keep_mask]
            self._relation.size = keep_count
            self._rebuild_edge_index()

    def export_dense(self, dtype: np.dtype = np.int8) -> np.ndarray:
        """导出当前关系到稠密矩阵（仅映射到已知本地 uid 空间）。

        Args:
            dtype: 输出矩阵 dtype。

        Returns:
            np.ndarray: 形状为 `(len(src_uids), len(dst_uids))` 的稠密矩阵。
        """
        dense = np.zeros((self.src_uids.shape[0], self.dst_uids.shape[0]), dtype=dtype)
        if self._relation.size == 0:
            return dense

        src_map = {int(uid): idx for idx, uid in enumerate(self.src_uids)}
        dst_map = {int(uid): idx for idx, uid in enumerate(self.dst_uids)}

        src_values = self._relation.d["src_uid"][: self._relation.size]
        dst_values = self._relation.d["dst_uid"][: self._relation.size]

        for src_value, dst_value in zip(src_values, dst_values):
            src_pos = src_map.get(int(src_value))
            dst_pos = dst_map.get(int(dst_value))
            if src_pos is None or dst_pos is None:
                continue
            dense[src_pos, dst_pos] = 1

        return dense

    def _rebuild_edge_index(self) -> None:
        """重建 `(src_uid, dst_uid) -> edge_idx` 映射。"""
        self._edge_index.clear()
        src_values = self._relation.d["src_uid"][: self._relation.size]
        dst_values = self._relation.d["dst_uid"][: self._relation.size]

        for rel_index, (src_value, dst_value) in enumerate(zip(src_values, dst_values)):
            self._edge_index[(int(src_value), int(dst_value))] = rel_index

    @staticmethod
    def _normalize_selector(selector: Any, size: int) -> np.ndarray:
        """将索引选择器规范化为整数数组。

        Args:
            selector: 支持 int/slice/ndarray/list/bool-mask。
            size: 当前轴长度。

        Returns:
            np.ndarray: 整数索引数组。

        Raises:
            ValueError: 当布尔掩码长度不匹配时抛出。
            TypeError: 当索引类型不支持时抛出。
        """
        if isinstance(selector, slice):
            start, stop, step = selector.indices(size)
            return np.arange(start, stop, step, dtype=np.int64)

        if isinstance(selector, (int, np.integer)):
            index = int(selector)
            if index < 0:
                index += size
            if index < 0 or index >= size:
                raise ValueError("索引越界。")
            return np.asarray([index], dtype=np.int64)

        array = np.asarray(selector)
        if array.dtype == np.bool_:
            flat = array.reshape(-1)
            if flat.shape[0] != size:
                raise ValueError("布尔掩码长度必须与轴长度一致。")
            return np.nonzero(flat)[0].astype(np.int64, copy=False)

        if np.issubdtype(array.dtype, np.integer):
            normalized = array.astype(np.int64, copy=False)
            normalized = np.where(normalized < 0, normalized + size, normalized)
            if np.any((normalized < 0) | (normalized >= size)):
                raise ValueError("索引越界。")
            return normalized

        raise TypeError(f"不支持的索引类型: {type(selector)}")

    @staticmethod
    def _normalize_vector_indices(indices: np.ndarray | list[int], size: int) -> np.ndarray:
        """将索引向量标准化为一维整数数组。

        Args:
            indices: 索引向量。
            size: 当前轴长度。

        Returns:
            np.ndarray: 一维整数数组。
        """
        normalized = RecsRelationMatrixProxy._normalize_selector(indices, size)
        return normalized.reshape(-1)

    @staticmethod
    def _expand_pairs(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """扩展为一一配对索引。

        规则：
        - 若可广播，则按广播结果展开。
        - 若不可广播但均为一维数组，则按笛卡尔积展开。

        Args:
            rows: 行索引数组。
            cols: 列索引数组。

        Returns:
            tuple[np.ndarray, np.ndarray]: 扩展后的一一配对行列索引。
        """
        try:
            row_b, col_b = np.broadcast_arrays(rows, cols)
            return row_b.reshape(-1), col_b.reshape(-1)
        except ValueError:
            row_1d = rows.reshape(-1)
            col_1d = cols.reshape(-1)
            row_expand = np.repeat(row_1d, col_1d.shape[0])
            col_expand = np.tile(col_1d, row_1d.shape[0])
            return row_expand, col_expand
