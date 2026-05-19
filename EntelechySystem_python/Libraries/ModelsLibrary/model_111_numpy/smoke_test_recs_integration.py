"""model_111_numpy 与 RECS 关系层接入的最小冒烟验证。"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from EntelechySystem_python.Libraries.ModelsLibrary.model_111_numpy.model import Model
from EntelechySystem_python.Libraries.ModelsLibrary.model_111_numpy.model_define import ModelDefine


def test_operation_units_relation_proxy() -> None:
    """验证 OperationUnits 的 RECS 关系代理基础行为。"""
    units = ModelDefine.OperationUnits(
        N_units=np.uint64(16),
        max_N_links=np.uint64(8),
        unit_type=np.uint8(2),
        init_gid=np.uint64(100),
    )

    units.connect_cartesian(np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64), include_self=True)
    assert units.links_id.edge_count == 4, "笛卡尔积连接后边数应为 4"

    units.connect_pairs(np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64))
    assert units.links_id.edge_count == 4, "重复连接不应新增重复边"

    units.disconnect_pairs(np.array([1], dtype=np.int64), np.array([3], dtype=np.int64))
    assert units.links_id.edge_count == 3, "删除一条边后边数应为 3"

    dense = units.export_links_dense()
    assert dense[0, 2] == 1
    assert dense[1, 3] == 0


def test_model_init_with_recs_relations() -> None:
    """验证 Model 初始化后控制-概念关系边已写入 RECS。"""
    gb = {
        "is_init_model": True,
        "计算神经元预留位总数量": 2 ** 12,
        "运作单元预留位总数量": 2 ** 16,
        "单个神经元连接预留位总数量": 8,
        "单个运作单元连接预留位总数量": 2 ** 8,
    }

    model = Model(gb)
    assert model.op_units_Control.links_id.edge_count > 0, "控制单元关系边应被创建"

    conception_uid_set = set(model.op_units_Conception.gid.astype(np.int64).tolist())
    relation = model.op_units_Control.links_id.relation
    dst_uids = relation.d["dst_uid"][: relation.size]
    has_cross_edges = any(int(uid) in conception_uid_set for uid in dst_uids)
    assert has_cross_edges, "应存在控制单元到概念单元的跨集合关系边"


if __name__ == "__main__":
    test_operation_units_relation_proxy()
    test_model_init_with_recs_relations()
    print("smoke_test_recs_integration: PASSED")
