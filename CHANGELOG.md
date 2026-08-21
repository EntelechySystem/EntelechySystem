# CHANGELOG

所有重要变更将记录在此文件中，遵循简洁明了的时间线记录方式。

## Unreleased

- 2026-08-18: **迁移本地场景 SDK 依赖**：按上游通知将 `world-model-kit` 迁移为 `scene-kit`，同步更新本地路径 `../scene-kit` 及 `scene_kit` 导入验证；历史条目保留旧名称。
- 2026-07-31: **Python 工具包统一命名为 `entelechy`**：`pyproject.toml` 的 `name` 由 `entelechysystem` 改为 `entelechy`；同步更新 `README.md`、`docs/开发者指南.md` 中的工具包名说明（源码包目录仍为 `EntelechySystem/`，导入路径不变）。
- 2026-07-31: **环境迁移至 uv (Python 3.14)**，彻底整理依赖体系：
  - `pyproject.toml`：直接依赖从 15 个精简为 6 个（`chardet`/`numpy`/`openpyxl`/`pandas` + 两个本地包），移除源码未使用的 `gymnasium`/`jupyter`/`matplotlib`/`pettingzoo`/`pygame`/`scipy`/`torchaudio`/`torchvision`/`pip`/`pandas-stubs` 等；`torch` 移入 `[project.optional-dependencies]`（模型无 torch 时自动退化为 NumPy）。
  - 新增 `[tool.uv.sources]`：以可编辑（editable）方式链接本地依赖 `../relation-entity-component-system`（RECS）与 `../world-model-kit`。
  - 新增 `scripts/uv_migrate.sh`：一键迁移脚本（备份→清理→重建→验证，幂等，支持 `--mirror` 国内镜像）。
  - `.gitignore`：新增 `old_requirements.txt`（迁移备份文件）。
  - 文档同步：`README.md`、`docs/开发者指南.md`、`docs/用户手册.md` 更新为 uv 工作流与新目录结构（`EntelechySystem_python/` → `EntelechySystem/`）。

- 2026-02-18: 新增文档目录 `docs/`，包含：
  - `docs/开发者指南.md` — 仓库结构、核心入口、配置体系与开发约定。
  - `docs/用户手册.md` — Windows 下快速上手步骤与配置/输出说明。
  - `docs/开发日志.md` — 日志规范与本次文档条目。
  - `docs/需求文档.md` — 阶段性功能/非功能需求与验收方式。
  - `docs/README.md` — docs 索引页（导航）。
- 2026-02-18: 更新 `README.md` 的“使用方法”部分，添加快速开始命令与 docs 链接。

## v0.0.8_alpha (2022-06-07)

- 初始版本信息见 `pyproject.toml` 与 `README.md`。
