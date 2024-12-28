"""
CIS 010 实验主程序
"""

from engine.simulator.simulator import simulator


def main():
    # %% 配置项
    config = dict(
        folderpath_config=r"EntelechySystem_python/Libraries/ConfigsLibrary/config_test",  # 配置项文件夹路径
        foldername_engine=r"EntelechyEngine",  # 引擎所在工程文件夹名称
        foldername_CIS=r"EntelechyEngine",  # 引擎所在工程文件夹名称
        folderpath_relpath_engine=r"../",  # 引擎所在工程文件夹相对本实验项目根路径文件夹之相对路径
        folderpath_relpath_outputData=r"../",  # 输出数据所在工程文件夹相对本实验项目根路径文件夹之相对路径
        is_auto_confirmation=True,  # 是否自动确认一些比较危险的操作例如删除、移动、复制文件等。默认 False；
        # #BUG #NOTE 下面的配置，如果要用 False，那么不能用 Python 控制台运行，只能在终端用命令行运行。否则运行的时候不会继续推进。
        is_use_xlsx_as_config_file=False,  # 是否使用 Excel 文件作为配置文件。默认 True；
        is_prerun_config_program=False,  # 是否在实验主程序之前，预先运行配置库相关的生成配置程序。默认 True；
    )

    # %% 运行模拟器
    simulator(config)

    pass  # function


if __name__ == '__main__':
    main()
