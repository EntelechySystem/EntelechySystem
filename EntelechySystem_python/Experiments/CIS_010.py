"""
CIS 010 实验主程序
"""

from engine.simulator.simulator import simulator


def main():
    # %% 配置项
    config = dict()
    config['folderpath_config'] = r"ComplexIntelligenceSystem_python/Libraries/ConfigsLibrary/config_test"  # 配置项文件夹路径
    config['foldername_engine'] = r"EntelechyEngine"  # 引擎所在工程文件夹名称
    config['folderpath_relpath_engine'] = r"../"  # 引擎所在工程文件夹相对本实验项目根路径文件夹之相对路径
    config['folderpath_relpath_outputData'] = r"../"  # 输出数据所在工程文件夹相对本实验项目根路径文件夹之相对路径
    config['is_prerun_config_program'] = True  # 是否在实验主程序之前，预先运行了配置库相关的生成配置程序。默认 True；
    config['is_auto_confirmation'] = True  # 是否自动确认一些比较危险的操作例如删除、移动、复制文件等。默认 False；

    # %% 运行模拟器
    simulator(config)

    pass  # function


if __name__ == '__main__':
    main()
