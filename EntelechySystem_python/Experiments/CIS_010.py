"""
CIS 010 实验主程序
"""

from Engine import engine


def main():
    # %% 配置项
    config = dict()
    config['folderpath_config'] = r"Libraries/configs_library/config_2024-03"  # 配置项文件夹路径
    # config['filename_config'] = r"config.py"  # 配置项文件夹路径
    config['foldername_engine'] = r"Engine"  # 引擎所在工程文件夹名称
    config['folderpath_relpath_engine'] = r"../"  # 引擎所在工程文件夹相对本实验项目根路径文件夹之相对路径
    config['folderpath_relpath_outputData'] = r"../"  # 输出数据所在工程文件夹相对本实验项目根路径文件夹之相对路径
    config['is_auto_confirmation'] = True  # 是否自动确认一些比较危险的操作例如删除、移动、复制文件等。默认 False；

    # %% 运行引擎
    engine(config)

    pass  # function


if __name__ == '__main__':
    main()

