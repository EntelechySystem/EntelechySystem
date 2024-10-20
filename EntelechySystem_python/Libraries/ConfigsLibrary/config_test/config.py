"""
配置项
"""

from engine.externals import pickle, Path, sys, base64, os
from engine.tools.DataManageTools import DataManageTools
from engine.tools.Tools import Tools


def main():

    # 导入配置文件，从 Excel 文件
    config = DataManageTools.load_configs_from_excel_to_dict(Path(folderpath_config), is_save_to_pyfile=True)
    # config = DataManageTools.load_configs_from_excel_to_dict(Path(r'.'), True)

    # 保存为 PKL 文件
    with open(Path(folderpath_config, r"config.pkl"), 'wb') as f:
        pickle.dump(config, f)

    print("运行完毕！")
    pass  # function



if __name__ == '__main__':
    # %% 设置工作目录。
    # folderpath_config=Tools.setup_working_directory()
    if Path(sys.argv[0]).name == Path(__file__).name:
        # 在控制台运行，切换到脚本所在的文件夹
        folderpath = Path(__file__).resolve().parent
        os.chdir(folderpath)
    else:
        # 通过其他脚本运行，执行特定的代码
        list_args = Tools.decode_args([*sys.argv[1:]])
        folderpath = list_args[0]
        pass  # if
    folderpath_config = folderpath
    main()


