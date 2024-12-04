"""
模型设置项  #TODO 未完成
"""

from engine.externals import pickle, Path, sys, base64, os
from engine.tools.DataManageTools import DataManageTools
from engine.tools.Tools import Tools


def main():
    # %% 设置工作目录。
    # folderpath_settings=Tools.setup_working_directory()
    if Path(sys.argv[0]).name == Path(__file__).name:
        # 在控制台运行，切换到脚本所在的文件夹
        folderpath = Path(__file__).resolve().parent
        os.chdir(folderpath)
    else:
        # 通过其他脚本运行，执行特定的代码
        list_args = Tools.decode_args([*sys.argv[1:]])
        folderpath = list_args[0]
        pass  # if
    folderpath_settings = folderpath

    # 导入配置文件，从 Excel 文件
    # settings = DataManageTools.load_configs_from_excel_to_dict(Path(folderpath_settings), is_save_to_pyfile=True)
    # settings = DataManageTools.load_configs_from_excel_to_dict(Path(r'.'), True)

    # 保存为 PKL 文件
    with open(Path(folderpath_settings, r"settings.pkl"), 'wb') as f:
        pickle.dump(settings, f)

    print("运行完毕！")
    pass  # function



if __name__ == '__main__':
    main()
