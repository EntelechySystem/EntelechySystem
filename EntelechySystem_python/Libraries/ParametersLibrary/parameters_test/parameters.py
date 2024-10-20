"""
程序：设置参数变量数据记录 parameter_variables 。
"""

from engine.externals import pickle, Path, sys, os, np, pd, itertools, deepcopy
from engine.tools.DataManageTools import DataManageTools
from engine.tools.Tools import Tools


def main():

    folderpath_parameters = folderpath

    # 从 Excel 文件导入参数文件为数据框
    df_parameters = pd.read_excel(str(Path(folderpath_parameters, r'parameters.xlsx')), engine='openpyxl')

    # 去除不启用的行
    df_parameters = df_parameters[df_parameters['是否启用'] == True]

    # 排序，依次按照 '是否启用'、'扫描层级优先级'、'exp_id' 排序
    df_parameters = df_parameters.sort_values(by=['是否启用', '扫描层级优先级', 'exp_id'], ignore_index=True)

    # 获取启用的行值所有的参数项作为组合参数批处理实验作业数据框之列名
    list_parameters = df_parameters['参数项'].tolist()
    list_works_columns = ['exp_id', '试验状态', '实验文件夹名'] + list_parameters

    # 新建【组合参数批处理实验作业数据框】
    df_works = pd.DataFrame(columns=list_works_columns)

    # %%

    # 根据各列列名获取 df_parameters 之对应的【扫描列表】，根据【扫描参数层级】依次展开各列表，形成组合参数。一个组合占据 df_works 一行。

    # 获取每一个参数的扫描列表字符串，转换为一个列表，作为字典的值
    dict_parameters_scan_list = dict()
    for i in range(len(df_parameters)):
        dict_parameters_scan_list[df_parameters['参数项'][i]] = eval(df_parameters['扫描列表'][i])

    # 获取各【正交组合组】、【一一对应组合组】、【上三角组合组】之参数集合，将相同的组放在一起
    set_正交组合组 = df_parameters['正交组合组'].unique()
    set_一一对应组合组 = df_parameters['一一对应组合组'].unique()
    set_上三角组合组 = df_parameters['上三角组合组'].apply(lambda x: eval(x)[0]).unique()

    dict_正交组合组 = dict()
    for g in set_正交组合组:
        if g is not np.nan and g != 0:
            dict_正交组合组[g] = df_parameters[df_parameters['正交组合组'] == g]['参数项'].tolist()  # 一个组合组字典，键是组名，值是参数项列表
        pass  # for

    dict_一一对应组合组 = dict()
    for g in set_一一对应组合组:
        if g is not np.nan and g != 0:
            dict_一一对应组合组[g] = df_parameters[df_parameters['一一对应组合组'] == g]['参数项'].tolist()  # 一个组合组字典，键是组名，值是参数项列表
        pass  # for

    dict_上三角组合组 = dict()
    for g in set_上三角组合组:
        if g is not np.nan and g != 0:
            indices = df_parameters['上三角组合组'].apply(lambda x: eval(x)[0]) == g
            dict_上三角组合组[g] = list(zip(df_parameters[indices]['参数项'].tolist(), [eval(v)[1] for v in df_parameters[indices]['上三角组合组'].tolist()]))  # 一个组合组字典，键是组名，值是参数项元组列表。每一个元素是一个元组，元组之第 1 位是参数项名，第 2 位是扫描层级优先级。
        pass  # for

    # 估算组合参数数量
    N_combinations_正交组合组 = 1
    for group_name, parameter_set in dict_正交组合组.items():
        N_combinations_正交组合组 *= np.prod([len(dict_parameters_scan_list[parameter]) for parameter in parameter_set])
        pass  # for
    N_combinations_一一对应组合组 = 1
    for group_name, parameter_set in dict_一一对应组合组.items():
        N_combinations_一一对应组合组 *= len(dict_parameters_scan_list[parameter_set[0]])
        pass  # for
    N_combinations_上三角组合组 = 0
    for group_name, parameter_set in dict_上三角组合组.items():
        N_combinations_上三角组合组_10 = 0
        dict_levels_count = dict()  # 当前上三角组合组之扫描层级优先级计数字典。键是扫描层级优先级，值是该优先级之计数。
        set_levels = []  # 当前上三角组合组之扫描层级优先级列表集合
        for parameter in parameter_set:
            set_levels.append(parameter[1])
            pass  # for
        set_levels = set(set_levels)
        for level in set_levels:
            dict_levels_count[level] = 0
            for parameter in parameter_set:
                if parameter[1] == level:
                    dict_levels_count[level] += 1
                    pass  # if
                pass  # for
            pass  # for
        for k, v in reversed(list(dict_levels_count.items())):
            N_combinations_上三角组合组_20 = 1
            for parameter in reversed(parameter_set):
                if parameter[1] == k:
                    N_combinations_上三角组合组_20 *= len(dict_parameters_scan_list[parameter[0]])
                    pass  # if
                pass  # for
            N_combinations_上三角组合组_10 += N_combinations_上三角组合组_20
            pass  # for
        N_combinations_上三角组合组 += N_combinations_上三角组合组_10 // 2  # #HACK 这个只是一个很粗略的估算，实际的组合数量可能会更少一些。
        pass  # for

    # 估算组合参数数量。BUG 如果存在某一个组合组的参数数量为 0，会导致整个估算的组合参数数量为 0。
    # 排除某一个组合组的参数数量为 0 的情况。
    if N_combinations_正交组合组 == 0:
        N_combinations_正交组合组 = 1
        pass  # if
    if N_combinations_一一对应组合组 == 0:
        N_combinations_一一对应组合组 = 1
        pass  # if
    if N_combinations_上三角组合组 == 0:
        N_combinations_上三角组合组 = 1
        pass  # if
    N_combinations = N_combinations_正交组合组 * N_combinations_一一对应组合组 * N_combinations_上三角组合组

    # 如果组合参数数量超过一千，暂停程序，提示用户确认是否继续。超过一万直接终止程序。
    print(f"\n组合参数数量大约为 {N_combinations}。")
    if N_combinations > 10000 and N_combinations < 1000000:
        print(rf"估计的组合参数数量超过 10'000 。请确认是否继续？")
        input("按回车键继续。")
    elif N_combinations > 100000:
        raise (rf"估计的组合参数数量超过 1'000'000 。终止程序！请重新设计参数。")
        pass  # if

    # 生成组合参数（#HACK 谨慎运行！！！可能的组合数量非常大，容易导致内存不够用）
    list_combinations_正交组合组 = []
    list_columns_name_正交组合组 = []
    for group_name, parameter_set in dict_正交组合组.items():
        combinations_正交组合组_single_group = list(itertools.product(*[dict_parameters_scan_list[parameter] for parameter in parameter_set]))
        list_combinations_正交组合组.extend(combinations_正交组合组_single_group)
        list_columns_name_正交组合组.extend(parameter_set)
        pass  # for
    df_combinations_正交组合组 = pd.DataFrame(list_combinations_正交组合组, columns=list_columns_name_正交组合组)

    list_combinations_一一对应组合组 = []
    list_columns_name_一一对应组合组 = []
    for group_name, parameter_set in dict_一一对应组合组.items():
        combinations_一一对应组合组_single_group = list(zip(*[dict_parameters_scan_list[parameter] for parameter in parameter_set]))
        list_combinations_一一对应组合组.extend(combinations_一一对应组合组_single_group)
        list_columns_name_一一对应组合组.extend(parameter_set)
        pass  # for
    df_combinations_一一对应组合组 = pd.DataFrame(list_combinations_一一对应组合组, columns=list_columns_name_一一对应组合组)

    list_combinations_上三角组合组 = []
    list_columns_name_上三角组合组 = []
    for group_name, parameter_set in dict_上三角组合组.items():

        set_levels = []  # 当前上三角组合组之扫描层级优先级列表集合
        for parameter in parameter_set:
            set_levels.append(parameter[1])
            pass  # for
        set_levels = set(set_levels)
        list_levels = [item[1] for item in dict_上三角组合组[group_name]]
        list_indices_set_levels = list(set(list(set_levels).index(level) for level in list_levels))  # 获取 set_levels 之各扫描层级优先级开始的索引
        list_indices_set_levels.append(len(list_levels))  # 添加最后一个索引

        # 先生成正交组合组，然后根据上三角组合规则，剔除不符合规则的组合项。
        combinations_上三角组合组_single_group = list(itertools.product(*[dict_parameters_scan_list[parameter[0]] for parameter in parameter_set]))

        # 根据上三角组合规则，删除不参与组合的参数值。
        list_drop = []  # 记录需要删除的组合之参数值之索引
        for i, combination in enumerate(combinations_上三角组合组_single_group):
            for l in range(len(list_levels) - 1):
                values_01 = combination[l:l + 1]
                values_02 = combination[l + 1:l + 2]
                if min(values_01) >= max(values_02):
                    list_drop.append(i)
                    break
                    pass  # if
                pass  # for
            pass  # for
        combinations_上三角组合组_single_group = [combination for i, combination in enumerate(combinations_上三角组合组_single_group) if i not in list_drop]
        list_combinations_上三角组合组.extend(combinations_上三角组合组_single_group)
        list_columns_name_上三角组合组.extend([parameter[0] for parameter in parameter_set])
        pass  # for
    df_combinations_上三角组合组 = pd.DataFrame(list_combinations_上三角组合组, columns=list_columns_name_上三角组合组)

    ## 生成最终的组合参数
    df_combinations_正交组合组 = df_combinations_正交组合组.assign(key=1)
    df_combinations_一一对应组合组 = df_combinations_一一对应组合组.assign(key=1)
    df_combinations_上三角组合组 = df_combinations_上三角组合组.assign(key=1)
    # 初始化一个数据框，用于合并
    df_works = pd.DataFrame({'key': [1]})
    df_works = pd.merge(df_works, df_combinations_正交组合组, on='key', how='outer')
    df_works = pd.merge(df_works, df_combinations_一一对应组合组, on='key', how='outer')
    df_works = pd.merge(df_works, df_combinations_上三角组合组, on='key', how='outer')
    df_works = df_works.drop('key', axis=1)

    df_works['exp_id'] = np.arange(len(df_works), dtype=int) + 1  # 添加 exp_id 列，exp_id 从 1 开始
    df_works = df_works[['exp_id'] + [col for col in df_works.columns if col != 'exp_id']]  # exp_id 列放在第一列

    df_works['works_status'] = '未开始'  # 添加作业状态列 works_states

    # 打印实际的组合参数数量
    print(f"\n实际的组合参数数量为 {len(df_works)}。")

    # 保存组合参数数据框
    print("开始保存 Excel 文件...")
    df_works.to_excel(str(Path(folderpath_parameters, r'parameters_works.xlsx')), index=False)
    print("保存 Excel 文件完成。")

    # 保存到 PKL 文件
    print("开始保存 PKL 文件...")
    with open(Path(folderpath_parameters, r"parameters_works.pkl"), 'wb') as f:
        pickle.dump(df_works, f)
    print("保存 PKL 文件完成。")

    print("运行完毕！")

    pass  # function


if __name__ == '__main__':
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
    main()
