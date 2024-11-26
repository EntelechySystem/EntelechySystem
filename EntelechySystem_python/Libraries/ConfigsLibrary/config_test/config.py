import sys

config = dict(
    神经元总数量=256,  # 数据类型：整数；配置类别：模型；备注：初始数量。默认 1'000'000。 ；
    运作单元总数量=64,  # 数据类型：整数；配置类别：模型；备注：初始数量。默认 10'000。 ；
    单个神经元最大连接数=8,  # 数据类型：整数；配置类别：模型；备注：初始数量。默认 1'000。 ；
    单个运作单元最大连接数=8,  # 数据类型：整数；配置类别：模型；备注：初始数量。默认 1'000。 ；
    is_use_xlsx_as_config_file=True,  # 数据类型：布尔值；配置类别：配置；备注：是否使用 xlsx 表格文件作为各种配置的默认选项。默认 False 。如果是 True，则采用名为 `XXX_dict.py` 的文件作为各种配置的默认选项。其中 XXX 表示配置项、设置项等； ；
    is_prerun_config_program=False,  # 数据类型：布尔值；配置类别：操作；备注：是否在实验主程序之前，预先运行配置库相关的生成配置程序。默认 True。如果要用 False，那么不能用 Python 控制台运行，只能在终端用命令行运行。否则运行的时候不会继续推进。 ；
    is_prerun_agents_program=True,  # 数据类型：布尔值；配置类别：操作；备注：是否在实验主程序之前，预先运行个体众库相关的生成参数的程序。默认 True； ；
    is_prerun_settings_program=True,  # 数据类型：布尔值；配置类别：操作；备注：是否在实验主程序之前，预先运行设置库相关的生成参数的程序。默认 True； ；
    is_prerun_parameters_program=True,  # 数据类型：布尔值；配置类别：操作；备注：是否在实验主程序之前，预先运行参数库相关的生成参数的程序。默认 True； ；
    is_auto_confirmation=True,  # 数据类型：布尔值；配置类别：操作；备注：是否自动确认一些比较危险的操作例如删除、移动、复制文件等。默认 False； ；
    is_auto_open_outputlog=False,  # 数据类型：布尔值；配置类别：操作；备注：是否自动打开输出日志文件。默认 False； ；
    is_enable_multiprocessing=False,  # 数据类型：布尔值；配置类别：操作；备注：是否启用多进程并行。默认 True； ；
    percent_core_for_multiprocessing=0.5,  # 数据类型：浮点数；配置类别：操作；备注：设置多进程的 CPU 核心数与该设备总的 CPU 核心数占比。默认 0.5； ；
    system_platform=sys.platform,  # 数据类型：代码段；配置类别：操作；备注：获取系统信息 ；
    高性能方案='自定义ECS方案',  # 数据类型：字符串；配置类别：初始化模型；备注：高性能方案。可选参数值为："稀疏矩阵方案", "自定义ECS方案", "顺序遍历单个Agent方案"。 ；
    is_ignore_warning=False,  # 数据类型：布尔值；配置类别：开发；备注：是否忽略警告。默认 False； ；
    test_logging=10,  # 数据类型：整数；配置类别：开发；备注：日志输出级别。调试级别是10，输出信息级别是20。具体见：[logging —— python的日志记录工具](https://docs.python.org/zh-cn/3.9/library/logging.html#levels) ；
    下标开始=False,  # 数据类型：整数；配置类别：开发；备注：在所使用的编程语言中，下标开始计数的值。例如 Python 下标开始的计数是 0 ，Julia 下标开始的计数是 1 。默认 1 。 ；
    engine_version='v0.0.7_alpha',  # 数据类型：字符串；配置类别：配置；备注：引擎版本。 ；
    folderpath_experiments_projects='Experiments/CIS_010',  # 数据类型：路径字符串；配置类别：路径；备注：实验程序文件所在文件夹 ；
    foldername_engine='EntelechyEngine',  # 数据类型：路径字符串；配置类别：路径；备注：模拟器所在工程文件夹名称 ；
    folderpath_relpath_engine='../',  # 数据类型：路径字符串；配置类别：路径；备注：模拟器所在工程文件夹相对本实验项目根路径文件夹之相对路径 ；
    folderpath_system='EntelechySystem_python/Libraries/SystemsLibrary/system_test',  # 数据类型：路径字符串；配置类别：路径；备注：配置项设置所在的文件夹 ；
    folderpath_config='EntelechySystem_python/Libraries/ConfigsLibrary/config_test',  # 数据类型：路径字符串；配置类别：路径；备注：配置项设置所在的文件夹 ；
    folderpath_parameters='EntelechySystem_python/Libraries/ParametersLibrary/parameters_test',  # 数据类型：路径字符串；配置类别：路径；备注：参数设置所在的文件夹 ；
    folderpath_relpath_CIS='../',  # 数据类型：路径字符串；配置类别：路径；备注：CIS所在工程文件夹相对本实验项目根路径文件夹之相对路径 ；
    folderpath_models='ComplexIntelligenceSystem/ComplexIntelligenceSystem_python/Libraries/ModelsLibrary/model_test',  # 数据类型：路径字符串；配置类别：路径；备注：模型所在的文件夹 ；
    folderpath_settings='ComplexIntelligenceSystem_python/Libraries/SettingsLibrary/settings_test',  # 数据类型：路径字符串；配置类别：路径；备注：设置项所在的文件夹 ；
    folderpath_relpath_ECS='../',  # 数据类型：路径字符串；配置类别：路径；备注：ECS所在工程文件夹相对本实验项目根路径文件夹之相对路径 ；
    folderpath_world_conception_knowledge='ElementalConceptionSystem/ElementalConceptionSystem_python/Libararies/ConceptionLibrary/conception_test',  # 数据类型：路径字符串；配置类别：路径；备注：世界环境文件夹路径 ；
    folderpath_relpath_AWS='../',  # 数据类型：路径字符串；配置类别：路径；备注：AWS所在工程文件夹相对本实验项目根路径文件夹之相对路径 ；
    folderpath_world_environment='AgentsWorldSystem/AgentsWorldSystem_python/Libraries/WorldLibrary/Virtual2DMiniNurseryEnv',  # 数据类型：路径字符串；配置类别：路径；备注：世界环境文件夹路径 ；
    folderpath_agents='AgentsWorldSystem/AgentsWorldSystem_python/Libraries/AgentsLibrary/agents_test',  # 数据类型：路径字符串；配置类别：路径；备注：模型所在的文件夹 ；
    folderpath_relpath_LMS='../',  # 数据类型：路径字符串；配置类别：路径；备注：LMS所在工程文件夹相对本实验项目根路径文件夹之相对路径 ；
    foldername_outputData='EntelechyData',  # 数据类型：路径字符串；配置类别：路径；备注：输出数据所在工程文件夹名称 ；
    folderpath_relpath_outputData='../',  # 数据类型：路径字符串；配置类别：路径；备注：输出数据所在工程文件夹相对本实验项目根路径文件夹之相对路径 ；
    folderpath_root_experiments='SimulationsData',  # 数据类型：路径字符串；配置类别：路径；备注：手动设置实验文件夹根路径； ；
    foldername_experiments_output_data='exp_output_data',  # 数据类型：路径字符串；配置类别：路径；备注：手动设置实验导出数据文件夹名称。默认"exp_output_data"； ；
    foldername_prefix_experiments='test',  # 数据类型：字符串；配置类别：命名；备注：手动设置初始生成的实验文件夹前缀名。默认"default"； ；
    foldername_set_manually='test',  # 数据类型：字符串；配置类别：命名；备注：手动设置生成的实验文件夹全名。只有在【type_of_experiments_foldername】是 "set manually" 的时候生效； ；
    is_datetime=True,  # 数据类型：布尔值；配置类别：命名；备注：是否使用日期时间作为实验文件夹名称的一部分。默认 True； ；
    type_of_experiments_foldername='default',  # 数据类型：字符串；配置类别：命名；备注：设置实验文件夹命名方式。取值："default"、"set manually"。默认"default"； ；
    schedule_operation='ref_schedule_operation',  # 数据类型：引用；配置类别：运行；备注：调度需要运作的程序； ；
    运行模式='交互式观察运行模式',  # 数据类型：字符串；配置类别：初始化模型；备注：运行模式。可选参数值为："批量实验作业运行模式", "交互式观察运行模式" ； ；
    program_预加载相关的实验和库文件程序=True,  # 数据类型：布尔值；配置类别：程序；备注：是否运行「预加载相关的实验和库文件程序」； ；
    # program_预加载相关的实验和库文件程序=False,  # 数据类型：布尔值；配置类别：程序；备注：是否运行「预加载相关的实验和库文件程序」； ；
    # program_实验组模拟程序=True,  # 数据类型：布尔值；配置类别：程序；备注：是否运行「实验组模拟程序」； ；
    program_实验组模拟程序=False,  # 数据类型：布尔值；配置类别：程序；备注：是否运行「实验组模拟程序」； ；
    program_可视化结果程序=False,  # 数据类型：布尔值；配置类别：程序；备注：是否运行「可视化结果程序」； ；
    # program_可视化结果程序=False,  # 数据类型：布尔值；配置类别：程序；备注：是否运行「可视化结果程序」； ；
    is_rerun_all_done_works_in_the_same_experiments=True,  # 数据类型：布尔值；配置类别：配置；备注：是否重新运行所有已经完成的实验。默认 False。如果为 True，则在实验运行之前，重置该实验组当中所有的实验作业运行状态为 "RAW"。 ；
    list_idsExperiment_to_run=[1],  # 数据类型：代码段；配置类别：实验；备注：设置要运行的实验编号列表。默认 None，表示运行所有实验。 ；
    is_develop_mode=True,  # 数据类型：布尔值；配置类别：开发；备注：是否处于开发模型状态。默认 False。默认情况下，模拟器通在子进程独立启用相关的程序。启用之后，在模拟器中，将通过函数调用的方式调用各个程序。启用之后，适合在 Python 3.11 开始的版本做断点调试。 ；
    is_maintain_files_in_simulator_when_develop_mode=True,  # 数据类型：布尔值；配置类别：开发；备注：如果 is_develop_mode == True ，那么是否保留模拟器里的需要保留的文件？默认 False。运行的时候只会运行模拟器里的，而不会运行外部导入的文件，运行后也不会将其删除。如果你想直接运行模拟器里的需要保留的文件，并且做开发这些文件相关的工作，建议开启此项。 ；
    is_init_model=True,  # 数据类型：布尔值；配置类别：开发；备注：是否初始化智能体模型。默认 False 。默认情况下，加载模型的时候是已经初始化过的继续需要运行的模型 ；
)
