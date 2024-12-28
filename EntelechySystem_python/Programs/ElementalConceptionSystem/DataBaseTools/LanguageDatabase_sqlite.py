"""
@File   : LanguageDatabase_sqlite.py
@Author : Yee Cube
@Date   : 2022/09/02
@Desc   : 语言数据库，基于sqlite。该语言数据库主要用于手动预置粗糙的语言数据。主要通过可视化工具例如Pychamr自带的数据库工具：连接数据库、构建表格、增删查改数据等。如果数据量较大，也考虑传统的SQL语句以实现增删查改。
"""

import sqlite3

conn = sqlite3.connect('identifier.sqlite')
cursor = conn.cursor()

## 备注
"""
默认已经通过可视化工具连接号数据库、创建好表格了
"""
#####

### 输入需要的SQL语句 ###
sql = \
"""

"""
########################


cursor.execute(sql)
conn.commit()
cursor.close()
conn.close()
