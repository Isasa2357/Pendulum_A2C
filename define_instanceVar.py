import os
import re

# コード入力
lines = list()
while True:
    line = input()
    if line == "":
        break
    lines.append(line)
code = ""
for line in lines:
    code += line.replace('\n', '').replace('\t', '').replace(' ', '')

# 変数名認識
code = code.replace('def__init__(self,', '').replace('):', '')

vars_raw = code.split(',')
vars = [var_raw.split(':')[0] for var_raw in vars_raw]
print(vars)

paste_code = list()
for var in vars:
    paste_code.append(f'self._{var} = {var}')

for code in paste_code:
    print(code)
