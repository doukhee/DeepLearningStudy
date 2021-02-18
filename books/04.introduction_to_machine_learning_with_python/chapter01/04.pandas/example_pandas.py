#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import IPython

# 회원 정보가 들어간 간단한 데이터셋을 생성한다
data = {
    "Name":['John', 'Anna', 'Peter', 'Linda'], 
    'Location':['New York', 'Paris', "Berlin", "London"],
    'Age':[24, 13, 53, 33]
    }
# pandas의 데이터 형태로 변환
data_frame = pd.DataFrame(data)
#IPython.display(data_frame)
# 확인을 위해 출력
print("{}\r\n".format(data_frame))
# Age열의 값이 30 이상인 모든 행을 선택
print(data_frame[data_frame.Age > 30])