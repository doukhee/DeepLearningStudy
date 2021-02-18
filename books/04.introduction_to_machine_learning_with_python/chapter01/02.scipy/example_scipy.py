#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
from scipy import sparse
# 대각선 원소는 1이고 나머지는 0인 2차원 numpy 배열을 만든다
eye = np.eye(4)
# 확인을 위한 출력
print("Numpy Array : \r\n{}".format(eye))
# Numpy 배열을 CSR포맷의 SciPy 희소 행렬로 변환
sparse_matrix = sparse.csr_matrix(eye)
# 확인을 위한 출력
print("SciPy CSR Array : \r\n{}".format(sparse_matrix))
# 주어진 모양과 유형의 새로운 배열을 반환합니다.
data = np.ones(4)
# numpy.arange()
# 주어진 간격 내에서 균일 한 간격의 값을 반환합니다.
# numpy.arange()
row_indices = np.arange(4)
# 주어진 간격 내에서 균일 한 간격의 값을 반환합니다.
col_indices = np.arange(4)
# coordinate 포맷으로 배열 생성 행렬의 위치를 별도의 매개변수로 전달 받는다
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
# 확인을 위한 출력
print(" COO 표현 : \r\n{}".format(eye_coo))