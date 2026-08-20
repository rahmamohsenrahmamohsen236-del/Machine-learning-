# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#TASK1

data = [10, 14, 5, 9, 2, 20, 6, 9, 14, 9]
import numpy as np
import statistics as st

print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Mode:", st.mode(data))
print("Standard Deviation:", np.std(data, ddof=1))


import statistics

data = [4, 12, 3, 7, 5, 9, 2]

# حساب المتوسط الحسابي والوسيط
mean_val = statistics.mean(data)
median_val = statistics.median(data)

print(f"Mean: {mean_val}")
print(f"Median: {median_val}")

import statistics

data = [150, 220, 180, 210, 300, 190, 250, 160]

# حساب المتوسط الحسابي والوسيط
mean_val = statistics.mean(data)
median_val = statistics.median(data)

print(f"Mean: {mean_val}")
print(f"Median: {median_val}")

import statistics

dataset_A = [1, 3, 2, 5, 3, 4, 3, 2, 8]
dataset_B = [12, 5, 8, 20, 14, 3]

# Dataset A
mode_A = statistics.mode(dataset_A)

# Dataset B (التعامل مع عدم وجود منوال)
try:
    mode_B = statistics.mode(dataset_B)
except statistics.StatisticsError:
    mode_B = "No Mode (لا يوجد منوال)"

print(f"Mode for Dataset A: {mode_A}")
print(f"Mode for Dataset B: {mode_B}")


import statistics

data = [1.5, 1.8, 2.0, 1.7]


std_dev = statistics.stdev(data)

print(f"Sample Standard Deviation (s): {std_dev:.4f}")



import numpy as np

data = [12, 5, 18, 9, 15, 6, 20, 11, 8]


q1 = np.percentile(data, 25, method='midpoint')
q3 = np.percentile(data, 75, method='midpoint')
iqr = q3 - q1

print(f"Q1: {q1}")
print(f"Q3: {q3}")
print(f"IQR: {iqr}")


import statistics
import numpy as np

data = [25, 40, 15, 30, 40, 50, 10, 70]

# 1. Mean
mean_val = statistics.mean(data)

# 2. Median
median_val = statistics.median(data)

# 3. Mode
mode_val = statistics.mode(data)

# 4. IQR
q1 = np.percentile(data, 25, method='midpoint')
q3 = np.percentile(data, 75, method='midpoint')
iqr = q3 - q1

# 5. Sample Standard Deviation
std_dev = statistics.stdev(data)

print(f"1. Mean: {mean_val}")
print(f"2. Median: {median_val}")
print(f"3. Mode: {mode_val}")
print(f"4. IQR: {iqr}")
print(f"5. Sample Standard Deviation (s): {std_dev:.2f}")

#TASK2

import numpy as np

M = np.array([
    [4, -2, 7, 0],
    [1, 9, 3, -5],
    [8, 6, -1, 2]
])


print("Dimensions:", M.shape)


print("m_1,3 =", M[0, 2])  
print("m_2,4 =", M[1, 3])  
print("m_3,2 =", M[2, 1]) 


import numpy as np

A = np.array([[3, 5], [-1, 4]])
B = np.array([[0, -2], [7, 1]])
C = np.array([[1, 2, 3], [4, 5, 6]])

# الجمع A + B
print("A + B =\n", A + B)

# 2. الطرح A - B
print("A - B =\n", A - B)

#الجمع A + C (غير ممكن)
try:
    print(A + C)
except ValueError:
    print("لا يمكن جمع A + C لأن أبعادهما مختلفة!")
    

import numpy as np

X = np.array([[2, -1], [3, 0]])
Y = np.array([[1, 4], [-2, 5]])

# 1. الضرب القياسي (3X)
print("3X =\n", 3 * X)


print("Element-wise (X * Y) =\n", X * Y)


print("Dot Product (X . Y) =\n", np.dot(X, Y))


import numpy as np

K = np.array([[5, 2], [3, 4]])

# المدور Transpose
print("Transpose (K^T) =\n", K.T)

# المحدد Determinant
det_K = np.linalg.det(K)
print("Determinant =", round(det_K))  # 14

# المعكوس Inverse
inv_K = np.linalg.inv(K)
print("Inverse (K^-1) =\n", inv_K)


import numpy as np

# 2x + 3y = 13
# 4x - y = 5

A = np.array([[2, 3], [4, -1]])
B = np.array([13, 5])

# حل المعادلة A * X = B
solution = np.linalg.solve(A, B)

print("x =", round(solution[0]))  
print("y =", round(solution[1])) 


import numpy as np


latency = np.array([120, 150, 130, 180, 150, 200, 110, 160])

# 1. المتوسط (Mean)
mean_val = np.mean(latency)

# 2. الوسيط (Median)
median_val = np.median(latency)

# 3. المنوال (Mode)
vals, counts = np.unique(latency, return_counts=True)
mode_val = vals[np.argmax(counts)]

print("Mean =", mean_val)
print("Median =", median_val)
print("Mode =", mode_val)


import numpy as np


errors = np.array([2, 4, 6, 8, 10])

std_val = np.std(errors, ddof=1)

q25, q75 = np.percentile(errors, [25, 75])
iqr_val = q75 - q25

print("Standard Deviation =", round(std_val, 3))
print("IQR =", iqr_val)




import numpy as np


area = np.array([1200.0, np.nan, 1800.0])


mean_area = np.nanmean(area)
area[np.isnan(area)] = mean_area
print("Area after imputation:", area)


area_scaled = (area - np.min(area)) / (np.max(area) - np.min(area))
print("Area Scaled (0 to 1):", area_scaled)


from sklearn.model_selection import train_test_split


data = list(range(100))

# تقسيم 80% تدريب و 20% اختبار
X_train, X_test = train_test_split(data, test_size=0.20, random_state=42)

print("عدد عينات التدريب (X_train):", len(X_train))  
print("عدد عينات الاختبار (X_test):", len(X_test))  


age = 4
actual_price = 16000

# 1. التنبؤ بالسعر
predicted_price = -2000 * age + 25000
print("Predicted Price:", predicted_price) 


residual = actual_price - predicted_price
print("Residual Error:", residual) 


