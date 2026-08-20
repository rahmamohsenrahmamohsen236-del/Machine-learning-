import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# 1. NUMPY & MATRIX OPERATIONS
# ==========================================

A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])

B = np.array([[6],
              [-11],
              [-3]])

print("--- Matrix A Shape & Dimensions ---")
print("Shape of A:", A.shape)  # (3, 3)
print("Dimensions:", A.ndim)  # 2D

# Matrix Indexing (الوصول للعناصر)
print("\nElement at Row 2, Column 3 (Index 1, 2):", A[1, 2])

# Matrix Multiplication (ضرب المصفوفات)
# ضرب عنصر بـ عنصر (Element-wise)
print("\nElement-wise A * A:\n", A * A)

# الضرب الرياضي (Dot Product)
C = np.dot(A, A)  # أو A @ A
print("\nMatrix Multiplication (A @ A):\n", C)


# ==========================================
# 2. SOLVING SYSTEM OF LINEAR EQUATIONS
# ==========================================
# لنفترض أن لدينا النظام الخطي التالي:
# 2x + 1y - 1z = 6
# -3x - 1y + 2z = -11
# -2x + 1y + 2z = -3

print("\n--- Solving Linear Equations (A * X = B) ---")

# الطريقة الأولى: استخدام المعكوس (A_inv * B)
A_inv = np.linalg.inv(A)
X_inv = np.dot(A_inv, B)
print("Solution using Inverse (x, y, z):\n", X_inv)

# الطريقة الثانية: المباشرة والأكثر دقة برمجياً (np.linalg.solve)
X_solve = np.linalg.solve(A, B)
print("\nSolution using np.linalg.solve (x, y, z):\n", X_solve)


# ==========================================
# 3. PANDAS & DATA FRAME INTEGRATION
# ==========================================

# تحويل البيانات والمصفوفات إلى جدول Pandas DataFrame
# تخيل أن لدينا بيانات 5 طلاب ومجموعات درجاتهم في 3 مواد
data_matrix = np.array([
    [85, 90, 78],
    [60, 70, 65],
    [92, 95, 98],
    [75, 80, 70],
    [50, 55, 60]
])

student_names = ['Alice', 'Bob', 'Charlie', 'David', 'Eva']
subjects = ['Math_LinearAlgebra', 'Statistics', 'Python_Programming']

# إنشاء DataFrame من مصفوفة NumPy
df_students = pd.DataFrame(data=data_matrix, index=student_names, columns=subjects)

print("\n--- Pandas DataFrame ---")
print(df_students)

# ربط مفاهيم Matrix Indexing مع Pandas (loc & iloc)
print("\nCharlie's Grades (using loc):")
print(df_students.loc['Charlie'])

print("\nStatistics Grade for Bob (Row Index 1, Col Index 1 using iloc):")
print(df_students.iloc[1, 1])

# العمليات الإحصائية على المصفوفة داخل Pandas
print("\n--- Summary Statistics (Mean per Subject) ---")
print(df_students.mean(axis=0))  # axis=0 يحسب المتوسط لكل عمود

# إضافة عمود جديد حسابي (إجمالي الدرجات باختبار المصفوفات)
df_students['Total_Score'] = df_students.sum(axis=1)
print("\n--- Final Table with Total Scores ---")
print(df_students)


# ==========================================
# 4. PANDAS BASIC OPERATIONS
# ==========================================

# 1. إنشاء جدول بيانات (DataFrame) من القاموس
data = {
    'Name': ['Ahmed', 'Mona', 'Ali', 'Sara'],
    'Age': [25, 30, 22, 28],
    'Salary': [5000, 8000, 4500, 7500]
}

df = pd.DataFrame(data)

# 2. عرض البيانات وتصفحها
print("\n--- 1. عرض أول صفين في الجدول ---")
print(df.head(2))

# 3. اختيار عمود معين
print("\n--- 2. عرض عمود الأسماء فقط ---")
print(df['Name'])

# 4. تصفية البيانات (Filtering)
# اختيار الأشخاص الذين أعمارهم أكبر من 24 سنة
print("\n--- 3. الأشخاص أكبر من 24 سنة ---")
filtered_df = df[df['Age'] > 24]
print(filtered_df)

# 5. إضافة عمود جديد حسابي
# إضافة عمود يعرض الراتب السنوي (الراتب * 12)
df['Yearly_Salary'] = df['Salary'] * 12

print("\n--- 4. الجدول بعد إضافة الراتب السنوي ---")
print(df)

# 6. حساب إحصائيات سريعة
print("\n--- 5. متوسط الأعمار ---")
print(df['Age'].mean())


# ==========================================
# 5. MATPLOTLIB VISUALIZATION
# ==========================================

# بيانات وهمية: العلاقة بين المساحة وسعر المنزل
area = np.array([1000, 1500, 1800, 2400, 3000])
price = np.array([150, 220, 260, 340, 420])

plt.figure(figsize=(8, 5))
plt.scatter(area, price, color='blue', marker='o', label='Houses')
plt.plot(area, price, color='red', linestyle='--', label='Trend Line') # خط الاتجاه

plt.title('House Price vs. Area (Matplotlib Example)')
plt.xlabel('Area (sqft)')
plt.ylabel('Price ($1000)')
plt.legend()
plt.grid(True)
plt.show()


# ==========================================
# 6. SEABORN VISUALIZATION
# ==========================================

# إنشاء جدول بيانات
data_seaborn = {
    'Age': [25, 30, 35, 40, 22, 48, 55, 60],
    'Salary': [50, 60, 70, 80, 45, 110, 120, 130],
    'Experience': [2, 5, 8, 12, 1, 18, 22, 25]
}
df_seaborn = pd.DataFrame(data_seaborn)

plt.figure(figsize=(10, 4))                   

# 1. رسم Boxplot لمعرفة القيم الشاذة في الراتب
plt.subplot(1, 2, 1)
sns.boxplot(y=df_seaborn['Salary'], color='skyblue')
plt.title('Salary Distribution & Outliers')

# 2. رسم Correlation Heatmap لمعرفة ارتباط المتغيرات
plt.subplot(1, 2, 2)
sns.heatmap(df_seaborn.corr(), annot=True, cmap='Blues')
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()


# ==========================================
# 7. SCIKIT-LEARN (MACHINE LEARNING)
# ==========================================

# 1. تجهيز بيانات وهمية
np.random.seed(42)
X = np.random.rand(100, 2) * 100  # ميزتين (Features)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 5  # المتغير المستهدف (Target)

# 2. تقسيم البيانات (80% تدريب / 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. عمل Scale للبيانات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # لاحظ transform فقط للـ Test!

# 4. بناء وتدريب نموذج Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 5. التنبؤ وتقييم النموذج
y_pred = model.predict(X_test_scaled)
print(f"Scikit-Learn Model R2 Score: {r2_score(y_test, y_pred) * 100:.2f}%")


# ==========================================
# 8. PYTORCH (DEEP LEARNING)
# ==========================================

# 1. تحويل البيانات إلى PyTorch Tensors
# نفترض أن لدينا بيانات (X) و مخرجات (y)
X_tensor = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
y_tensor = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32) # العلاقة y = 2x

# 2. بناء شبكة عصبية بسيطة (Linear Neural Network)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1) # طبقة واحدة

    def forward(self, x):
        return self.linear(x)

model = SimpleNN()

# 3. تحديد دالة الخطأ (Loss Function) والمحسن (Optimizer)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. حلقة التدريب (Training Loop)
for epoch in range(100):
    # Forward Pass: التنبؤ
    y_pred = model(X_tensor)
    
    # حساب الخطأ
    loss = criterion(y_pred, y_tensor)
    
    # Backward Pass: حساب المشتقات وتحديث الأوزان
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. التنبؤ برقم جديد (مثلاً x = 5)
test_val = torch.tensor([[5.0]], dtype=torch.float32)
predicted = model(test_val)
print(f"PyTorch Predicted output for x=5: {predicted.item():.2f}")