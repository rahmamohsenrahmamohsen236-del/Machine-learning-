import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
data = {
    'Age': [25, 30, 35, 40, 22, 48, 55, 60],
    'Salary': [50, 60, 70, 80, 45, 110, 120, 130],
    'Experience': [2, 5, 8, 12, 1, 18, 22, 25]
}
df = pd.DataFrame(data)


plt.figure(figsize=(10, 4))                   


plt.subplot(1, 2, 1)
sns.boxplot(y=df['Salary'], color='skyblue')
plt.title('Salary Distribution & Outliers')

plt.subplot(1, 2, 2)
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()




np.random.seed(42)
X = np.random.rand(100, 2) * 100  
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 5  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 


model = LinearRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
print(f"Scikit-Learn Model R2 Score: {r2_score(y_test, y_pred) * 100:.2f}%")
...

# 1. تحويل البيانات إلى PyTorch Tensors
# نفترض أن لدينا بيانات 입력 (X) و مخرجات (y)
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


criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


for epoch in range(100):
    # Forward Pass: التنبؤ
    y_pred = model(X_tensor)
    

    loss = criterion(y_pred, y_tensor)
    

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


test_val = torch.tensor([[5.0]], dtype=torch.float32)
predicted = model(test_val)
print(f"PyTorch Predicted output for x=5: {predicted.item():.2f}")







