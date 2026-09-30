import math

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 0]

w1, w2, w3, w4 = 0.1, 0.2, 0.3, 0.4
b1, b2 = 0.5, 0.6
w5, w6 = 0.7, 0.8
b3 = 0.9
lr = 0.5

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_turev(x):
    return x * (1 - x)

for _ in range(10000):
    for i in range(len(X)):
        x1, x2 = X[i][0], X[i][1]
        
        h1 = sigmoid(x1 * w1 + x2 * w2 + b1)
        h2 = sigmoid(x1 * w3 + x2 * w4 + b2)
        
        out = sigmoid(h1 * w5 + h2 * w6 + b3)
        
        d_out = (Y[i] - out) * sigmoid_turev(out)
        
        d_h1 = (d_out * w5) * sigmoid_turev(h1)
        d_h2 = (d_out * w6) * sigmoid_turev(h2)
        
        w5 += lr * d_out * h1
        w6 += lr * d_out * h2
        b3 += lr * d_out
        
        w1 += lr * d_h1 * x1
        w2 += lr * d_h1 * x2
        b1 += lr * d_h1
        
        w3 += lr * d_h2 * x1
        w4 += lr * d_h2 * x2
        b2 += lr * d_h2

for i in range(len(X)):
    x1, x2 = X[i][0], X[i][1]
    h1 = sigmoid(x1 * w1 + x2 * w2 + b1)
    h2 = sigmoid(x1 * w3 + x2 * w4 + b2)
    out = sigmoid(h1 * w5 + h2 * w6 + b3)
    
    print(f"Girdi: {X[i]} -> Tahmin: {out:.4f} (Yuvarlanmis: {round(out)})")
