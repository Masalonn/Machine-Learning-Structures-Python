import math
import random
import matplotlib.pyplot as plt

inputs = [
    [-1.0, -1.0, 1.0],
    [-1.0,  1.0, 1.0],
    [ 1.0, -1.0, 1.0],
    [ 1.0,  1.0, 1.0],
]

targets = [0.0, 1.0, 1.0, 0.0]

# Sigmoida + pochodna
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(y):
    return y * (1.0 - y)

# Parametry sieci
n_inputs = 3
n_hidden = 2
eta = 0.3
epochs = 5000

random.seed(42)

w_input_hidden = [
    [random.uniform(-0.5, 0.5) for _ in range(n_inputs)]
    for _ in range(n_hidden)
]

w_hidden_output = [random.uniform(-0.5, 0.5) for _ in range(n_hidden + 1)]

errors = []  # Lista do wykresu błędu

# Trening
for epoch in range(epochs):
    total_error = 0.0

    for x_vec, t in zip(inputs, targets):

        # Przejście w przód
        hidden_out = []
        for j in range(n_hidden):
            s = sum(w_input_hidden[j][i] * x_vec[i] for i in range(n_inputs))
            hidden_out.append(sigmoid(s))

        hidden_with_bias = hidden_out + [1.0]

        s_out = sum(w_hidden_output[i] * hidden_with_bias[i] for i in range(n_hidden + 1))
        y = sigmoid(s_out)

        error = t - y
        total_error += 0.5 * (error ** 2)

        delta_out = error * sigmoid_derivative(y)

        deltas_hidden = []
        for j in range(n_hidden):
            d = sigmoid_derivative(hidden_out[j]) * w_hidden_output[j] * delta_out
            deltas_hidden.append(d)

        # Korekta wag
        for i in range(n_hidden):
            w_hidden_output[i] += eta * delta_out * hidden_out[i]
        w_hidden_output[-1] += eta * delta_out  # bias

        for j in range(n_hidden):
            for i in range(n_inputs):
                w_input_hidden[j][i] += eta * deltas_hidden[j] * x_vec[i]

    errors.append(total_error)  # logowanie błędu

# Funkcja testowa sieci
def forward(x_vec):
    hidden_out = []
    for j in range(n_hidden):
        s = sum(w_input_hidden[j][i] * x_vec[i] for i in range(n_inputs))
        hidden_out.append(sigmoid(s))

    hidden_with_bias = hidden_out + [1.0]
    s_out = sum(w_hidden_output[i] * hidden_with_bias[i] for i in range(n_hidden + 1))
    return sigmoid(s_out)

print("Wyniki XOR po uczeniu:")
for (x1, x2, bias), t in zip(inputs, targets):
    y = forward([x1, x2, bias])
    y_bin = 1 if y >= 0.5 else 0
    print(f"({x1}, {x2}) -> y={y:.4f}, oczekiwane={int(t)}, wynik={y_bin}")

plt.plot(errors)
plt.xlabel("Epoka")
plt.ylabel("Błąd")
plt.title("Spadek błędu podczas uczenia XOR")
plt.grid(True)
plt.show()
