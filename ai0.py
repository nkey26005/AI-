import math
import random
from typing import List, Tuple


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class TinyNeuralNetwork:
    """Максимально простая нейросеть: 2 входа -> 2 скрытых нейрона -> 1 выход."""

    def __init__(self, seed: int = 42) -> None:
        random.seed(seed)

        # Веса слоя input -> hidden (2x2)
        self.w_ih = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_h = [0.0, 0.0]

        # Веса слоя hidden -> output (2x1)
        self.w_ho = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_o = 0.0

    def forward(self, x: List[float]) -> Tuple[List[float], float]:
        h = [0.0, 0.0]
        for j in range(2):
            h_raw = x[0] * self.w_ih[0][j] + x[1] * self.w_ih[1][j] + self.b_h[j]
            h[j] = sigmoid(h_raw)

        out_raw = h[0] * self.w_ho[0] + h[1] * self.w_ho[1] + self.b_o
        y_pred = sigmoid(out_raw)
        return h, y_pred

    def train(self, data: List[Tuple[List[float], float]], epochs: int = 5000, lr: float = 0.1) -> None:
        for _ in range(epochs):
            for x, y_true in data:
                h, y_pred = self.forward(x)

                # dLoss/dOut для MSE
                d_loss_d_ypred = 2.0 * (y_pred - y_true)

                # Выходной слой
                d_ypred_d_outraw = y_pred * (1.0 - y_pred)
                d_loss_d_outraw = d_loss_d_ypred * d_ypred_d_outraw

                # Градиенты для w_ho и b_o
                d_loss_d_w_ho0 = d_loss_d_outraw * h[0]
                d_loss_d_w_ho1 = d_loss_d_outraw * h[1]
                d_loss_d_b_o = d_loss_d_outraw

                # Градиенты для скрытого слоя
                d_loss_d_h0 = d_loss_d_outraw * self.w_ho[0]
                d_loss_d_h1 = d_loss_d_outraw * self.w_ho[1]

                d_h0_d_hraw0 = h[0] * (1.0 - h[0])
                d_h1_d_hraw1 = h[1] * (1.0 - h[1])

                d_loss_d_hraw0 = d_loss_d_h0 * d_h0_d_hraw0
                d_loss_d_hraw1 = d_loss_d_h1 * d_h1_d_hraw1

                d_loss_d_w_ih00 = d_loss_d_hraw0 * x[0]
                d_loss_d_w_ih10 = d_loss_d_hraw0 * x[1]
                d_loss_d_w_ih01 = d_loss_d_hraw1 * x[0]
                d_loss_d_w_ih11 = d_loss_d_hraw1 * x[1]
                d_loss_d_b_h0 = d_loss_d_hraw0
                d_loss_d_b_h1 = d_loss_d_hraw1

                # Обновления весов
                self.w_ho[0] -= lr * d_loss_d_w_ho0
                self.w_ho[1] -= lr * d_loss_d_w_ho1
                self.b_o -= lr * d_loss_d_b_o

                self.w_ih[0][0] -= lr * d_loss_d_w_ih00
                self.w_ih[1][0] -= lr * d_loss_d_w_ih10
                self.w_ih[0][1] -= lr * d_loss_d_w_ih01
                self.w_ih[1][1] -= lr * d_loss_d_w_ih11
                self.b_h[0] -= lr * d_loss_d_b_h0
                self.b_h[1] -= lr * d_loss_d_b_h1

    def predict(self, x: List[float]) -> float:
        _, y_pred = self.forward(x)
        return y_pred


def main() -> None:
    # XOR — классический тест, который решает только сеть со скрытым слоем
    data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    net = TinyNeuralNetwork(seed=42)
    net.train(data, epochs=5000, lr=0.1)

    print("Проверка после обучения:")
    for x, y_true in data:
        y_pred = net.predict(x)
        print(f"x={x} -> pred={y_pred:.4f} (ожидалось {y_true})")


if __name__ == "__main__":
    main()
