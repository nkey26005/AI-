import unittest

from ai0 import TinyNeuralNetwork


class TestTinyNeuralNetwork(unittest.TestCase):
    def test_xor_learning(self) -> None:
        data = [
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0),
        ]

        net = TinyNeuralNetwork(seed=42)
        net.train(data, epochs=5000, lr=0.1)

        preds = [net.predict(x) for x, _ in data]

        # Проверяем бинарную классификацию порогом 0.5
        classes = [1.0 if p >= 0.5 else 0.0 for p in preds]
        expected = [y for _, y in data]

        self.assertEqual(classes, expected)


if __name__ == "__main__":
    unittest.main()
