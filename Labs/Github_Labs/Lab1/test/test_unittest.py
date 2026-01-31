import sys
import os
import unittest

# Get the path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import calculator


class TestCalculator(unittest.TestCase):

    def test_fun1(self):
        self.assertEqual(calculator.fun1(2, 3), 5)
        self.assertEqual(calculator.fun1(5, 0), 5)
        self.assertEqual(calculator.fun1(-1, 1), 0)
        self.assertEqual(calculator.fun1(-1, -1), -2)

    def test_fun2(self):
        self.assertEqual(calculator.fun2(2, 3), -1)
        self.assertEqual(calculator.fun2(5, 0), 5)
        self.assertEqual(calculator.fun2(-1, 1), -2)
        self.assertEqual(calculator.fun2(-1, -1), 0)

    def test_fun3(self):
        self.assertEqual(calculator.fun3(2, 3), 6)
        self.assertEqual(calculator.fun3(5, 0), 0)
        self.assertEqual(calculator.fun3(-1, 1), -1)
        self.assertEqual(calculator.fun3(-1, -1), 1)

    def test_fun4(self):
        # fun4(x,y) = (x+y) + (x-y) + (x*y)
        self.assertEqual(
            calculator.fun4(2, 3),
            (2 + 3) + (2 - 3) + (2 * 3)   # 5 + (-1) + 6 = 10
        )
        self.assertEqual(
            calculator.fun4(5, 0),
            (5 + 0) + (5 - 0) + (5 * 0)   # 5 + 5 + 0 = 10
        )
        self.assertEqual(
            calculator.fun4(-1, -1),
            (-1 + -1) + (-1 - -1) + (-1 * -1)  # -2 + 0 + 1 = -1
        )

    def test_fun5_divide_basic(self):
        self.assertEqual(calculator.fun5_divide(10, 2), 5)

    def test_fun5_divide_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            calculator.fun5_divide(10, 0)

    def test_fun5_divide_type_error(self):
        with self.assertRaises(ValueError):
            calculator.fun5_divide("10", 2)


if __name__ == '__main__':
    unittest.main()