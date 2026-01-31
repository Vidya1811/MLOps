def fun1(x, y):
    """
    Adds two numbers together.
    Args:
        x (int/float): First number.
        y (int/float): Second number.
    Returns:
        int/float: Sum of x and y.
        Raises:
        ValueError: If x or y is not a number.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    
    return x + y

def fun2(x, y):
    """
    Subtracts two numbers.
    Args:
        x (int/float): First number.
        y (int/float): Second number.
    Returns:
        int/float: Difference of x and y.
        Raises:
        ValueError: If x or y is not a number.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    return x - y

def fun3(x, y):
    """
    Multiplies two numbers together.
    Args:
        x (int/float): First number.
        y (int/float): Second number.
    Returns:
        int/float: Product of x and y.
        Raises:
        ValueError: If either x or y is not a number.
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    return x * y

def fun4(x, y):
    """
    Combines results of fun1, fun2, and fun3 for the same inputs (x, y),
    then returns their sum.

    fun4(x, y) = fun1(x, y) + fun2(x, y) + fun3(x, y)

    Args:
        x (int/float): First number.
        y (int/float): Second number.
    Returns:
        int/float: Combined result.
    Raises:
        ValueError: If x or y is not a number.
    """
    # Reuse validation already implemented in fun1/fun2/fun3
    add_res = fun1(x, y)
    sub_res = fun2(x, y)
    mul_res = fun3(x, y)
    return add_res + sub_res + mul_res


def fun5_divide(x, y):
    """
    Divides x by y.

    Args:
        x (int/float): Numerator
        y (int/float): Denominator
    Returns:
        float: x / y
    Raises:
        ValueError: If x or y is not a number
        ZeroDivisionError: If y == 0
    """
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError("Both inputs must be numbers.")
    if y == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return x / y


# f1_op = fun1(2,3)
# f2_op = fun2(2,3)
# f3_op = fun3(2,3)
# f4_op = fun4(f1_op,f2_op,f3_op)

