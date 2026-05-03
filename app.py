import streamlit as st
import sympy as sp
import numpy as np

# --- Helper Functions for Parsing ---
def parse_function(func_str, var_name='x'):
    var = sp.Symbol(var_name)
    expr = sp.sympify(func_str)
    return var, expr, sp.lambdify(var, expr, 'numpy')

def parse_ode(func_str):
    x, y = sp.symbols('x y')
    expr = sp.sympify(func_str)
    return sp.lambdify((x, y), expr, 'numpy')

# --- Lab 1: Root Finding Methods ---
def bisection(f, a, b, tol=1e-5, max_iter=100):
    if f(a) * f(b) >= 0:
        return None, "Bisection method fails. f(a) and f(b) must have opposite signs.", []
    steps = []
    c = a
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        steps.append({
            "iter": i,
            "a": a,
            "b": b,
            "c": c,
            "f(c)": fc,
            "interval": abs(b - a)
        })
        if fc == 0 or (b - a) / 2 < tol:
            break
        if fc * f(a) < 0:
            b = c
        else:
            a = c
    return c, None, steps

def newton_raphson(f_expr, var, x0, tol=1e-5, max_iter=100):
    f = sp.lambdify(var, f_expr, 'numpy')
    f_prime = sp.lambdify(var, sp.diff(f_expr, var), 'numpy')
    steps = []
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        fpx = f_prime(x)
        if fpx == 0:
            return None, "Zero derivative. Newton-Raphson fails.", steps
        x_new = x - fx / fpx
        err = abs(x_new - x)
        steps.append({
            "iter": i,
            "x": x,
            "f(x)": fx,
            "f'(x)": fpx,
            "x_new": x_new,
            "error": err
        })
        if err < tol:
            return x_new, None, steps
        x = x_new
    return x, None, steps

def secant_method(f, x0, x1, tol=1e-5, max_iter=100):
    steps = []
    for i in range(1, max_iter + 1):
        fx0 = f(x0)
        fx1 = f(x1)
        if fx1 - fx0 == 0:
            return None, "Division by zero. Secant method fails.", steps
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        err = abs(x_new - x1)
        steps.append({
            "iter": i,
            "x0": x0,
            "x1": x1,
            "f(x0)": fx0,
            "f(x1)": fx1,
            "x_new": x_new,
            "error": err
        })
        if err < tol:
            return x_new, None, steps
        x0, x1 = x1, x_new
    return x1, None, steps

# --- Lab 2: Interpolation Methods ---
def lagrange_interpolation(x_pts, y_pts, x_val):
    n = len(x_pts)
    result = 0.0
    steps = []
    for i in range(n):
        term = y_pts[i]
        for j in range(n):
            if i != j:
                term = term * (x_val - x_pts[j]) / (x_pts[i] - x_pts[j])
        result += term
        steps.append({
            "i": i,
            "L_i(x)": term,
            "partial_sum": result
        })
    return result, steps

def forward_diff_table(y_pts):
    n = len(y_pts)
    table = np.zeros((n, n))
    table[:, 0] = y_pts
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]
    return table

def newton_forward(x_pts, y_pts, x_val):
    table = forward_diff_table(y_pts)
    h = x_pts[1] - x_pts[0]
    p = (x_val - x_pts[0]) / h
    result = table[0][0]
    p_term = 1.0
    fact = 1
    steps = [{"i": 0, "term": table[0][0], "partial_sum": result}]
    for i in range(1, len(x_pts)):
        p_term *= (p - (i - 1))
        fact *= i
        term = (p_term * table[0][i]) / fact
        result += term
        steps.append({
            "i": i,
            "term": term,
            "partial_sum": result
        })
    return result, table, steps

def newton_backward(x_pts, y_pts, x_val):
    table = forward_diff_table(y_pts)
    n = len(x_pts)
    h = x_pts[1] - x_pts[0]
    p = (x_val - x_pts[-1]) / h
    result = table[n - 1][0]
    p_term = 1.0
    fact = 1
    steps = [{"i": 0, "term": table[n - 1][0], "partial_sum": result}]
    for i in range(1, n):
        p_term *= (p + (i - 1))
        fact *= i
        term = (p_term * table[n - 1 - i][i]) / fact
        result += term
        steps.append({
            "i": i,
            "term": term,
            "partial_sum": result
        })
    return result, table, steps

# --- Lab 3: Numerical Integration ---
def trapezoidal(f, a, b):
    fa = f(a)
    fb = f(b)
    result = (b - a) / 2 * (fa + fb)
    steps = [{"x": a, "f(x)": fa}, {"x": b, "f(x)": fb}]
    return result, steps

def simpsons_13(f, a, b):
    m = (a + b) / 2
    fa = f(a)
    fm = f(m)
    fb = f(b)
    result = (b - a) / 6 * (fa + 4 * fm + fb)
    steps = [{"x": a, "f(x)": fa}, {"x": m, "f(x)": fm}, {"x": b, "f(x)": fb}]
    return result, steps

def composite_trapezoidal(f, a, b, n):
    h = (b - a) / n
    result = 0.0
    steps = []
    for i in range(n + 1):
        x = a + i * h
        fx = f(x)
        coef = 1 if i == 0 or i == n else 2
        result += coef * fx
        steps.append({"i": i, "x": x, "f(x)": fx, "coef": coef})
    return (h / 2) * result, steps

# --- Lab 4: Differential Equations ---
def euler_method(f, x0, y0, h, x_end):
    n = int((x_end - x0) / h)
    steps = []
    for i in range(1, n + 1):
        fxy = f(x0, y0)
        y_new = y0 + h * fxy
        steps.append({
            "iter": i,
            "x": x0,
            "y": y0,
            "f(x,y)": fxy,
            "y_new": y_new
        })
        y0 = y_new
        x0 = x0 + h
    return y0, steps

def modified_euler(f, x0, y0, h, x_end):
    n = int((x_end - x0) / h)
    steps = []
    for i in range(1, n + 1):
        fxy = f(x0, y0)
        y_predict = y0 + h * fxy
        fxy_next = f(x0 + h, y_predict)
        y_new = y0 + (h / 2) * (fxy + fxy_next)
        steps.append({
            "iter": i,
            "x": x0,
            "y": y0,
            "predict": y_predict,
            "f(x,y)": fxy,
            "f(x+h,yp)": fxy_next,
            "y_new": y_new
        })
        y0 = y_new
        x0 = x0 + h
    return y0, steps

def rk4_method(f, x0, y0, h, x_end):
    n = int((x_end - x0) / h)
    steps = []
    for i in range(1, n + 1):
        k1 = h * f(x0, y0)
        k2 = h * f(x0 + h / 2, y0 + k1 / 2)
        k3 = h * f(x0 + h / 2, y0 + k2 / 2)
        k4 = h * f(x0 + h, y0 + k3)
        y_new = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        steps.append({
            "iter": i,
            "x": x0,
            "y": y0,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "y_new": y_new
        })
        y0 = y_new
        x0 = x0 + h
    return y0, steps

# ==========================================
# STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="Numerical Computing Labs", layout="wide")
st.title("Numerical Computing Labs (CS2008)")

tab1, tab2, tab3, tab4 = st.tabs(["Lab 1: Roots", "Lab 2: Interpolation", "Lab 3: Integration", "Lab 4: Differential Eqs"])

with tab1:
    st.header("Lab 1: Solution of Nonlinear Equations")
    st.markdown("Methods: Bisection, Newton-Raphson, Secant")
    method1 = st.selectbox("Select method:", ["Bisection", "Newton-Raphson", "Secant"], key="lab1_method")
    func_input = st.text_input("Enter function f(x):", "x**3 - x - 2")
    col1, col2 = st.columns(2)
    a_val = col1.number_input("Interval start (a) / Initial guess 0:", value=1.0)
    b_val = col2.number_input("Interval end (b) / Initial guess 1:", value=2.0)
    col3, col4 = st.columns(2)
    tol = col3.number_input("Tolerance:", value=1e-5, format="%.7f")
    max_iter = col4.number_input("Max iterations:", value=50, min_value=1, step=1)

    if st.button("Calculate Root"):
        try:
            var, expr, f = parse_function(func_input)
            if method1 == "Bisection":
                root, err, steps = bisection(f, a_val, b_val, tol=tol, max_iter=max_iter)
            elif method1 == "Newton-Raphson":
                root, err, steps = newton_raphson(expr, var, a_val, tol=tol, max_iter=max_iter)
            else:
                root, err, steps = secant_method(f, a_val, b_val, tol=tol, max_iter=max_iter)

            if err:
                st.error(err)
            else:
                st.success(f"**Root:** {root}")
                st.subheader("Iterations")
                st.dataframe(steps, use_container_width=True)
        except Exception as e:
            st.error(f"Error parsing function: {e}")

with tab2:
    st.header("Lab 2: Interpolation")
    st.markdown("Methods: Lagrange, Newton Forward, Newton Backward")
    method2 = st.selectbox("Select method:", ["Lagrange", "Newton Forward", "Newton Backward"], key="lab2_method")
    x_input = st.text_input("Enter X values (comma separated):", "1, 2, 3, 4")
    y_input = st.text_input("Enter Y values (comma separated):", "1, 8, 27, 64")
    val_to_find = st.number_input("Value to interpolate (x):", value=2.5)

    if st.button("Interpolate"):
        try:
            x_pts = np.array(list(map(float, x_input.split(','))))
            y_pts = np.array(list(map(float, y_input.split(','))))
            if len(x_pts) != len(y_pts):
                st.error("X and Y must have the same length.")
            else:
                if method2 == "Lagrange":
                    result, steps = lagrange_interpolation(x_pts, y_pts, val_to_find)
                    st.success(f"**Interpolated value:** {result}")
                    st.subheader("Steps")
                    st.dataframe(steps, use_container_width=True)
                else:
                    diffs = np.diff(x_pts)
                    if not np.allclose(diffs, diffs[0]):
                        st.error("Newton methods require equally spaced X values.")
                    else:
                        if method2 == "Newton Forward":
                            result, table, steps = newton_forward(x_pts, y_pts, val_to_find)
                        else:
                            result, table, steps = newton_backward(x_pts, y_pts, val_to_find)
                        st.success(f"**Interpolated value:** {result}")
                        st.subheader("Forward Difference Table")
                        st.dataframe(table, use_container_width=True)
                        st.subheader("Steps")
                        st.dataframe(steps, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

with tab3:
    st.header("Lab 3: Numerical Integration")
    st.markdown("Methods: Trapezoidal, Simpson's 1/3, Composite Trapezoidal")
    method3 = st.selectbox("Select method:", ["Trapezoidal", "Simpson's 1/3", "Composite Trapezoidal"], key="lab3_method")
    int_func = st.text_input("Enter function to integrate f(x):", "sin(x)")
    col1, col2, col3 = st.columns(3)
    lower = col1.number_input("Lower bound (a):", value=0.0)
    upper = col2.number_input("Upper bound (b):", value=3.14159)
    n_intervals = col3.number_input("Intervals for Composite (n):", value=10, min_value=1, step=1)

    if st.button("Integrate"):
        try:
            _, _, f_int = parse_function(int_func)
            if method3 == "Trapezoidal":
                result, steps = trapezoidal(f_int, lower, upper)
            elif method3 == "Simpson's 1/3":
                result, steps = simpsons_13(f_int, lower, upper)
            else:
                result, steps = composite_trapezoidal(f_int, lower, upper, int(n_intervals))
            st.success(f"**Integral value:** {result}")
            st.subheader("Steps")
            st.dataframe(steps, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

with tab4:
    st.header("Lab 4: Differential Equations")
    st.markdown("Methods: Euler, Modified Euler, 4-RK")
    method4 = st.selectbox("Select method:", ["Euler", "Modified Euler", "RK4"], key="lab4_method")
    st.latex(r"\frac{dy}{dx} = f(x,y)")
    ode_func = st.text_input("Enter function f(x, y):", "x + y")
    col1, col2, col3, col4 = st.columns(4)
    x0 = col1.number_input("Initial x0:", value=0.0)
    y0 = col2.number_input("Initial y0:", value=1.0)
    h = col3.number_input("Step size (h):", value=0.1)
    x_end = col4.number_input("Target X:", value=0.5)

    if st.button("Solve ODE"):
        try:
            f_ode = parse_ode(ode_func)
            if method4 == "Euler":
                result, steps = euler_method(f_ode, x0, y0, h, x_end)
            elif method4 == "Modified Euler":
                result, steps = modified_euler(f_ode, x0, y0, h, x_end)
            else:
                result, steps = rk4_method(f_ode, x0, y0, h, x_end)
            st.success(f"**Result at x = {x_end}:** {result}")
            st.subheader("Steps")
            st.dataframe(steps, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
