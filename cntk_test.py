import cntk
import numpy as np

x = cntk.input_variable(3)
y = cntk.input_variable(3)

input_dict = {
    x: np.asarray([[1, 2, 3]], dtype=np.float32),
    y: np.asarray([[4, 5, 6]], dtype=np.float32)
}

f = x + y

print("x + y = ", f.eval(input_dict))
