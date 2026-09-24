"""使用者確認的 d=1 模型核心；不預設參數共用方式或訓練排程。

X: 收盤價。z=diff(X): 一階差分，單位仍為價格，不是報酬率。
z[t] = a + phi*z[t-1] + theta*e[t-1] + e[t]
e[t+1] = c + b*e[t] + u[t+1]

未來 u 的條件均值為 0。a、c 可以分開；若使用者選擇共用，
呼叫端須令 c=a，且共同參數的梯度等於兩欄梯度之和。
這只是數學核心，不是完成的回測，也沒有替使用者選定訓練設定。
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameters:
    a: float
    phi: float
    theta: float
    b: float
    c: float

    def array(self):
        return np.array([self.a, self.phi, self.theta, self.b, self.c])


def forecast(x, z, e, parameters):
    """只能傳入 t 日已知的 X、差分和殘差；不接受未來價格。"""
    a, phi, theta, b, c = parameters.array()
    e1 = c + b * e
    z1 = a + phi * z + theta * e + e1
    e2 = c + b * e1
    z2 = a + phi * z1 + theta * e1 + e2
    x1 = x + z1
    x2 = x1 + z2
    if np.any(np.asarray(x1) <= 0) or np.any(np.asarray(x2) <= 0):
        raise ValueError("模型預測非正價格；回測需明確指定處理規則。")
    return dict(e1=e1, e2=e2, z1=z1, z2=z2,
                x1=x1, x2=x2, predicted_target=z2 / x1)


def path_and_gradient(prices, parameters, *, initial_delta, initial_residual):
    """逐日預測，以及對 [a, phi, theta, b, c] 的完整解析梯度。

    殘差會依參數重算，梯度穿過整條殘差遞迴。初始化明確由呼叫端傳入。
    缺值政策尚未確認，因此此核心拒絕缺值，不暗自補值或刪除資料。
    第 k 個輸出對應 prices[k+1] 當日的訊號，不是該日已實現報酬。
    """
    x = np.asarray(prices, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError("需要至少兩個有限且為正的價格，缺值必須先按確認的規則處理。")
    a, phi, theta, b, c = parameters.array()
    zprev = float(initial_delta)
    eprev = float(initial_residual)
    deprev = np.zeros(5)
    predictions, gradients, residuals = [], [], []
    for xt, zt in zip(x[1:], np.diff(x)):
        et = zt - a - phi * zprev - theta * eprev
        de = np.array([-1., -zprev, -eprev, 0., 0.]) - theta * deprev
        e1 = c + b * et
        de1 = np.array([0., 0., 0., et, 1.]) + b * de
        z1 = a + phi * zt + theta * et + e1
        dz1 = np.array([1., zt, et, 0., 0.]) + theta * de + de1
        e2 = c + b * e1
        de2 = np.array([0., 0., 0., e1, 1.]) + b * de1
        z2 = a + phi * z1 + theta * e1 + e2
        dz2 = np.array([1., z1, e1, 0., 0.]) + phi * dz1 + theta * de1 + de2
        denominator = xt + z1
        if denominator <= 0 or denominator + z2 <= 0:
            raise ValueError("模型預測非正價格。")
        predictions.append(z2 / denominator)
        gradients.append((dz2 * denominator - z2 * dz1) / denominator**2)
        residuals.append(et)
        zprev, eprev, deprev = zt, et, de
    return np.array(predictions), np.array(gradients), np.array(residuals)


def target_mse(prices, targets, mature_mask, parameters, *, initial_delta, initial_residual):
    """普通 Target MSE。呼叫端必須依 exit_date<=fit_asof 建立 mature_mask。

    targets、mature_mask 與 prices 等長；未到期／缺失標籤不進 loss。
    這裡不決定暖身窗、每日一次或重跑歷史等尚未確認的研究選擇。
    """
    predictions, gradients, _ = path_and_gradient(
        prices, parameters, initial_delta=initial_delta, initial_residual=initial_residual)
    y = np.asarray(targets, dtype=float)
    mask = np.asarray(mature_mask, dtype=bool)
    if y.shape != np.asarray(prices).shape or mask.shape != y.shape:
        raise ValueError("價格、標籤和到期旗標必須具有相同形狀。")
    use = mask[1:] & np.isfinite(y[1:])
    if not use.any():
        raise ValueError("沒有可用且已到期的 Target。")
    error = predictions[use] - y[1:][use]
    return float(np.mean(error**2)), np.mean(2 * error[:, None] * gradients[use], axis=0)
