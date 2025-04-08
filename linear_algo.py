from math import sqrt

def golden_ratio(f, l, r, eps, max_steps_count):
    count_iter = 2
    invphi: float = (-1 + sqrt(5)) / 2  # 1/phi
    path = []
    while abs(l - r) > eps and count_iter < max_steps_count:
        x1 = r - (r - l) * invphi
        x2 = l + (r - l) * invphi
        if f([x1]) < f([x2]):
            r = x2
            path.append([r])
        else:
            l = x1
            path.append([x1])
        count_iter += 1
    path.append([min(r, l)])
    return path


def dichotomy(f, a, b, eps, max_steps_count):
    ak = a
    bk = b
    k = 0
    path = []
    while abs(ak - bk) > eps and k < max_steps_count:
        ck = (ak + bk) / 2
        left_point = (ak + ck) / 2
        right_point = (ck + bk) / 2
        f_left = f([left_point])
        f_right = f([right_point])
        f_center = f([ck])
        if f_left < f_center:
            bk = ck
        elif f_right < f_center:
            ak = ck
        else:
            ak = left_point
            bk = right_point
        path.append([ck])
        k += 1
    result = (ak + bk) / 2
    path.append([result])
    return path
