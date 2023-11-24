# This is a sample Python script.
import random


def prime_factorization(n):
    factors = []
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def mobius(n):
    factors = prime_factorization(n)
    unique_factors = set(factors)

    if len(factors) != len(unique_factors):
        return 0

    return (-1) ** len(factors)


def euler(n):
    result = n

    i = 2
    while i * i <= n:
        if n % i == 0:
            while n % i == 0:
                n //= i
            result -= result // i
        i += 1

    if n > 1:
        result -= result // n

    return result


def gcd(a, b):
    while a != 0 and b != 0:
        if a > b:
            a = a % b
        else:
            b = b % a

    return (a + b)


def lcm(a, b):
    return abs(a * b) // gcd(a, b)


# task2
def extended_gcd(a, b):
    x, xx, y, yy = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a % b
        x, xx = xx, x - xx * q
        y, yy = yy, y - yy * q
    return (a, x, y)


def mod_inverse(a, m):
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise Exception('Inverse does not exist')
    else:
        return x % m


def chinese_remainder_theorem(congruences):
    # congruences is a list of tuples (a_i, m_i)
    M = 1
    for _, m in congruences:
        M *= m

    x = 0
    for a, m in congruences:
        M_i = M // m
        inv = mod_inverse(M_i, m)
        x += a * M_i * inv

    return x % M


# task 3
def legendre_symbol(a, p):
    if a % p == 0:
        return 0
    elif pow(a, (p - 1) // 2, p) == 1:
        return 1
    else:
        return -1


def jacobi_symbol(a, p):
    if p % 2 == 0 or p <= 0:
        raise ValueError("Jacobi symbol is defined for odd positive integers only.")
    if a == 0:
        return 0

    result = 1
    for e in prime_factorization(p):
        result *= legendre_symbol(a, e)
    return result


# task 6
def convert_to_base(num, base):
    result = []
    while num > 0:
        result.append(num % base)
        num //= base
    return result[::-1]




def cipolla_mult(pair1, pair2, w, p):
    a, b = pair1
    c, d = pair2
    return ((a * c + b * d * w) % p, (a * d + b * c) % p)


def cipolla(n, p):
    phi = p - 1
    aa = 0
    for i in range(1, p):
        if legendre_symbol(i*i-n, p) == -1:
            aa = i
            break

    x1 = (aa, 1)
    x2 = cipolla_mult(x1, x1, aa * aa - n, p)
    exponent = convert_to_base((p + 1) // 2, 2)

    for i in range(1, len(exponent)):
        if exponent[i] == 0:
            x2 = cipolla_mult(x2, x1, aa * aa - n, p)
            x1 = cipolla_mult(x1, x1, aa * aa - n, p)
        else:
            x1 = cipolla_mult(x1, x2, aa * aa - n, p)
            x2 = cipolla_mult(x2, x2, aa * aa - n, p)

    return x1[0], (-x1[0] % p)


if __name__ == '__main__':
    # 1
    print(mobius(4))
    print(euler(42))
    a = 115
    b = 1
    print(gcd(115, 1), lcm(a, b))
    # 2
    congruences = [(1, 2), (2, 3), (6, 7)]
    print(chinese_remainder_theorem(congruences))
    # 3
    a = 3
    b = 9
    print(legendre_symbol(a, b))
    print(jacobi_symbol(a, b))
    # 6
    r = cipolla(2, 7)
    print("Roots of 2 mod 7:", r[0], r[1], r[0] ** 2 % 7, r[1] ** 2 % 7 == 2, 2 % 7 == 2)
    r = cipolla(8218, 10007)
    print("Roots of 8218 mod 10007:", r[0], r[1], r[0] ** 2 % 10007 == 8218, r[1] ** 2 % 10007 == 8218)
    r = cipolla(56, 101)
    print("Roots of 56 mod 101:", r[0], r[1], r[0] ** 2 % 101 == 56, r[1] ** 2 % 101 == 56)
    r = cipolla(1, 11)
    print("Roots of 1 mod 11:", r[0], r[1], r[0] ** 2 % 11 == 1, r[1] ** 2 % 11 == 1)
