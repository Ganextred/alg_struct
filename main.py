# This is a sample Python script.

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

def gcd(a,b):
    while a != 0 and b != 0:
        if a > b:
            a = a % b
        else:
            b = b % a

    return (a + b)

def lcm(a, b):
    return abs(a * b) // gcd(a, b)


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


if __name__ == '__main__':
    print(mobius(4))
    print(euler(42))
    a=115
    b=1
    print(gcd(115,1), lcm(a,b))

    congruences = [(1, 2), (2, 3), (6, 7)]
    print(chinese_remainder_theorem(congruences))

