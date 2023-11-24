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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(mobius(4))
    print(euler(42))
    a=115
    b=1
    print(gcd(115,1), lcm(a,b))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
