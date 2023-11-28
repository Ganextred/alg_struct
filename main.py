# This is a sample Python script.
import math
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

    a %= p
    result = 1
    while a != 0:
        while a % 2 == 0:
            a /= 2
            n_mod_8 = p % 8
            if n_mod_8 in (3, 5):
                result = -result
        a, p = p, a
        if a % 4 == 3 and p % 4 == 3:
            result = -result
        a %= p
    if p == 1:
        return result
    else:
        return 0


# task 4
def rho_pollard(n):
    def g(x):
        return (x ** 2 + 1) % n

    x = 2  # starting value
    y = x
    d = 1

    while d == 1:
        x = g(x)
        y = g(g(y))
        d = gcd(abs(x - y), n)

    if d == n:
        return None  # Failure, couldn't factorize
    else:
        return d


# task 5
def baby_step_giant_step(a, b, n):
    m = int(n ** 0.5) + 1

    baby_steps = {pow(a, i, n): i for i in range(m)}

    giant_step_multiplier = a ** (m * (n - 2)) % n

    for j in range(m):
        current_value = (b * (giant_step_multiplier ** j)) % n

        # Check if the value matches a baby step
        if current_value in baby_steps:
            return j * m + baby_steps[current_value]

    return None  # No match found


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
        if legendre_symbol(i * i - n, p) == -1:
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


# task 7
def solovay_strassen(n, k=5):
    if n == 2:
        return True

    if n % 2 == 0 or n == 1:
        return False

    for _ in range(k):
        a = random.randint(2, n - 1)
        jacobi = jacobi_symbol(a, n)
        power = pow(a, (n - 1) // 2, n)

        if jacobi % n != power:
            return False

    return True


# task 8
def sieve_eratosthenes(n):
    prime = [True for i in range(n + 1)]
    p = 2
    while (p * p <= n):

        # If prime[p] is not
        # changed, then it is a prime
        if (prime[p] == True):

            # Update all multiples of p
            for i in range(p * p, n + 1, p):
                prime[i] = False
        p += 1
    res = []
    # Print all prime numbers
    for p in range(2, n + 1):
        if prime[p]:
            res.append(p)
    return res


first_primes_list = sieve_eratosthenes(1000)


def is_prime(num):
    for prm in first_primes_list:
        if num % prm == 0:
            return False
    if num < 2 or num % 2 == 0 or num % 3 == 0 or num % 5 == 0:
        return False
    if not solovay_strassen(num, 10):
        return False
    print("prime", num)
    return True


def generate_prime(bits):
    while True:
        candidate = random.getrandbits(bits)
        if is_prime(candidate):
            return candidate


def generate_keypair(bits):
    p = generate_prime(bits)
    q = generate_prime(bits)

    n = p * q
    phi = (p - 1) * (q - 1)

    # Choose public exponent e
    e = random.randint(2, phi - 1)
    while gcd(e, phi) != 1:
        e = random.randint(2, phi - 1)

    # Calculate private exponent d
    d = mod_inverse(e, phi)

    return (n, e), (n, d)


def encrypt(message, public_key):
    n, e = public_key
    cipher_text = [pow(ord(char), e, n) for char in message]
    return cipher_text


def decrypt(cipher_text, private_key):
    n, d = private_key
    decrypted_text = [chr(pow(char, d, n)) for char in cipher_text]
    return ''.join(decrypted_text)


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
    # 4
    print("Task 4, Pollard's rho algorithm:")
    print(rho_pollard(291))
    print()
    # 5
    print("Task 5, Baby-step giant-step")
    print(baby_step_giant_step(5, 3, 23))
    print()
    # 6
    print("Task 6, Cipolla")
    r = cipolla(2, 7)
    print("Roots of 2 mod 7:", r[0], r[1], r[0] ** 2 % 7, r[1] ** 2 % 7 == 2, 2 % 7 == 2)
    r = cipolla(8218, 10007)
    print("Roots of 8218 mod 10007:", r[0], r[1], r[0] ** 2 % 10007 == 8218, r[1] ** 2 % 10007 == 8218)
    r = cipolla(56, 101)
    print("Roots of 56 mod 101:", r[0], r[1], r[0] ** 2 % 101 == 56, r[1] ** 2 % 101 == 56)
    r = cipolla(1, 11)
    print("Roots of 1 mod 11:", r[0], r[1], r[0] ** 2 % 11 == 1, r[1] ** 2 % 11 == 1)
    # 7
    print("Task 7, Solovay Strassen")
    i = 283
    print(i, "is prime:", solovay_strassen(i))
    i = 287
    print(i, "is prime:", solovay_strassen(i))
    # 8
    print(first_primes_list)

    print("Task 8, RSA")
    bits = 128
    public_key, private_key = generate_keypair(bits)
    message = "Supper secret message!"
    encrypted_message = encrypt(message, public_key)
    print("Encrypted message:", encrypted_message)
    decrypted_message = decrypt(encrypted_message, private_key)
    print("Decrypted message:", decrypted_message)
