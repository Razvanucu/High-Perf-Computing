import sys
import random


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <n>")
        sys.exit(1)

    try:
        n = int(sys.argv[1])

        if n <= 0:
            raise ValueError

    except ValueError:
        print("Error: n must be a positive integer.")
        sys.exit(1)

    matrix = [
        [random.random() for _ in range(n)]
        for _ in range(n)
    ]

    filename = f"Mat{n}.txt"

    with open(filename, "w") as f:
        f.write(f"{n}\n")

        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")

    print(f"Matrix written to {filename}")


if __name__ == "__main__":
    main()