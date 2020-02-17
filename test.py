n = list(map(int, input().split()))
while not (n[0] == 0 and n[1] == 0):
    if n[1] % n[0] == 0:
        print("factor")
    elif n[0] % n[1] == 0:
        print("multiple")
    else:
        print("neither")
    n = list(map(int, input().split()))
