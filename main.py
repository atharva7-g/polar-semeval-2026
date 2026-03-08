def concat(x, y):
    return int(str(x) + str(y))


x = int(input())
y = 1

while concat(x, y) % (x + y) != 0:
    y += 1

print(y)
