out = []
for i in range(1000):
    out.append(i*512)

with open("test_system.txt", "a+") as f:
    f.write(str(out))