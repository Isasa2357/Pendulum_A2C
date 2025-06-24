from SamplingTree import SamplingTree
import time

tree = SamplingTree(10000)

count = [0] * 10000
for i in range(10000):
    tree.add(i + 1)

iters = 10000000
batch_size = 32
start = time.time()
for _ in range(iters):
    samples, _ = tree.get_samples(batch_size)
    for sample in samples:
        count[int(sample) - 1] += 1
end = time.time()
print(end - start)

for i in range(iters):
    prob = (i + 1) / tree.total()
    pred = count[i] / (iters * batch_size)
    if (not(prob * 0.95 <= pred and pred <= prob * 1.05)):
        print(f'{i}, {count[i]}, {pred}, {prob}')