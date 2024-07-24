import csv
import matplotlib.pyplot as plt
import re

def sentence_to_list(sentence):
    label = []
    nums = re.findall(r'\d', sentence)
    for num in nums:
        label.append(int(num))
    return label

def analysis(name):
    x = [0] * 512
    with open(name, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            label = row[2]
            label = sentence_to_list(label)
            for k, l in enumerate(label):
                if l == 1:
                    x[k] += 1
    return x

name = '../inputs_2024-7-24-13.csv'
x = analysis(name)

# プロットの部分
plt.figure(figsize=(10, 5))
plt.plot(x)
plt.title('Frequency of Ones in Each Position')
plt.xlabel('Position')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
