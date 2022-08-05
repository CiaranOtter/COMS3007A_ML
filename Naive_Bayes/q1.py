from sklearn.model_selection import train_test_split 
import numpy as np

with open("simple-food-reviews.txt") as f:
    temp = f.readlines()
f.close()

stats = np.empty(3);

def clean(data):
    for i in range(len(data)):
        data[i] = data[i].replace("\n", "")

training_data, testing_data = train_test_split(temp, test_size=0.25, random_state=25)

clean(temp)
print(temp)

print(training_data, "\n")
print(testing_data, "\n")
print(stats)