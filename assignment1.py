import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

train = pd.read_csv('mitbih_train.csv', header=None)
x_train = train.drop(187, axis=1)
y_train = train[187]
test = pd.read_csv('mitbih_test.csv', header=None)
x_test = test.drop(187, axis=1)
y_test = test[187]

x = [0,1,2,3,4]
counts = []
y = list(y_train)
for i in x:
    counts.append(y.count(i))

plt.hist(y)
plt.xlabel("Classes")
plt.ylabel("Count")
plt.show()

model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

class_names = [0,1,2,3,4]
y_pred = model.predict(x_test)
y_true = y_test

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

# Display the values in each cell of the confusion matrix
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()