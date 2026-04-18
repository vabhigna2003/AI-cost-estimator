import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Example input (adjust based on your states)
# age, sex, bmi, children, smoker + state columns (dummy)
input_data = np.array([[30, 1, 25.0, 1, 0] + [0]*25])

prediction = model.predict(input_data)

print("Predicted Medical Cost:", prediction[0])