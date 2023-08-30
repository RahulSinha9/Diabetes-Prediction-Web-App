import pickle

import numpy as np

load_model = pickle.load(open("C:/Users/rahul/Desktop/webapp/trained_model.sav", 'rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)
input_data_as_np_array = np.asarray(input_data)
input_data_reshape = input_data_as_np_array.reshape(1,-1)
prediction = load_model.predict(input_data_reshape)
print(prediction)
if (prediction[0]==0):
    print('The person is not diabetic')
else:
    print("The person is diabetic")
