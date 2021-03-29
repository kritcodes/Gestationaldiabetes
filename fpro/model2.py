import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



model = pickle.load(open('model.pkl','rb'))

preg = input('Enter the Pregnancy month: ')
glucose = input('Enter the Glucose amount: ')
bloodpres = input('Enter the Blood Pressure level: ')
skinthick = input('Enter the Skin Thickness: ')
insulin = input ('Enter the insulin: ')
bmi = input ('Enter the  BMI: ')
dpf = input ('Enter the Diabetes Pedigree Function: ')
age = input ('Enter your age: ')

pred = model.predict([[preg,glucose,bloodpres,skinthick,insulin,bmi,dpf,age]])

if pred == 1:
    print ('The person is Diagnosed Diabetes')
    
else:
    print('The person is not Diagnosed Diabetes')