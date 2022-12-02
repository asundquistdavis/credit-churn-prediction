from pickle import load
import numpy as np

model = load(open('../Models/d-rfc.pkl', 'rb'))

# 'age': <int>
# 'numdep': <int>
# 'gender': in ['female?', 'male?']
# 'education level': in ['college?', 'doctorate?', 'graduate?', 'high school?', 'post grad?', 'uneducated?', 'education unk?']
# 'marrital status': in ['single?', 'married?', 'divorced?', 'marrital status unk?']
# 'income level': in ['income 1', 'income 2', 'income 3', 'income 4', 'income 5', 'income unk']

age_range = range(20, 70)
numdep_range = range(4)
gender_range = range(2)
edclvl_range = range(7)
marsta_range = range(4)
income_range = range(6)

model_space