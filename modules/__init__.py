import pandas as pd

test = pd.read_csv(r"C:\Users\Adarsh Tiwari\OneDrive\Desktop\IIT IMP Documents\2nd sem\Projects\Project Data Science 1\test.csv")
print(test.info())

mask = (
    (test['Gender'] == 'Female') &
    (test['Driving_License'] == 1) &
    (test['Previously_Insured'] == 0) &
    (test['Vehicle_Damage'] == 'Yes')
)

print(len(test.loc[mask, ['id', 'Annual_Premium']]))