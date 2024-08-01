# Decision trees
Here I'm implementing C4.5 algorithm to build decision trees without the pruning step using the recursive idea of splitting the dataset and select the best attribute in each level of recursion.
Here is what the decision tree look like:
```.
Success ratio: 0.765625
|__ 
    |__ Glucose<154.500000
        |__ Pregnancies<13.500000
            |__ BloodPressure<109.000000
                |__ Insulin<540.000000 -> 0
                |__ Insulin>=540.000000 -> 1
            |__ BloodPressure>=109.000000 -> 1
        |__ Pregnancies>=13.500000 -> 1
    |__ Glucose>=154.500000
        |__ BMI<23.100000 -> 0
        |__ BMI>=23.100000
            |__ DiabetesPedigreeFunction<2.233000
                |__ BMI<46.100000 -> 1
                |__ BMI>=46.100000 -> 0
            |__ DiabetesPedigreeFunction>=2.233000 -> 0
```
The building process took around ~0.8~ ~0.08~ 0.03 seconds using the dataset below and with a i5-11400H cpu:
https://github.com/Anny8910/Decision-Tree-Classification-on-Diabetes-Dataset/blob/master/diabetes_dataset.csv
