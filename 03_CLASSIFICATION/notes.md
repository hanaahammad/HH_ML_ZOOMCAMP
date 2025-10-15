# week 3 homework

Question 6, my result is 100 however from the hint I undrestood the right answer is the smallest c of multple records so I changed it to 0.1

 ```python
accuracy_list = []
for c in C:
    print(c)
    model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    model.predict(X_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    converted_decision = (y_pred >= 0.5) 
    #converted_decision

    (y_val == converted_decision).mean()
    df_pred = pd.DataFrame()
    df_pred['probability'] = y_pred
    df_pred['prediction'] = converted_decision.astype(int)
    df_pred['actual'] = y_val
    df_pred['correct'] = df_pred.prediction == df_pred.actual
    print(df_pred.correct.mean())
    accuracy_list.append([c,round(df_pred.correct.mean() ,3)])

print(accuracy_list)
sorted_matrix = sorted(accuracy_list, key=lambda row: row[1])
sorted_matrix


[[0.01, np.float64(0.66)],
 [0.1, np.float64(0.791)],
 [1, np.float64(0.791)],
 [10, np.float64(0.791)],
 [100, np.float64(0.796)]]
