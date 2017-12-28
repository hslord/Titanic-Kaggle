import pandas as pd
import matplotlib.pyplot as plt
'''
orig_df = pd.read_csv('train.csv')
orig_df.head()
orig_df.describe() #nothing implausible
orig_df.info()
'''Age has 177 nulls
Cabin has 687 nulls - very likely not important as its cabin#
Embarked has 2 nulls
Drop for sure - Name, Cabin, Ticket'''
col_dropped_df = orig_df.drop(['Name', 'Cabin', 'Ticket'], axis=1)
col_dropped_df.info()

'''Embarked'''
col_dropped_df[col_dropped_df['Embarked'].isnull() == True] #these look like normal passengers
col_dropped_df.groupby(['Embarked']).count() #3 options - Q (77), C (168), S (644)
#dummify
embarked_dummies = pd.get_dummies(col_dropped_df['Embarked'], drop_first=True, dummy_na=True)

'''Sex'''
col_dropped_df.groupby(['Sex']).count() #314 females, 577 males
#dummify
sex_dummies = pd.get_dummies(col_dropped_df['Sex'], drop_first=True, dummy_na=False)

'''concatenate dummified columns, drop orig'''
drop_again = col_dropped_df.drop(['Embarked', 'Sex'], axis=1)
dummified_df = pd.concat((drop_again, embarked_dummies, sex_dummies), axis=1)
dummified_df.info()

'''check null ages'''
dummified_df[dummified_df['Age'].isnull() == True].describe() #don't see pattern
#going to make null ages median ages. will make column noting this
dummified_df['null_age'] = dummified_df['Age'].isnull() == True

dummified_df.loc[dummified_df['null_age'] == True, 'Age'] = dummified_df['Age'].median()
final_df = dummified_df

final_df.corr()
#highest survived correlations are as follows:
#Male -0.54, Pclass -0.34, Fare 0.26 (collinear with class), S -0.16, null_age -0.09

pd.scatter_matrix(final_df, figsize=(20,20))
plt.show()'''

def final_df(f):
    orig_df = pd.read_csv(f)
    col_dropped_df = orig_df.drop(['Name', 'Cabin', 'Ticket'], axis=1)
    embarked_dummies = pd.get_dummies(col_dropped_df['Embarked'], drop_first=True, dummy_na=True)
    sex_dummies = pd.get_dummies(col_dropped_df['Sex'], drop_first=True, dummy_na=False)
    drop_again = col_dropped_df.drop(['Embarked', 'Sex'], axis=1)
    dummified_df = pd.concat((drop_again, embarked_dummies, sex_dummies), axis=1)
    dummified_df['null_age'] = dummified_df['Age'].isnull() == True
    dummified_df.loc[dummified_df['null_age'] == True, 'Age'] = dummified_df['Age'].median()
    final_df = dummified_df
    return final_df
