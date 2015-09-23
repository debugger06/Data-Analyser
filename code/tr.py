import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("/Users/vessilli/Desktop/andy/test/data/sunil_2m.csv")

print df.columns

def LogisticRegression(x,y):
	lst = list(set(y + x))
	ddf = df[lst]
	#x=["COST","Gender",'Age','Education']
	#y=["CLICKS"]
	for i in x:
		if ddf[i].dtype=='object':
			fillvalue = ddf[i].value_counts()
			fillvalue = fillvalue.index[0]
		else:
			fillvalue = np.mean(ddf[i])
		ddf[i] = ddf[i].fillna(fillvalue)
	for i in y:
		fillvalue = ddf[i].value_counts()
		fillvalue = fillvalue.index[0]
		ddf[i] = ddf[i].fillna(fillvalue)
	categorical = []
	nonCategorical = []
	for i in x:
		if ddf[i].dtype=="object":
			categorical.append(i)
			print i
		else:
			print i
			nonCategorical.append(i)
	data = ddf[y+nonCategorical]
	for j in categorical:
		#dummy_b = self.get_dummies(ddf,j)
		dummy_b = pd.get_dummies(ddf[j],prefix=j)
		dummy_columns = dummy_b.columns

	cols = list(dummy_columns[1:len(dummy_columns)])
	data[cols] = dummy_b[dummy_columns[1:len(dummy_columns)]]
	data['intercept'] = 1.0
	columns = data.columns
	y = columns[0]
	x = columns[1:len(columns)]
	print data.head()
	logit = sm.Logit(data[y], data[x])
	result = logit.fit()


"""
def LogisticRegression(df):
	print df.head()
	x=["COST","Gender"]
	y=["CLICKS"]

	print df[x].head()
	print df[y].head()
	print df[y].dtypes
	#print df[x].dtypes

	logit = sm.Logit(df["CLICKS"], df["COST","Gender"])
"""

#LogisticRegression(df)