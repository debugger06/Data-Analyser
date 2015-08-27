import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("data/sunil_2m.csv")


def LogisticRegression(df):
	print df.head()
	x=["COST","Gender"]
	y=["CLICKS"]

	print df[x].head()
	print df[y].head()
	print df[y].dtypes
	#print df[x].dtypes

	logit = sm.Logit(df["CLICKS"], df["COST","Gender"])


LogisticRegression(df)