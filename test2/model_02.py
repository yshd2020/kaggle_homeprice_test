import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#学習データ・テストデータの読み込み
train=pd.read_csv("./data/train.csv")
test=pd.read_csv("./data/test.csv")

#trainデータの"GrLivArea"と"TotalBsmtSF"の外れ値を除去し、trainを定義する
train = train.drop(train[train["GrLivArea"] > 4500].index)
train = train.drop(train[train["TotalBsmtSF"] > 4000].index)

#使用する説明変数の定義
features=["OverallQual","GrLivArea","GarageCars","TotalBsmtSF"]

#説明変数
X=train[features]

#目的変数
Y=train["SalePrice"]

#学習データ・テストデータへの分割
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#モデルの作成
model=LinearRegression()

#学習データを使用してモデルの学習
model.fit(X_train,Y_train)

#データの予測
Y_pred=model.predict(X_test)

#モデルの評価
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

#提出用データを予測
#予測用データのGarageCars,GarageArea,TotalBsmtSFにある欠損値をmedianで埋める
X_submit = test[features]
X_submit = test[features].fillna(train[features].median())
SalePrice_pred = model.predict(X_submit)


#予測結果を提出用のデータフレームにまとめる
submission = pd.DataFrame({
    'Id':test["Id"],
    'SalePrice': SalePrice_pred
})

#提出用ファイルの書き出し
submission.to_csv('submission_HousePriceAdvanced_LinearRegression_2.csv', index = False )
