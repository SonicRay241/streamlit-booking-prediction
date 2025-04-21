from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from numpy.typing import ArrayLike
import joblib

EncoderType = LabelEncoder | OneHotEncoder | OrdinalEncoder

class Classifier(XGBClassifier):
    def __init__(self, **kwargs):
        self.encoders: dict[str, EncoderType] = {}
        self.target_encoder: EncoderType = None
        super().__init__(**kwargs)

    def add_encoder(self, colname: str, encoder: EncoderType | str):
        if not isinstance(encoder, EncoderType | str):
            raise TypeError(
                f"The object loaded must be an instance of LabelEncoder | OneHotEncoder | OrdinalEncoder | str, but got an instance of {type(encoder).__name__}"
            )
        
        if isinstance(encoder, str):
            encoder = joblib.load(encoder)

            if not isinstance(encoder, EncoderType):
                raise TypeError(
                    f"The object loaded must be an instance of LabelEncoder | OneHotEncoder | OrdinalEncoder, but got an instance of {type(encoder).__name__}"
                )
        

        self.encoders.update({ colname: encoder })

    def set_target_encoder(self, encoder: EncoderType | str):
        if not isinstance(encoder, EncoderType | str):
            raise TypeError(
                f"The object loaded must be an instance of LabelEncoder | OneHotEncoder | OrdinalEncoder | str, but got an instance of {type(encoder).__name__}"
            )

        if isinstance(encoder, str):
            encoder = joblib.load(encoder)

            if not isinstance(encoder, EncoderType):
                raise TypeError(
                    f"The object loaded must be an instance of LabelEncoder | OneHotEncoder | OrdinalEncoder, but got an instance of {type(encoder).__name__}"
                )
        
        self.target_encoder = encoder

    def encode_feature(self, column_name: str, value: ArrayLike):
        encoder = self.encoders.get(column_name)
        if encoder is None:
            raise KeyError(
                f"Encoder for column {column_name} is has not been added. Use the method add_encoder() to add mode feature encoder"
            )
        
        return encoder.transform(value)
    
    def encode_target(self, value: ArrayLike):
        return self.target_encoder.transform(value)
    
    def set_params(self, params: dict[str, any]):
        return super().set_params(**params)

    def train(self, dataframe: DataFrame, target_name: str):
        if target_name not in dataframe.columns:
            raise KeyError(f"Target column '{target_name}' is not in the dataframe")

        y = dataframe[target_name]
        x = dataframe.drop(columns=target_name)

        for k, v in self.encoders.items():
            x[k] = v.fit_transform(x[k])

        y = self.target_encoder.fit_transform(y)

        super().fit(x, y)

    def predict(self, x: DataFrame, inverse_target=True):
        for k, v in self.encoders.items():
            x[k] = v.transform(x[k])

        if inverse_target:
            return self.target_encoder.inverse_transform(super().predict(x))

        return super().predict(x)
    
    def eval(self, dataframe: DataFrame, target_name: str):
        if target_name not in dataframe.columns:
            raise KeyError(f"Target column '{target_name}' is not in the dataframe")

        pred = self.predict(dataframe.drop(columns=target_name), inverse_target=False)
        y_true_enc = self.target_encoder.transform(dataframe[target_name])
        
        return {
            "accuracy": accuracy_score(y_true_enc, pred),
            "f1_score": f1_score(y_true_enc, pred)
        }
    
    def train_and_eval(
        self,
        dataframe: DataFrame,
        target_name: str,
        test_size: float = 0.2,
        seed: int | None = None
    ):
        if target_name not in dataframe.columns:
            raise KeyError(f"Target column '{target_name}' is not in the dataframe")

        train_df, test_df = train_test_split(dataframe, test_size=test_size, random_state=seed)

        self.train(train_df, target_name)

        return self.eval(test_df, target_name)

    @classmethod
    def load_model(cls, path: str):
        instance = joblib.load(path)

        if not isinstance(instance, cls):
            raise TypeError(
                f"The object loaded must be an instance of {cls.__name__}, but got an instance of {type(instance).__name__}"
            )
        
        return instance
    
    def save_model(self, path: str):
        joblib.dump(self, path)

# Retrain with best params
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/Dataset_B_hotel.csv")
    df.drop(columns="Booking_ID", inplace=True)
    df["type_of_meal_plan"] = df["type_of_meal_plan"].fillna("Not Selected")
    df.dropna(subset="required_car_parking_space", inplace=True)
    df.dropna(subset="avg_price_per_room", inplace=True)
    df.drop_duplicates(inplace=True)

    mymodel = Classifier()

    mymodel.add_encoder("market_segment_type", "encoders/market_segment_type_encoder.pkl")
    mymodel.add_encoder("room_type_reserved", "encoders/room_type_encoder.pkl")
    mymodel.add_encoder("type_of_meal_plan", "encoders/type_of_meal_encoder.pkl")
    mymodel.set_target_encoder("encoders/target_encoder.pkl")

    mymodel.set_params({
        'colsample_bytree': 0.8,
        'gamma': 0,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'scale_pos_weight': 1,
        'subsample': 1.0
    })

    result = mymodel.train_and_eval(df, "booking_status", seed=42)
    print("Evaluation results:")
    for k, v in result.items():
        print(f"    {k}: {v}")

    mymodel.save_model("models/oop_model.pkl")