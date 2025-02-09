import pandas as pd

def limpieza(filepath: str):
    df = pd.read_excel(filepath)
    drop_columns = ["studyName", "Sample Number", "Region", "Individual ID", "Comments","Date Egg"]
    df = df.drop(columns=drop_columns, errors="ignore")
    df = pd.get_dummies(df, columns=["Clutch Completion"], drop_first=True)
    numeric_columns = [
        "Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Body Mass (g)", 
        "Delta 15 N (o/oo)", "Delta 13 C (o/oo)"
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    df["Sex"] = df["Sex"].apply(lambda x: "FEMALE" if x not in ["MALE", "FEMALE"] else x)
    return df
