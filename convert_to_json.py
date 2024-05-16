import pandas as pd


def convert_to_json():

    csv_file = pd.DataFrame(pd.read_csv(
        "assets/auto_train.csv", sep=",", header=0, index_col=False))
    csv_file.to_json("assets/auto_train.json", orient="records", date_format="epoch",
                     double_precision=10, force_ascii=True, date_unit="ms", default_handler=None)
