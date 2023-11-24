

def show_outcome_distrib(df: pd.DataFrame):
    count = ""
    if isinstance(df, pd.DataFrame):
        count = df['Outcome'].value_counts()
    else:
        count = df.value_counts()

    count.plot(kind = 'pie', explode = [0, 0.1], figsize = (6,6), autopct = '%1.1f%%', shadow = True)

    plt.ylabel("OUtcome: Normal vs Abnormal")
    plt.legend(['Normal', 'Abnormal'])
    plt.show()