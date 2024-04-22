import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from typing import Literal

models = [

]
models.append(LogisticRegression(solver='liblinear'))
models.append(SVC())
models.append(KNeighborsClassifier(n_neighbors=7))
models.append(DecisionTreeClassifier())
models.append(RandomForestClassifier())
models.append(GaussianNB())




def ensemble_methods(models, X, option : Literal["hard", "soft"] = "hard"):
    print(f"list of models = {models}")

    arg_max_models = [model for model in models if hasattr(model, "decision_function")] # [N]
    prob_models = [model for model in models if not(hasattr(model, "decision_function"))] # [N, output]

    def hard_voting() ->np.ndarray:
        Ys = []
        for model in models:
            Ys.append(model.predict(X))

        Ys = np.stack(Ys)

        hard_voting_result = np.sum(Ys, axis=0) >= (Ys.shape[0] / 2)
        return hard_voting_result.astype(int)
    
    def soft_voting() ->np.ndarray:
        probs = []
        for model in arg_max_models:
            probs.append(model.decision_function(X))

        for model in prob_models:
            probs.append(model.predict_proba(X)[:, 1]) # second col

        probs = np.stack(probs)

        soft_voting_result = np.average(probs, axis=0) >= 0.5
        return soft_voting_result.astype(int)
    
    return {
        "hard" : hard_voting,
        "soft" : soft_voting
    }[option]()





if __name__ == "__main__":
    # model = SVC()
    # X = np.random.random((1000,10))
    # Y = np.random.randint(0,2,size=(1000))

    # for i in models:
    #     i.fit(X,Y)
    
    # print(ensemble_methods(models, X, "hard"))



    from functools import cache

    from time import sleep,time


    class T():
        def __init__(self) -> None:
            ...

        @cache
        def caching_test(self, x):
            print(x)
            sleep(x)
            return x
        

    z = T()

    start = time()
    z.caching_test(2)
    z.caching_test(2)
    end = time()

    print(end - start )