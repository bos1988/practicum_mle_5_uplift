import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


# реализуем подсчёт Qini Score:

def qini_df(df, print_optimum=False):
    len_df = len(df)
    # 1. Отранжируем выборку по значению uplift в порядке убывания
    ranked = df.sort_values(by='uplift_score', ascending=False).reset_index(drop=True)
    # ranked["n"] = ranked["uplift_score"].rank(pct=True, ascending=False)  # старое неверное решение, т.к. для одинаковых `score` задается одинаковы `n`
    ranked["n"] = np.arange(len_df) / len_df

    N_c = sum(ranked['target_class'] <= 1)
    N_t = sum(ranked['target_class'] >= 2)

    # посчитаем в отсортированном датафрейме основные показатели,
    # которые используются при расчёте Qini Score
    ranked['n_c1'] = 0
    ranked['n_t1'] = 0
    ranked.loc[ranked.target_class == 1,'n_c1'] = 1
    ranked.loc[ranked.target_class == 3,'n_t1'] = 1
    ranked['n_c1/nc'] = ranked.n_c1.cumsum() / N_c
    ranked['n_t1/nt'] = ranked.n_t1.cumsum() / N_t

    # посчитаем Qini-кривую
    ranked['uplift'] = ranked['n_t1/nt'] - ranked['n_c1/nc']

    # добавим случайную кривую
    ranked['random_uplift'] = ranked["n"] * ranked['uplift'].iloc[-1]

    # добавим оптимум
    negative_bound = 1 - ranked['n_c1/nc'].iloc[-1]
    ranked['optimum'] = ranked['n_t1/nt'].iloc[-1]
    ranked['optimum'] = ranked['optimum'].mask(ranked['optimum'] > ranked["n"], ranked["n"])
    ranked['optimum'] = ranked['optimum'].mask(
        ranked["n"] > negative_bound, ranked['optimum'] - ranked["n"] + negative_bound
    )

    # немного кода для визуализации
    plt.figure(figsize=(10,8))
    plt.plot(ranked['n'], ranked['uplift'], color='b')
    plt.plot(ranked['n'], ranked['random_uplift'], linestyle="--", color='r')
    if print_optimum:
        plt.plot(ranked['n'], ranked['optimum'], linestyle="--", color='g')
    plt.show()

    # считаем qini-score как разность между площадью под кривыми модели и случайного решения
    auc_model = auc(ranked['n'], ranked['uplift'])
    auc_random = auc(ranked['n'], ranked['random_uplift'])
    qini_score = auc_model - auc_random
    return qini_score
    
    # тоже самое, что через auc (немного другая точность - разница незначительна):
    return (ranked['uplift'] - ranked['random_uplift']).sum() / df.shape[0]

