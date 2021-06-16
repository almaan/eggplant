import pandas as pd
import numpy as np

def match_data_frames(df_a,
                      df_b,
                      )->pd.DataFrame:


    union_cols = df_a.columns.union(df_b.columns)
    n_vars = len(union_cols)
    n_a = df_a.shape[0]
    n_b = df_b.shape[0]

    new_df = pd.DataFrame(np.ones((n_a + n_b,n_vars))*np.nan,
                          columns = union_cols,
                          index = df_a.index.append(df_b.index),
                          )

    new_df.loc[df_a.index,df_a.columns] = df_a.values
    new_df.loc[df_b.index,df_b.columns] = df_b.values

    return new_df


