import re
import pandas as pd
import streamlit as st
from streamlit_helpers import *


my_file = open("D0D1_definitions.text", 'r')
text = my_file.read()

text = re.sub("\n\n", "\n", text)
text = re.sub("Definition\n", "Definition: ", text)
text = re.sub("Commonly Confused With\n", "Commonly Confused With: ", text)
text = "\n".join(re.findall(".*: .*", text))

pair_list = []

d0_chunk_list = text.split("D0: ")
for chunk in d0_chunk_list[1:]:
    split_chunk = chunk.split("D1: ")
    d0 = split_chunk[0].strip()
    for d1_chunk in split_chunk[1:]:
        mini_parts = d1_chunk.split("\n")
        d1 = mini_parts[0]
        pair_def = mini_parts[1][12:]
        confusions = mini_parts[2][len("Commonly Confused With: "):]
        pair_list.append([d0, d1, pair_def, confusions])

print(len(pair_list))

df = pd.DataFrame(data=pair_list, columns=["D0", "D1", "Defn", "Confusion"])

shape_and_memory(df)

#df["Keywords"] = regex_vectorized("All other (.*) D1 label pairs depending on (.*)", df['Confusion'], ret="findall")
filtered_df = filter_form(df)

st.write(filtered_df)
shape_and_memory(filtered_df)

l_col, r_col = st.beta_columns(2)
l_col.write(filtered_df[["D0", "D1"]].groupby("D0").count())
r_col.write(filtered_df[["D0", "D1"]].groupby("D1").count())

st.write(filtered_df[["D0", "Confusion"]].groupby("Confusion").count())

idx = st.slider("Starting line", value=0, min_value=0, max_value=len(filtered_df), step=5)
st.table(filtered_df.iloc[idx:idx+20])
