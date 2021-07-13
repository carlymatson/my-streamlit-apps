import re
import pandas as pd
import streamlit as st
#from streamlit_helpers import *
import streamlit_helpers as sthf


print("---Running---")
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



df = pd.DataFrame(data=pair_list, columns=["D0", "D1", "Defn", "Confusion"])


with st.beta_expander("Filter Form"):
    filtered_df = sthf.filter_form2(df)

st.write(filtered_df)

st.markdown("-"*10)
st.write("Before Filtering")
sthf.shape_and_memory(df)
st.write("After Filtering")
sthf.shape_and_memory(filtered_df)
st.markdown("-"*10)

if len(filtered_df) == 0:
    st.stop()

st.write("Value Counts")
l_col, r_col = st.beta_columns(2)
l_col.write(filtered_df["D0"].value_counts())
r_col.write(filtered_df["D1"].value_counts())
#st.write(filtered_df["Defn"].value_counts())
st.write(filtered_df["Confusion"].value_counts())

st.markdown("-"*10)

st.write("Full Text Display")
sthf.display_page(filtered_df, 10)

print("Session state after run:", st.session_state)