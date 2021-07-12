import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
from streamlit_helpers import *


### Select CSV ###

file_choice = st.sidebar.radio("How to access file", ["Directory", "Upload"])
if file_choice == "Directory":
    # Could make this work and display more smoothly.
    p_hub = Path('/hub')
    dir_list = list(p_hub.iterdir())
    p_dir = st.sidebar.selectbox("Choose subdirectory", dir_list)
    path_list = list(Path(p_dir).glob('**/*.csv'))
    if len(path_list) == 0:
        st.write("There are no CSVs in this subdirectory.")
        st.stop()
    my_file = st.selectbox("Choose file", path_list)
    my_df = load_form(my_file)
else:
    my_file = st.file_uploader("Upload CSV")
    if my_file == None:
        st.write("No file.")
        st.stop()
    try:
        my_df = pd.read_csv(my_file)
    except:
        st.write("Didn't work!")
        st.stop()


### Load CSV ###


shape_and_memory(my_df)

### Filter, Style, and Display DataFrame ###
featured_df = feature_form(my_df)
filtered_df = filter_form(featured_df)
styled_df = style_form(filtered_df.head(100))
st.write(styled_df)


### Analysis ###
st.subheader("Analysis of Modified DataFrame")

shape_and_memory(filtered_df)


# Map #

map_on = st.sidebar.checkbox("Show map", value=True)
if map_on:
    num_plotted = show_map(filtered_df)
    if num_plotted != None:
        st.write("Number of points on map:", num_plotted)

# Column statistics #

l_col, r_col = st.beta_columns(2)
col_to_analyze = l_col.selectbox("Column to analyze", list(filtered_df.columns))
# If the dataframe has categorical variables, use one of them.
n_unique = filtered_df[col_to_analyze].nunique()
if n_unique == 0:
    st.write("Column is empty.")
    st.stop()

if is_numeric_dtype(filtered_df[col_to_analyze]):
    fig, ax = plt.subplots()
    discrete = is_integer(filtered_df[col_to_analyze])
    sns.histplot(data=filtered_df, x=col_to_analyze, ax = ax, discrete=discrete, kde=True)
    st.write(fig)
elif get_coltype(filtered_df[col_to_analyze]) == 'text':
    num_lines = len(filtered_df)
    idx = r_col.number_input("Starting line number", value=0, min_value=0, max_value=num_lines, step=10)
    st.table(filtered_df[col_to_analyze].iloc[idx: idx+10])
elif n_unique <= 200:
    st.write("Value Counts")
    l_col, r_col = st.beta_columns((1,3))
    val_counts = filtered_df[col_to_analyze].value_counts()
    l_col.dataframe(val_counts)
    val_counts = val_counts.head(25)
    fig, ax = plt.subplots(figsize=(10, len(val_counts)/4 + 1))
    sns.barplot(x=val_counts, y=val_counts.index, ax = ax)
    ax.set_title("Value Counts (Top 25)")
    r_col.write(fig)
    if False:
        fig, ax = plt.subplots()
        val_counts.plot.pie(ax = ax)
        st.write(fig)

else:
    st.write(filtered_df[col_to_analyze].head(500))
