import streamlit as st
import pandas as pd
import numpy as np
import math
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
import re


def regex_search(str_, regex = "", ret=''):
    if len(regex) < 1:
        return True
    results = re.findall(regex, str_, flags=re.IGNORECASE)
    if ret.startswith("First") and results:
        return results[0]
    if ret.startswith("All"):
        return "|".join(results)
    if ret.startswith("Number"):
        return len(results)
    return bool(results)

# Vectorize the function so that it can be applied to entire columns at once.
regex_vectorized = np.vectorize(regex_search, excluded=['regex', 'ret'])


def perform_op(op, a, b):
    try:
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        if op == "/": return a / b
        if op == "^": return a ** b
        return False
    except Exception as e:
        print("Operation could not be performed: ", e)
        return False

# Not handled: Parentheses. Operations returning false are treated as zero (no error).
def parse_alg_exp(s, df):
    s = re.sub("[*]{2}", "^", s)
    for ops in ["[+-]", "[*/]", "\^"]:
        while re.search(ops, s):
            op_list = re.findall(ops, s)
            parts = re.split(ops, s)
            total = parse_alg_exp(parts[0], df)
            for idx, op in enumerate(op_list):
                part = parse_alg_exp(parts[idx+1], df)
                total = perform_op(op, total, part)
            return total
    s = s.strip()
    try: return int(s)
    except: pass
    try: return float(s)
    except: pass
    try: return df[s]
    except:
        print("Couldn't parse %s"%(s))
        return False

@st.cache
def load_csv(filename, args):
    df = pd.read_csv(filename, low_memory=False)
    print("Read it again")
    return df

@st.cache
def get_slice(df, filter_list = []):
    slice = df
    for filter in filter_list:
        filter_type, col, val = filter
        if filter_type == 'range':
            slice = slice[(slice[col] >= val[0]) & (slice[col] <= val[1])]
        elif filter_type == 'multiselect':
            slice = slice[slice[col].isin(val)]
        else:
            slice = slice[regex_vectorized(slice[col], val)]
    return slice

def avg_word_count(series): # This could be more generally useful.
    if series.dtype != object:
        return 0
    word_count = series.str.count(' ') + 1
    average_count = word_count.sum() / len(series)
    return average_count

def is_integer(ser):
    if str(ser.dtype).startswith('int'): return True
    try:
        ser.astype('Int64')
        return True
    except:
        return False

# FIXME I don't like relying on this naming scheme. It would be nice to have more robust logic.
def get_coltype(series):
    coltype = ""
    if series.dtype == object and avg_word_count(series) > 5:
        coltype = 'text'
    elif str(series.dtype).startswith('float'):
        coltype = 'float'
    elif series.nunique() <= 20 and coltype != 'float':
        coltype = 'smallcat'
    elif is_integer(series):
        coltype = 'int'
    else:
        coltype = str(series.dtype).rstrip('0123456789')
    print("Type of %s is %s"%(series.name, coltype))
    return coltype

def get_coltype2(series):
    coltype = ""
    if series.dtype == object:
        if series.nunique() <= 20:
            coltype = 'smallcat'
        else:
            word_counts = regex_vectorized(series, " ",  "Number of matches") + 1
            if word_counts.sum() / len(word_counts) > 5:
                coltype = 'text'
            else:
                coltype = 'string'
    elif str(series.dtype).startswith('float'):
        coltype = 'float'
    elif series.nunique() <= 10:
        coltype = 'smallcat'
    elif is_integer(series):
        coltype = 'int'
    else:
        coltype = str(series.dtype).rstrip('0123456789')
    print("*Type of %s is %s"%(series.name, coltype))
    return coltype

def get_memory_w_unit(df):
    mem_units = ['B', 'KB', 'MB', 'GB']
    mem_use = df.memory_usage().sum()
    if mem_use != 0:
        mem_log_floor = math.floor(math.log(mem_use, 1024))
        mem_unit = mem_units[mem_log_floor]
        mem_use = mem_use / 1024**mem_log_floor
    else:
        mem_unit = 'B'
    return mem_use, mem_unit

def shape_and_memory(df):
    mem_use, mem_unit = get_memory_w_unit(df)
    l_col, r_col = st.beta_columns(2)
    with l_col: st.write("Shape:", df.shape)
    with r_col: st.write("Size (in %s):"%(mem_unit), mem_use)

# FIXME Return sliced/renamed/dropnulled df or if impossible return False
def show_map(df, lat=None, lon=None):
    if lat == None or lon == None:
        col_list = list(df.columns)
        for col in col_list:
            if lat == None and re.search('^lat[itude]*$|^y$', col, flags = re.IGNORECASE):
                print("Latitude: %s"%(col))
                lat = col
            elif lon == None and re.search('^lon[gitude]*$|^lng$|^x$', col, flags = re.IGNORECASE):
                print("Longitude: %s"%(col))
                lon = col
    if lat == None or lon == None:
        return
    try:
        rename_dict = {lat: 'lat', lon: 'lon'}
        map_df = df[[lat, lon]].copy()
        map_df.rename(columns = rename_dict, inplace=True)
        map_df = map_df.dropna()
        st.map(map_df)
        return len(map_df)
    except:
        print("Something went wrong with map.")

def filter_widget_dev(col, df):
    nunique = df[col].nunique()
    coltype = df[col].dtype
    if nunique <= 10 or (coltype == object and nunique <= 20):
        filter_type = 'multiselect'
        value_list = list(df[col].unique())
        value = st.multiselect("Which values should be included?", value_list, value_list)
    elif is_numeric_dtype(coltype): # or datetime or timedelta.
        filter_type = 'range'
        min_ = df[col].min()
        max_ = df[col].max() # Manage any type weirdness.
        value = st.slider(col, value=(min_, max_), min_value=min_, max_value=max_)

# FIXME Without reference to my personal coltypes.
def filter_widget(col, df):
    coltype = get_coltype(df[col])
    filter_type = ""
    if coltype == 'smallcat' or coltype == 'bool':
        filter_type = 'multiselect'
        value_list = list(df[col].unique())
        value = st.multiselect("Which values should be included?", value_list, value_list)
    elif coltype == 'int' or coltype == 'float':
        filter_type = 'range'
        min_ = df[col].min()
        max_ = df[col].max()
        if coltype == 'int':
            min_ = int(min_)
            max_ = int(max_)
        else:
            min_ = float(min_)
            max_ = float(max_)
        value = st.slider(col, value=(min_, max_), min_value=min_, max_value=max_)
    else: # coltype == 'text' or coltype == 'string' (or...?)
        filter_type = 'regex'
        value = st.text_input("Contains regular expression (case insensitive)", "")
    return filter_type, value

def filter_form(df, key='filt'):
    with st.beta_expander("Filter DataFrame"):
        filter_cols = st.multiselect("Columns to filter", list(df.columns), [], key=key + "_cols")
        with st.form("Filter"):
            mask_list = []
            for col in filter_cols:
                filter_type, value = filter_widget(col, df) # key = key + "_" + col
                mask_list.append((filter_type, col, value))
            st.form_submit_button("Apply Filters")
        filtered_df = get_slice(df, mask_list)
    return filtered_df

def style_form(df, key='style'):
    with st.sidebar:
        with st.beta_expander("Style DataFrame"):
            int_cols = [col for col in list(df.columns) if is_integer(df[col])]
            with st.form("Styling"):
                int_cols = st.multiselect("Columns to format as integers", list(df.columns), int_cols)
                st.form_submit_button("Apply Styling")
    col_styler = {col: '{:.0f}' for col in int_cols}
    styled_df = df.style.format(col_styler)
    return styled_df

def load_form(filename, key='load'):
    print("File:", filename)
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print("Didn't work. ", e)
    df = pd.read_csv(filepath_or_buffer = filename, nrows = 100)
    print("So far so good...")
    args = {}
    with st.sidebar:
        with st.beta_expander("CSV Settings"):
            with st.form("CSV"):
                use_first_row = st.checkbox("Use first row as header", value=True)
                index_rows = st.multiselect("Header rows", [0,1], [0])
                index_cols = st.multiselect("Index columns", [0,1], [])
                use_cols = st.multiselect("Columns to use", list(df.columns), list(df.columns))
                date_cols = st.multiselect("Columns with dates", list(df.columns), [])
                st.write("Pick latitude and longitude columns")
                st.form_submit_button("Apply Settings")
    if use_first_row: # This is awkward.
        use_first_row = 0
    else:
        use_first_row = None
    # FIXME Whyyyy is this getting pissy all of a sudden???
    my_df = load_csv(filename, args)
    return my_df.copy()

feature_types = ["Algebra", "Regex"]
def feature_form(df, key='feat'):
    with st.beta_expander("Create New Features"):
        #st.write("Buggy! Don't try to use at the moment.")
        l_col, r_col = st.beta_columns(2)
        num_feats = l_col.number_input("Number of algebraic expressions", min_value=0, value=0, step=1)
        num_regex = r_col.number_input("Number of regex features", min_value=0, value=0, step=1)
        with st.form("Features"):
            for n in range(num_feats):
                widget_key = key + "_alg_%d"%(n)
                col_name, res = feature_widget(df, 'Algebra', widget_key)
                #st.write(pd.Series(res).head(10).rename(col_name))
                st.markdown("---")
                #col_new, series = feature_widget(df, newtype, key=key)
                df[col_name] = res
            for n in range(num_regex):
                widget_key = key + "_re_%d"%(n)
                col_name, res = feature_widget(df, 'Regex', widget_key)
                #st.write(pd.Series(res).head(10).rename(col_name))
                st.markdown("---")
                #col_new, series = feature_widget(df, newtype, key=key)
                df[col_name] = res
            st.form_submit_button("Create new features")
    return df


def feature_widget(df, ftype, key = "new_feature"):

    #ftype = r_col.selectbox("Feature type", feature_types, key = key + "_type")
    if ftype == 'Algebra': # Linear combination
        l_col, r_col = st.beta_columns((1,2))
        col_new = l_col.text_input("Name of new column", key, key = key + "_name")
        exp = r_col.text_input("Algebraic expression (no parentheses)", key= key + "_input")
        result = parse_alg_exp(exp, df)
        print("Parsed: ", result)
        return col_new, result
    elif ftype == 'Regex': # Regex
        l_col, m_col, r_col = st.beta_columns(3)
        col_new = l_col.text_input("Name of new column", key, key = key + "_name")
        text_cols = ['text']
        text_col = m_col.selectbox("Choose text column to search", text_cols)
        re_type = r_col.selectbox("Regex function", ["Find", "Findall", "Number"])
        input_re = st.text_input("Enter string or regular expression to find")
        result = regex_vectorized(df[text_col], input_re, 'All')
        return col_new, result
# To-Do:
# Figure out key-naming situation, particularly in filter and feature.
