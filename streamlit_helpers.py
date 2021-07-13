from pandas.core.dtypes.common import is_datetime64_any_dtype, is_timedelta64_dtype
import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
from pandas.api.types import is_numeric_dtype, is_timedelta64_dtype, is_datetime64_any_dtype
import re
from fuzzywuzzy import process
import base64


### Functions that render to the Streamlit app. ###

def navigate_directories(parent = '.'):
    if parent == None:
        return []
    p = Path(parent)
    csv_list = list(p.glob('*.csv'))
    subdir_list = [None] + list(f for f in p.iterdir() if f.is_dir())
    p_subdir = st.sidebar.selectbox("Choose subdirectory", subdir_list, format_func = get_end_path)
    if p_subdir != None:
        csv_list = navigate_directories(p_subdir)
    else:
        csv_list = list(p.glob('*.csv'))
    return csv_list

def shape_and_memory(df):
    mem_use, mem_unit = get_memory_w_unit(df)
    l_col, r_col = st.beta_columns(2)
    with l_col: st.write("Shape:", df.shape)
    with r_col: st.write("Size (in %s):"%(mem_unit), mem_use)

def file_writer(df): # Add option to write or download. Check for max size or use try/except.
    file_to_write = st.text_input("Enter name of file to write to", value = "myfile.csv")
    my_link = download_link(df, file_to_write, "Get Download Link")
    st.markdown(my_link, unsafe_allow_html = True)


### The following functions are cached to reduce computation time. ###

@st.cache
def load_csv(filename, args):
    df = pd.read_csv(filename, low_memory=False, parse_dates=True)
    return df

@st.cache
def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

@st.cache
def get_slice2(df, filter_list = []):
    slice = df
    for filter in filter_list:
        try:
            filter_type = filter["wtype"]
            col = filter["colname"]
            val = filter["value"]
            if filter_type.lower() == "slider" or filter_type == "Date Input":
                slice = slice[(slice[col] >= val[0]) & (slice[col] <= val[1])]
            elif filter_type.lower() == 'multiselect':
                slice = slice[slice[col].isin(val)]
            else:
                slice = slice[regex_vectorized(slice[col], val)]
        except Exception as e:
            print("Error getting slice:", e)
    return slice

@st.cache
def get_nunique(series):
    return series.nunique()

@st.cache
def get_memory_w_unit(df):
    mem_units = ['bytes', 'KB', 'MB', 'GB']
    mem_use = df.memory_usage().sum()
    if mem_use != 0:
        mem_log_floor = math.floor(math.log(mem_use, 1024))
        mem_use = mem_use / 1024**mem_log_floor
    else:
        mem_log_floor = 0
    return mem_use, mem_units[mem_log_floor]

@st.cache
def get_confusion_matrix(df, col1, col2):
    grouped = df.groupby([col1, col2])[col1].count()
    grouped = grouped.reset_index(name="count")
    grouped['col1_total'] = grouped.groupby[col1]["count"].transform("sum")
    grouped['percentages'] = grouped["count"] / grouped["col1_total"]
    confusion_matrix = pd.pivot_table(data=grouped, index=col1, columns=col2, values="percentages")
    return confusion_matrix


### These functions assist with creating new features in the Streamlit app. ###
# FIXME Re-do using the regex module (rather than the default re)
def regex_search(str_, regex = "", ret=''):
    if len(regex) < 1:
        return True
    results = re.findall(regex, str(str_), flags=re.IGNORECASE)
    if ret.startswith("Find") and results and len(results) > 0:
        return results[0]
    if ret.startswith("Findall"):
        return "|".join(results)
    if ret.startswith("Number"):
        return len(results)
    return bool(results)

# Vectorize the function so that it can be applied to entire dataframe columns at once.
regex_vectorized = np.vectorize(regex_search, excluded=['regex', 'ret'])

def perform_op(op, a, b):
    try:
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        if op == "/": return a / b
        if op == "^": return a ** b
        # If none of the above, it is an invalid operation.
        return False
    except Exception as e:
        print("Operation could not be performed: ", e)
        return False

# Not handled: Parentheses. Operations returning false are currently treated as zero (no error).
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


### Type-handling and helper functions  ###

def is_range_dtype(series):
    result = (
        is_numeric_dtype(series) or 
        is_datetime64_any_dtype(series) or 
        is_timedelta64_dtype(series)
        )
    return result

def get_end_path(path):
    stem = str(path).split("/")[-1]
    return stem

def get_map_data(df, lat = None, lon = None):
    if lat == None:
        lat, fuzzscore = process.extractOne("latitude", list(df.columns))
    if lon == None:
        lon, fuzzscore = process.extractOne("longitude", list(df.columns))
    try:
        rename_dict = {lat: 'latitude', lon: 'longitude'}
        map_df = df[[lat, lon]]
        map_df = map_df.rename(columns = rename_dict)
        map_df = map_df.dropna()
        return map_df
    except Exception as e:
        return None

### More rendering ###
def map_and_filter(df, lat = None, lon = None):
    # Before slicing, get latitude and longitude
    map_df = get_map_data(df, lat, lon)
    if map_df != None and len(map_df) > 0:
        st.map(df[["latitude", "longitude"]].dropna().head(100))
        # Use form? Then I could just pass widget specifications such as 
        # (latitude, slider) and (longitude, slider) and get back filtered df.

# Add controls for lines per page?
def display_page(df, lines_per_page=10):
    num_lines = len(df)
    idx = st.slider("Starting line number", value=0, min_value=0, max_value=num_lines, step=lines_per_page)
    st.table(df.iloc[idx: idx+10])

# Get min/max and cast to streamlit-friendly types.
def get_slider_bounds(series):
    if str(series.dtype).startswith("int"):
        min_ = int(series.min())
        max_ = int(series.max())
    elif str(series.dtype).startswith("float"):
        min_ = float(series.min())
        max_ = float(series.max())
    else:
        min_ = series.min()
        max_ = series.max()
    return min_, max_


# This function should set up the columns (unless they are provided), 
# put any common kwargs in a dictionary, and call appropriate function.
# Call to creating series or index should be cached.
def render_widget(wtype, kwargs): # kwargs should include label, key, help, etc.
    result = None
    cols = st.beta_columns((6,1))
    if wtype == "feature_regex":
        #result = widget_regex(kwargs)
        return result
    if wtype == "feature_fuzzy":
        #result = widget_fuzzy(kwargs)
        return result
    try:
        widget_func = getattr(cols[0], wtype.replace(" ", "_").lower())
        result = widget_func(**kwargs)
    except Exception as e:
        print("Error rendering widget:", e)
        cols[0].write("Couldn't render widget.")
        return None, cols
    return result, cols

def render_form(widget_list, editable = True, form_submit_name="Apply Changes", key="form"):
    values = []
    with st.form(key.replace("_", " ").title()): # Or not format?
        for widget in widget_list:
            wtype = widget["wtype"]
            kwargs = widget["kwargs"]
            wkey = key + "__%s"%(widget["colname"]) 
            kwargs["key"] = wkey
            val, cols = render_widget(wtype, kwargs) # append_cols = (1,)*editable
            if editable:
                do_erase = cols[-1].checkbox("X", key=wkey + "_del", help="Delete field on next update")
                if do_erase:
                    widget_list.remove(widget)
                    st.experimental_rerun()
            values.append(val)
            widget["value"] = val
        st.form_submit_button(form_submit_name)
    return values

def filter_form2(df, key="filterform"):
    col1, col2 = st.beta_columns((3,3))
    all_cols = list(df.columns)
    col_to_filter = col1.selectbox("Column to filter", all_cols, key = key + "_col")
    possible_filter_types = ["Slider", "Multiselect", "Date Input", "Text Input"]
    type_of_filter = col2.selectbox("Type of filter", possible_filter_types, key = key + "_type")
    make_new_field = st.button("Add New Filter")
    if 'filter_fields' not in st.session_state:
        st.session_state.filter_fields = []
    if make_new_field:
        kwargs = {"label": "Filtering: %s"%(col_to_filter)}
        if type_of_filter == "Slider" or type_of_filter == "Date Input":
            min_, max_ = get_slider_bounds(df[col_to_filter])
            kwargs["min_value"] = min_
            kwargs["max_value"] = max_
            kwargs["value"] = (min_, max_)
        elif type_of_filter == "Multiselect":
            options = list(df[col_to_filter].unique())[:20]
            kwargs["options"] = options
            kwargs["default"] = options
        st.session_state.filter_fields.append({
            "wtype": type_of_filter, 
            "kwargs": kwargs, 
            "colname": col_to_filter
            })
    form_values = render_form(st.session_state.filter_fields, "Filter Form", key=key)
    filtered_df = get_slice2(df, st.session_state.filter_fields)
    return filtered_df

# FIXME Finish implementing.
def feature_form2(df, key="featureform"):
    feature_types = ["Text", "Expression"]
    text_feature_types = ["Regex", "Fuzzywuzzy", "Spacy", "NLPretext"]
    make_new_field = st.button("Create New Form Field")
    if 'filter_fields' not in st.session_state:
        st.session_state.filter_fields = []
    if make_new_field:
        # Get min&max or options, label, help, and pass as kwargs
        st.session_state.filter_fields.append((col_to_filter, type_of_filter))
    #form_values = render_form(st.session_state.filter_fields, "Filter Form", key=key)

# Not currently used anywhere - just rethinking how compound widget rendering might work.
def widget_stats():
    # Necessities:
    # Columns: Shape (unless some are passed). In dictionary, or separate?
    # Key: form + "__" + wtype + "__" + column
    # Some values (min, max, options) need to be computed to be entered.

    slider_kwargs = { # Should this also be packaged in a list?
        "label": "Filtering:", 
        "key": "mykey", 
        "help": "Adjust slider to change range of allowed values",
        "min_value": 0,
        "max_value": 10,
        "value": (0,10)
        }
    multi_kwargs = {
        "label": "Filtering:", 
        "key": "mykey", 
        "help": "Adjust slider to change range of allowed values",
        "options": [],
        "default": []
    }
    regex_widget = [
        {
            "wtype": "selectbox", 
            "kwargs": {
                "label": "Regex Function",
                "help": "Select function from regex module to use",
                "key": "mykey",
                "options": ["Findall", "Find", "Count", "Boolean"]
            }
        },
        {
            "wtype": "text_input", 
            "kwargs": {
                "label": "Substring or regular expression",
                "help": "Enter substring or regular expression for which to search",
                "key": "mykeyagain"
            }
        }
    ]
    fuzzy_widget = [
        {
            "wtype": "selectbox", 
            "kwargs": {
                "label": "Fuzzywuzzy Scorer",
                "help": "Select scoring function from fuzz submodule of fuzzywuzzy",
                "key": "mykey",
                "options": ["WRatio", "ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"],
                "format_func": lambda s: s.replace("_", " ").title()
            }
        },
        {
            "wtype": "text_input", 
            "kwargs": {
                "label": "Substring or regular expression to fuzzily match",
                "help": "Enter substring for which to fuzzily search",
                "key": "mykeyagain"
            }
        }
    ]
    return