import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import math as math
from ipywidgets import interact
import ipywidgets as ipw

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.io import show, output_notebook, push_notebook

import time

output_notebook()
st.write("hello word")
