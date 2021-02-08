import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#bokeh
from bokeh.palettes import brewer, Spectral10
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, Line
from bokeh.models.tools import HoverTool
from bokeh.transform import linear_cmap
from bokeh.io import output_notebook, show

columns2 = ['Player', 
            'Outcome',
            'Kills',
            'Deaths',
            '+/-',
            'Flag Captures',
            'Flag Attempts',
            'Score',
            'Time Played',
            'Damage Dealt',
            'Damage Taken',
            'DR',
            'DKR',
            'DDF',
            'Laser DD',
            'Laser DT',
            'Missile DD',
            'Missile DT',
            'Grenade DD',
            'Grenade DT',
            'Bouncy DD',
            'Bouncy DT']

"""Interactive chart to display relationship between two statistical categories for CTA Joy games.\n 
Individual player stats have been aggregrated across known names and each player's category averages are used to build each chart."""

avg_player_stats = pd.read_csv("https://raw.githubusercontent.com/ctaela/cta_dive/master/data/avg_player_stats_all.csv")

def plot_stats(x, y, df):
    
    """
    Create Linear Regression Line
    """
    a, b = df[x], df[y]

    X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.33, random_state=42)

    X_train = np.array(X_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    regr = linear_model.LinearRegression()

    regr.fit(X_train, y_train)

    df[y + " STD"] = df[y].apply(lambda a: round((a-df[y].mean())/df[y].std()))
    df[y + " rank"] = df[y].rank(ascending=False)
    df[x + " rank"] = df[x].rank(ascending=False)
    
    mapper = linear_cmap(field_name=y + " STD", palette=brewer["RdBu"][len(df[y + " STD"].unique())], 
                         low=min(df[y + " STD"].unique()), high=max(df[y + " STD"].unique()))
    
    source = ColumnDataSource(df)
    p = figure(x_range=(df[x].min() - df[x].std(), df[x].max() + df[x].std()), 
               y_range=(df[y].min() - df[y].std(), df[y].max() + df[y].std()))
    r1 = p.circle(x=x, y=y,
            source=source, size=10, color=mapper, legend_group= y + " STD")

    p.title.text = y + " vs. " + x
    p.title.align = "center"
    p.xaxis.axis_label = x
    p.yaxis.axis_label = y
    p.legend.location = 'top_left'
    p.legend.title = "St. Dev's from Avg " + y

    p.add_tools(HoverTool(renderers=[r1], tooltips=[
                    ("Player", "@Player"),
                    (y, "@{" + y +"}{0.000}"),
                    (y + " Rank", "#@{" + y + " rank}"),
                    (x, "@{" + x +"}{0}"),
                    (x + " Rank", "#@{" + x + " rank}")]))

    line_x = [df[x].min().item() - df[x].std().item(), df[x].max().item() + df[x].std().item()]
    line_y = [(line_x[0]*regr.coef_.item()) + regr.intercept_.item(), (line_x[1]*regr.coef_.item()) + regr.intercept_.item()]
    r2 = p.line(line_x, line_y, line_width=2, color="black")
    p.add_tools(HoverTool(renderers=[r2], 
                          tooltips=[(x, "$x{0000}"),
                          ("Predicted " + y, "$y")]))

    st.bokeh_chart(p)

y_option = st.selectbox(
    'Which category will be on the y-axis?',
    ['Kills',
    'Deaths',
    '+/-',
    'Flag Captures',
    'Flag Attempts',
    'Score',
    'Time Played',
    'Damage Dealt',
    'Damage Taken',
    'DR',
    'DKR',
    'DDF',
    'Laser DD',
    'Laser DT',
    'Missile DD',
    'Missile DT',
    'Grenade DD',
    'Grenade DT',
    'Bouncy DD',
    'Bouncy DT',
    'Rating'])

x_option = st.selectbox(
    'Which category will be on the x-axis?',
    ['Kills',
    'Deaths',
    '+/-',
    'Flag Captures',
    'Flag Attempts',
    'Score',
    'Time Played',
    'Damage Dealt',
    'Damage Taken',
    'DR',
    'DKR',
    'DDF',
    'Laser DD',
    'Laser DT',
    'Missile DD',
    'Missile DT',
    'Grenade DD',
    'Grenade DT',
    'Bouncy DD',
    'Bouncy DT',
    'Rating'])

st.write('Note:  Minimum of 50 games played.')

plot_stats(x_option, y_option, avg_player_stats)

