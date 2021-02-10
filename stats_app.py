import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#bokeh
from bokeh.palettes import brewer, Spectral10
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, LabelSet, Line
from bokeh.models.tools import HoverTool
from bokeh.transform import linear_cmap
from bokeh.io import output_notebook, show

"""Interactive chart to display relationship between two statistical categories for CTA Joy games.\n 
Individual player stats have been aggregrated across known names and each player's category averages are used to build each chart."""

avg_player_stats = pd.read_csv("https://raw.githubusercontent.com/ctaela/cta_dive/master/data/avg_player_stats_all.csv")

def plot_stats(x_axis, y_axis, df, highlight=[]):
    
    """
    Create Linear Regression Line
    """
    a, b = df[x_axis], df[y_axis]

    X_train, X_test, y_train, y_test = train_test_split(a, b, test_size=0.33, random_state=42)

    X_train = np.array(X_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    regr = linear_model.LinearRegression()

    regr.fit(X_train, y_train)

    df[y_axis + " STD"] = df[y_axis].apply(lambda a: round((a-df[y_axis].mean())/df[y_axis].std()))
    df[y_axis + " rank"] = df[y_axis].rank(ascending=False)
    df[x_axis + " rank"] = df[x_axis].rank(ascending=False)
    
    mapper = linear_cmap(field_name=y_axis + " STD", palette=brewer["RdBu"][len(df[y_axis + " STD"].unique())], 
                         low=min(df[y_axis + " STD"].unique()), high=max(df[y_axis + " STD"].unique()))
    
    source = ColumnDataSource(df)
    source2 = ColumnDataSource(df[df["Player"].isin(highlight)])
    
    p = figure(x_range=(df[x_axis].min() - df[x_axis].std(), df[x_axis].max() + df[x_axis].std()), 
               y_range=(df[y_axis].min() - df[y_axis].std(), df[y_axis].max() + df[y_axis].std()))
    
    r1 = p.circle(x=x_axis, y=y_axis,
            source=source, size=10, color=mapper, line_color="black", legend_group= y_axis + " STD")

    p.title.text = y_axis + " vs. " + x_axis
    p.title.align = "center"
    p.xaxis.axis_label = x_axis
    p.yaxis.axis_label = y_axis
    p.legend.location = 'top_left'
    p.legend.title = "St. Dev's from Avg " + y_axis
    p.background_fill_color = "#dddddd"
    p.background_fill_alpha = 0.1
    
    line_x = [df[x_axis].min().item() - df[x_axis].std().item(), df[x_axis].max().item() + df[x_axis].std().item()]
    line_y = [(line_x[0]*regr.coef_.item()) + regr.intercept_.item(), (line_x[1]*regr.coef_.item()) + regr.intercept_.item()]
    r2 = p.line(line_x, line_y, line_width=2, color="black")

    p.add_tools(HoverTool(renderers=[r1], tooltips=[
                    ("Player", "@Player"),
                    (y_axis, "@{" + y_axis +"}{0.000}"),
                    (y_axis + " Rank", "#@{" + y_axis + " rank}"),
                    (x_axis, "@{" + x_axis +"}{0}"),
                    (x_axis + " Rank", "#@{" + x_axis + " rank}")]))

    
    p.add_tools(HoverTool(renderers=[r2], 
                          tooltips=[(x_axis, "$x_axis{0000}"),
                          ("Predicted " + y_axis, "$y_axis")]))
       
    labels = LabelSet(x=x_axis, 
                         y=y_axis, text="Player", y_offset=8,
                  text_font_size="11px", text_color="#555555",
                  source=source2, text_align='center')
    
    p.add_layout(labels)

    st.bokeh_chart(p)

y_option = st.selectbox(
    'Which category will be on the y-axis?',
    ['Kills',
    'Deaths',
    '+/-',
    'Flag Captures',
    'Flag Attempts',
    'Score',
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

st.text('Note:  Only players with a minimum of 50 games are displayed.')

highlight_players = st.multiselect("Select players to highlight on the chart.", avg_player_stats["Player"].unique())

plot_stats(x_option, y_option, avg_player_stats, highlight_players)

