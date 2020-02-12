import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data.split_dataset import SplitDataset


class DataAnalyzer:
    def maxlen(self, split_ds: SplitDataset, token_col: str) -> int:
        return max(
            [df[token_col].apply(len).max() for _, df in split_ds]
        )

    def plot_class_balance(self, split_ds: SplitDataset, target_col: str = 'target') -> go.Figure:
        fig = go.Figure(layout=go.Layout(title='Class Balance', xaxis={'title': target_col}))
        x = split_ds.train_df[target_col].unique()
        y = np.bincount(split_ds.train_df[target_col])
        fig.add_trace(go.Bar(y=y, x=x))
        return fig

    def plot_number_of_tokens(self, split_ds: SplitDataset, token_col: str = 'tokens') -> go.Figure:
        bar_data = {
            'x': [name for name, _ in split_ds],
            'y': [df[token_col].apply(len).sum() for _, df in split_ds]
        }
        fig = go.Figure(
                data=[go.Bar(**bar_data)],
                layout=go.Layout(title=f'Number of {token_col}', xaxis={'title': 'Data Source'}, yaxis_type='log')
            )
        return fig

    def length_plot(self, split_ds: SplitDataset, column: str) -> go.Figure:
        lengths = {name: df[column].apply(len) for name, df in split_ds}
        fig = go.Figure(layout=go.Layout(title=f'{column} length'))
        for key in lengths:
            fig.add_trace(go.Histogram(x=lengths[key], name=key))
        fig.update_layout(barmode='overlay')
        fig.update_layout(xaxis_title=f'Length of {column}')
        fig.update_traces(opacity=0.75)
        fig.update_traces(histnorm='probability')
        fig.update_traces(bingroup=1)
        return fig
    
