
import subprocess
import codecs
import torchtext.data as data
import ast
import numpy as np
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10
import random

train_path = '/home/alvin/git_repository/transformer_tutorial/log/epoch400/m30k_deen_shr.train.log'
valid_path = '/home/alvin/git_repository/transformer_tutorial/log/epoch400/m30k_deen_shr.valid.log'

def read_text(filename):
    fc = subprocess.getoutput('file -b --mime-encoding %s' %filename)
    with codecs.open(filename, 'r', encoding=fc) as f:
        content = f.read().splitlines()
    #next(content,null)
    return content

def get_example(path):
    contents = read_text(path)[1:]  # remove header
    list_from_csv = list(map(lambda x: x.split(','), contents))
    EPOCH = data.Field()
    LOSS = data.Field()
    PPL = data.Field()
    ACC = data.Field()
    fields = [('epoch',EPOCH),('loss',LOSS),('ppl',PPL),('acc',ACC)]   # 配合 .json 文字檔的格
    examples = list(map(lambda x:data.Example.fromlist(x,fields),list_from_csv))
    return examples,fields

def main():

    examples,fields = get_example(train_path)
    train_output = data.Dataset(examples=examples,fields=fields)
    examples,fields = get_example(valid_path)
    valid_output = data.Dataset(examples=examples,fields=fields)
    
    node_num = len(train_output)
    xs = [list(range(node_num)) for i in range(2)]
    train_ppl = np.concatenate(list(train_output.acc))
    valid_ppl = np.concatenate(list(valid_output.acc))
    #ys = np.array(np.append(train_ppl,valid_ppl),dtype=float).tolist()
    ys = [train_ppl.tolist(), valid_ppl.tolist()]
    source = ColumnDataSource(data=dict(
        x = xs,
        y = ys,
        color = (Category10[3])[0:2],
        group = ['train','valid']))
    p3 = figure(plot_width=600, plot_height=300,   x_axis_label='Epoch(s)',y_axis_label='Accuracy')
   
    p3.multi_line(
        xs='x',
        ys='y',
        legend_group='group',
        source=source,
        line_color='color'
)
        
    #Add hover tools, basically an invisible line
    source2 = ColumnDataSource(dict(
        invisible_xs= np.concatenate(xs).tolist(),
        invisible_ys= np.concatenate(ys).tolist(),
        group = ['train']*node_num + ['valid'] * node_num))
    line = p3.line(
        'invisible_xs',
        'invisible_ys',
        source=source2,
        alpha=0)
    hover = HoverTool(tooltips =[
        ('dataset','@group'),
        ('epoch','@invisible_xs'),
        ('accuracy','@invisible_ys')])
    hover.renderers = [line]
    p3.add_tools(hover)
    p3.legend.location = "top_left"
    show(p3)


if __name__ == "__main__":
    main()
