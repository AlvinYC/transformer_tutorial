
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
    fields = [('epoch',EPOCH),('loss',LOSS),('ppl',PPL),('Accuracy',ACC)]   # 配合log檔的格式
    examples = list(map(lambda x:data.Example.fromlist(x,fields),list_from_csv))
    return examples,fields

def get_curve(dataset, target):
    data_curve = eval('dataset'+ '.' + target)
    curve = np.concatenate(list(data_curve))          # m * 1 * 1 -> m * 1 (string)
    return curve


def main():

    train_path = '/home/alvin/git_repository/transformer_tutorial/log/epoch400/m30k_deen_shr.train.log'
    valid_path = '/home/alvin/git_repository/transformer_tutorial/log/epoch400/m30k_deen_shr.valid.log'
    target_field = 'loss'   # set target filed to be y-axis, need one of fileds which get_example() setup

    # text -> example -> data.Dataset
    # get example from txt file, if text format is changed, only update get_example() function 
    examples,fields = get_example(train_path)
    train_dataset = data.Dataset(examples=examples,fields=fields)
    examples,fields = get_example(valid_path)
    valid_dataset = data.Dataset(examples=examples,fields=fields)
    
    # retrieve x/y value from data.Dataset object
    # get x-axis value
    node_num = len(train_dataset)
    xs = [list(range(node_num)) for i in range(2)]              # multiline (two) need  2  duplicate x-axis
    # get y-axis value
    curve1= get_curve(train_dataset,target_field)
    curve2= get_curve(valid_dataset,target_field)
    ys = [curve1.tolist(), curve2.tolist()]                                 # [flost_list, float_list]

    # build Bokeh data format
    source = ColumnDataSource(data=dict(
        x = xs,
        y = ys,
        color = (Category10[3])[0:2],
        group = ['train','valid']))
    p3 = figure(plot_width=600, plot_height=300,   x_axis_label='Epoch(s)',y_axis_label=target_field)
   
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
        (target_field,'@invisible_ys')])
    hover.renderers = [line]
    p3.add_tools(hover)
    p3.legend.location = "top_left"
    p3.legend.click_policy="mute"
    show(p3)


if __name__ == "__main__":
    main()
