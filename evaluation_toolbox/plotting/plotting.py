import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import os


#%%
def get_figures(n_rows,n_cols,figsize=(10,10),sharex=True,sharey=True):
    """ Get figure and axes object
    args:
        n_rows: number of rows
        n_cols: number of columns
        figsize: size of figure
        sharex: share x axis
    returns: figure object, axes object"""
    return plt.subplots(n_rows, n_cols, sharex=sharex, sharey=sharey, figsize=figsize)
    

def add_font(font_path='/mimer/NOBACKUP/groups/brainage/thesis_brainage/evaluation_toolbox/fonts/NimbusSans.otf'):
    """ Register a single font file with matplotlib """
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    return font_name


def set_r_params(small=14, medium=16, big=18):
    """ Set rc parameters for plots """
    font_name = add_font()  # Register and get font name
    mpl.rcParams.update()
    plt.rc('font', family=font_name)
    plt.rc('font', size=medium)         # controls default text sizes
    plt.rc('axes', titlesize=big)       # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small)    # legend fontsize
    plt.rc('savefig', dpi=300)          # figure resolution
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)

def set_size(fig, a, b):
    """ Set size of figure
    args:   
        fig: figure object
        a: width in inches
        b: height in inches
    returns: figure object"""
    fig.set_size_inches(a, b)
    fig.set_tight_layout(True)
    return fig

def set_style(grid,offleft=5,offbottom=5,spinewidth=1.4):
    """ Set style of plot, and apply tight layout
    args:
        grid: seaborn.FacetGrid object
        offleft: offset of left spine
        offbottom: offset of bottom spine
        spinewidth: width of spines
    returns: seaborn.FacetGrid object"""
    grid.despine(top=True, right=True, left=False, bottom=False, offset={'left': offleft, 'bottom': offbottom})

    for i in range(0,len(grid.axes[0][:])):
        grid.axes[0][i].grid(axis='y', color='C7', linestyle='--', lw=.8)
        grid.axes[0][i].tick_params(which='major', direction='out', length=3, width=spinewidth, bottom=True, left=True)
        grid.axes[0][i].tick_params(which='minor', direction='out', length=2, width=spinewidth/2, bottom=True, left=True)
        plt.setp(grid.axes[0][i].spines.values(), linewidth=spinewidth)
    grid.tight_layout()

    return grid

def set_style_ax(fig,axes,both_axes=True,offleft=5,offbottom=5,spinewidth=1.4):
    """ Set style of plot, and apply tight layout
    args:
        fig: figure object
        axes: axes object
        both_axes: highlight values on both axes
        offleft: offset of left spine
        offbottom: offset of bottom spine
        spinewidth: width of spines
    returns: figure object"""
    sns.despine(fig,top=True, right=True, left=False, bottom=False, offset={'left': offleft, 'bottom': offbottom})

    for ax in axes.flatten():
        if both_axes:
            ax.grid(axis='both', color='C7', linestyle='--', lw=.8)
        else:
            ax.grid(axis='y', color='C7', linestyle='--', lw=.8)
        ax.tick_params(which='major', direction='out', length=3, width=spinewidth, bottom=True, left=True)
        ax.tick_params(which='minor', direction='out', length=2, width=spinewidth/2, bottom=True, left=True)
        plt.setp(ax.spines.values(), linewidth=spinewidth)
    fig.tight_layout()

    return fig

def save_figure(plot, file):
    """Saves a figure to a file.

    Parameters
    ----------
    plot : matplotlib.figure.Figure
        The figure to save.
    file : str
        The file to save the figure to.
    """
    plot.savefig(file, dpi=300, bbox_inches='tight')