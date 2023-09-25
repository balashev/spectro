import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
#import matplotlib.pyplot as plt


def cmap_from_color(color, r=0, c='w', alpha=1, name='mycmap', bins=128):
    """
    return cmap object for given color. Cmap is color --> c, d by default is white (1,1,1)
    parameters:
        - color      : color
        - r          : if r=1 then reverse order
        - c          : second color can be set as tuple (x_r,x_g,x_b) of str ('w', 'k', ...)
    """
    if isinstance(c, str):
        d = {'w': (1, 1, 1), 'k': (0, 0, 0)}
        c = d[c]
    cdict = {}
    if isinstance(color, str):
        if '#' in color:
            color = mcolors.hex2color(color)
        else:
            color = mcolors.hex2color(mcolors.cnames[color])
    print(color, c)
    for i, col in enumerate(['red', 'green', 'blue']):
        if r:
            cdict[col] = (0.0, color[i], color[i]), (1.0, c[i], c[i])
        else:
            cdict[col] = (0.0, c[i], c[i]), (1.0, color[i], color[i])

    if alpha < 0:
        cdict = {
            **cdict,
            'alpha': (
                (0.0, alpha, alpha),
                (1.0, 1.0, 1.0),
            ),
        }
    print(cdict)

    return mcolors.LinearSegmentedColormap(name, cdict, N=bins)
    #mpl.colormaps.register(mcolors.LinearSegmentedColormap.from_list(name, cdict, N=bins))
    #return plt.get_cmap(name)
    #plt.register_cmap(name=name, cmap=cdict) #, lut=bins)
    #return plt.get_cmap(name)


class rectangle():
    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.right = self.left + width
        self.bottom = self.top - height
        self.data = [left, top - height, width, height]

    def __str__(self):
        return str(list([self.left, self.bottom, self.right - self.left, self.top - self.bottom]))

    def __repr__(self):
        return str(list([self.left, self.bottom, self.right - self.left, self.top - self.bottom]))


class rect_param():
    """
    class for setting position and size of axis object of the regular grid plot:
    parameters:
        - n_rows      : number of rows
        - n_cols      : number of columns
        - row_offset  : offset between rows
        - col_offset  : offset between columns
        - width       : width of the grid
        - height      : height of the grid
        - v_indend    :
        - order       : order of the panel, 'v' for vertical and 'h' for horizontal
    """
    def __init__(self, n_rows=1, n_cols=1, row_offset=0, col_offset=0,
                 width=1.0, height=1.0,  order='v', v_indent=0.05, h_indent=0.03):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.row_offset = row_offset
        self.col_offset = col_offset
        self.width = width
        self.height = height
        self.v_indent = v_indent
        self.h_indent = h_indent
        self.order = order

def specify_rects(rect_pars):
    """
    function to form rects from the list of regular grid plot objects <rect_param>:
    parameters:
        - rect_pars   : list of rect_param object. can be just one object.

        example:
            rects = [rect_param(n_rows=2, n_cols=1, order='v', height=0.2, row_offset=0.02),
                     rect_param(n_rows=10, n_cols=2, order='v', height=0.75)
                     ]
            ps.specify_rects(rects)
    """
    rects = []
    if not isinstance(rect_pars, list):
        rect_pars = [rect_pars]
    left = rect_pars[0].v_indent
    top = 1
    for r in rect_pars:
        panel_h = (r.height - r.h_indent - r.row_offset * (r.n_rows - 1)) / r.n_rows
        panel_w = (r.width - r.v_indent - r.col_offset * (r.n_cols - 1)) / r.n_cols
        for i in range(r.n_rows * r.n_cols):
            if r.order == 'v':
                col = i // r.n_rows
                row = i % r.n_rows
            if r.order == 'h':
                col = i % r.n_cols
                row = i // r.n_cols
            rects.append(rectangle(left + (panel_w + r.col_offset) * col, top - (panel_h + r.row_offset) * row, panel_w, panel_h))
            # [left+(panel_w+r.col_offset)*col, top-panel_h*(row+1)-r.row_offset*row, panel_w, panel_h])
        top -= r.height
        top -= r.h_indent
        # if r.row_offset == 0:
        #    top -= r.h_indent

    return rects