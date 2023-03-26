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
