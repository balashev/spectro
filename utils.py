from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

def cmap_from_color(color, r=0, c='w', name='mycmap'):
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
        color = mcolors.hex2color(mcolors.cnames[color])
    print(color, c)
    for i, col in enumerate(['red', 'green', 'blue']):
        if r:
            cdict[col] = (0.0, color[i], color[i]), (1.0, c[i], c[i])
        else:
            cdict[col] = (0.0, c[i], c[i]), (1.0, color[i], color[i])

    plt.register_cmap(name=name, data=cdict)
    return plt.get_cmap(name)
