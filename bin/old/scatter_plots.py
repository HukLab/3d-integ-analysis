import matplotlib.pyplot as plt

get_index_mask = lambda df, val: df.index == val

def plot_by_key(df, key, mean_index_val='all', label='', color='black'):
    df2d = df[df.dotmode == '2d']
    df3d = df[df.dotmode == '3d']
    pts = df2d.join(df3d, lsuffix='_2d', rsuffix='_3d')
    index_mask = get_index_mask(pts, mean_index_val)
    if index_mask.any():
        pts[index_mask].plot(key + '_2d', key + '_3d', label='', marker='>', linestyle='', color=color)
    pts[~index_mask].plot(key + '_2d', key + '_3d', label=label, marker='o', linestyle='', color=color)
    ymin, ymax = df[key].min(), df[key].max()
    ymin -= ymin*.05
    ymax += ymax*.05
    return ymin, ymax

def plot_info(key, (ymin, ymax), legend=False, loc=None):
    plt.plot([ymin, ymax], [ymin, ymax], linestyle='dashdot', color='gray')
    plt.xlabel('2d')
    plt.ylabel('3d')
    plt.title(key)
    if legend:
        if loc:
            plt.legend(loc=loc)
        else:
            plt.legend()
