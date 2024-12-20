from matplotlib import pyplot as plt


def add_threshold(ax, threshold):
    ax.axline((threshold, 0), (threshold, 40))


def create_histogram_plot(hist):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Histogram')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Brightness')
    ax.plot(range(len(hist)), hist)
    return ax


def plot_histogram_with_thresholds(hist, thresholds=[]):
    ax = create_histogram_plot(hist)
    for threshold in thresholds:
        add_threshold(ax, threshold)
    plt.show()
