import plotly.graph_objects as go
from source.utils import learning_manager


def plot(algos):
    g_acc = go.Figure()
    g_time = go.Figure()
    for algo in algos:
        discriminant = learning_manager.discriminants[algo]
        times = []
        sizes = []
        accs = []
        name = ""
        print(algo)
        for file in discriminant:
            data = discriminant[file]
            times.append(data.avg_time)
            accs.append(data.avg_acc)
            sizes.append(data.size)
            name = data.NAME
            print(file)
        g_acc.add_trace(go.Scatter(x=sizes, y=accs, name=name))
        g_time.add_trace(go.Scatter(x=sizes, y=times, name=name))
        print(times)
        print(accs)
        print(sizes)

    g_acc.update_layout(title='Average accuracy by size of data',
                        xaxis_title='Training Data Size',
                        yaxis_title='Prediction Accuracy')
    g_time.update_layout(title='Average time by size of data',
                         xaxis_title='Training Data Size',
                         yaxis_title='Time to construct predictor')
    g_acc.show()
    g_time.show()


def plot_all():
    plot(learning_manager.algo_names.keys())
