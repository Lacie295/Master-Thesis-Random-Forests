import plotly.graph_objects as go
from source.utils import learning_manager


def plot(algos):
    print("Building graphs.")
    g_acc = go.Figure()
    g_time = go.Figure()
    for algo in algos:
        discriminant = learning_manager.discriminants[algo]
        datas = []
        name = ""

        for file in discriminant:
            data = discriminant[file]
            acc = data.avg_acc
            time = data.avg_time
            size = data.size
            datas.append((acc, time, size))
            name = data.NAME

        datas = sorted(datas, key=lambda i: i[2])

        accs = [i[0] for i in datas]
        times = [i[1] for i in datas]
        sizes = [i[2] for i in datas]
        g_acc.add_trace(go.Scatter(x=sizes, y=accs, name=name))
        g_time.add_trace(go.Scatter(x=sizes, y=times, name=name))

    g_acc.update_layout(title='Average accuracy by size of data',
                        xaxis_title='Training Data Size',
                        yaxis_title='Prediction Accuracy')
    g_time.update_layout(title='Average time by size of data',
                         xaxis_title='Training Data Size',
                         yaxis_title='Time to construct predictor')
    g_acc.write_image("plots/acc.png")
    g_time.write_image("plots/time.png")


def plot_all():
    plot(learning_manager.algo_names.keys())
