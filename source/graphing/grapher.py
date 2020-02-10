import plotly.graph_objects as go
from source.utils import learning_manager, file_manager


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
                         yaxis_title='Time to construct predictor (s)')
    g_acc.update_yaxes(type="log")
    g_time.update_yaxes(type="log")
    g_acc.update_xaxes(rangemode="tozero")
    g_time.update_xaxes(rangemode="tozero")
    g_acc.write_image("plots/acc.png")
    g_time.write_image("plots/time.png")

    if "DL8-forest" in algos:
        discriminant = learning_manager.discriminants["DL8-forest"]
        for file in discriminant:
            layout = go.Layout(title='Frequency of attributes by depth',
                               xaxis=dict(type='category', title='Attribute number'),
                               yaxis=dict(title='Number of customers'))
            g_spread = go.Figure(layout=layout)
            data = discriminant[file]
            depth_map = {}
            total = {}
            for i in data.depth_map:
                d = data.depth_map[i]
                for depth in d:
                    attrs = d[depth]
                    if depth not in depth_map:
                        depth_map[depth] = {}
                    for attr in attrs:
                        if attr not in depth_map[depth]:
                            depth_map[depth][attr] = attrs[attr]
                        else:
                            depth_map[depth][attr] += attrs[attr]
                        if attr not in total:
                            total[attr] = attrs[attr]
                        else:
                            total[attr] += attrs[attr]
            print(depth_map)

            for depth in depth_map:
                keys = [k for k, v in sorted(total.items(), key=lambda item: -item[1])]
                values = [depth_map[depth][k] if k in depth_map[depth] else 0 for k in keys]
                g_spread.add_trace(go.Bar(x=keys, y=values, name="Depth " + str(depth)))

            g_spread.write_image("plots/spead_" + file.split("/")[-1].split(".")[0] + ".png")


def plot_all():
    plot(learning_manager.algo_names.keys())


def table(algos):
    print("\\begin{tabular}{ll|" + ("l" * len(algos)) + "}")
    s = "Dataset & Size"
    for algo in algos:
        s += " & " + algo
    s += "\\\\"
    print(s)
    print("\\hline")
    files = sorted(file_manager.data_sets, key=lambda a: learning_manager.discriminants[algos[0]][a].size)
    for file in files:
        s = file.split("/")[-1].split(".")[0] + " & " + str(learning_manager.discriminants[algos[0]][file].size)
        max_acc = max([learning_manager.discriminants[d][file].avg_acc for d in algos])
        for algo in algos:
            d_acc = learning_manager.discriminants[algo][file].avg_acc
            s += " & {0:.2f}\\%".format(round(d_acc * 100, 2)) if d_acc < max_acc else \
                " & \\textcolor{{uclgreen}}{{{0:.2f}\\%}}".format(round(d_acc * 100, 2))
        s += "\\\\"
        print(s)
    print("\\end{tabular}")


def table_all():
    table(list(learning_manager.algo_names.keys()))
