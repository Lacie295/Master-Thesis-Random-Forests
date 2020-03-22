import plotly.graph_objects as go
from plotly.subplots import make_subplots
from source.utils import learning_manager, file_manager
import sys


def plot(algos):
    print("Building graphs.")
    # Create a graph for classification accuracy and for build time
    g_acc = go.Figure()
    g_time = go.Figure()
    for algo in algos:
        # For each discriminant create a line in the graoh
        discriminant = learning_manager.discriminants[algo]
        datas = []
        name = ""

        # Create a list of assiciated accuracy, time and sample size for all files
        for file in discriminant:
            data = discriminant[file]
            acc = data.avg_acc
            time = data.avg_time
            size = data.size
            datas.append((acc, time, size))
            name = data.NAME

        # Sort data by sample size
        datas = sorted(datas, key=lambda i: i[2])

        # Create individual accuracy, time and size arrays and add them to their respective plots.
        accs = [i[0] for i in datas]
        times = [i[1] for i in datas]
        sizes = [i[2] for i in datas]
        g_acc.add_trace(go.Scatter(x=sizes, y=accs, name=name))
        g_time.add_trace(go.Scatter(x=sizes, y=times, name=name))

    # Make the graph look nice
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

    # Make the DL8-forest specific graphs
    for algo in algos:
        if algo == "DL8-forest" or algo == "OptDL8-forest":
            discriminant = learning_manager.discriminants[algo]
            for file in discriminant:
                # Start with the attribute spread graph
                layout = go.Layout(title='Frequency of attributes by depth',
                                   xaxis=dict(type='category', title='Attribute number (sorted by total %'),
                                   yaxis=dict(title='Frequency (%)'))
                g_spread = go.Figure(layout=layout)

                # Get the data
                data = discriminant[file]
                depth_map = {}
                total = {}

                total_count = 0
                depth_count = {}

                # Flatten the depth_map of each iteration over the file (such that depth_map[d] contains all attributes of
                # all trees at depth d)
                for i in data.depth_map:
                    d = data.depth_map[i]
                    for depth in d:
                        attrs = d[depth]
                        if depth not in depth_count:
                            depth_count[depth] = 0

                        for attr in attrs:
                            depth_count[depth] += attrs[attr]
                            total_count += attrs[attr]

                # Transform the depth_map counts into ratios and keep track of the total ratio of each attribute
                for i in data.depth_map:
                    d = data.depth_map[i]
                    for depth in d:
                        attrs = d[depth]
                        if depth not in depth_map:
                            depth_map[depth] = {}

                        for attr in attrs:
                            if attr not in depth_map[depth]:
                                depth_map[depth][attr] = 100 * attrs[attr] / depth_count[depth]
                            else:
                                depth_map[depth][attr] += 100 * attrs[attr] / depth_count[depth]
                            if attr not in total:
                                total[attr] = 100 * attrs[attr] / total_count
                            else:
                                total[attr] += 100 * attrs[attr] / total_count

                # Add a trace for each depth, making sure it's sorted by total attribute ratio
                for depth in depth_map:
                    keys = [k for k, v in sorted(total.items(), key=lambda item: -item[1])]
                    values = [depth_map[depth][k] if k in depth_map[depth] else 0 for k in keys]
                    g_spread.add_trace(go.Bar(x=keys, y=values, name="Depth " + str(depth)))

                g_spread.write_image("plots/spread/spread_" + file.split("/")[-1].split(".")[0] + ".png")

        if algo == "DL8-forest":
            discriminant = learning_manager.discriminants[algo]
            for file in discriminant:
                data = discriminant[file]

                # Get the unanimity for the file
                unanimity = data.unanimity
                n_estimators = data.n_estimators

                # Transform the list of unanimity ratios into a count of unanimity
                unan = [0] * (n_estimators[0] + 1)
                for row in unanimity:
                    for col in row:
                        unan[col] += 1

                # Add unanimity to graph
                layout = go.Layout(title='Tree unanimity in DL8Forest',
                                   xaxis=dict(type='category', title='Number of trees in agreement'),
                                   yaxis=dict(title='Frequency (#)'))
                g_unan = go.Figure(layout=layout)
                g_unan.add_trace(go.Scatter(x=list(range(n_estimators[0] + 1)), y=unan))
                g_unan.write_image("plots/unan/unan_" + file.split("/")[-1].split(".")[0] + ".png")

        if algo == "OptDL8-forest":
            discriminant = learning_manager.discriminants[algo]
            for file in discriminant:
                print("graphing:" + file)
                data = discriminant[file]
                ns = list(range(1, max(data.n_estimators) + 1))
                acc = []

                for n in ns:
                    sys.stdout.write("\rn = " + str(n))
                    sys.stdout.flush()
                    accs = data.check_acc_with_n_trees(n)
                    acc.append(accs)
                print()

                layout = go.Layout(title='Forest accuracy with n trees',
                                   xaxis=dict(rangemode="tozero", title='Number of trees used'),
                                   yaxis=dict(rangemode="tozero", title='Prediction accuracy'))
                subplot_titles = ["Test accuracy on forest " + str(i) for i in range(len(data.n_estimators))]
                subplot_titles.extend(["Train accuracy on forest " + str(i) for i in range(len(data.n_estimators))])
                g_f_acc = make_subplots(rows=2, cols=len(data.n_estimators),
                                        shared_xaxes=True, shared_yaxes=True, x_title='Number of trees used',
                                        y_title='Prediction accuracy',
                                        subplot_titles=tuple(subplot_titles))
                g_f_acc.update_layout(height=1200, width=200 + 500 * len(data.n_estimators),
                                      title_text='Forest accuracy with n trees on ' + file)

                for i in range(len(acc[0])):
                    n_estimators = data.n_estimators[i]
                    g_f_acc.add_trace(go.Scatter(x=ns[:n_estimators], y=acc[:n_estimators, i], mode='lines',
                                                 name="Forest #" + str(i)), row=1, col=i + 1)

                acc = []

                for n in ns:
                    sys.stdout.write("\rn = " + str(n))
                    sys.stdout.flush()
                    accs = data.check_acc_with_n_trees(n, test=False)
                    acc.append(accs)
                print()

                for i in range(len(acc[0])):
                    n_estimators = data.n_estimators[i]
                    g_f_acc.add_trace(go.Scatter(x=ns[:n_estimators], y=acc[:n_estimators, i], mode='lines',
                                                 name="Forest #" + str(i)), row=2, col=i + 1)

                g_f_acc.write_image("plots/acc/acc_" + file.split("/")[-1].split(".")[0] + ".png")


def plot_all():
    plot(learning_manager.algo_names.keys())


def table(algos):
    # Generate a latex table containing all the accuracies for each algorithm
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
