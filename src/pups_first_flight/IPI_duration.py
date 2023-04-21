
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='whitegrid', font_scale=1.5,
        rc={"lines.linewidth": 1, 'grid.linestyle': '--'})

c_list = sns.color_palette("muted", 7)
target_sheet = ["418", "422", "423", "424", "425", "427", "430"]
flapping_sheets = ["422", "430"]
true_flight_sheets = ["418", "423", "424", "425", "427"]


def modify_data(df_data):
    df_data = df_data.sort_values(by="Time to landing [s]", ascending=False)

    return df_data


def make_flapping_fig(df_targets):
    labels_list = ["batC", "batI"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))

    for idx, df_target in enumerate(df_targets):
        df_target = modify_data(df_target)
        sns.lineplot(x=df_target["Time to landing [s]"],
                     y=df_target["IPI [ms]"], marker='o', markersize=15, ax=ax1, c=c_list[idx], label=labels_list[idx])
    ax1.set_xlim((0, 1))
    ax1.set_ylim((0, 150))
    ax1.invert_xaxis()
    for idx, df_target in enumerate(df_targets):
        df_target = modify_data(df_target)
        sns.lineplot(x=df_target["Time to landing [s]"],
                     y=df_target["Duration [ms]"], marker='o', markersize=15, ax=ax2, c=c_list[idx], label=labels_list[idx])
    ax1.set_xlim((0, 1))

    ax2.set_xlim((0, 1))
    ax2.set_ylim((0, 3.5))
    ax1.invert_xaxis()
    ax2.invert_xaxis()
    plt.tight_layout()
    ax1.legend()
    ax2.legend()
    plt.savefig("data/pups_first_flight/flapping.png")


def make_true_flight_fig(df_targets):
    labels_list = ["batA", "batD", "batE", "batF", "batG"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))

    for idx, df_target in enumerate(df_targets):
        df_target = modify_data(df_target)
        sns.lineplot(x=df_target["Time to landing [s]"],
                     y=df_target["IPI [ms]"], marker='o', markersize=15, ax=ax1, c=c_list[2 + idx], label=labels_list[idx])
    ax1.set_xlim((0, 1))
    ax1.set_ylim((0, 150))
    ax1.invert_xaxis()
    for idx, df_target in enumerate(df_targets):
        df_target = modify_data(df_target)
        sns.lineplot(x=df_target["Time to landing [s]"],
                     y=df_target["Duration [ms]"], marker='o', markersize=15, ax=ax2, c=c_list[2 + idx], label=labels_list[idx])
    ax1.set_xlim((0, 1))

    ax2.set_xlim((0, 1))
    ax2.set_ylim((0, 3.5))
    ax1.invert_xaxis()
    ax2.invert_xaxis()
    plt.tight_layout()
    ax1.legend()
    ax2.legend()
    plt.savefig("data/pups_first_flight/True_flight.png")


def main(target_data):
    df_target_data = pd.read_excel(target_data, sheet_name=target_sheet)
    flapping_data = [df_target_data[target_ID]
                       for target_ID in flapping_sheets]
    true_flight_data = [df_target_data[target_ID]
                        for target_ID in true_flight_sheets]
    make_flapping_fig(flapping_data)
    make_true_flight_fig(true_flight_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=False)
    args = parser.parse_args()
    main(args.data)
