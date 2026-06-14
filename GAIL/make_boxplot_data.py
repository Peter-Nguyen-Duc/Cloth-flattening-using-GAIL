from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np

import scipy.stats as stats


def load_data_from_tag(log_dir, tag):
    """
    SUMMARY:
        Extracts scalar values for a specified tag from TensorBoard event logs.

    ARGS:
        log_dir (str): Directory path where the TensorBoard event logs are stored.
        tag (str): The specific tag (scalar name) to extract values for.

    RETURNS:
        val (list): A list of scalar values corresponding to the specified tag.
    """


    event = EventAccumulator(log_dir)
    event.Reload()

    # E. g. get all values and steps of a scalar called 'environment/reward_IRL_avg'
    val = [s.value for s in event.Scalars(tag)]

    return val


# latest policy learning/AIRL/Saved_models/AIRL_FLICK_3_TASK_26-01-15_07-21-22
# directories = ["learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_15-07-13",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_15-16-43",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_15-20-37",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_15-25-00"]

# best policy learning/AIRL/Saved_models/AIRL_FLICK_3_TASK_26-01-15_07-21-22
# directories = ["learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_16-23-15",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_16-26-45",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_16-31-17",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_16-35-44"]

# "learning/AIRL/Saved_models/AIRL_FLICK_3_TASK_26-01-16_19-34-51"
# # best policy
# directories = ["learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_16-46-32",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_16-49-52",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_16-54-16",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_16-58-27"]

# Latest - learning/AIRL/Saved_models/AIRL_FLICK_3_TASK_26-01-16_19-34-51
# directories = ["learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_21-41-36",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_21-46-34",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_21-51-39",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_21-56-29"]



# latest - no domain randomization
# learning/AIRL/Saved_models/AIRL_FLICK_3_TASK_25-09-11_02-08-54
# directories = ["learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_17-06-16",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_17-09-44",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_17-13-18",
#                "learning/AIRL/logs/AIRL_FLICK_3_TASK_26-01-18_17-16-23"]


directories = [
"GAIL/Saved_models/validation_mass_finetuned_80_steps/0_15/logs/GAIL_CLOTH_TASK_26-06-14_10-37-22",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/0_15/logs/GAIL_CLOTH_TASK_26-06-14_10-40-15",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/0_15/logs/GAIL_CLOTH_TASK_26-06-14_10-44-33",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/0_15/logs/GAIL_CLOTH_TASK_26-06-14_10-48-43",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/0_15/logs/GAIL_CLOTH_TASK_26-06-14_10-52-53",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/0_15/logs/GAIL_CLOTH_TASK_26-06-14_10-57-16",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/0_15/logs/GAIL_CLOTH_TASK_26-06-14_11-01-44",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/0_15/logs/GAIL_CLOTH_TASK_26-06-14_11-06-00",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/0_15/logs/GAIL_CLOTH_TASK_26-06-14_11-10-19",
]

diffusion_policies = [
    
"GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_10-52-10",
"GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_10-56-24",
"GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_11-00-38",
"GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_11-05-26",
"GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_11-10-00",
"GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_11-14-37",
"GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_11-18-42",
"GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_11-23-00",
"GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_11-27-04",

]





add_diffusion_policy_results = True 


name = "test_plot"

diffusion_model = [load_data_from_tag(dirs, "environment/reward_env_end" ) for dirs in diffusion_policies]

boxplot_values_env_end = [load_data_from_tag(dirs, "environment/reward_env_end" ) for dirs in directories]
IRL_values_env_average_end = [load_data_from_tag(dirs, "environment/reward_IRL_avg" ) for dirs in directories]



# --- NEW: Scaling and offset parameters for discriminator data ---
scale_factor = 1  # Adjust this to scale the IRL rewards
offset = -1.2        # Adjust this to offset the IRL rewards
y_coef = 0.0


y_max = 1.7
y_min = -1


y_max_d = 1.7
y_min_d = -1



# # Calculate the scaled y-axis limits
# scaled_ylim_bottom = (y_min * scale_factor) + offset
# scaled_ylim_top = (y_max * scale_factor) + offset




# Apply scaling and offset to IRL_values_env_average_end
scaled_IRL_values = [[(val * scale_factor) + offset +  i * y_coef for val in vals] for i, vals in enumerate(IRL_values_env_average_end)]


x_positions = np.arange(1, len(directories) + 1)  # x-axis positions for

means = [np.mean(vals) for vals in scaled_IRL_values]
stds = [np.std(vals) for vals in scaled_IRL_values]

confidence_intervals = []

alpha = 0.05

for i in range(len(scaled_IRL_values)):

    CI = stats.norm.interval(1 - alpha, loc=means[i], scale=stds[i]/np.sqrt(len(scaled_IRL_values[i])))

    error_margin = CI[0]
    margin_of_error2 = error_margin - means[i]


    confidence_intervals.append(margin_of_error2)



# ----------------------------------

# Apply scaling and offset to IRL_values_env_average_end
diffusion_model = [vals for i, vals in enumerate(diffusion_model)]



diffusion_means = [np.mean(vals) for vals in diffusion_model]
diffusion_stds = [np.std(vals) for vals in diffusion_model]






# Colors for the boxplots: blue for trained, red for unseen
# Trained environment: Mediumseagreen
# unseen environment: Orange
# Baseline environment: cornflowerblue

colors = ['Orange', 'cornflowerblue', 'Orange', 'Orange', 'Orange', 'Orange', 'Orange', 'Orange', 'Orange']



# # Plot boxplots
# plt.figure(figsize=(10, 6))

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(10, 6))
# Plot the Discriminator Reward line and fill on the secondary y-axis
ax1.plot(x_positions, means, color='darkred', linestyle='--', linewidth=1.5)
ax1.fill_between(
    x_positions,
    [-val + means[i] for i, val in enumerate(stds)],
    [val + means[i] for i, val in enumerate(stds)],
    color="#ff0303",
    alpha=0.20,
)


if add_diffusion_policy_results:
    ax1.plot(x_positions, diffusion_means, color='black', linestyle='-', linewidth=1.5)
    ax1.fill_between(
        x_positions,
        [-val + diffusion_means[i] for i, val in enumerate(diffusion_stds)],
        [val + diffusion_means[i] for i, val in enumerate(diffusion_stds)],
        color="gray",
        alpha=0.50,
    )




bplot = ax1.boxplot(boxplot_values_env_end, patch_artist=True, widths=0.6,
            boxprops=dict(facecolor='none', color='blue', linewidth=1),
            medianprops=dict(color='red', linewidth=0.5),
            whiskerprops=dict(color='black', linewidth=1),
            capprops=dict(color='black', linewidth=1),
            flierprops=dict(marker='o', markersize=3, markerfacecolor='none', markeredgecolor='black'),)




for i, patch in enumerate(bplot['boxes']):
    if False:  # Indice specifico per cui vuoi rimuovere il contorno
        patch.set(edgecolor=colors[i], linewidth=2.5)  # Rimuove il contorno
    else:
        patch.set(edgecolor="black", linewidth=1)  # Mantiene il contorno per gli altri

    # patch.set_facecolor(colors[i])


for i in range(len(boxplot_values_env_end)):
    if False: # Replace with the indices of the boxplots you want to adjust
        plt.setp(bplot['medians'][i], color='cornflowerblue', linewidth=3)  # Hide median line
    # elif i == 2 or i == 3 or i == 4 or i==6:
    #     plt.setp(bplot['medians'][i], color='Orange', linewidth=2)  # Keep median line for others
    # elif i==7:
    #     plt.setp(bplot['medians'][i], color='Mediumseagreen', linewidth=2)  # Keep median line for others
    else:
        plt.setp(bplot['medians'][i], color='red', linewidth=1)  # Keep median line for others


# for patch, color in zip(bplot['boxes'], colors):
#     patch.set(color=color, edgecolor="black")

# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)


# Set labels for primary y-axis
ax1.set_ylabel('Performance metric', fontsize=22)
ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])

# For mass
ax1.set_xlabel('Cloth node mass [Kg]', fontsize=22)
ax1.set_xticklabels(["0.005", "0.010", "0.030", "0.050", "0.075", "0.100", "0.125", "0.150", "0.175"])


# # for cloth spacing
# ax1.set_xlabel('Cloth node spacing [m]', fontsize=22)
# ax1.set_xticklabels(["0.025", "0.030", "0.035", "0.040", "0.045", "0.050", "0.055", "0.060", "0.065"])




ax1.tick_params(axis='y', labelsize=14)

ax1.grid(axis='y', linestyle='--', alpha=0.7)

ax1.set_ylim(y_min, y_max)

# # Plot the scaled mean line for IRL_values_env_average_end
# plt.plot(x_positions, means, color='darkred', linestyle='-', linewidth=2, label=f'Discriminator Reward')

# # Fill between for bounds
# plt.fill_between(x_positions, [-val + means[i] for i, val in enumerate(stds)],  [val + means[i] for i, val in enumerate(stds)], color="#ff0303", alpha=0.10, label='Standard deviation')


# # Add edges to the filled bounds
# plt.plot(x_positions, [-val + means[i] for i, val in enumerate(stds)], color='#0062ff', alpha=0.9, linewidth=1.0)
# plt.plot(x_positions, [val + means[i] for i, val in enumerate(stds)], color='#0062ff', alpha=0.9, linewidth=1.0)

# Create a secondary y-axis (right) for the Discriminator Reward
ax2 = ax1.twinx()


# ax2.plot(x_positions, [-val + means[i] for i, val in enumerate(stds)], color="#000000", alpha=0.9, linewidth=1.0)
# ax2.plot(x_positions, [val + means[i] for i, val in enumerate(stds)], color="#000000", alpha=0.9, linewidth=1.0)

# Generate scaled y-axis ticks
iter_y_axis = y_min_d
original_ticks = np.array([iter_y_axis + i*0.5 for i in range(10)]) # Adjust the number of ticks as needed


scaled_ticks = (original_ticks * scale_factor) + offset



# Set the scaled y-axis ticks and labels
ax2.set_yticks(scaled_ticks, labels = [f"{tick:.2f}" for tick in original_ticks])
print(original_ticks)
print(scaled_ticks)

# # Set the scaled y-axis limits
# ax2.set_ylim(scaled_ylim_bottom, scaled_ylim_top)

# Customize secondary y-axis
ax2.set_ylabel('Discriminator Reward', fontsize=22)
ax2.tick_params(axis='y', labelsize=14)

ax2.set_ylim(y_min_d, y_max_d)



# Set outline colors for each box

title_font_size = 20
label_font_size = 22
tick_font_size = 14
legend_font_size = 13

# Customize the plot
# plt.title('Baseline policy - Node mass 0.01 kg', fontsize=title_font_size, pad=20)
# plt.ylabel('Ground truth rewards', fontsize=label_font_size) 
# plt.xlabel('Cloth node mass [kg]', fontsize=label_font_size)
# plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9], ["0.005", "0.010", "0.030", "0.050","0.075", "0.100", "0.125", "0.150", "0.175"], fontsize=tick_font_size  )

plt.yticks(fontsize=tick_font_size)
# Add legend

# plt.plot([], [], color='Mediumseagreen', label='Finetuned Environment', linewidth=5)
plt.plot([], [], color='Black', label='Diffusion policy', linewidth=5)
plt.plot([], [], color='Orange', label='Unseen Environment', linewidth=5)
plt.plot([], [], color='cornflowerblue', label='Baseline Environment', linewidth=5)
plt.plot([], [], color='firebrick', label='Discriminator Reward w. standard deviation', linewidth=5)
plt.legend(fontsize=legend_font_size,loc='lower right',)  

# Show the plot
plt.savefig("plots/"+ name + ".png")