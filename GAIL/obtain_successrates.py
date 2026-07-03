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
"GAIL/Saved_models/validation_mass_finetuned_80_steps/mass_randomization_best/logs/GAIL_CLOTH_TASK_26-06-14_11-51-23",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/mass_randomization_best/logs/GAIL_CLOTH_TASK_26-06-14_11-54-11",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/mass_randomization_best/logs/GAIL_CLOTH_TASK_26-06-14_11-56-44",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/mass_randomization_best/logs/GAIL_CLOTH_TASK_26-06-14_11-59-05",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/mass_randomization_best/logs/GAIL_CLOTH_TASK_26-06-14_12-01-34",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/mass_randomization_best/logs/GAIL_CLOTH_TASK_26-06-14_12-04-15",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/mass_randomization_best/logs/GAIL_CLOTH_TASK_26-06-14_12-07-02",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/mass_randomization_best/logs/GAIL_CLOTH_TASK_26-06-14_12-09-44",
"GAIL/Saved_models/validation_mass_finetuned_80_steps/mass_randomization_best/logs/GAIL_CLOTH_TASK_26-06-14_12-12-25",
]

diffusion_policies = [
    
"GAIL/Saved_models/diffusion_policies/500_training_steps/diffusion_model_mass_envs/logs/GAIL_CLOTH_TASK_26-06-12_09-49-42",
"GAIL/Saved_models/diffusion_policies/500_training_steps/diffusion_model_mass_envs/logs/GAIL_CLOTH_TASK_26-06-12_09-54-49",
"GAIL/Saved_models/diffusion_policies/500_training_steps/diffusion_model_mass_envs/logs/GAIL_CLOTH_TASK_26-06-12_09-58-42",
"GAIL/Saved_models/diffusion_policies/500_training_steps/diffusion_model_mass_envs/logs/GAIL_CLOTH_TASK_26-06-12_10-02-57",
"GAIL/Saved_models/diffusion_policies/500_training_steps/diffusion_model_mass_envs/logs/GAIL_CLOTH_TASK_26-06-12_10-07-07",
"GAIL/Saved_models/diffusion_policies/500_training_steps/diffusion_model_mass_envs/logs/GAIL_CLOTH_TASK_26-06-12_10-11-20",
"GAIL/Saved_models/diffusion_policies/500_training_steps/diffusion_model_mass_envs/logs/GAIL_CLOTH_TASK_26-06-12_10-14-40",
"GAIL/Saved_models/diffusion_policies/500_training_steps/diffusion_model_mass_envs/logs/GAIL_CLOTH_TASK_26-06-12_10-17-41",
"GAIL/Saved_models/diffusion_policies/500_training_steps/diffusion_model_mass_envs/logs/GAIL_CLOTH_TASK_26-06-12_10-20-40",
]




# directories = [
# "IRL_cloth_size_domain_randomization_data/validation_of_cloth_size_policies/data_cloth_size_randomization_policy/logs/GAIL_CLOTH_TASK_26-06-03_19-42-27",
# "IRL_cloth_size_domain_randomization_data/validation_of_cloth_size_policies/data_cloth_size_randomization_policy/logs/GAIL_CLOTH_TASK_26-06-03_19-47-10",
# "IRL_cloth_size_domain_randomization_data/validation_of_cloth_size_policies/data_cloth_size_randomization_policy/logs/GAIL_CLOTH_TASK_26-06-03_19-53-50",
# "IRL_cloth_size_domain_randomization_data/validation_of_cloth_size_policies/data_cloth_size_randomization_policy/logs/GAIL_CLOTH_TASK_26-06-03_20-01-44",
# "IRL_cloth_size_domain_randomization_data/validation_of_cloth_size_policies/data_cloth_size_randomization_policy/logs/GAIL_CLOTH_TASK_26-06-03_20-09-48",
# "IRL_cloth_size_domain_randomization_data/validation_of_cloth_size_policies/data_cloth_size_randomization_policy/logs/GAIL_CLOTH_TASK_26-06-03_20-17-17",
# "IRL_cloth_size_domain_randomization_data/validation_of_cloth_size_policies/data_cloth_size_randomization_policy/logs/GAIL_CLOTH_TASK_26-06-03_20-24-20",
# "IRL_cloth_size_domain_randomization_data/validation_of_cloth_size_policies/data_cloth_size_randomization_policy/logs/GAIL_CLOTH_TASK_26-06-03_20-30-56",
# "IRL_cloth_size_domain_randomization_data/validation_of_cloth_size_policies/data_cloth_size_randomization_policy/logs/GAIL_CLOTH_TASK_26-06-03_20-37-57",
# ]

# diffusion_policies = [
    
# "GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_10-52-10",
# "GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_10-56-24",
# "GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_11-00-38",
# "GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_11-05-26",
# "GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_11-10-00",
# "GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_11-14-37",
# "GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_11-18-42",
# "GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_11-23-00",
# "GAIL/Saved_models/diffusion_policies/339_training_steps/cloth_size/logs/GAIL_CLOTH_TASK_26-06-14_11-27-04",

# ]





add_diffusion_policy_results = True 


name = "test_plot"

diffusion_model = [load_data_from_tag(dirs, "environment/reward_env_end" ) for dirs in diffusion_policies]

boxplot_values_env_end = [load_data_from_tag(dirs, "environment/reward_env_end" ) for dirs in directories]
IRL_values_env_average_end = [load_data_from_tag(dirs, "environment/reward_IRL_avg" ) for dirs in directories]




diffusion_model = [vals for i, vals in enumerate(diffusion_model)]



success_irl = 0
success_diffusion = 0

# cloth size
labels = ["0.025", "0.030", "0.035", "0.040", "0.045", "0.050", "0.055", "0.060", "0.065"]

# # cloth mass
# labels = ["0.005", "0.010", "0.030", "0.050", "0.075", "0.100", "0.125", "0150", "0.175"]


sucessrates_irl = []
sucessrates_diffusion = []

measure_type = "successrate" # [sucessrate, performance_metric]

success_treshold = 0.80


for i, list_of_vals in enumerate(boxplot_values_env_end):

    if measure_type == "successrate":
        
        success_IRL = np.array([int(reward > success_treshold) for reward in list_of_vals])
        success_Diffusion = np.array([int(reward > success_treshold)  for reward in diffusion_model[i]])


        success_IRL = np.sum(success_IRL)/len(success_IRL)


        success_Diffusion = np.sum(success_Diffusion)/len(success_Diffusion)

    elif measure_type == "performance_metric":

        success_IRL = np.sum(list_of_vals)/len(list_of_vals)


        success_Diffusion = np.sum(diffusion_model[i])/len(diffusion_model[i])


    print("\n")
    print(f"environment: {labels[i]}")
    print(f"success IRL -- ({success_treshold}): {success_IRL}", )
    print(f"success diffusion -- ({success_treshold}): {success_Diffusion}", )
    print("------")


    sucessrates_irl.append(success_IRL)
    sucessrates_diffusion.append(success_Diffusion)


print("avg success rate IRL: ", np.mean(np.array(sucessrates_irl)))
print("avg success rate diffusion: ", np.mean(np.array(sucessrates_diffusion)))
