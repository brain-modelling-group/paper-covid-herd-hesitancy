import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("paper", font_scale=1.8)

filename = '../inputs/aus_air_vaccine_data.csv'
df  = pd.read_csv(filename, parse_dates=['date'])
this_date = "2021-08-17"
df_sub = df[(df["date"] == this_date)]


f, ax = plt.subplots(figsize=(10,5.5))


ages_lb = np.array(df_sub["age_lb"][:-1])
ages_lb[0] -= 1 
ages_labels = ["16-19", "20-24", "25-29",
               "30-34", "35-39", "40-44",
               "45-49", "50-54", "55-59",
               "60-64", "65-69", "70-74",
               "75-79", "80-84", "85-89",
               "90-94", "95+"]

ax.bar(ages_lb, 100*np.ones(ages_lb.shape) , width=3.9, color="black", alpha=0.05)
ax.bar(ages_lb, np.array(df_sub["first_dose_perc"][:-1]), width=3.9, alpha=0.5, label="partially vaccinated", color="#02b5b9")
ax.bar(ages_lb, np.array(df_sub["second_dose_perc"][:-1]), width=3.9, alpha=0.5, label="fully vaccinated", color="#144387")

ax.plot([0, 100],[20,20] , ls=":", color="black", alpha=0.5)
ax.plot([0, 100],[50,50] , ls=":", color="black", alpha=0.5)
ax.plot([0, 100],[80,80] , ls=":", color="black", alpha=0.5)

ax.set_xlim([12, 98])
ax.set_ylabel("% vaccinated \n(relative to group size)")
ax.set_xlabel("age group")
ax.set_xticks(ages_lb)
ax.set_xticklabels(ages_labels, rotation=45)
title_string = f"QLD age-specific vaccination coverage on {this_date}"
ax.set_title(title_string)

handler1, label1 = ax.get_legend_handles_labels()
ax.legend(handler1, label1, loc="upper right", frameon=True)

f.tight_layout()
plt.show()
