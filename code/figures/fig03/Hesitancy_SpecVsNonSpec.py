#!/usr/bin/env python
# coding: utf-8

"""
Figure 03
"""
import numpy as np
import matplotlib.pyplot as plt

# Mock up fig. 4 - https://melbourneinstitute.unimelb.edu.au/publications/research-insights/ttpn/vaccination-report
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,5.5))

# Group by date
date_labels = ["Aug 2020", "Jan 2021", "Apr 2021"]

#Categorize by age
age_labels = ['18-24','25-34','35-44','45-54','55-64','65+'] 


AgeGroupSize = [466430, 733009, 679427, 669171, 605806, 831881]
AgeGroupFraction = np.divide(AgeGroupSize,np.sum(AgeGroupSize))

# Data of not willing to vaccinate
not_willing_18_24   = np.multiply([9.2, 14.5, 15.00],AgeGroupFraction[0])
not_willing_25_34 =   np.multiply([17.10, 29.80,  16.20],AgeGroupFraction[1])
not_willing_35_44   =  np.multiply([11.70, 24.70, 17.40],AgeGroupFraction[2])
not_willing_45_54   =  np.multiply([16.30, 27.00, 15.30],AgeGroupFraction[3])
not_willing_55_64   =  np.multiply([10.40, 12.10, 13.30],AgeGroupFraction[4])
not_willing_65_plus   = np.multiply( [8.00, 9.40, 8.00],AgeGroupFraction[5])




xticks = np.arange(len(date_labels)) 
width = 0.15  # the width of the bars
width_bars = 0.14
d1 = ax[0].bar(xticks-2.5*width, not_willing_18_24   , width_bars, color="#66c2a5",  label=age_labels[0])
d2 = ax[0].bar(xticks-1.5*width, not_willing_25_34 ,   width_bars, color="#fc8d62", label=age_labels[1])
d3 = ax[0].bar(xticks-1/2*width, not_willing_35_44  , width_bars, color="#8da0cb", label=age_labels[2])
d4 = ax[0].bar(xticks+1/2*width, not_willing_45_54   , width_bars, color="#c26683",  label=age_labels[3])
d5 = ax[0].bar(xticks+1.5*width, not_willing_55_64 ,   width_bars, color="#a566c2", label=age_labels[4])
d6 = ax[0].bar(xticks+2.5*width, not_willing_65_plus  , width_bars, color="#fad232", label=age_labels[5])
# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Proportion of Adult Population \n Who are Hesitant (\%)')
ax[0].set_title('Hesitancy by Age Group')
ax[0].set_xticks(xticks)
ax[0].set_xticklabels(date_labels)
ax[0].set_ylim([0, 8])
ax[0].legend()

ax[0].bar_label(d1, labels = np.round(not_willing_18_24,2), padding=3)
ax[0].bar_label(d2, labels = np.round(not_willing_25_34,2), padding=3)
ax[0].bar_label(d3, labels = np.round(not_willing_35_44,2), padding=3)
ax[0].bar_label(d4, labels = np.round(not_willing_45_54,2), padding=3)
ax[0].bar_label(d5, labels = np.round(not_willing_55_64,2), padding=3)
ax[0].bar_label(d6, labels = np.round(not_willing_65_plus,2), padding=3)

## schematic
NonSpec_Hes = 1/6*(not_willing_18_24[2] + not_willing_25_34[2] + 
    not_willing_35_44[2]+not_willing_45_54[2]+not_willing_55_64[2]+not_willing_65_plus[2])
# Empirical data of not willing to vaccinate
not_willing_18_24_ax1    =   [not_willing_18_24[1], NonSpec_Hes]
not_willing_25_34_ax1    =   [not_willing_25_34[1], NonSpec_Hes]
not_willing_35_44_ax1   =  [not_willing_35_44[1], NonSpec_Hes]
not_willing_45_54_ax1    =   [not_willing_45_54[1], NonSpec_Hes]
not_willing_55_64_ax1    =   [not_willing_55_64[1], NonSpec_Hes]
not_willing_65_plus_ax1   =  [not_willing_65_plus[1],  NonSpec_Hes]

hesitancy_labels = ["Specific", "Non Specific"]
xticks_ax1 = np.arange(len(hesitancy_labels))  

ax[1].bar(xticks_ax1-2.5*width, not_willing_18_24_ax1   , width_bars, color="#66c2a5",  label=age_labels[0])
ax[1].bar(xticks_ax1-1.5*width, not_willing_25_34_ax1 ,   width_bars, color="#fc8d62", label=age_labels[1])
ax[1].bar(xticks_ax1-1/2*width, not_willing_35_44_ax1  , width_bars, color="#8da0cb", label=age_labels[2])
ax[1].bar(xticks_ax1+1/2*width, not_willing_45_54_ax1   , width_bars, color="#c26683",  label=age_labels[3])
ax[1].bar(xticks_ax1+1.5*width, not_willing_55_64_ax1 ,   width_bars, color="#a566c2", label=age_labels[4])
ax[1].bar(xticks_ax1+2.5*width, not_willing_65_plus_ax1  , width_bars, color="#fad232", label=age_labels[5])



# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('Proportion of Adult Population \n Who are Hesitant (\%)')
ax[1].set_title('Hesitancy Distribution Across Age Groups')
ax[1].set_xticks(xticks_ax1)
ax[1].set_xticklabels(hesitancy_labels)
ax[1].set_ylim([0, 8])
ax[1].legend()


fig.tight_layout()
plt.savefig('Hesitancy_Spec_vs_NonSpec.pdf')
plt.show()
totals = not_willing_18_24+not_willing_25_34+not_willing_35_44+not_willing_45_54+not_willing_55_64+not_willing_65_plus
total_Aug = totals[0]
total_Jan = totals[1]
total_Apr = totals[2]