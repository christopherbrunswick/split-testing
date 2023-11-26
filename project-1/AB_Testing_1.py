#!/usr/bin/env python
# coding: utf-8

# # Split Testing On A Kaggle Dataset
# 
# Article Used: https://kishantongs.medium.com/a-b-testing-kaggle-dataset-469f3ecd2da7#:~:text=Data%20Collection%3A%20A%2FB%20tests,measurements%20depending%20on%20the%20objective. 
# 
# Article Used: https://towardsdatascience.com/ab-testing-with-python-e5964dd66143
# 
# 1. <strong>Definition</strong>: A/B testing, also known as split testing, is a method of comparing two versions of a webpage, app, email, or other marketing assets to determine which one performs better. It’s a crucial technique used in data-driven decision-making to optimize various aspects of a product or marketing strategy.
# 
# 
# 2. <strong>Objective</strong>: A/B testing is conducted to improve specific key performance indicators (KPIs) or metrics, such as click-through rates, conversion rates, revenue, or any other measurable goal. The objective is to identify changes that lead to a statistically significant improvement in these metrics.
# 
# 
# 3. <strong>Two Variations</strong>: A/B testing involves creating two versions of the asset you want to test: the control (A) and the variant (B). The control is typically the existing or current version, while the variant is the modified version with the changes you want to test.
# 
# 4. <strong>Random Assignment</strong>: Users or a sample of users are randomly assigned to one of the two groups: the control group or the variant group. This randomization helps ensure that the groups are comparable and that any differences in performance can be attributed to the changes you’ve made.
# 
# 5. <strong>Testing Period</strong>: The control and variant versions are simultaneously presented to their respective groups for a specific testing period. During this time, data on user interactions and behavior are collected.
# 
# 6. <strong>Data Collection</strong>: A/B tests collect data on how each group interacts with the asset. This data can include clicks, conversions, purchases, engagement metrics, or any other relevant measurements depending on the objective.
# 
# 7. <strong>Statistical Analysis</strong>: After the testing period, statistical analysis is performed to compare the performance of the control and variant groups. Common statistical tests used include t-tests, chi-square tests, or regression analysis, depending on the type of data and the metric being tested.
# 
# 8. <strong>Statistical Significance</strong>: The analysis determines whether the observed differences between the two groups are statistically significant. In other words, it assesses whether the changes made in the variant group had a meaningful impact on the chosen metrics or if the differences could have occurred by random chance.
# 
# 9. <strong>Interpretation</strong>: If the test shows that the variant group outperforms the control group with statistical significance, it suggests that the changes made in the variant are likely beneficial. If there’s no significant difference or if the control group performs better, it may indicate that the changes are not effective.
# 
# 10. <strong>Implementation</strong>: If the variant proves to be better, the changes are often implemented on a broader scale, such as across the entire website, app, or marketing campaign.
# 
# 11. <strong>Iterative Process</strong>: A/B testing is an iterative process. The insights gained from one test can inform future tests and refinements, leading to continuous optimization.
# 
# 12. <strong>Ethical Considerations</strong>: It’s essential to conduct A/B testing ethically and transparently, ensuring that users’ privacy and trust are respected. Clearly communicate when users are part of an experiment, and consider the potential impact on user experience.
# 
# 13. <strong>Sample Size and Duration</strong>: Determining an appropriate sample size and duration for the test is crucial to ensure the results are statistically valid. Factors like traffic volume and the expected effect size influence these decisions.
# 
# 14. <strong>Multiple Variants</strong>: While A/B testing compares two variations (A and B), multivariate testing involves testing multiple variations of elements simultaneously to determine the best combination.

# # Potential Scenario 

# ### Baseline Conversion Rate and Lift
# 
# Let’s imagine you work on the product team at a medium-sized online e-commerce business. The UX designer worked really hard on a new version of the product page, with the hope that it will lead to a higher conversion rate. The product manager (PM) told you that the current conversion rate is about 15% on average throughout the year, and that the team would be happy with an increase of 3%, meaning that the new design will be considered a success if it raises the conversion rate to 18%.

# ### Forming Hypothesis
# 
# First things first, we want to make sure we formulate a hypothesis at the start of our project. This will make sure our interpretation of the results is correct as well as rigorous.
# 
# Given we don’t know if the new design will perform better or worse (or the same?) as our current design, we’ll choose a <strong>two-tailed test</strong>:
# 
# <strong>Hₒ: p = pₒ</strong>
# 
# <strong>Hₐ: p ≠ pₒ</strong>
# 
# where p and pₒ stand for the conversion rate of the new and old design, respectively. We’ll also set a confidence level of 95%:
# 
# <strong>α = 0.05</strong>
# 
# The α value is a threshold we set, by which we say “if the probability of observing a result as extreme or more (p-value) is lower than α, then we reject the Null hypothesis”. Since our α=0.05 (indicating 5% probability), our confidence (1 — α) is 95%.
# 
# Don’t worry if you are not familiar with the above, all this really means is that whatever conversion rate we observe for our new design in our test, we want to be 95% confident it is statistically different from the conversion rate of our old design, before we decide to reject the Null hypothesis Hₒ.

# ### Choosing Variables
# 
# For our test we’ll need two groups:
# 
# A <strong>control group</strong> - They'll be shown the old design
# 
# A <strong>treatment (or experimental) group</strong> - They'll be shown the new design
# 
# This will be our Independent Variable. The reason we have two groups even though we know the baseline conversion rate is that we want to control for other variables that could have an effect on our results, such as seasonality: by having a control group we can directly compare their results to the treatment group, because the only systematic difference between the groups is the design of the product page, and we can therefore attribute any differences in results to the designs.
# 
# For our Dependent Variable (i.e. what we are trying to measure), we are interested in capturing the conversion rate. A way we can code this is by each user session with a binary variable:
# 
# 
# <strong>0</strong> - The user did not buy the product during this user session
# 
# <strong>1</strong> - The user bought the product during this user session
# 
# This way, we can easily calculate the mean for each group to get the conversion rate of each design.

# ### Choosing Sample Size
# 
# It is important to note that since we won’t test the whole user base (our population), the conversion rates that we’ll get will inevitably be only estimates of the true rates.
# 
# The number of people (or user sessions) we decide to capture in each group will have an effect on the precision of our estimated conversion rates: the larger the sample size, the more precise our estimates (i.e. the smaller our confidence intervals), the higher the chance to detect a difference in the two groups, if present.
# 
# On the other hand, the larger our sample gets, the more expensive (and impractical) our study becomes.
# 
# So how many people should we have in each group?
# 
# The sample size we need is estimated through something called Power analysis, and it depends on a few factors:
# 
# <strong>Power of the test (1 — β)</strong> — This represents the probability of finding a statistical difference between the groups in our test when a difference is actually present. This is usually set at 0.8 by convention (here’s more info on statistical power, if you are curious)
# 
# 
# <strong>Alpha value (α)</strong> — The critical value we set earlier to 0.05
# 
# 
# <strong>Effect size</strong> — How big of a difference we expect there to be between the conversion rates
# 
# 
# Since our team would be happy with a difference of 3%, we can use 15% and 18% to calculate the effect size we expect.

# In[1]:


# Packages imports
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

get_ipython().run_line_magic('matplotlib', 'inline')

# Some plot styling preferences
plt.style.use('seaborn-whitegrid')
font = {'family' : 'Helvetica',
        'weight' : 'bold',
        'size'   : 14}

mpl.rc('font', **font)
effect_size = sms.proportion_effectsize(0.15, 0.18)    # Calculating effect size based on our expected rates

required_n = sms.NormalIndPower().solve_power(
    effect_size, 
    power=0.8, 
    alpha=0.05, 
    ratio=1
    )                                                  # Calculating sample size needed

required_n = ceil(required_n)                          # Rounding up to next whole number                          

print(required_n)


# For each group (control and experiment) we would need <strong>at least 2399 observations</strong> based on our power analysis. In practice, with our power parameter set to 0.8, we have about 80% chance to detect the actual difference in our conversion rate as statistically significant in our test with the sample size we calculated.

# ### Collecting and Preparing Data

# ### 1. Read data into pandas dataframe
# ### 2. Check and clean data

# In[2]:


df = pd.read_csv('E:/datasets/ab_data.csv')
df.head(5)


# In[3]:


df.info()


# In[3]:


df.describe().T


# In[4]:


df.isnull().sum()


# ### Visualizing random variables

# In[14]:


# finding outliers of this random variable
import logging
logging.getLogger('matplotlib.font_manager').disabled=True

def visualize(df, feature):
    data = df[feature]
    plt.subplot(121)
    plot_1 = sns.countplot(df, x=feature)
    return plot_1


# In[15]:


visualize(df, 'group')


# In[30]:


df['group'].unique()


# In[16]:


visualize(df, 'landing_page')


# In[31]:


df['landing_page'].unique()


# In[17]:


visualize(df, 'converted')


# In[32]:


df['converted'].unique()


# In[23]:


df['converted'].value_counts()


# In[24]:


df['converted'].value_counts().sum()


# In[7]:


#no duplicate indices
df.duplicated().sum()


# In[5]:


pd.crosstab(df['group'], df['landing_page'])


# In[8]:


#users that may have been sampled multiple times
session_counts = df['user_id'].value_counts(ascending=False)
multi_users = session_counts[session_counts > 1].count()
print(f'There are {multi_users} users that appear multiple times in the dataset')


# ### 3. Randomly sample 2399 observations from the dataframe for each group

# We don't want to sample the same users twice so lets remove the users that appear more than once 

# In[9]:


users_to_drop = session_counts[session_counts > 1].index
df = df[~df['user_id'].isin(users_to_drop)]
print(f'There are now {df.shape[0]} entries')


# In[10]:


control_sample = df[df['group'] == 'control'].sample(n=required_n, random_state=42)
treatment_sample = df[df['group'] == 'treatment'].sample(n=required_n, random_state=42)

ab_test = pd.concat([control_sample, treatment_sample], axis=0)
ab_test.reset_index(drop=True, inplace=True)
ab_test


# In[11]:


ab_test.info()


# In[13]:


ab_test['group'].value_counts()


# ### Visualizing Results

# In[14]:


conversion_rates = ab_test.groupby('group')['converted']

std_p = lambda x: np.std(x, ddof=0)              # Std. deviation of the proportion
se_p = lambda x: stats.sem(x, ddof=0)            # Std. error of the proportion (std / sqrt(n))

conversion_rates = conversion_rates.agg([np.mean, std_p, se_p])
conversion_rates.columns = ['conversion_rate', 'std_deviation', 'std_error']


conversion_rates.style.format('{:.3f}')


# There is a noticeable difference between our two designs. Our new design performed much better than our old design.

# In[20]:


get_ipython().system('pip install logger')


# In[23]:


import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

plt.figure(figsize=(8,6))

sns.barplot(x=ab_test['group'], y=ab_test['converted'])

plt.ylim(0, 0.17)
plt.title('Conversion rate by group', pad=20)
plt.xlabel('Group', labelpad=15)
plt.ylabel('Converted (proportion)', labelpad=15);


# A difference can visually be seen now let us test to see if the difference is statistically significant

# ### Testing Our Hypothesis

# In[24]:


from statsmodels.stats.proportion import proportions_ztest, proportion_confint
control_results = ab_test[ab_test['group'] == 'control']['converted']
treatment_results = ab_test[ab_test['group'] == 'treatment']['converted']
n_con = control_results.count()
n_treat = treatment_results.count()
successes = [control_results.sum(), treatment_results.sum()]
nobs = [n_con, n_treat]

z_stat, pval = proportions_ztest(successes, nobs=nobs)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05)

print(f'z statistic: {z_stat:.2f}')
print(f'p-value: {pval:.3f}')
print(f'ci 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'ci 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')


# ### Conclusion 

# Since 0.153 > 0.05, we cannot reject the Null hypothesis Hₒ, which means that our new design did not perform significantly different (let alone better) than our old one. If we look at the confidence interval for our treatment group neither our baseline conversion rate or target conversion rate is within the interval, however the interval is closer to the baseline conversion rate of 15% which may indicate that it is more likely that the true conversion rate of the new design is similar to our baseline, rather than the 18% target conversion rate. Therefore, our new design is definitely not an improvement of our old design.
