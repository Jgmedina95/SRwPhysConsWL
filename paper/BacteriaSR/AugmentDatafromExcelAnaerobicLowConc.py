import pandas as pd
import numpy as np


data = pd.read_excel('20240401_LactateVariation_LowerConc_02.xlsx',header=[0,1],sheet_name=1)

conc= [0,10,25,50,75,100,125,150,175,200]   #Because of issues with float values, we will use the concentration*10 first to create the columns, and then divide by 10 at the end 
#rename columns
print(data.keys())
for i in conc:
    #rename columns from {i} uM to {i}
    data[i] = data[i] - data['Control']
averages_df = data.groupby(level=0, axis=1).mean()
std_df = data.groupby(level=0, axis=1).std()

averages_df = averages_df.drop('Control', axis=1)
std_df = std_df.drop('Control', axis=1)
#add standard deviation to the dataframe
averages_df = averages_df.join(std_df, lsuffix='_mean', rsuffix='_std')
averages_df['Time'] = np.linspace(0, 48, 193)*60 #minutes
mean_columns = [col for col in averages_df.columns if 'mean' in col]
std_columns = [col for col in averages_df.columns if 'std' in col]

# Reshape the DataFrame for averages
df_means = pd.melt(averages_df, id_vars=['Time'], value_vars=mean_columns,
                   var_name='Conc', value_name='Avg')
df_means['Conc'] = df_means['Conc'].str.replace('_mean', '').astype(int)

# Reshape the DataFrame for standard deviations
df_stds = pd.melt(averages_df, id_vars=['Time'], value_vars=std_columns,
                  var_name='Conc', value_name='Std')
df_stds['Conc'] = df_stds['Conc'].str.replace('_std', '').astype(int)

# Merge the two DataFrames on 'Time' and 'Concentration'
df_combined = pd.merge(df_means, df_stds, on=['Time', 'Conc'])

# Optionally, sort by 'Time' and 'Concentration' if needed
df_combined.sort_values(by=['Time', 'Conc'], inplace=True)

# Reset index if desired
df_combined.reset_index(drop=True, inplace=True)

aerobic_df = pd.DataFrame(columns=['Time', 'QDConc', 'Value'])

N=5 #number of samples to draw for augmentation

# Iterate over each row and sample N points using the average and std dev
for index, row in df_combined.iterrows():
    samples = np.random.normal(loc=row['Avg'], scale=row['Std'], size=N)
    temp_df = pd.DataFrame({
        'Time': row['Time'],
        'Conc': row['Conc'],
        'Value': samples
    })
    aerobic_df = pd.concat([aerobic_df, temp_df], ignore_index=True)

# sampled_df now has the structure you want
#print(sampled_df.head(20))
aerobic_df['Anaerobic'] = 0
aerobic_df['Temperature'] = 303.15   #30 degrees celsius
#divide QDConc by 10 
aerobic_df['Conc'] = aerobic_df['Conc']/10
aerobic_df.to_csv('Aerobic_AugmentedLowConc_03.csv', index=True)
